use std::collections::VecDeque;
use std::time::{Duration, Instant};

use eframe::{egui, egui::Vec2};
use egui::{Align2, Color32, FontFamily, FontId, RichText, TextStyle};
use egui_plot::{Corner, Legend, Line, Plot, PlotBounds, PlotPoints, Text};
use sia::clock::{LinuxClock, Timestamp};
use sia::collection::Collector;
use sia::model::{Capability, MetricDescriptor, Reading, Unavailable, Unit};

const HISTORY_CAPACITY: usize = 300;
const SAMPLE_PERIOD: Duration = Duration::from_secs(1);

#[derive(Default)]
struct RollingSeries {
    points: VecDeque<[f64; 2]>,
}

impl RollingSeries {
    fn push(&mut self, x: f64, y: f64) {
        if self.points.len() == HISTORY_CAPACITY {
            self.points.pop_front();
        }
        self.points.push_back([x, y]);
    }

    fn last(&self) -> Option<f64> {
        self.points.back().map(|point| point[1])
    }

    fn bounds(&self, minimum: f64, scale: f64) -> Option<(f64, f64)> {
        let mut low = f64::INFINITY;
        let mut high = f64::NEG_INFINITY;
        for [x, y] in &self.points {
            if *x >= minimum && y.is_finite() {
                low = low.min(y * scale);
                high = high.max(y * scale);
            }
        }
        if low.is_finite() && high.is_finite() {
            Some((low, high))
        } else {
            None
        }
    }

    fn segments(&self, minimum: f64, scale: f64) -> Vec<PlotPoints> {
        let mut segments = Vec::new();
        let mut points = Vec::new();
        for [x, y] in &self.points {
            if *x < minimum {
                continue;
            }
            if y.is_finite() {
                points.push([*x, *y * scale]);
            } else if !points.is_empty() {
                segments.push(PlotPoints::from(std::mem::take(&mut points)));
            }
        }
        if !points.is_empty() {
            segments.push(PlotPoints::from(points));
        }
        segments
    }
}

struct Track {
    descriptor: MetricDescriptor,
    series: RollingSeries,
    label: String,
    visible: bool,
    color: Color32,
}

impl Track {
    fn drawable(&self) -> bool {
        self.visible
            && !matches!(
                self.descriptor.capability,
                Capability::Unavailable(Unavailable::Unsupported(_))
            )
    }
}

struct SensorGroup {
    key: String,
    display: String,
    temperatures: Vec<usize>,
    frequencies: Vec<usize>,
    warn: f64,
    hot: f64,
}

fn classify(raw: &str) -> (String, String, f64, f64) {
    let name = raw.to_lowercase();
    let (key, display, warn, hot) = if name.contains("coretemp")
        || name.contains("k10temp")
        || name.contains("zen")
        || name.contains("cpu")
    {
        ("cpu", "CPU", 90.0, 100.0)
    } else if name.contains("amdgpu") {
        ("gpu", "GPU (amdgpu)", 85.0, 95.0)
    } else if name.contains("nvidia") || name.contains("gpu") {
        ("gpu", "GPU (nvidia)", 85.0, 95.0)
    } else if name.contains("nvme") {
        ("nvme", "NVMe SSD", 70.0, 80.0)
    } else if name.contains("spd") {
        ("ramspd", "Memory (SPD Hub)", 70.0, 85.0)
    } else if name.contains("iwlwifi") {
        ("wifi", "Wi-Fi Controller (iwlwifi)", 80.0, 90.0)
    } else if name.contains("r8169") {
        ("eth", "Ethernet Controller (r8169)", 80.0, 90.0)
    } else if name.contains("igc") {
        ("eth", "Ethernet Controller (igc)", 80.0, 90.0)
    } else if name.contains("e1000") {
        ("eth", "Ethernet Controller (e1000)", 80.0, 90.0)
    } else if name.contains("r8125") {
        ("eth", "Ethernet Controller (r8125)", 80.0, 90.0)
    } else if name.contains("acpitz") {
        ("acpi", "System Temperature (acpitz)", 80.0, 95.0)
    } else if name.contains("pch") || name.contains("isa") {
        ("chipset", "Chipset", 85.0, 95.0)
    } else {
        (raw, raw, 90.0, 100.0)
    };
    (key.to_string(), display.to_string(), warn, hot)
}

fn nice_label(group: &str, raw: &str) -> String {
    let name = raw.to_lowercase();
    match group {
        "CPU" if name.contains("package") => "CPU (Package)".to_string(),
        "CPU" if name.contains("tctl") || name.contains("tdie") => "CPU (Composite)".to_string(),
        "CPU" if name.starts_with("core ") => raw.replace("Core ", "CPU (Core ") + ")",
        "GPU (amdgpu)" | "GPU (nvidia)" if name.contains("edge") => "GPU (Edge)".to_string(),
        "GPU (amdgpu)" | "GPU (nvidia)" if name.contains("hotspot") => "GPU (Hotspot)".to_string(),
        "NVMe SSD" => raw.replace("Composite", "SSD"),
        _ => raw.to_string(),
    }
}

fn rank(key: &str) -> u8 {
    match key {
        "cpu" => 0,
        "gpu" => 1,
        "nvme" => 2,
        "ramspd" => 3,
        "wifi" => 4,
        "eth" => 5,
        _ => 6,
    }
}

fn theme_color(key: &str) -> Color32 {
    match key {
        "cpu" => Color32::from_rgb(244, 67, 54),
        "gpu" => Color32::from_rgb(33, 150, 243),
        "nvme" => Color32::from_rgb(255, 152, 0),
        "ramspd" => Color32::from_rgb(76, 175, 80),
        "wifi" => Color32::from_rgb(0, 150, 136),
        "eth" => Color32::from_rgb(171, 71, 188),
        _ => Color32::from_rgb(158, 158, 158),
    }
}

fn tint(color: Color32, index: usize) -> Color32 {
    let factor = (index as f32 * 0.08).min(0.8);
    let channel = |value: u8| (value as f32 + (255.0 - value as f32) * factor) as u8;
    Color32::from_rgb(channel(color.r()), channel(color.g()), channel(color.b()))
}

fn frequency_color(index: usize) -> Color32 {
    let colors = [
        [244, 67, 54],
        [33, 150, 243],
        [76, 175, 80],
        [255, 152, 0],
        [156, 39, 176],
        [121, 85, 72],
        [63, 81, 181],
        [0, 150, 136],
        [205, 220, 57],
        [233, 30, 99],
        [158, 158, 158],
        [255, 87, 34],
        [3, 169, 244],
        [139, 195, 74],
        [171, 71, 188],
        [255, 238, 88],
        [38, 198, 218],
        [141, 110, 99],
        [120, 144, 156],
    ];
    let [red, green, blue] = colors[index % colors.len()];
    Color32::from_rgb(red, green, blue)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LegendPlacement {
    Footer,
    Side,
}

struct App {
    collector: Collector<LinuxClock>,
    tracks: Vec<Track>,
    groups: Vec<SensorGroup>,
    origin: Option<Timestamp>,
    seconds: f64,
    samples: u64,
    start: Instant,
    last_tick: Instant,
    error: Option<String>,
    display_window_secs: f64,
    legend_place: LegendPlacement,
    ui_font_size: f32,
    ui_font_color: Color32,
    pending_ui_font_size: f32,
    pending_ui_font_color: Color32,
    live_font_preview: bool,
    gpu_mem_effective: bool,
}

impl App {
    fn new() -> Self {
        let mut app = Self {
            collector: sia::native::collector(),
            tracks: Vec::new(),
            groups: Vec::new(),
            origin: None,
            seconds: 0.0,
            samples: 0,
            start: Instant::now(),
            last_tick: Instant::now(),
            error: None,
            display_window_secs: 120.0,
            legend_place: LegendPlacement::Footer,
            ui_font_size: 14.0,
            ui_font_color: Color32::WHITE,
            pending_ui_font_size: 14.0,
            pending_ui_font_color: Color32::WHITE,
            live_font_preview: false,
            gpu_mem_effective: false,
        };
        app.sample();
        app
    }

    fn register(&mut self, descriptor: MetricDescriptor) {
        if let Some(track) = self.tracks.iter_mut().find(|track| {
            track.descriptor.metric_id == descriptor.metric_id
                && track.descriptor.entity_id == descriptor.entity_id
        }) {
            track.descriptor = descriptor;
            return;
        }
        // Initialization status has no device identity and does not create
        // pretend device traces. It remains available in the typed batch.
        if descriptor.entity_id == "provider:nvidia" {
            return;
        }
        let (key, display, warn, hot) = classify(&descriptor.entity_name);
        let index = self.tracks.len();
        let label = nice_label(&display, &descriptor.display_name);
        let mut color = theme_color(&key);
        if descriptor.metric_id == "memory.occupancy" {
            color = theme_color("ramspd");
        } else if descriptor.metric_id == "gpu.memory.occupancy" {
            color = theme_color("nvme");
        }
        let visible = descriptor.unit != Unit::Celsius && descriptor.metric_id != "gpu.clock.video";
        let unit = descriptor.unit;
        self.tracks.push(Track {
            descriptor,
            series: RollingSeries::default(),
            label,
            visible,
            color,
        });
        if unit != Unit::Celsius && unit != Unit::Hertz {
            return;
        }
        let group_index = match self.groups.iter().position(|group| group.key == key) {
            Some(index) => index,
            None => {
                self.groups.push(SensorGroup {
                    key,
                    display,
                    temperatures: Vec::new(),
                    frequencies: Vec::new(),
                    warn,
                    hot,
                });
                self.groups.len() - 1
            }
        };
        let group = &mut self.groups[group_index];
        if unit == Unit::Celsius {
            self.tracks[index].color = tint(color, group.temperatures.len());
            group.temperatures.push(index);
        } else {
            self.tracks[index].color = frequency_color(group.frequencies.len());
            group.frequencies.push(index);
        }
    }

    fn sample(&mut self) {
        let batch = match self.collector.collect() {
            Ok(batch) => batch,
            Err(error) => {
                self.error = Some(error.to_string());
                for track in &mut self.tracks {
                    track.series.push(self.seconds, f64::NAN);
                }
                return;
            }
        };
        self.error = None;
        let elapsed = self
            .origin
            .as_ref()
            .and_then(|origin| batch.observed_at.elapsed_since(origin));
        if batch.discontinuity || elapsed.is_none() {
            self.origin = Some(batch.observed_at.clone());
            self.seconds = 0.0;
            for track in &mut self.tracks {
                track.series = RollingSeries::default();
            }
        } else if let Some(nanoseconds) = elapsed {
            self.seconds = nanoseconds as f64 / 1_000_000_000.0;
        }
        let previous_groups: Vec<String> =
            self.groups.iter().map(|group| group.key.clone()).collect();
        for descriptor in batch.descriptors {
            self.register(descriptor);
        }
        for group in &mut self.groups {
            if previous_groups.contains(&group.key) {
                continue;
            }
            let preferred = group.temperatures.iter().copied().find(|index| {
                let name = self.tracks[*index].label.to_lowercase();
                name.contains("package") || name.contains("composite") || name.contains("core)")
            });
            if let Some(index) = preferred.or_else(|| group.temperatures.first().copied()) {
                self.tracks[index].visible = true;
            }
        }
        self.groups.sort_by_key(|group| rank(&group.key));
        for track in &mut self.tracks {
            let sample = batch.samples.iter().find(|sample| {
                sample.metric_id == track.descriptor.metric_id
                    && sample.entity_id == track.descriptor.entity_id
            });
            let value = match sample.map(|sample| &sample.reading) {
                Some(Reading::Available(value)) => value.as_f64(),
                _ => f64::NAN,
            };
            track.series.push(self.seconds, value);
        }
        self.samples += 1;
    }

    fn scale(&self, track: &Track) -> f64 {
        if track.descriptor.unit == Unit::Hertz {
            if self.gpu_mem_effective && track.descriptor.metric_id == "gpu.clock.memory" {
                2.0 / 1_000_000_000.0
            } else {
                1.0 / 1_000_000_000.0
            }
        } else {
            1.0
        }
    }

    fn plot(&self, ui: &mut egui::Ui, unit: Unit, id: &str, height: f32) {
        let (minimum, maximum) = if self.seconds > self.display_window_secs {
            (self.seconds - self.display_window_secs, self.seconds)
        } else {
            (0.0, self.display_window_secs)
        };
        let mut low = f64::INFINITY;
        let mut high = f64::NEG_INFINITY;
        for track in &self.tracks {
            if track.descriptor.unit != unit || !track.drawable() {
                continue;
            }
            if let Some((a, b)) = track.series.bounds(minimum, self.scale(track)) {
                low = low.min(a);
                high = high.max(b);
            }
        }
        if unit == Unit::Percent {
            low = 0.0;
            high = 100.0;
        } else {
            if !low.is_finite() || !high.is_finite() || (high - low).abs() < 1e-6 {
                (low, high) = if unit == Unit::Celsius {
                    (0.0, 120.0)
                } else {
                    (0.1, 10.0)
                };
            }
            let padding = if unit == Unit::Celsius {
                ((high - low) * 0.1).max(2.0)
            } else {
                ((high - low) * 0.08).max(0.05)
            };
            low = (low - padding).max(0.0);
            high = (high + padding).min(if unit == Unit::Celsius { 130.0 } else { 12.0 });
        }
        let mut plot = Plot::new(id)
            .height(height)
            .allow_scroll(true)
            .allow_zoom(true);
        if unit == Unit::Percent {
            plot = plot.legend(Legend::default().position(Corner::LeftTop));
        }
        plot.show(ui, |plot_ui| {
            plot_ui.set_plot_bounds(PlotBounds::from_min_max([minimum, low], [maximum, high]));
            for track in &self.tracks {
                if track.descriptor.unit != unit || !track.drawable() {
                    continue;
                }
                let mut label = track.label.clone();
                if self.gpu_mem_effective && track.descriptor.metric_id == "gpu.clock.memory" {
                    label.push_str(" (effective)");
                }
                let duplicates = self
                    .tracks
                    .iter()
                    .filter(|other| other.label == track.label)
                    .count();
                if duplicates > 1 {
                    label.push_str(&format!(" [{}]", track.descriptor.entity_id));
                }
                for points in track.series.segments(minimum, self.scale(track)) {
                    plot_ui.line(Line::new(points).name(&label).color(track.color));
                }
            }
            for tick in 0..=4 {
                let value = low + (high - low) * tick as f64 / 4.0;
                let label = match unit {
                    Unit::Percent => format!("{value:.0}%"),
                    Unit::Hertz => format!("{value:.2} GHz"),
                    _ => format!("{value:.0}"),
                };
                plot_ui
                    .text(Text::new([maximum, value].into(), &label).anchor(Align2::RIGHT_CENTER));
                if unit == Unit::Percent {
                    plot_ui.text(
                        Text::new([minimum, value].into(), label).anchor(Align2::LEFT_CENTER),
                    );
                }
            }
        });
    }

    fn legend_items(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Legend:").strong());
        for group in &self.groups {
            for index in &group.temperatures {
                let track = &self.tracks[*index];
                if !track.drawable() {
                    continue;
                }
                let mut label = track.label.clone();
                if let Some(value) = track.series.last() {
                    if value >= group.hot {
                        label.push_str(" 🔥");
                    } else if value >= group.warn {
                        label.push_str(" 🥵");
                    }
                }
                ui.horizontal(|ui| {
                    ui.colored_label(track.color, "●");
                    let response = ui.label(label);
                    if let Capability::Unavailable(reason) = &track.descriptor.capability {
                        response.on_hover_text(reason.reason());
                    }
                });
            }
        }
    }

    fn settings(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                ui.heading("Display");
                ui.horizontal(|ui| {
                    ui.label("Window (seconds before scroll):");
                    ui.add(egui::Slider::new(
                        &mut self.display_window_secs,
                        30.0..=900.0,
                    ));
                    egui::ComboBox::from_label("Legend placement")
                        .selected_text(match self.legend_place {
                            LegendPlacement::Footer => "Footer",
                            LegendPlacement::Side => "Side",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.legend_place,
                                LegendPlacement::Footer,
                                "Footer",
                            );
                            ui.selectable_value(
                                &mut self.legend_place,
                                LegendPlacement::Side,
                                "Side strip",
                            );
                        });
                    ui.separator();
                    ui.label("Font size");
                    let response = ui.add(egui::Slider::new(
                        &mut self.pending_ui_font_size,
                        10.0..=22.0,
                    ));
                    if self.live_font_preview || response.drag_stopped() {
                        self.ui_font_size = self.pending_ui_font_size;
                    }
                    ui.label("Font color");
                    ui.color_edit_button_srgba(&mut self.pending_ui_font_color);
                    if self.live_font_preview {
                        self.ui_font_color = self.pending_ui_font_color;
                    }
                    ui.separator();
                    if ui.button("Apply font").clicked() {
                        self.ui_font_size = self.pending_ui_font_size;
                        self.ui_font_color = self.pending_ui_font_color;
                    }
                    ui.toggle_value(&mut self.live_font_preview, "Live preview");
                });
                ui.separator();
                ui.heading("Sensors");
                egui::Grid::new("sensor_grid")
                    .num_columns(2)
                    .striped(true)
                    .min_col_width(500.0)
                    .spacing([18.0, 8.0])
                    .show(ui, |ui| {
                        for group in &self.groups {
                            egui::CollapsingHeader::new(&group.display)
                                .id_source(format!("group_{}", group.key))
                                .default_open(false)
                                .show(ui, |ui| {
                                    ui.with_layout(
                                        egui::Layout::left_to_right(egui::Align::TOP),
                                        |ui| {
                                            let inner = (ui.available_width()
                                                - ui.spacing().item_spacing.x)
                                                .max(0.0);
                                            let left = (inner * 0.7).min((inner - 250.0).max(0.0));
                                            let right = (inner - left).max(0.0);
                                            let layout = egui::Layout::top_down(egui::Align::LEFT);
                                            ui.allocate_ui_with_layout(
                                                egui::vec2(left, 0.0),
                                                layout,
                                                |ui| {
                                                    ui.label(
                                                        RichText::new("Temperatures").strong(),
                                                    );
                                                    for index in &group.temperatures {
                                                        track_checkbox(
                                                            ui,
                                                            &mut self.tracks[*index],
                                                        );
                                                    }
                                                },
                                            );
                                            ui.allocate_ui_with_layout(
                                                egui::vec2(right, 0.0),
                                                layout,
                                                |ui| {
                                                    ui.label(RichText::new("Frequencies").strong());
                                                    ui.horizontal(|ui| {
                                                        if ui.button("All").clicked() {
                                                            for index in &group.frequencies {
                                                                self.tracks[*index].visible = true;
                                                            }
                                                        }
                                                        if ui.button("None").clicked() {
                                                            for index in &group.frequencies {
                                                                self.tracks[*index].visible = false;
                                                            }
                                                        }
                                                    });
                                                    if group.key == "gpu" {
                                                        ui.checkbox(
                                                            &mut self.gpu_mem_effective,
                                                            "Show memory as effective (x2)",
                                                        );
                                                    }
                                                    for index in &group.frequencies {
                                                        track_checkbox(
                                                            ui,
                                                            &mut self.tracks[*index],
                                                        );
                                                    }
                                                },
                                            );
                                        },
                                    );
                                });
                            ui.end_row();
                        }
                    });
            });
    }
}

fn track_checkbox(ui: &mut egui::Ui, track: &mut Track) {
    let response = ui.checkbox(&mut track.visible, &track.label);
    if let Capability::Unavailable(reason) = &track.descriptor.capability {
        response.on_hover_text(reason.reason());
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut style: egui::Style = (*ctx.style()).clone();
        style.visuals.override_text_color = Some(self.ui_font_color);
        for text_style in [
            TextStyle::Heading,
            TextStyle::Body,
            TextStyle::Button,
            TextStyle::Small,
        ] {
            style.text_styles.insert(
                text_style,
                FontId::new(self.ui_font_size, FontFamily::Proportional),
            );
        }
        style.text_styles.insert(
            TextStyle::Monospace,
            FontId::new(self.ui_font_size, FontFamily::Monospace),
        );
        ctx.set_style(style);
        if self.last_tick.elapsed() >= SAMPLE_PERIOD {
            self.sample();
            self.last_tick = Instant::now();
        }
        ctx.request_repaint_after(Duration::from_millis(16));
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("SIA - System Information Analyzer - © David Crawley 2025");
                ui.separator();
                ui.label(format!("Uptime: {}s", self.start.elapsed().as_secs()));
                ui.separator();
                ui.label(format!("Samples: {}", self.samples));
                for metric in ["cpu.utilization", "memory.occupancy"] {
                    if let Some(track) = self
                        .tracks
                        .iter()
                        .find(|track| track.descriptor.metric_id == metric)
                    {
                        ui.separator();
                        let value = track.series.last().filter(|value| value.is_finite());
                        let label = track.label.trim_end_matches(" %");
                        ui.label(match value {
                            Some(value) => format!("{label}: {value:.0}%"),
                            None => format!("{label}: unavailable"),
                        });
                    }
                }
            });
            if let Some(error) = &self.error {
                ui.label(error);
            }
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.set_min_size(Vec2::new(1200.0, 880.0));
            ui.heading("Utilization");
            self.plot(ui, Unit::Percent, "util", 220.0);
            ui.separator();
            ui.heading("Temperatures (°C)");
            self.plot(ui, Unit::Celsius, "temps", 260.0);
            ui.separator();
            ui.heading("Frequencies (GHz)");
            self.plot(ui, Unit::Hertz, "freq", 240.0);
            match self.legend_place {
                LegendPlacement::Footer => {
                    ui.horizontal_wrapped(|ui| self.legend_items(ui));
                }
                LegendPlacement::Side => {
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| self.legend_items(ui));
                    });
                }
            }
            ui.separator();
            self.settings(ui);
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1230.0, 1130.0])
            .with_min_inner_size([950.0, 700.0])
            .with_title("SIA - System Information Analyzer - © David Crawley 2025"),
        ..Default::default()
    };
    eframe::run_native(
        "SIA - System Information Analyzer",
        options,
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
}
