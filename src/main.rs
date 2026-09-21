use eframe::{egui, egui::Vec2};
use egui::{Align2, Color32, FontFamily, FontId, RichText, TextStyle};
use egui_plot::{Corner, Legend, Line, Plot, PlotBounds, PlotPoints, Text};
use sia::clock::MonotonicClock;
use sia::provider::HostProvider;
use sia::{Clock, Collection, Collector, MetricDescriptor, MonitorRole};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

#[cfg(feature = "nvidia")]
use sia::GpuClock;

#[derive(Default, Clone)]
struct RollingSeries {
    xs: VecDeque<f64>,
    ys: VecDeque<f64>,
    cap: usize,
}

impl RollingSeries {
    fn new(cap: usize) -> Self {
        Self {
            xs: VecDeque::with_capacity(cap),
            ys: VecDeque::with_capacity(cap),
            cap,
        }
    }

    fn push(&mut self, x: f64, y: f64) {
        if self.xs.len() == self.cap {
            self.xs.pop_front();
            self.ys.pop_front();
        }
        self.xs.push_back(x);
        self.ys.push_back(y);
    }

    fn points_after(&self, x_min: f64) -> PlotPoints {
        self.points_after_scaled(x_min, 1.0)
    }

    fn points_after_scaled(&self, x_min: f64, div: f64) -> PlotPoints {
        let mut out = Vec::with_capacity(self.xs.len());
        for (x, y) in self.xs.iter().zip(self.ys.iter()) {
            if *x >= x_min {
                out.push([*x, *y / div]);
            }
        }
        PlotPoints::from(out)
    }

    fn min_max_y(&self, x_min: f64, x_max: f64) -> Option<(f64, f64)> {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for (x, y) in self.xs.iter().zip(self.ys.iter()) {
            if *x >= x_min && *x <= x_max && y.is_finite() {
                min = min.min(*y);
                max = max.max(*y);
            }
        }
        if min.is_finite() && max.is_finite() {
            Some((min, max))
        } else {
            None
        }
    }

    fn last_y(&self) -> Option<f64> {
        self.ys.back().copied().filter(|value| value.is_finite())
    }
}

#[derive(Clone, Debug)]
struct SensorItem {
    name: String,
    idx: usize,
    visible: bool,
    color: Color32,
}

#[derive(Clone, Debug)]
struct SensorGroup {
    key: String,
    display: String,
    items: Vec<SensorItem>,
    visible: bool,
    warn: f64,
    hot: f64,
}

struct FrequencyItem {
    idx: usize,
    core: usize,
    visible: bool,
    color: Color32,
}

fn classify(raw: &str) -> (String, String, f64, f64) {
    let raw_lower = raw.to_lowercase();
    let r = raw_lower.as_str();
    let (key, display, warn, hot) = if r.contains("coretemp")
        || r.contains("k10temp")
        || r.contains("zen")
        || r.contains("cpu")
    {
        ("cpu", "CPU", 90.0, 100.0)
    } else if r.contains("amdgpu") {
        ("gpu", "GPU (amdgpu)", 85.0, 95.0)
    } else if r.contains("nvidia") || r.contains("gpu") {
        ("gpu", "GPU (nvidia)", 85.0, 95.0)
    } else if r.contains("nvme") {
        ("nvme", "NVMe SSD", 70.0, 80.0)
    } else if r.contains("spd") {
        ("ramspd", "Memory (SPD Hub)", 70.0, 85.0)
    } else if r.contains("iwlwifi") {
        ("wifi", "Wi‑Fi Controller (iwlwifi)", 80.0, 90.0)
    } else if r.contains("r8169") {
        ("eth", "Ethernet Controller (r8169)", 80.0, 90.0)
    } else if r.contains("igc") {
        ("eth", "Ethernet Controller (igc)", 80.0, 90.0)
    } else if r.contains("e1000") {
        ("eth", "Ethernet Controller (e1000)", 80.0, 90.0)
    } else if r.contains("r8125") {
        ("eth", "Ethernet Controller (r8125)", 80.0, 90.0)
    } else if r.contains("acpitz") {
        ("acpi", "System Temperature (acpitz)", 80.0, 95.0)
    } else if r.contains("pch") || r.contains("isa") {
        ("chipset", "Chipset", 85.0, 95.0)
    } else {
        (raw, raw, 90.0, 100.0)
    };
    (key.into(), display.into(), warn, hot)
}

fn nice_label(group: &str, raw: &str) -> String {
    let lower = raw.to_lowercase();
    match group {
        "CPU" => {
            if lower.contains("package") {
                "CPU (Package)".into()
            } else if lower.contains("tctl") || lower.contains("tdie") {
                "CPU (Composite)".into()
            } else if lower.starts_with("core ") {
                raw.replace("Core ", "CPU (Core ") + ")"
            } else {
                raw.into()
            }
        }
        "GPU (amdgpu)" | "GPU (nvidia)" => {
            if lower.contains("edge") {
                "GPU (Edge)".into()
            } else if lower.contains("hotspot") {
                "GPU (Hotspot)".into()
            } else {
                raw.into()
            }
        }
        "NVMe SSD" => raw.replace("Composite", "SSD"),
        _ => raw.into(),
    }
}

fn palette() -> Vec<Color32> {
    vec![
        Color32::from_rgb(244, 67, 54),
        Color32::from_rgb(33, 150, 243),
        Color32::from_rgb(76, 175, 80),
        Color32::from_rgb(255, 152, 0),
        Color32::from_rgb(156, 39, 176),
        Color32::from_rgb(121, 85, 72),
        Color32::from_rgb(63, 81, 181),
        Color32::from_rgb(0, 150, 136),
        Color32::from_rgb(205, 220, 57),
        Color32::from_rgb(233, 30, 99),
        Color32::from_rgb(158, 158, 158),
        Color32::from_rgb(255, 87, 34),
        Color32::from_rgb(3, 169, 244),
        Color32::from_rgb(139, 195, 74),
        Color32::from_rgb(171, 71, 188),
        Color32::from_rgb(255, 238, 88),
        Color32::from_rgb(38, 198, 218),
        Color32::from_rgb(141, 110, 99),
        Color32::from_rgb(120, 144, 156),
    ]
}

fn theme_color(kind: &str) -> Color32 {
    match kind {
        "cpu" => Color32::from_rgb(244, 67, 54),
        "gpu" => Color32::from_rgb(33, 150, 243),
        "nvme" => Color32::from_rgb(255, 152, 0),
        "ramspd" => Color32::from_rgb(76, 175, 80),
        "wifi" => Color32::from_rgb(0, 150, 136),
        "eth" => Color32::from_rgb(171, 71, 188),
        _ => Color32::from_rgb(158, 158, 158),
    }
}

fn tint(color: Color32, factor: f32) -> Color32 {
    let tint_channel = |value: u8| {
        let value = value as f32;
        (value + (255.0 - value) * factor).clamp(0.0, 255.0) as u8
    };
    Color32::from_rgba_unmultiplied(
        tint_channel(color.r()),
        tint_channel(color.g()),
        tint_channel(color.b()),
        color.a(),
    )
}

fn group_rank(key: &str) -> i32 {
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

fn cpu_sensor_key(name: &str) -> (u8, i32, String) {
    let lower = name.to_lowercase();
    let tier = if lower.contains("package") || lower.contains("composite") {
        0
    } else if lower.contains("cpu (core ") {
        1
    } else {
        3
    };
    let mut index = i32::MAX;
    if let Some(start) = lower.find("cpu (core ") {
        if let Some(end) = lower[start + 11..].find(')') {
            index = lower[start + 11..start + 11 + end]
                .parse()
                .unwrap_or(i32::MAX);
        }
    }
    (tier, index, lower)
}

fn build_groups(descriptors: &[MetricDescriptor]) -> Vec<SensorGroup> {
    let mut map: BTreeMap<String, SensorGroup> = BTreeMap::new();
    for (idx, descriptor) in descriptors.iter().enumerate() {
        let MonitorRole::Temperature { source_name, label } = &descriptor.role else {
            continue;
        };
        // Preserve the baseline's separately added NVML core-temperature line.
        if descriptor.provider == "nvml" {
            continue;
        }
        let (key, display, warn, hot) = classify(source_name);
        let entry = map.entry(key.clone()).or_insert(SensorGroup {
            key,
            display: display.clone(),
            items: vec![],
            visible: true,
            warn,
            hot,
        });
        entry.items.push(SensorItem {
            name: nice_label(&display, label),
            idx,
            visible: false,
            color: Color32::WHITE,
        });
    }
    for group in map.values_mut() {
        let mut showed = false;
        for item in &mut group.items {
            let name = item.name.to_lowercase();
            let preferred =
                name.contains("composite") || name.contains("package") || name.contains("core)");
            if !showed
                && (preferred
                    || group.display.contains("Wi‑Fi Controller")
                    || group.display.contains("Ethernet Controller")
                    || group.display.contains("System Temperature"))
            {
                item.visible = true;
                showed = true;
            }
        }
        if !showed {
            if let Some(first) = group.items.first_mut() {
                first.visible = true;
            }
        }
        let base = theme_color(&group.key);
        for (index, item) in group.items.iter_mut().enumerate() {
            item.color = tint(base, index as f32 * 0.08);
        }
        if group.display.starts_with("CPU") {
            group
                .items
                .sort_by(|a, b| cpu_sensor_key(&a.name).cmp(&cpu_sensor_key(&b.name)));
        } else if group.display.starts_with("GPU") {
            fn tier(name: &str) -> u8 {
                let name = name.to_lowercase();
                if name.contains("edge") {
                    0
                } else if name.contains("hotspot") {
                    1
                } else {
                    2
                }
            }
            group
                .items
                .sort_by(|a, b| tier(&a.name).cmp(&tier(&b.name)).then(a.name.cmp(&b.name)));
        } else {
            group.items.sort_by(|a, b| a.name.cmp(&b.name));
        }
    }
    let mut groups: Vec<_> = map.into_values().collect();
    for (idx, descriptor) in descriptors.iter().enumerate() {
        let MonitorRole::Temperature { label, .. } = &descriptor.role else {
            continue;
        };
        if descriptor.provider != "nvml" {
            continue;
        }
        let item = SensorItem {
            name: label.clone(),
            idx,
            visible: true,
            color: Color32::WHITE,
        };
        if let Some(group) = groups.iter_mut().find(|group| group.key == "gpu") {
            group.items.push(item);
        } else {
            groups.push(SensorGroup {
                key: "gpu".into(),
                display: "GPU".into(),
                items: vec![item],
                visible: true,
                warn: 85.0,
                hot: 95.0,
            });
        }
    }
    groups.sort_by_key(|group| group_rank(&group.key));
    groups
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LegendPlacement {
    Footer,
    Side,
}

struct FrequencyLine {
    idx: usize,
    name: String,
    divisor: f64,
    color: Option<Color32>,
}

struct App {
    start: Instant,
    origin_ns: u64,
    collector: Box<dyn Collection>,
    descriptors: Vec<MetricDescriptor>,
    series: Vec<RollingSeries>,
    cpu_util: usize,
    ram_util: usize,
    gpu_util: usize,
    vram_util: usize,
    frequencies: Vec<FrequencyItem>,
    groups: Vec<SensorGroup>,
    seconds: f64,
    sample_count: usize,
    sample_period: Duration,
    last_tick: Instant,
    display_window_secs: f64,
    legend_place: LegendPlacement,
    ui_font_size: f32,
    ui_font_color: Color32,
    pending_ui_font_size: f32,
    pending_ui_font_color: Color32,
    live_font_preview: bool,
    #[cfg(feature = "nvidia")]
    gpu_frequency_visible: [bool; 4],
    #[cfg(feature = "nvidia")]
    gpu_mem_effective: bool,
}

impl App {
    fn new(capacity: usize, sample_hz: f64) -> Self {
        let sample_period = Duration::from_secs_f64((1.0 / sample_hz).max(0.05));
        let provider = HostProvider::new();
        let clock = MonotonicClock::new();
        let origin_ns = clock.now().ns;
        let collector = Box::new(Collector::new(provider, clock, sample_period));
        let descriptors = collector.descriptors().to_vec();
        let index_for = |role| {
            descriptors
                .iter()
                .position(|descriptor| descriptor.role == role)
                .expect("Host provider must declare the four monitor utilization roles")
        };
        let cpu_util = index_for(MonitorRole::CpuUtilization);
        let ram_util = index_for(MonitorRole::RamUtilization);
        let gpu_util = index_for(MonitorRole::GpuUtilization);
        let vram_util = index_for(MonitorRole::VramUtilization);
        let palette = palette();
        let mut frequencies = Vec::new();
        for (idx, descriptor) in descriptors.iter().enumerate() {
            if let MonitorRole::CpuFrequency { core } = &descriptor.role {
                frequencies.push(FrequencyItem {
                    idx,
                    core: *core,
                    visible: true,
                    color: palette[frequencies.len() % palette.len()],
                });
            }
        }
        let groups = build_groups(&descriptors);
        let series = descriptors
            .iter()
            .map(|_| RollingSeries::new(capacity))
            .collect();
        Self {
            start: Instant::now(),
            origin_ns,
            collector,
            descriptors,
            series,
            cpu_util,
            ram_util,
            gpu_util,
            vram_util,
            frequencies,
            groups,
            seconds: 0.0,
            sample_count: 0,
            sample_period,
            last_tick: Instant::now(),
            display_window_secs: 120.0,
            legend_place: LegendPlacement::Footer,
            ui_font_size: 14.0,
            ui_font_color: Color32::WHITE,
            pending_ui_font_size: 14.0,
            pending_ui_font_color: Color32::WHITE,
            live_font_preview: false,
            #[cfg(feature = "nvidia")]
            gpu_frequency_visible: [true, true, true, false],
            #[cfg(feature = "nvidia")]
            gpu_mem_effective: false,
        }
    }

    fn sample(&mut self) {
        let batch = self.collector.collect();
        self.seconds = batch.observed_at.ns.saturating_sub(self.origin_ns) as f64 / 1e9;
        self.sample_count += 1;
        for (series, sample) in self.series.iter_mut().zip(batch.samples) {
            // Model frequencies are canonical Hz. Retain the existing monitor's
            // kHz CPU and MHz GPU histories and their GHz display conversions.
            let divisor = match sample.descriptor.role {
                MonitorRole::CpuFrequency { .. } => 1000.0,
                MonitorRole::GpuFrequency(_) => 1_000_000.0,
                _ => 1.0,
            };
            let value = sample
                .value
                .map(|value| value / divisor)
                .unwrap_or(f64::NAN);
            let timestamp = if sample.timestamp.clock_domain == batch.observed_at.clock_domain {
                sample.timestamp.ns
            } else {
                sample.observed_at.ns
            };
            let seconds = timestamp.saturating_sub(self.origin_ns) as f64 / 1e9;
            series.push(seconds, value);
        }
    }

    fn frequency_lines(&self) -> Vec<FrequencyLine> {
        let mut lines: Vec<_> = self
            .frequencies
            .iter()
            .filter(|item| item.visible)
            .map(|item| FrequencyLine {
                idx: item.idx,
                name: format!("CPU Core {}", item.core),
                divisor: 1_000_000.0,
                color: Some(item.color),
            })
            .collect();
        // Use the descriptor roles, never a presentation-side device query.
        for (idx, descriptor) in self.descriptors.iter().enumerate() {
            #[cfg(feature = "nvidia")]
            if let MonitorRole::GpuFrequency(clock) = &descriptor.role {
                let (slot, name) = match clock {
                    GpuClock::Graphics => (0, "GPU Graphics"),
                    GpuClock::Sm => (1, "GPU SM"),
                    GpuClock::Memory => (2, "GPU Memory"),
                    GpuClock::Video => (3, "GPU Video"),
                };
                if self.gpu_frequency_visible[slot] {
                    let effective = *clock == GpuClock::Memory && self.gpu_mem_effective;
                    lines.push(FrequencyLine {
                        idx,
                        name: if effective {
                            "GPU Memory (effective)".into()
                        } else {
                            name.into()
                        },
                        divisor: 1000.0 / if effective { 2.0 } else { 1.0 },
                        color: None,
                    });
                }
            }
            #[cfg(not(feature = "nvidia"))]
            let _ = (idx, descriptor);
        }
        lines.shrink_to_fit();
        lines
    }

    fn legend_items(&self, ui: &mut egui::Ui) {
        for group in &self.groups {
            if !group.visible {
                continue;
            }
            for item in &group.items {
                if !item.visible {
                    continue;
                }
                let mut text = item.name.clone();
                let last = self.series[item.idx].last_y();
                let hot = last.map(|value| value >= group.hot).unwrap_or(false);
                let warn = last.map(|value| value >= group.warn).unwrap_or(false);
                if hot {
                    text.push_str(" 🔥");
                } else if warn {
                    text.push_str(" 🥵");
                }
                ui.horizontal(|ui| {
                    ui.colored_label(item.color, "●");
                    ui.label(text);
                });
            }
        }
    }

    fn footer_legend(&self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Legend:").strong());
            self.legend_items(ui);
        });
    }

    fn side_legend(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new("Legend").strong());
                self.legend_items(ui);
            });
        });
    }

    fn display_settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("Display");
        ui.horizontal(|ui| {
            ui.label("Window (seconds) before scroll):");
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
                    ui.selectable_value(&mut self.legend_place, LegendPlacement::Footer, "Footer");
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
    }

    fn sensor_settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("Sensors");
        egui::Grid::new("sensor_grid")
            .num_columns(2)
            .striped(true)
            .min_col_width(500.0)
            .spacing([18.0, 8.0])
            .show(ui, |ui| {
                for group in &mut self.groups {
                    if group.display.starts_with("CPU") {
                        egui::CollapsingHeader::new("CPU")
                            .id_source("grp_cpu")
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
                                                    RichText::new("Core temperatures").strong(),
                                                );
                                                for item in &mut group.items {
                                                    ui.checkbox(&mut item.visible, &item.name);
                                                }
                                            },
                                        );
                                        ui.allocate_ui_with_layout(
                                            egui::vec2(right, 0.0),
                                            layout,
                                            |ui| {
                                                ui.label(
                                                    RichText::new("Core frequencies").strong(),
                                                );
                                                ui.horizontal(|ui| {
                                                    if ui.button("All").clicked() {
                                                        for item in &mut self.frequencies {
                                                            item.visible = true;
                                                        }
                                                    }
                                                    if ui.button("None").clicked() {
                                                        for item in &mut self.frequencies {
                                                            item.visible = false;
                                                        }
                                                    }
                                                });
                                                for item in &mut self.frequencies {
                                                    ui.checkbox(
                                                        &mut item.visible,
                                                        format!("CPU Core {}", item.core),
                                                    );
                                                }
                                            },
                                        );
                                    },
                                );
                            });
                    } else if group.display.starts_with("GPU") {
                        egui::CollapsingHeader::new("GPU")
                            .id_source("grp_gpu")
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
                                                ui.label(RichText::new("Temperatures").strong());
                                                for item in &mut group.items {
                                                    ui.checkbox(&mut item.visible, &item.name);
                                                }
                                            },
                                        );
                                        ui.allocate_ui_with_layout(
                                            egui::vec2(right, 0.0),
                                            layout,
                                            |ui| {
                                                ui.label(RichText::new("Frequencies").strong());
                                                #[cfg(feature = "nvidia")]
                                                {
                                                    ui.checkbox(
                                                        &mut self.gpu_frequency_visible[0],
                                                        "GPU Graphics",
                                                    );
                                                    ui.checkbox(
                                                        &mut self.gpu_frequency_visible[1],
                                                        "GPU SM",
                                                    );
                                                    ui.checkbox(
                                                        &mut self.gpu_mem_effective,
                                                        "Show memory as effective (x2)",
                                                    );
                                                    ui.checkbox(
                                                        &mut self.gpu_frequency_visible[2],
                                                        "GPU Memory",
                                                    );
                                                    ui.checkbox(
                                                        &mut self.gpu_frequency_visible[3],
                                                        "GPU Video",
                                                    );
                                                }
                                            },
                                        );
                                    },
                                );
                            });
                    } else {
                        egui::CollapsingHeader::new(group.display.clone())
                            .id_source(format!("grp_other_{}", group.display))
                            .default_open(false)
                            .show(ui, |ui| {
                                for item in &mut group.items {
                                    ui.checkbox(&mut item.visible, &item.name);
                                }
                            });
                    }
                    ui.end_row();
                }
            });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut style = (*ctx.style()).clone();
        style.visuals.override_text_color = Some(self.ui_font_color);
        style.text_styles = [
            (
                TextStyle::Heading,
                FontId::new(self.ui_font_size, FontFamily::Proportional),
            ),
            (
                TextStyle::Body,
                FontId::new(self.ui_font_size, FontFamily::Proportional),
            ),
            (
                TextStyle::Monospace,
                FontId::new(self.ui_font_size, FontFamily::Monospace),
            ),
            (
                TextStyle::Button,
                FontId::new(self.ui_font_size, FontFamily::Proportional),
            ),
            (
                TextStyle::Small,
                FontId::new(self.ui_font_size, FontFamily::Proportional),
            ),
        ]
        .into();
        ctx.set_style(style);
        if self.last_tick.elapsed() >= self.sample_period {
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
                ui.label(format!("Samples: {}", self.sample_count));
                ui.separator();
                ui.label(format!(
                    "CPU: {:.0}%",
                    self.series[self.cpu_util].last_y().unwrap_or(0.0)
                ));
                ui.separator();
                ui.label(format!(
                    "RAM: {:.0}%",
                    self.series[self.ram_util].last_y().unwrap_or(0.0)
                ));
            });
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.set_min_size(Vec2::new(1200.0, 880.0));
            let (xmin, xmax) = if self.seconds > self.display_window_secs {
                (self.seconds - self.display_window_secs, self.seconds)
            } else {
                (0.0, self.display_window_secs)
            };
            ui.heading("Utilization");
            Plot::new("util")
                .height(220.0)
                .allow_scroll(true)
                .allow_zoom(true)
                .legend(Legend::default().position(Corner::LeftTop))
                .show(ui, |plot_ui| {
                    plot_ui.set_plot_bounds(PlotBounds::from_min_max([xmin, 0.0], [xmax, 100.0]));
                    let mut value = 0.0;
                    while value <= 100.0 + 1e-6 {
                        plot_ui.text(
                            Text::new([xmin, value].into(), format!("{value:.0}%"))
                                .anchor(Align2::LEFT_CENTER),
                        );
                        value += 25.0;
                    }
                    for (index, name, color) in [
                        (self.cpu_util, "CPU %", "cpu"),
                        (self.gpu_util, "GPU %", "gpu"),
                        (self.ram_util, "RAM %", "ramspd"),
                        (self.vram_util, "VRAM %", "nvme"),
                    ] {
                        plot_ui.line(
                            Line::new(self.series[index].points_after(xmin))
                                .name(name)
                                .color(theme_color(color)),
                        );
                    }
                    let mut value = 0.0;
                    while value <= 100.0 + 1e-6 {
                        plot_ui.text(
                            Text::new([xmax, value].into(), format!("{value:.0}%"))
                                .anchor(Align2::RIGHT_CENTER),
                        );
                        value += 25.0;
                    }
                });
            ui.separator();
            ui.heading("Temperatures (°C)");
            Plot::new("temps")
                .height(260.0)
                .allow_scroll(true)
                .allow_zoom(true)
                .show(ui, |plot_ui| {
                    let mut min = f64::INFINITY;
                    let mut max = f64::NEG_INFINITY;
                    for group in &self.groups {
                        if !group.visible {
                            continue;
                        }
                        for item in &group.items {
                            if item.visible {
                                if let Some((a, b)) = self.series[item.idx].min_max_y(xmin, xmax) {
                                    min = min.min(a);
                                    max = max.max(b);
                                }
                            }
                        }
                    }
                    if !min.is_finite() || !max.is_finite() || (max - min).abs() < 1e-6 {
                        min = 0.0;
                        max = 120.0;
                    }
                    let pad = ((max - min) * 0.1).max(2.0);
                    min = (min - pad).max(0.0);
                    max = (max + pad).min(130.0);
                    plot_ui.set_plot_bounds(PlotBounds::from_min_max([xmin, min], [xmax, max]));
                    for group in &self.groups {
                        if !group.visible {
                            continue;
                        }
                        for item in &group.items {
                            if item.visible {
                                plot_ui.line(
                                    Line::new(self.series[item.idx].points_after(xmin))
                                        .name(format!("{}: {}", group.display, item.name))
                                        .color(item.color),
                                );
                            }
                        }
                    }
                    let step = (max - min) / 4.0;
                    let mut value = min;
                    while value <= max + 1e-6 {
                        plot_ui.text(
                            Text::new([xmax, value].into(), format!("{value:.0}"))
                                .anchor(Align2::RIGHT_CENTER),
                        );
                        value += step;
                    }
                });
            ui.separator();
            ui.heading("Frequencies (GHz)");
            let lines = self.frequency_lines();
            Plot::new("freq")
                .height(240.0)
                .allow_scroll(true)
                .allow_zoom(true)
                .show(ui, |plot_ui| {
                    let mut min = f64::INFINITY;
                    let mut max = f64::NEG_INFINITY;
                    for line in &lines {
                        if let Some((a, b)) = self.series[line.idx].min_max_y(xmin, xmax) {
                            min = min.min(a / line.divisor);
                            max = max.max(b / line.divisor);
                        }
                    }
                    if !min.is_finite() || !max.is_finite() || (max - min).abs() < 1e-6 {
                        min = 0.1;
                        max = 10.0;
                    }
                    let pad = ((max - min) * 0.08).max(0.05);
                    min = (min - pad).max(0.0);
                    max = (max + pad).min(12.0);
                    plot_ui.set_plot_bounds(PlotBounds::from_min_max([xmin, min], [xmax, max]));
                    for line in &lines {
                        let mut plot_line = Line::new(
                            self.series[line.idx].points_after_scaled(xmin, line.divisor),
                        )
                        .name(&line.name);
                        if let Some(color) = line.color {
                            plot_line = plot_line.color(color);
                        }
                        plot_ui.line(plot_line);
                    }
                    let step = (max - min) / 4.0;
                    let mut value = min;
                    while value <= max + 1e-6 {
                        plot_ui.text(
                            Text::new([xmax, value].into(), format!("{value:.2} GHz"))
                                .anchor(Align2::RIGHT_CENTER),
                        );
                        value += step;
                    }
                });
            match self.legend_place {
                LegendPlacement::Footer => self.footer_legend(ui),
                LegendPlacement::Side => self.side_legend(ui),
            }
            ui.separator();
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    self.display_settings(ui);
                    ui.separator();
                    self.sensor_settings(ui);
                });
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
        Box::new(|_cc| Ok(Box::new(App::new(5 * 60, 1.0)))),
    )
}
