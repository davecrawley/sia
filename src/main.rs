use eframe::{egui, egui::Vec2};
use egui::{Align2, Color32, FontFamily, FontId, RichText, TextStyle};
use egui_plot::{Corner, Legend, Line, Plot, PlotBounds, PlotPoints, Text};
use sia::clock::{Clock, SystemClock};
use sia::collection::{Collection, Collector};
use sia::model::{GpuClock, MetricDescriptor, MetricKind, SampleStatus};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

#[derive(Default, Clone)]
struct RollingSeries {
    points: VecDeque<[f64; 2]>,
    cap: usize,
}

impl RollingSeries {
    fn new(cap: usize) -> Self {
        Self {
            points: VecDeque::with_capacity(cap),
            cap,
        }
    }

    fn push(&mut self, x: f64, y: f64) {
        if self.points.len() == self.cap {
            self.points.pop_front();
        }
        self.points.push_back([x, y]);
    }

    fn points_after_scaled(&self, x_min: f64, divisor: f64) -> PlotPoints {
        PlotPoints::from(
            self.points
                .iter()
                .filter(|point| point[0] >= x_min)
                .map(|point| [point[0], point[1] / divisor])
                .collect::<Vec<_>>(),
        )
    }

    fn min_max_y(&self, x_min: f64, x_max: f64) -> Option<(f64, f64)> {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for [x, y] in &self.points {
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
        self.points.back().map(|point| point[1])
    }
}

struct SensorItem {
    name: String,
    idx: usize,
    visible: bool,
    color: Color32,
}

struct SensorGroup {
    key: String,
    display: String,
    items: Vec<SensorItem>,
    warn: f64,
    hot: f64,
}

fn classify(raw: &str) -> (String, String, f64, f64) {
    let lower = raw.to_lowercase();
    let (key, display, warn, hot) = if lower.contains("coretemp")
        || lower.contains("k10temp")
        || lower.contains("zen")
        || lower.contains("cpu")
    {
        ("cpu", "CPU", 90.0, 100.0)
    } else if lower.contains("amdgpu") {
        ("gpu", "GPU (amdgpu)", 85.0, 95.0)
    } else if lower.contains("nvidia") || lower.contains("gpu") {
        ("gpu", "GPU (nvidia)", 85.0, 95.0)
    } else if lower.contains("nvme") {
        ("nvme", "NVMe SSD", 70.0, 80.0)
    } else if lower.contains("spd") {
        ("ramspd", "Memory (SPD Hub)", 70.0, 85.0)
    } else if lower.contains("iwlwifi") {
        ("wifi", "Wi‑Fi Controller (iwlwifi)", 80.0, 90.0)
    } else if lower.contains("r8169") {
        ("eth", "Ethernet Controller (r8169)", 80.0, 90.0)
    } else if lower.contains("igc") {
        ("eth", "Ethernet Controller (igc)", 80.0, 90.0)
    } else if lower.contains("e1000") {
        ("eth", "Ethernet Controller (e1000)", 80.0, 90.0)
    } else if lower.contains("r8125") {
        ("eth", "Ethernet Controller (r8125)", 80.0, 90.0)
    } else if lower.contains("acpitz") {
        ("acpi", "System Temperature (acpitz)", 80.0, 95.0)
    } else if lower.contains("pch") || lower.contains("isa") {
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

fn group_rank(key: &str) -> u8 {
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

fn build_groups(descriptors: &[MetricDescriptor]) -> Vec<SensorGroup> {
    let mut map: BTreeMap<String, SensorGroup> = BTreeMap::new();
    for (idx, descriptor) in descriptors.iter().enumerate() {
        if descriptor.provider == "nvml" {
            continue;
        }
        if let MetricKind::Temperature {
            sensor_name,
            sensor_label,
        } = &descriptor.kind
        {
            let (key, display, warn, hot) = classify(sensor_name);
            let group = map.entry(key.clone()).or_insert(SensorGroup {
                key,
                display: display.clone(),
                items: Vec::new(),
                warn,
                hot,
            });
            group.items.push(SensorItem {
                name: nice_label(&display, sensor_label),
                idx,
                visible: false,
                color: Color32::WHITE,
            });
        }
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
        for (index, item) in group.items.iter_mut().enumerate() {
            item.color = tint(theme_color(&group.key), index as f32 * 0.08);
        }
        if group.display.starts_with("CPU") {
            group.items.sort_by_key(|item| {
                let name = item.name.to_lowercase();
                let tier = if name.contains("package") || name.contains("composite") {
                    0
                } else if name.contains("cpu (core ") {
                    1
                } else {
                    3
                };
                let index = name
                    .split("cpu (core ")
                    .nth(1)
                    .and_then(|rest| rest.split(')').next())
                    .and_then(|number| number.parse::<i32>().ok())
                    .unwrap_or(i32::MAX);
                (tier, index, name)
            });
        } else if group.display.starts_with("GPU") {
            group.items.sort_by_key(|item| {
                let name = item.name.to_lowercase();
                let tier = if name.contains("edge") {
                    0
                } else if name.contains("hotspot") {
                    1
                } else {
                    2
                };
                (tier, item.name.clone())
            });
        } else {
            group.items.sort_by(|a, b| a.name.cmp(&b.name));
        }
    }
    let mut groups: Vec<_> = map.into_values().collect();
    // Preserve the baseline's additional first-NVIDIA-device temperature line.
    for (idx, descriptor) in descriptors.iter().enumerate() {
        if descriptor.provider != "nvml"
            || !matches!(&descriptor.kind, MetricKind::Temperature { .. })
        {
            continue;
        }
        let item = SensorItem {
            name: "GPU (Core)".into(),
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

struct App {
    collection: Box<dyn Collection>,
    descriptors: Vec<MetricDescriptor>,
    series: Vec<RollingSeries>,
    frequency_visible: Vec<bool>,
    frequency_colors: Vec<Color32>,
    groups: Vec<SensorGroup>,
    start: Instant,
    origin_ns: u64,
    seconds: f64,
    samples: usize,
    sample_period: Duration,
    last_tick: Instant,
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
    fn new(collection: Box<dyn Collection>, origin_ns: u64) -> Self {
        let descriptors: Vec<_> = collection
            .capabilities()
            .into_iter()
            .map(|capability| capability.descriptor)
            .collect();
        let groups = build_groups(&descriptors);
        let palette = palette();
        let mut cpu_index = 0;
        let frequency_colors = descriptors
            .iter()
            .map(|descriptor| {
                if matches!(&descriptor.kind, MetricKind::CpuFrequency { .. }) {
                    let color = palette[cpu_index % palette.len()];
                    cpu_index += 1;
                    color
                } else {
                    Color32::WHITE
                }
            })
            .collect();
        let frequency_visible = descriptors
            .iter()
            .map(|descriptor| {
                !matches!(&descriptor.kind, MetricKind::GpuFrequency(GpuClock::Video))
            })
            .collect();
        let series = descriptors
            .iter()
            .map(|_| RollingSeries::new(300))
            .collect();
        Self {
            collection,
            descriptors,
            series,
            frequency_visible,
            frequency_colors,
            groups,
            start: Instant::now(),
            origin_ns,
            seconds: 0.0,
            samples: 0,
            sample_period: Duration::from_secs(1),
            last_tick: Instant::now(),
            display_window_secs: 120.0,
            legend_place: LegendPlacement::Footer,
            ui_font_size: 14.0,
            ui_font_color: Color32::WHITE,
            pending_ui_font_size: 14.0,
            pending_ui_font_color: Color32::WHITE,
            live_font_preview: false,
            gpu_mem_effective: false,
        }
    }

    fn sample(&mut self) {
        let batch = match self.collection.collect() {
            Ok(batch) => batch,
            Err(_) => return,
        };
        self.seconds = batch.observation_ns.saturating_sub(self.origin_ns) as f64 / 1e9;
        self.samples += 1;
        for sample in batch.samples {
            if let Some(index) = self
                .descriptors
                .iter()
                .position(|descriptor| descriptor.same_series(&sample.descriptor))
            {
                let time = sample.mono_ns.saturating_sub(self.origin_ns) as f64 / 1e9;
                let value = if sample.status == SampleStatus::Ok {
                    sample.value.unwrap_or(f64::NAN)
                } else {
                    f64::NAN
                };
                // Model frequencies use canonical Hz. The monitor retains its
                // existing kHz/MHz histories and GHz display conversions.
                let value = match &sample.descriptor.kind {
                    MetricKind::CpuFrequency { .. } => value / 1_000.0,
                    MetricKind::GpuFrequency(_) => value / 1_000_000.0,
                    _ => value,
                };
                self.series[index].push(time, value);
            }
        }
    }

    fn index_of(&self, kind: &MetricKind) -> Option<usize> {
        self.descriptors
            .iter()
            .position(|descriptor| &descriptor.kind == kind)
    }

    fn last_value(&self, kind: &MetricKind) -> f64 {
        self.index_of(kind)
            .and_then(|index| self.series[index].last_y())
            .unwrap_or(0.0)
    }

    fn plot_utilization(&self, ui: &mut egui::Ui, xmin: f64, xmax: f64) {
        ui.heading("Utilization");
        Plot::new("util")
            .height(220.0)
            .allow_scroll(true)
            .allow_zoom(true)
            .legend(Legend::default().position(Corner::LeftTop))
            .show(ui, |plot_ui| {
                plot_ui.set_plot_bounds(PlotBounds::from_min_max([xmin, 0.0], [xmax, 100.0]));
                for tick in 0..=4 {
                    let value = tick as f64 * 25.0;
                    plot_ui.text(
                        Text::new([xmin, value].into(), format!("{value:.0}%"))
                            .anchor(Align2::LEFT_CENTER),
                    );
                    plot_ui.text(
                        Text::new([xmax, value].into(), format!("{value:.0}%"))
                            .anchor(Align2::RIGHT_CENTER),
                    );
                }
                for (kind, name, color) in [
                    (MetricKind::CpuUtilization, "CPU %", "cpu"),
                    (MetricKind::GpuUtilization, "GPU %", "gpu"),
                    (MetricKind::RamOccupancy, "RAM %", "ramspd"),
                    (MetricKind::VramOccupancy, "VRAM %", "nvme"),
                ] {
                    let points = self
                        .index_of(&kind)
                        .map(|index| self.series[index].points_after_scaled(xmin, 1.0))
                        .unwrap_or_else(|| PlotPoints::from(Vec::<[f64; 2]>::new()));
                    plot_ui.line(Line::new(points).name(name).color(theme_color(color)));
                }
            });
    }

    fn plot_temperatures(&self, ui: &mut egui::Ui, xmin: f64, xmax: f64) {
        ui.heading("Temperatures (°C)");
        Plot::new("temps")
            .height(260.0)
            .allow_scroll(true)
            .allow_zoom(true)
            .show(ui, |plot_ui| {
                let mut min = f64::INFINITY;
                let mut max = f64::NEG_INFINITY;
                for group in &self.groups {
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
                    for item in &group.items {
                        if item.visible {
                            plot_ui.line(
                                Line::new(self.series[item.idx].points_after_scaled(xmin, 1.0))
                                    .name(format!("{}: {}", group.display, item.name))
                                    .color(item.color),
                            );
                        }
                    }
                }
                for tick in 0..=4 {
                    let value = min + (max - min) * tick as f64 / 4.0;
                    plot_ui.text(
                        Text::new([xmax, value].into(), format!("{value:.0}"))
                            .anchor(Align2::RIGHT_CENTER),
                    );
                }
            });
    }

    fn frequency_divisor(&self, kind: &MetricKind) -> Option<f64> {
        match kind {
            MetricKind::CpuFrequency { .. } => Some(1_000_000.0),
            MetricKind::GpuFrequency(GpuClock::Memory) if self.gpu_mem_effective => Some(500.0),
            MetricKind::GpuFrequency(_) => Some(1_000.0),
            _ => None,
        }
    }

    fn plot_frequencies(&self, ui: &mut egui::Ui, xmin: f64, xmax: f64) {
        ui.heading("Frequencies (GHz)");
        Plot::new("freq")
            .height(240.0)
            .allow_scroll(true)
            .allow_zoom(true)
            .show(ui, |plot_ui| {
                let mut min = f64::INFINITY;
                let mut max = f64::NEG_INFINITY;
                for (index, descriptor) in self.descriptors.iter().enumerate() {
                    if !self.frequency_visible[index] {
                        continue;
                    }
                    if let Some(divisor) = self.frequency_divisor(&descriptor.kind) {
                        if let Some((a, b)) = self.series[index].min_max_y(xmin, xmax) {
                            min = min.min(a / divisor);
                            max = max.max(b / divisor);
                        }
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
                for (index, descriptor) in self.descriptors.iter().enumerate() {
                    if !self.frequency_visible[index] {
                        continue;
                    }
                    if let Some(divisor) = self.frequency_divisor(&descriptor.kind) {
                        let label = if descriptor.kind == MetricKind::GpuFrequency(GpuClock::Memory)
                            && self.gpu_mem_effective
                        {
                            "GPU Memory (effective)"
                        } else {
                            &descriptor.display_name
                        };
                        let mut line =
                            Line::new(self.series[index].points_after_scaled(xmin, divisor))
                                .name(label);
                        if matches!(&descriptor.kind, MetricKind::CpuFrequency { .. }) {
                            line = line.color(self.frequency_colors[index]);
                        }
                        plot_ui.line(line);
                    }
                }
                for tick in 0..=4 {
                    let value = min + (max - min) * tick as f64 / 4.0;
                    plot_ui.text(
                        Text::new([xmax, value].into(), format!("{value:.2} GHz"))
                            .anchor(Align2::RIGHT_CENTER),
                    );
                }
            });
    }

    fn legend_items(&self, ui: &mut egui::Ui) {
        for group in &self.groups {
            for item in &group.items {
                if !item.visible {
                    continue;
                }
                let mut text = item.name.clone();
                let value = self.series[item.idx].last_y().unwrap_or(f64::NAN);
                if value >= group.hot {
                    text.push_str(" 🔥");
                } else if value >= group.warn {
                    text.push_str(" 🥵");
                }
                ui.horizontal(|ui| {
                    ui.colored_label(item.color, "●");
                    ui.label(text);
                });
            }
        }
    }

    fn legends(&self, ui: &mut egui::Ui) {
        match self.legend_place {
            LegendPlacement::Footer => {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("Legend:").strong());
                    self.legend_items(ui);
                });
            }
            LegendPlacement::Side => {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(RichText::new("Legend").strong());
                        self.legend_items(ui);
                    });
                });
            }
        }
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
                    let cpu = group.display.starts_with("CPU");
                    let gpu = group.display.starts_with("GPU");
                    let title = if cpu {
                        "CPU"
                    } else if gpu {
                        "GPU"
                    } else {
                        &group.display
                    };
                    let id = if cpu {
                        "grp_cpu".to_owned()
                    } else if gpu {
                        "grp_gpu".to_owned()
                    } else {
                        format!("grp_other_{}", group.display)
                    };
                    egui::CollapsingHeader::new(title)
                        .id_source(id)
                        .default_open(false)
                        .show(ui, |ui| {
                            if cpu || gpu {
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
                                                    RichText::new(if cpu {
                                                        "Core temperatures"
                                                    } else {
                                                        "Temperatures"
                                                    })
                                                    .strong(),
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
                                                    RichText::new(if cpu {
                                                        "Core frequencies"
                                                    } else {
                                                        "Frequencies"
                                                    })
                                                    .strong(),
                                                );
                                                if cpu {
                                                    ui.horizontal(|ui| {
                                                        let all = ui.button("All").clicked();
                                                        let none = ui.button("None").clicked();
                                                        if all || none {
                                                            for (index, descriptor) in
                                                                self.descriptors.iter().enumerate()
                                                            {
                                                                if matches!(
                                                                    &descriptor.kind,
                                                                    MetricKind::CpuFrequency { .. }
                                                                ) {
                                                                    self.frequency_visible[index] =
                                                                        all;
                                                                }
                                                            }
                                                        }
                                                    });
                                                }
                                                for (index, descriptor) in
                                                    self.descriptors.iter().enumerate()
                                                {
                                                    let show = if cpu {
                                                        matches!(
                                                            &descriptor.kind,
                                                            MetricKind::CpuFrequency { .. }
                                                        )
                                                    } else {
                                                        matches!(
                                                            &descriptor.kind,
                                                            MetricKind::GpuFrequency(_)
                                                        )
                                                    };
                                                    if show {
                                                        if descriptor.kind
                                                            == MetricKind::GpuFrequency(
                                                                GpuClock::Memory,
                                                            )
                                                        {
                                                            ui.checkbox(
                                                                &mut self.gpu_mem_effective,
                                                                "Show memory as effective (x2)",
                                                            );
                                                        }
                                                        ui.checkbox(
                                                            &mut self.frequency_visible[index],
                                                            &descriptor.display_name,
                                                        );
                                                    }
                                                }
                                            },
                                        );
                                    },
                                );
                            } else {
                                for item in &mut group.items {
                                    ui.checkbox(&mut item.visible, &item.name);
                                }
                            }
                        });
                    ui.end_row();
                }
            });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut style: egui::Style = (*ctx.style()).clone();
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
                ui.label(format!("Samples: {}", self.samples));
                ui.separator();
                ui.label(format!(
                    "CPU: {:.0}%",
                    self.last_value(&MetricKind::CpuUtilization)
                ));
                ui.separator();
                ui.label(format!(
                    "RAM: {:.0}%",
                    self.last_value(&MetricKind::RamOccupancy)
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
            self.plot_utilization(ui, xmin, xmax);
            ui.separator();
            self.plot_temperatures(ui, xmin, xmax);
            ui.separator();
            self.plot_frequencies(ui, xmin, xmax);
            self.legends(ui);
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
    let mut clock = SystemClock::new();
    let origin_ns = clock.now_ns().expect("monotonic clock unavailable");
    let collection = Collector::new(
        sia::providers::local_providers(),
        clock,
        Duration::from_secs(1),
    );
    eframe::run_native(
        "SIA - System Information Analyzer",
        options,
        Box::new(move |_cc| Ok(Box::new(App::new(Box::new(collection), origin_ns)))),
    )
}
