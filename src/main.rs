use eframe::{egui, egui::Vec2};
use egui::{Align2, Color32, FontFamily, FontId, RichText, TextStyle};
use egui_plot::{Corner, Legend, Line, Plot, PlotBounds, PlotPoints, Text};
use sia::clock::NativeClock;
use sia::collection::Collector;
use sia::model::{CollectionBatch, MetricDescriptor, Timestamp, Unit};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

#[derive(Default)]
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
        if self.cap == 0 {
            return;
        }
        if self.points.len() == self.cap {
            self.points.pop_front();
        }
        self.points.push_back([x, y]);
    }

    fn has_values(&self) -> bool {
        self.points.iter().any(|point| point[1].is_finite())
    }

    fn last_y(&self) -> Option<f64> {
        self.points
            .back()
            .map(|point| point[1])
            .filter(|value| value.is_finite())
    }

    fn min_max(&self, xmin: f64, xmax: f64, scale: f64) -> Option<(f64, f64)> {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &[x, y] in &self.points {
            if x >= xmin && x <= xmax && y.is_finite() {
                min = min.min(y * scale);
                max = max.max(y * scale);
            }
        }
        if min.is_finite() && max.is_finite() {
            Some((min, max))
        } else {
            None
        }
    }

    fn draw(&self, ui: &mut egui_plot::PlotUi, xmin: f64, scale: f64, name: &str, color: Color32) {
        let mut segment = Vec::new();
        for &[x, y] in &self.points {
            if x < xmin {
                continue;
            }
            if y.is_finite() {
                segment.push([x, y * scale]);
            } else if !segment.is_empty() {
                ui.line(
                    Line::new(PlotPoints::from(std::mem::take(&mut segment)))
                        .name(name)
                        .color(color),
                );
            }
        }
        if !segment.is_empty() {
            ui.line(Line::new(PlotPoints::from(segment)).name(name).color(color));
        }
    }
}

struct ViewMetric {
    descriptor: MetricDescriptor,
    series: RollingSeries,
    visible: bool,
    color: Color32,
}

struct SensorItem {
    name: String,
    index: usize,
    visible: bool,
    color: Color32,
}

struct SensorGroup {
    key: String,
    display: String,
    items: Vec<SensorItem>,
    visible: bool,
    warn: f64,
    hot: f64,
}

fn classify(raw: &str) -> (String, String, f64, f64) {
    let raw_lower = raw.to_lowercase();
    let (key, display, warn, hot) = if raw_lower.contains("coretemp")
        || raw_lower.contains("k10temp")
        || raw_lower.contains("zen")
        || raw_lower.contains("cpu")
    {
        ("cpu", "CPU", 90.0, 100.0)
    } else if raw_lower.contains("amdgpu") {
        ("gpu", "GPU (amdgpu)", 85.0, 95.0)
    } else if raw_lower.contains("nvidia") || raw_lower.contains("gpu") {
        ("gpu", "GPU (nvidia)", 85.0, 95.0)
    } else if raw_lower.contains("nvme") {
        ("nvme", "NVMe SSD", 70.0, 80.0)
    } else if raw_lower.contains("spd") {
        ("ramspd", "Memory (SPD Hub)", 70.0, 85.0)
    } else if raw_lower.contains("iwlwifi") {
        ("wifi", "Wi-Fi Controller (iwlwifi)", 80.0, 90.0)
    } else if raw_lower.contains("r8169") {
        ("eth", "Ethernet Controller (r8169)", 80.0, 90.0)
    } else if raw_lower.contains("igc") {
        ("eth", "Ethernet Controller (igc)", 80.0, 90.0)
    } else if raw_lower.contains("e1000") {
        ("eth", "Ethernet Controller (e1000)", 80.0, 90.0)
    } else if raw_lower.contains("r8125") {
        ("eth", "Ethernet Controller (r8125)", 80.0, 90.0)
    } else if raw_lower.contains("acpitz") {
        ("acpi", "System Temperature (acpitz)", 80.0, 95.0)
    } else if raw_lower.contains("pch") || raw_lower.contains("isa") {
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

fn palette(index: usize) -> Color32 {
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
    let [r, g, b] = colors[index % colors.len()];
    Color32::from_rgb(r, g, b)
}

fn tint(color: Color32, factor: f32) -> Color32 {
    let channel = |value: u8| {
        let value = f32::from(value);
        (value + (255.0 - value) * factor).clamp(0.0, 255.0) as u8
    };
    Color32::from_rgb(channel(color.r()), channel(color.g()), channel(color.b()))
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

fn build_groups(metrics: &[ViewMetric]) -> Vec<SensorGroup> {
    let mut groups: BTreeMap<String, SensorGroup> = BTreeMap::new();
    for (index, metric) in metrics.iter().enumerate() {
        if metric.descriptor.unit != Unit::Celsius || !metric.series.has_values() {
            continue;
        }
        let (key, display, warn, hot) = classify(&metric.descriptor.entity_display_name);
        let group = groups.entry(key.clone()).or_insert_with(|| SensorGroup {
            key,
            display: display.clone(),
            items: Vec::new(),
            visible: true,
            warn,
            hot,
        });
        group.items.push(SensorItem {
            name: nice_label(&display, &metric.descriptor.display_name),
            index,
            visible: false,
            color: Color32::WHITE,
        });
    }
    for (key, display) in [("cpu", "CPU"), ("gpu", "GPU")] {
        if metrics.iter().any(|metric| {
            metric.descriptor.unit == Unit::Hertz
                && metric.series.has_values()
                && (metric.descriptor.metric_id == "cpu.frequency") == (key == "cpu")
        }) {
            groups.entry(key.into()).or_insert_with(|| SensorGroup {
                key: key.into(),
                display: display.into(),
                items: Vec::new(),
                visible: true,
                warn: 90.0,
                hot: 100.0,
            });
        }
    }
    for group in groups.values_mut() {
        let preferred = group
            .items
            .iter()
            .position(|item| {
                let name = item.name.to_lowercase();
                name.contains("package") || name.contains("composite") || name.contains("core)")
            })
            .unwrap_or(0);
        for (index, item) in group.items.iter_mut().enumerate() {
            item.visible = index == preferred || group.key == "gpu";
            item.color = tint(theme_color(&group.key), index as f32 * 0.08);
        }
        group.items.sort_by_key(|item| {
            let name = item.name.to_lowercase();
            let tier = if name.contains("package") || name.contains("composite") {
                0
            } else if name.contains("cpu (core ") || name.contains("edge") {
                1
            } else if name.contains("hotspot") {
                2
            } else {
                3
            };
            let core = name
                .strip_prefix("cpu (core ")
                .and_then(|suffix| suffix.strip_suffix(')'))
                .and_then(|number| number.parse::<usize>().ok())
                .unwrap_or(usize::MAX);
            (tier, core, name)
        });
    }
    let mut groups: Vec<_> = groups.into_values().collect();
    groups.sort_by_key(|group| rank(&group.key));
    groups
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LegendPlacement {
    Footer,
    Side,
}

struct App {
    collector: Collector<NativeClock>,
    metrics: Vec<ViewMetric>,
    groups: Vec<SensorGroup>,
    capacity: usize,
    origin: Option<Timestamp>,
    start: Instant,
    seconds: f64,
    sample_count: usize,
    sample_period: Duration,
    last_tick: Instant,
    collection_error: Option<String>,
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
    fn new(capacity: usize, sample_hz: f64) -> Self {
        let mut app = Self {
            collector: sia::native_collector(),
            metrics: Vec::new(),
            groups: Vec::new(),
            capacity,
            origin: None,
            start: Instant::now(),
            seconds: 0.0,
            sample_count: 0,
            sample_period: Duration::from_secs_f64((1.0 / sample_hz).max(0.05)),
            last_tick: Instant::now(),
            collection_error: None,
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

    fn sample(&mut self) {
        match self.collector.collect() {
            Ok(batch) => self.ingest(batch),
            Err(error) => {
                self.collection_error = Some(error.to_string());
                for metric in &mut self.metrics {
                    metric.series.push(self.seconds, f64::NAN);
                }
            }
        }
    }

    /// The same ingestion path accepts captured production batches without an
    /// eframe window, allowing a verifier to render them in a standalone egui frame.
    fn ingest(&mut self, batch: CollectionBatch) {
        self.collection_error = None;
        let relative = self
            .origin
            .as_ref()
            .and_then(|origin| batch.observed_at.elapsed_since(origin));
        if batch.discontinuity || relative.is_none() {
            self.origin = Some(batch.observed_at.clone());
            self.seconds = 0.0;
            for metric in &mut self.metrics {
                metric.series.points.clear();
            }
        } else if let Some(relative) = relative {
            self.seconds = relative.as_secs_f64();
        }
        self.sample_count += 1;
        for descriptor in batch.descriptors {
            if let Some(metric) = self.metrics.iter_mut().find(|metric| {
                metric.descriptor.metric_id == descriptor.metric_id
                    && metric.descriptor.entity_id == descriptor.entity_id
            }) {
                metric.descriptor = descriptor;
            } else {
                let visible = descriptor.metric_id != "gpu.clock.video";
                self.metrics.push(ViewMetric {
                    descriptor,
                    series: RollingSeries::new(self.capacity),
                    visible,
                    color: palette(self.metrics.len()),
                });
            }
        }
        for metric in &mut self.metrics {
            let sample = batch.samples.iter().find(|sample| {
                sample.metric_id == metric.descriptor.metric_id
                    && sample.entity_id == metric.descriptor.entity_id
            });
            let value = sample
                .and_then(|sample| sample.value.as_ref().ok())
                .and_then(|value| value.as_f64())
                .unwrap_or(f64::NAN);
            let time = sample
                .and_then(|sample| {
                    self.origin
                        .as_ref()
                        .and_then(|origin| sample.observed_at.elapsed_since(origin))
                })
                .map(|elapsed| elapsed.as_secs_f64())
                .unwrap_or(self.seconds);
            metric.series.push(time, value);
        }
        // Rebuild when capability/history changes too, not only when descriptors
        // are first seen: a previously unavailable sensor may have just recovered.
        let mut groups = build_groups(&self.metrics);
        for group in &mut groups {
            if let Some(old) = self.groups.iter().find(|old| old.key == group.key) {
                group.visible = old.visible;
                for item in &mut group.items {
                    if let Some(old_item) = old.items.iter().find(|old| old.index == item.index) {
                        item.visible = old_item.visible;
                        item.color = old_item.color;
                    }
                }
            }
        }
        self.groups = groups;
    }

    fn latest(&self, id: &str) -> String {
        self.metrics
            .iter()
            .filter(|metric| metric.descriptor.metric_id == id)
            .filter(|metric| id != "cpu.utilization" || metric.descriptor.entity_id == "system:cpu")
            .find_map(|metric| metric.series.last_y())
            .map(|value| format!("{value:.0}%"))
            .unwrap_or_else(|| "unavailable".into())
    }

    fn frequency_scale(&self, metric: &ViewMetric) -> f64 {
        if self.gpu_mem_effective && metric.descriptor.metric_id == "gpu.clock.memory" {
            2.0 / 1_000_000_000.0
        } else {
            1.0 / 1_000_000_000.0
        }
    }

    fn utilization(&self, ui: &mut egui::Ui, xmin: f64, xmax: f64) {
        ui.heading("Utilization, capacity occupancy and pressure (%)");
        Plot::new("util")
            .height(220.0)
            .allow_scroll(true)
            .allow_zoom(true)
            .legend(Legend::default().position(Corner::LeftTop))
            .show(ui, |plot| {
                plot.set_plot_bounds(PlotBounds::from_min_max([xmin, 0.0], [xmax, 100.0]));
                for metric in self.metrics.iter().filter(|metric| {
                    metric.visible
                        && metric.descriptor.unit == Unit::Percent
                        && metric.series.has_values()
                }) {
                    metric.series.draw(
                        plot,
                        xmin,
                        1.0,
                        &metric.descriptor.display_name,
                        metric.color,
                    );
                }
                for tick in 0..=4 {
                    let y = f64::from(tick) * 25.0;
                    for (x, anchor) in [(xmin, Align2::LEFT_CENTER), (xmax, Align2::RIGHT_CENTER)] {
                        plot.text(Text::new([x, y].into(), format!("{y:.0}%")).anchor(anchor));
                    }
                }
            });
    }

    fn temperatures(&self, ui: &mut egui::Ui, xmin: f64, xmax: f64) {
        ui.heading("Temperatures (°C)");
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for group in self.groups.iter().filter(|group| group.visible) {
            for item in group.items.iter().filter(|item| item.visible) {
                if let Some((low, high)) = self.metrics[item.index].series.min_max(xmin, xmax, 1.0)
                {
                    min = min.min(low);
                    max = max.max(high);
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
        Plot::new("temps")
            .height(260.0)
            .allow_scroll(true)
            .allow_zoom(true)
            .show(ui, |plot| {
                plot.set_plot_bounds(PlotBounds::from_min_max([xmin, min], [xmax, max]));
                for group in self.groups.iter().filter(|group| group.visible) {
                    for item in group.items.iter().filter(|item| item.visible) {
                        self.metrics[item.index].series.draw(
                            plot,
                            xmin,
                            1.0,
                            &format!("{}: {}", group.display, item.name),
                            item.color,
                        );
                    }
                }
                for tick in 0..=4 {
                    let y = min + (max - min) * f64::from(tick) / 4.0;
                    plot.text(
                        Text::new([xmax, y].into(), format!("{y:.0}")).anchor(Align2::RIGHT_CENTER),
                    );
                }
            });
    }

    fn frequencies(&self, ui: &mut egui::Ui, xmin: f64, xmax: f64) {
        ui.heading("Frequencies (GHz)");
        let frequencies: Vec<_> = self
            .metrics
            .iter()
            .filter(|metric| {
                metric.visible
                    && metric.descriptor.unit == Unit::Hertz
                    && metric.series.has_values()
            })
            .collect();
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for metric in &frequencies {
            if let Some((low, high)) =
                metric
                    .series
                    .min_max(xmin, xmax, self.frequency_scale(metric))
            {
                min = min.min(low);
                max = max.max(high);
            }
        }
        if !min.is_finite() || !max.is_finite() || (max - min).abs() < 1e-6 {
            min = 0.1;
            max = 10.0;
        }
        let pad = ((max - min) * 0.08).max(0.05);
        min = (min - pad).max(0.0);
        max = (max + pad).min(12.0);
        Plot::new("freq")
            .height(240.0)
            .allow_scroll(true)
            .allow_zoom(true)
            .show(ui, |plot| {
                plot.set_plot_bounds(PlotBounds::from_min_max([xmin, min], [xmax, max]));
                for metric in &frequencies {
                    let name = if self.gpu_mem_effective
                        && metric.descriptor.metric_id == "gpu.clock.memory"
                    {
                        format!("{} (effective x2)", metric.descriptor.display_name)
                    } else {
                        metric.descriptor.display_name.clone()
                    };
                    metric.series.draw(
                        plot,
                        xmin,
                        self.frequency_scale(metric),
                        &name,
                        metric.color,
                    );
                }
                for tick in 0..=4 {
                    let y = min + (max - min) * f64::from(tick) / 4.0;
                    plot.text(
                        Text::new([xmax, y].into(), format!("{y:.2} GHz"))
                            .anchor(Align2::RIGHT_CENTER),
                    );
                }
            });
    }

    fn legend_entries(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Legend:").strong());
        for group in self.groups.iter().filter(|group| group.visible) {
            for item in group.items.iter().filter(|item| item.visible) {
                let series = &self.metrics[item.index].series;
                if !series.has_values() {
                    continue;
                }
                let mut text = item.name.clone();
                if let Some(value) = series.last_y() {
                    if value >= group.hot {
                        text.push_str(" 🔥");
                    } else if value >= group.warn {
                        text.push_str(" 🥵");
                    }
                } else {
                    text.push_str(" — unavailable");
                }
                ui.horizontal(|ui| {
                    ui.colored_label(item.color, "●");
                    ui.label(text);
                });
            }
        }
    }

    fn details(&self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Current values and capabilities").show(ui, |ui| {
            for metric in &self.metrics {
                let value = metric
                    .series
                    .last_y()
                    .map(|value| match metric.descriptor.unit {
                        Unit::Bytes => format!("{value:.0} bytes"),
                        Unit::Percent => format!("{value:.2}%"),
                        Unit::Celsius => format!("{value:.1} °C"),
                        Unit::Hertz => format!("{value:.0} Hz"),
                        _ => format!("{value}"),
                    })
                    .unwrap_or_else(|| "unavailable".into());
                ui.label(format!("{}: {}", metric.descriptor.display_name, value))
                    .on_hover_text(format!(
                        "{}\n{:?}\n{:?}",
                        metric.descriptor.source_semantics,
                        metric.descriptor.temporal_semantics,
                        metric.descriptor.capability
                    ));
            }
        });
        let status = self.collector.session_status();
        if status.environment_changed {
            ui.label("Environment changed: device topology or reset. Earlier observations remain in the trace.");
        }
        egui::CollapsingHeader::new("Device lifecycle").show(ui, |ui| {
            for device in &status.devices {
                ui.label(format!(
                    "{} [{}] — {:?}, generation {}; PCI {:?}; driver {} {:?}",
                    device.display_name,
                    device.entity_id,
                    device.state,
                    device.generation,
                    device.pci_address,
                    device.driver,
                    device.driver_version
                ));
            }
            for event in &status.events {
                ui.label(format!(
                    "{} ns: {} generation {} {:?}: {}",
                    event.observed_at.mono_ns,
                    event.entity_id,
                    event.generation,
                    event.state,
                    event.reason
                ));
            }
        });
    }

    fn settings(&mut self, ui: &mut egui::Ui) {
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
                    LegendPlacement::Side => "Side strip",
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
        ui.separator();
        ui.heading("Sensors");
        egui::Grid::new("sensor_grid")
            .num_columns(2)
            .striped(true)
            .min_col_width(500.0)
            .spacing([18.0, 8.0])
            .show(ui, |ui| {
                for group in &mut self.groups {
                    egui::CollapsingHeader::new(&group.display)
                        .id_source(format!("grp_{}", group.key))
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.with_layout(egui::Layout::left_to_right(egui::Align::TOP), |ui| {
                                let inner =
                                    (ui.available_width() - ui.spacing().item_spacing.x).max(0.0);
                                let left = (inner * 0.7).min((inner - 250.0).max(0.0));
                                let layout = egui::Layout::top_down(egui::Align::LEFT);
                                ui.allocate_ui_with_layout(egui::vec2(left, 0.0), layout, |ui| {
                                    ui.label(RichText::new("Temperatures").strong());
                                    for item in &mut group.items {
                                        ui.checkbox(&mut item.visible, &item.name);
                                    }
                                });
                                if group.key == "cpu" || group.key == "gpu" {
                                    ui.allocate_ui_with_layout(
                                        egui::vec2((inner - left).max(0.0), 0.0),
                                        layout,
                                        |ui| {
                                            ui.label(RichText::new("Frequencies").strong());
                                            let is_cpu = group.key == "cpu";
                                            let matches = |metric: &&mut ViewMetric| {
                                                metric.descriptor.unit == Unit::Hertz
                                                    && metric.series.has_values()
                                                    && (metric.descriptor.metric_id
                                                        == "cpu.frequency")
                                                        == is_cpu
                                            };
                                            if is_cpu {
                                                ui.horizontal(|ui| {
                                                    for (label, visible) in
                                                        [("All", true), ("None", false)]
                                                    {
                                                        if ui.button(label).clicked() {
                                                            for metric in self
                                                                .metrics
                                                                .iter_mut()
                                                                .filter(matches)
                                                            {
                                                                metric.visible = visible;
                                                            }
                                                        }
                                                    }
                                                });
                                            } else {
                                                ui.checkbox(
                                                    &mut self.gpu_mem_effective,
                                                    "Show memory as effective (x2)",
                                                );
                                            }
                                            for metric in self.metrics.iter_mut().filter(matches) {
                                                ui.checkbox(
                                                    &mut metric.visible,
                                                    &metric.descriptor.display_name,
                                                );
                                            }
                                        },
                                    );
                                }
                            });
                        });
                    ui.end_row();
                }
            });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut style = (*ctx.style()).clone();
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
                ui.label(format!("CPU: {}", self.latest("cpu.utilization")));
                ui.separator();
                ui.label(format!("RAM: {}", self.latest("memory.occupancy")));
            });
            if let Some(error) = &self.collection_error {
                ui.label(error);
            }
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.set_min_size(Vec2::new(1200.0, 880.0));
            let (xmin, xmax) = if self.seconds > self.display_window_secs {
                (self.seconds - self.display_window_secs, self.seconds)
            } else {
                (0.0, self.display_window_secs)
            };
            self.utilization(ui, xmin, xmax);
            ui.separator();
            self.temperatures(ui, xmin, xmax);
            ui.separator();
            self.frequencies(ui, xmin, xmax);
            match self.legend_place {
                LegendPlacement::Footer => {
                    ui.horizontal_wrapped(|ui| self.legend_entries(ui));
                }
                LegendPlacement::Side => {
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| self.legend_entries(ui));
                    });
                }
            }
            ui.separator();
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    self.details(ui);
                    self.settings(ui);
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
