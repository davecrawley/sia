use eframe::{egui, egui::Vec2};
use egui::{Align2, Color32, FontFamily, FontId, RichText, TextStyle};
use egui_plot::{Legend, Line, Plot, PlotBounds, PlotPoints, Text};
use sia::collection::MetricProvider;
use sia::presentation::{current_value, CurrentValue};
use sia::providers::{HostProvider, CPU_UTILIZATION, RAM_UTILIZATION};
use sia::{
    CanonicalUnit, CollectionBatch, Collector, EntityId, MetricDescriptor, MetricId, NativeClock,
    SampleStatus, SampleValue, SessionModel,
};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct SeriesPoint {
    time: f64,
    interval_start: Option<f64>,
    value: Option<f64>,
}

#[derive(Clone)]
struct Series {
    values: VecDeque<SeriesPoint>,
    capacity: usize,
}

impl Series {
    fn new(capacity: usize) -> Self {
        Self {
            values: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, point: SeriesPoint) {
        if self.values.len() == self.capacity {
            self.values.pop_front();
        }
        self.values.push_back(point);
    }

    fn last(&self) -> Option<f64> {
        self.values.back().and_then(|point| point.value)
    }

    fn min_max(&self, start: f64, end: f64, scale: f64) -> Option<(f64, f64)> {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for point in &self.values {
            if point.time >= start && point.time <= end {
                if let Some(value) = point.value {
                    min = min.min(value / scale);
                    max = max.max(value / scale);
                }
            }
        }
        (min.is_finite() && max.is_finite()).then_some((min, max))
    }

    fn segments(&self, start: f64, scale: f64) -> Vec<PlotPoints> {
        let mut result = Vec::new();
        let mut current = Vec::new();
        for point in &self.values {
            if point.time < start {
                continue;
            }
            if let Some(value) = point.value {
                let _native_interval_start = point.interval_start;
                current.push([point.time, value / scale]);
            } else if !current.is_empty() {
                result.push(PlotPoints::from(std::mem::take(&mut current)));
            }
        }
        if !current.is_empty() {
            result.push(PlotPoints::from(current));
        }
        result
    }
}

#[derive(Clone)]
struct Trace {
    metric: MetricId,
    entity: EntityId,
    label: String,
    provider: String,
    visible: bool,
    color: Color32,
    scale: f64,
    series: Series,
}

#[derive(Clone)]
struct SensorGroup {
    name: String,
    traces: Vec<usize>,
    visible: bool,
    warn: f64,
    hot: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LegendPlacement {
    Footer,
    Side,
}

fn color(index: usize) -> Color32 {
    const COLORS: [(u8, u8, u8); 12] = [
        (244, 67, 54),
        (33, 150, 243),
        (76, 175, 80),
        (255, 152, 0),
        (156, 39, 176),
        (0, 150, 136),
        (63, 81, 181),
        (205, 220, 57),
        (233, 30, 99),
        (255, 87, 34),
        (3, 169, 244),
        (158, 158, 158),
    ];
    let (r, g, b) = COLORS[index % COLORS.len()];
    Color32::from_rgb(r, g, b)
}

fn classify(descriptor: &MetricDescriptor) -> (String, f64, f64) {
    let text = format!(
        "{} {} {}",
        descriptor.display_name, descriptor.provider, descriptor.source_semantics
    )
    .to_lowercase();
    if text.contains("cpu") || text.contains("coretemp") || text.contains("k10temp") {
        ("CPU".into(), 90.0, 100.0)
    } else if text.contains("gpu") || text.contains("nvidia") || text.contains("amdgpu") {
        ("GPU".into(), 85.0, 95.0)
    } else if text.contains("nvme") {
        ("NVMe SSD".into(), 70.0, 80.0)
    } else if text.contains("memory") || text.contains("spd") {
        ("Memory".into(), 70.0, 85.0)
    } else if text.contains("wifi") {
        ("Wi-Fi Controller".into(), 80.0, 90.0)
    } else if text.contains("ethernet") {
        ("Ethernet Controller".into(), 80.0, 90.0)
    } else {
        (descriptor.provider.clone(), 90.0, 100.0)
    }
}

struct App {
    collector: Collector<NativeClock, HostProvider>,
    model: SessionModel,
    origin_ns: Option<u64>,
    elapsed: f64,
    sample_period: Duration,
    last_tick: Instant,
    traces: Vec<Trace>,
    groups: Vec<SensorGroup>,
    cpu_key: (MetricId, EntityId),
    ram_key: (MetricId, EntityId),
    display_window_secs: f64,
    legend_place: LegendPlacement,
    font_size: f32,
    font_color: Color32,
    pending_font_size: f32,
    pending_font_color: Color32,
    live_preview: bool,
}

impl App {
    fn new(capacity: usize, sample_hz: f64) -> Self {
        let provider = HostProvider::new();
        let targets = provider.targets();
        let descriptors = provider.descriptors();
        let mut traces = Vec::new();
        for target in targets {
            let descriptor = target.descriptor;
            if matches!(
                descriptor.canonical_unit,
                CanonicalUnit::Percent | CanonicalUnit::Celsius | CanonicalUnit::Hertz
            ) {
                let scale = if descriptor.canonical_unit == CanonicalUnit::Hertz {
                    1_000_000_000.0
                } else {
                    1.0
                };
                traces.push(Trace {
                    metric: descriptor.metric_id,
                    entity: target.entity_id,
                    label: descriptor.display_name,
                    provider: descriptor.provider,
                    visible: true,
                    color: color(traces.len()),
                    scale,
                    series: Series::new(capacity),
                });
            }
        }
        let mut groups: Vec<SensorGroup> = Vec::new();
        for (index, trace) in traces.iter().enumerate() {
            let Some(descriptor) = descriptors
                .iter()
                .find(|item| item.metric_id == trace.metric)
            else {
                continue;
            };
            if descriptor.canonical_unit != CanonicalUnit::Celsius {
                continue;
            }
            let (name, warn, hot) = classify(descriptor);
            if let Some(group) = groups.iter_mut().find(|group| group.name == name) {
                group.traces.push(index);
            } else {
                groups.push(SensorGroup {
                    name,
                    traces: vec![index],
                    visible: true,
                    warn,
                    hot,
                });
            }
        }
        Self {
            collector: Collector::new(NativeClock, provider),
            model: SessionModel::new(descriptors),
            origin_ns: None,
            elapsed: 0.0,
            sample_period: Duration::from_secs_f64((1.0 / sample_hz).max(0.05)),
            last_tick: Instant::now(),
            traces,
            groups,
            cpu_key: (CPU_UTILIZATION.into(), "system".into()),
            ram_key: (RAM_UTILIZATION.into(), "system".into()),
            display_window_secs: 120.0,
            legend_place: LegendPlacement::Footer,
            font_size: 14.0,
            font_color: Color32::WHITE,
            pending_font_size: 14.0,
            pending_font_color: Color32::WHITE,
            live_preview: false,
        }
    }

    fn sample(&mut self) {
        let batch = self.collector.collect();
        let origin = *self
            .origin_ns
            .get_or_insert(batch.observation_time.monotonic_ns);
        self.elapsed =
            batch.observation_time.monotonic_ns.saturating_sub(origin) as f64 / 1_000_000_000.0;
        self.update_traces(&batch, origin);
        self.model.ingest(batch);
    }

    fn update_traces(&mut self, batch: &CollectionBatch, origin: u64) {
        for trace in &mut self.traces {
            let matching = batch.samples.iter().filter(|sample| {
                sample.metric_id == trace.metric && sample.entity_id == trace.entity
            });
            for sample in matching {
                let value = if sample.status == SampleStatus::Ok {
                    sample.value.as_ref().and_then(SampleValue::numeric)
                } else {
                    None
                }
                .filter(|value| value.is_finite());
                trace.series.push(SeriesPoint {
                    time: sample.observation_time.monotonic_ns.saturating_sub(origin) as f64
                        / 1_000_000_000.0,
                    interval_start: sample.interval_start.map(|time| {
                        time.monotonic_ns.saturating_sub(origin) as f64 / 1_000_000_000.0
                    }),
                    value,
                });
            }
        }
    }

    fn trace_is_plottable(&self, trace: &Trace) -> bool {
        !matches!(
            current_value(&self.model, &trace.metric, &trace.entity),
            CurrentValue::Unsupported(_)
        )
    }

    fn state_text(&self, trace: &Trace) -> String {
        match current_value(&self.model, &trace.metric, &trace.entity) {
            CurrentValue::Value(_) => "ok".into(),
            CurrentValue::Missing => "missing".into(),
            CurrentValue::Stale => "stale".into(),
            CurrentValue::Unsupported(_) => "unsupported".into(),
            CurrentValue::PermissionDenied(_) => "permission denied".into(),
            CurrentValue::TemporarilyUnavailable(_) => "unavailable".into(),
            CurrentValue::Error(_) => "error".into(),
        }
    }

    fn summary(&self, key: &(MetricId, EntityId)) -> String {
        match current_value(&self.model, &key.0, &key.1) {
            CurrentValue::Value(value) => value
                .numeric()
                .map(|value| format!("{value:.0}%"))
                .unwrap_or_else(|| "state".into()),
            CurrentValue::Stale => "stale".into(),
            CurrentValue::TemporarilyUnavailable(_) => "unavailable".into(),
            CurrentValue::PermissionDenied(_) => "permission denied".into(),
            CurrentValue::Unsupported(_) => "unsupported".into(),
            CurrentValue::Error(_) => "error".into(),
            CurrentValue::Missing => "missing".into(),
        }
    }

    fn draw_trace(plot: &mut egui_plot::PlotUi, trace: &Trace, start: f64) {
        for (index, points) in trace
            .series
            .segments(start, trace.scale)
            .into_iter()
            .enumerate()
        {
            let name = if index == 0 {
                trace.label.clone()
            } else {
                String::new()
            };
            plot.line(Line::new(points).name(name).color(trace.color));
        }
    }

    fn legend(&self, ui: &mut egui::Ui, vertical: bool) {
        let draw = |ui: &mut egui::Ui| {
            ui.label(RichText::new("Legend").strong());
            for group in &self.groups {
                if !group.visible {
                    continue;
                }
                for index in &group.traces {
                    let trace = &self.traces[*index];
                    if !trace.visible || !self.trace_is_plottable(trace) {
                        continue;
                    }
                    let mut label = trace.label.clone();
                    if let Some(value) = trace.series.last() {
                        if value >= group.hot {
                            label.push_str(" 🔥");
                        } else if value >= group.warn {
                            label.push_str(" 🥵");
                        }
                    }
                    let state = self.state_text(trace);
                    ui.horizontal(|ui| {
                        ui.colored_label(trace.color, "●");
                        ui.label(format!("{label} [{state}]"));
                    });
                }
            }
        };
        if vertical {
            ui.vertical(draw);
        } else {
            ui.horizontal_wrapped(draw);
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut style = (*ctx.style()).clone();
        style.visuals.override_text_color = Some(self.font_color);
        style.text_styles = [
            (
                TextStyle::Heading,
                FontId::new(self.font_size, FontFamily::Proportional),
            ),
            (
                TextStyle::Body,
                FontId::new(self.font_size, FontFamily::Proportional),
            ),
            (
                TextStyle::Monospace,
                FontId::new(self.font_size, FontFamily::Monospace),
            ),
            (
                TextStyle::Button,
                FontId::new(self.font_size, FontFamily::Proportional),
            ),
            (
                TextStyle::Small,
                FontId::new(self.font_size, FontFamily::Proportional),
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
                ui.label(format!("Uptime: {:.0}s", self.elapsed));
                ui.separator();
                ui.label(format!("CPU: {}", self.summary(&self.cpu_key)));
                ui.separator();
                ui.label(format!("RAM: {}", self.summary(&self.ram_key)));
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.set_min_size(Vec2::new(1200.0, 880.0));
            let start = (self.elapsed - self.display_window_secs).max(0.0);
            let end = self.elapsed.max(self.display_window_secs);
            ui.heading("Utilization");
            Plot::new("util")
                .legend(Legend::default())
                .height(220.0)
                .show(ui, |plot| {
                    plot.set_plot_bounds(PlotBounds::from_min_max([start, 0.0], [end, 100.0]));
                    for trace in &self.traces {
                        if trace.visible
                            && self.trace_is_plottable(trace)
                            && trace.scale == 1.0
                            && (trace.metric.0.contains("utilization")
                                || trace.metric.0 == RAM_UTILIZATION)
                        {
                            Self::draw_trace(plot, trace, start);
                        }
                    }
                    for value in [0.0, 25.0, 50.0, 75.0, 100.0] {
                        plot.text(
                            Text::new([end, value].into(), format!("{value:.0}%"))
                                .anchor(Align2::RIGHT_CENTER),
                        );
                    }
                });
            ui.separator();
            ui.heading("Temperatures (°C)");
            Plot::new("temps").height(260.0).show(ui, |plot| {
                let mut min = f64::INFINITY;
                let mut max = f64::NEG_INFINITY;
                for group in &self.groups {
                    if group.visible {
                        for index in &group.traces {
                            let trace = &self.traces[*index];
                            if trace.visible && self.trace_is_plottable(trace) {
                                if let Some((a, b)) = trace.series.min_max(start, end, 1.0) {
                                    min = min.min(a);
                                    max = max.max(b);
                                }
                            }
                        }
                    }
                }
                if !min.is_finite() || (max - min).abs() < 0.001 {
                    min = 0.0;
                    max = 120.0;
                }
                plot.set_plot_bounds(PlotBounds::from_min_max(
                    [start, (min - 3.0).max(0.0)],
                    [end, (max + 3.0).min(130.0)],
                ));
                for group in &self.groups {
                    if group.visible {
                        for index in &group.traces {
                            let trace = &self.traces[*index];
                            if trace.visible && self.trace_is_plottable(trace) {
                                Self::draw_trace(plot, trace, start);
                            }
                        }
                    }
                }
            });
            ui.separator();
            ui.heading("Frequencies (GHz)");
            Plot::new("freq").height(240.0).show(ui, |plot| {
                let mut max: f64 = 1.0;
                for trace in &self.traces {
                    if trace.visible && self.trace_is_plottable(trace) && trace.scale > 1.0 {
                        if let Some((_, value)) = trace.series.min_max(start, end, trace.scale) {
                            max = max.max(value);
                        }
                    }
                }
                plot.set_plot_bounds(PlotBounds::from_min_max(
                    [start, 0.0],
                    [end, (max * 1.1).max(1.0)],
                ));
                for trace in &self.traces {
                    if trace.visible && self.trace_is_plottable(trace) && trace.scale > 1.0 {
                        Self::draw_trace(plot, trace, start);
                    }
                }
            });
            match self.legend_place {
                LegendPlacement::Footer => self.legend(ui, false),
                LegendPlacement::Side => self.legend(ui, true),
            }
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Display");
                ui.horizontal(|ui| {
                    ui.label("Window (seconds) before scroll:");
                    ui.add(egui::Slider::new(
                        &mut self.display_window_secs,
                        30.0..=900.0,
                    ));
                    egui::ComboBox::from_label("Legend placement")
                        .selected_text(if self.legend_place == LegendPlacement::Footer {
                            "Footer"
                        } else {
                            "Side"
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
                    ui.label("Font size");
                    let response =
                        ui.add(egui::Slider::new(&mut self.pending_font_size, 10.0..=22.0));
                    ui.label("Font color");
                    ui.color_edit_button_srgba(&mut self.pending_font_color);
                    if self.live_preview || response.drag_stopped() {
                        self.font_size = self.pending_font_size;
                        self.font_color = self.pending_font_color;
                    }
                    if ui.button("Apply font").clicked() {
                        self.font_size = self.pending_font_size;
                        self.font_color = self.pending_font_color;
                    }
                    ui.toggle_value(&mut self.live_preview, "Live preview");
                });
                ui.heading("Sensors");
                for group_index in 0..self.groups.len() {
                    let name = self.groups[group_index].name.clone();
                    let trace_indices = self.groups[group_index].traces.clone();
                    egui::CollapsingHeader::new(name).show(ui, |ui| {
                        ui.checkbox(&mut self.groups[group_index].visible, "Show group");
                        for index in trace_indices {
                            let state = self.state_text(&self.traces[index]);
                            let trace = &mut self.traces[index];
                            ui.checkbox(
                                &mut trace.visible,
                                format!("{} ({}) — {}", trace.label, trace.provider, state),
                            );
                        }
                    });
                }
                egui::CollapsingHeader::new("Frequencies").show(ui, |ui| {
                    for index in 0..self.traces.len() {
                        if self.traces[index].scale > 1.0 {
                            let state = self.state_text(&self.traces[index]);
                            let trace = &mut self.traces[index];
                            ui.checkbox(&mut trace.visible, format!("{} — {}", trace.label, state));
                        }
                    }
                });
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
        Box::new(|_| Ok(Box::new(App::new(5 * 60, 1.0)))),
    )
}
