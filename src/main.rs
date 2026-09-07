use eframe::{egui, egui::Vec2};
use egui::Color32;
use egui_plot::{Corner, Legend, Line, Plot, PlotBounds, PlotPoints};
use sia::{
    default_live_source, CanonicalUnit, EntityId, MetricDescriptor, MetricId, MetricSample,
    SampleStatus, SessionSource,
};
use std::{
    collections::{BTreeMap, VecDeque},
    time::{Duration, Instant},
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SeriesKey {
    metric_id: MetricId,
    entity_id: EntityId,
}
struct Series {
    descriptor: MetricDescriptor,
    entity_id: EntityId,
    points: VecDeque<[f64; 2]>,
}
impl Series {
    fn new(descriptor: MetricDescriptor, entity_id: EntityId) -> Self {
        Self {
            descriptor,
            entity_id,
            points: VecDeque::with_capacity(300),
        }
    }
    fn push(&mut self, time: f64, value: f64) {
        if self.points.len() == 300 {
            self.points.pop_front();
        }
        self.points.push_back([time, value]);
    }
    fn points(&self, since: f64, divisor: f64) -> PlotPoints {
        PlotPoints::from(
            self.points
                .iter()
                .filter(|p| p[0] >= since)
                .map(|p| [p[0], p[1] / divisor])
                .collect::<Vec<_>>(),
        )
    }
}

struct App {
    source: Box<dyn SessionSource>,
    descriptors: BTreeMap<MetricId, MetricDescriptor>,
    series: BTreeMap<SeriesKey, Series>,
    origin: Option<u64>,
    latest: f64,
    samples: usize,
    last_poll: Instant,
    started: Instant,
    window: f64,
}
impl App {
    fn new() -> Self {
        let source = default_live_source();
        let descriptors = source
            .descriptors()
            .iter()
            .cloned()
            .map(|d| (d.metric_id.clone(), d))
            .collect();
        Self {
            source: Box::new(source),
            descriptors,
            series: BTreeMap::new(),
            origin: None,
            latest: 0.0,
            samples: 0,
            last_poll: Instant::now()
                .checked_sub(Duration::from_secs(1))
                .unwrap_or_else(Instant::now),
            started: Instant::now(),
            window: 120.0,
        }
    }
    fn poll(&mut self) {
        if let Ok(samples) = self.source.poll() {
            for sample in samples {
                self.ingest(sample);
            }
        }
    }
    fn ingest(&mut self, sample: MetricSample) {
        self.samples += 1;
        if sample.status != SampleStatus::Ok {
            return;
        }
        let Some(value) = sample.value.as_ref().and_then(|v| v.as_f64()) else {
            return;
        };
        let Some(descriptor) = self.descriptors.get(&sample.metric_id).cloned() else {
            return;
        };
        let origin = *self.origin.get_or_insert(sample.observation_time.0);
        let time = sample.observation_time.0.saturating_sub(origin) as f64 / 1e9;
        self.latest = self.latest.max(time);
        let key = SeriesKey {
            metric_id: sample.metric_id,
            entity_id: sample.entity_id.clone(),
        };
        self.series
            .entry(key)
            .or_insert_with(|| Series::new(descriptor, sample.entity_id))
            .push(time, value);
    }
    fn latest_value(&self, id: &str) -> Option<f64> {
        self.series
            .values()
            .find(|s| s.descriptor.metric_id.0 == id)
            .and_then(|s| s.points.back().map(|p| p[1]))
    }
    fn matching(&self, unit: &CanonicalUnit) -> Vec<&Series> {
        self.series
            .values()
            .filter(|s| &s.descriptor.unit == unit)
            .collect()
    }
}

impl eframe::App for App {
    fn update(&mut self, context: &egui::Context, _: &mut eframe::Frame) {
        if self.last_poll.elapsed() >= Duration::from_secs(1) {
            self.poll();
            self.last_poll = Instant::now();
        }
        context
            .request_repaint_after(Duration::from_secs(1).saturating_sub(self.last_poll.elapsed()));
        egui::TopBottomPanel::top("top").show(context, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("SIA - System Information Analyzer - © David Crawley 2025");
                ui.separator();
                ui.label(format!("Uptime: {}s", self.started.elapsed().as_secs()));
                ui.separator();
                ui.label(format!("Samples: {}", self.samples));
                ui.separator();
                ui.label(format!(
                    "CPU: {:.0}%",
                    self.latest_value("system.cpu.utilization").unwrap_or(0.0)
                ));
                ui.separator();
                ui.label(format!(
                    "RAM: {:.0}%",
                    self.latest_value("system.memory.utilization")
                        .unwrap_or(0.0)
                ));
            })
        });
        egui::CentralPanel::default().show(context, |ui| {
            ui.horizontal(|ui| {
                ui.label("Window (seconds):");
                ui.add(egui::Slider::new(&mut self.window, 30.0..=900.0));
                ui.separator();
                ui.label(format!("Clock: {}", self.source.metadata().clock_domain));
            });
            let max_time = self.latest.max(self.window);
            let min_time = (max_time - self.window).max(0.0);
            draw_plot(
                ui,
                "utilization",
                "Utilization (%)",
                self.matching(&CanonicalUnit::Percent),
                min_time,
                max_time,
                1.0,
                Some((0.0, 100.0)),
            );
            draw_plot(
                ui,
                "temperatures",
                "Temperatures (°C)",
                self.matching(&CanonicalUnit::Celsius),
                min_time,
                max_time,
                1.0,
                None,
            );
            draw_plot(
                ui,
                "frequencies",
                "Frequencies (GHz)",
                self.matching(&CanonicalUnit::Hertz),
                min_time,
                max_time,
                1e9,
                None,
            );
            ui.small("Unavailable capabilities are omitted; missing observations appear as gaps.");
        });
    }
}

fn draw_plot(
    ui: &mut egui::Ui,
    id: &str,
    title: &str,
    series: Vec<&Series>,
    min_time: f64,
    max_time: f64,
    divisor: f64,
    fixed_y: Option<(f64, f64)>,
) {
    ui.separator();
    ui.heading(title);
    Plot::new(id)
        .height(240.0)
        .legend(Legend::default().position(Corner::LeftTop))
        .show(ui, |plot_ui| {
            if let Some((min_y, max_y)) = fixed_y {
                plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                    [min_time, min_y],
                    [max_time, max_y],
                ));
            }
            for (index, item) in series.into_iter().enumerate() {
                plot_ui.line(
                    Line::new(item.points(min_time, divisor))
                        .name(format!(
                            "{} ({})",
                            item.descriptor.display_name, item.entity_id.0
                        ))
                        .color(series_color(&item.descriptor.metric_id.0, index)),
                );
            }
        });
}
fn series_color(id: &str, index: usize) -> Color32 {
    let base = if id.contains("cpu") {
        (244, 67, 54)
    } else if id.contains("gpu") {
        (33, 150, 243)
    } else if id.contains("memory") {
        (76, 175, 80)
    } else if id.contains("thermal") {
        (255, 152, 0)
    } else {
        (156, 39, 176)
    };
    let factor = (index % 5) as f32 * 0.1;
    let light = |v: u8| (v as f32 + (255.0 - v as f32) * factor).min(255.0) as u8;
    Color32::from_rgb(light(base.0), light(base.1), light(base.2))
}
fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1230.0, 1000.0])
            .with_min_inner_size(Vec2::new(800.0, 600.0))
            .with_title("SIA - System Information Analyzer - © David Crawley 2025"),
        ..Default::default()
    };
    eframe::run_native(
        "SIA - System Information Analyzer",
        options,
        Box::new(|_| Ok(Box::new(App::new()))),
    )
}
