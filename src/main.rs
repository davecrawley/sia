use eframe::{egui, egui::Vec2};
use egui::{Align2, Color32, RichText};
use egui_plot::{Corner, Legend, Line, Plot, PlotBounds, PlotPoints, Text};
use sia::{
    default_live_source, CanonicalUnit, EntityId, MetricDescriptor, MetricId, MetricSample,
    SampleStatus, SessionSource,
};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SeriesKey {
    metric_id: MetricId,
    entity_id: EntityId,
}

struct Series {
    descriptor: MetricDescriptor,
    entity_id: EntityId,
    points: VecDeque<[f64; 2]>,
    capacity: usize,
}

impl Series {
    fn new(descriptor: MetricDescriptor, entity_id: EntityId, capacity: usize) -> Self {
        Self {
            descriptor,
            entity_id,
            points: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, time: f64, value: f64) {
        if self.points.len() == self.capacity {
            self.points.pop_front();
        }
        self.points.push_back([time, value]);
    }

    fn points_after(&self, minimum_time: f64, divisor: f64) -> PlotPoints {
        PlotPoints::from(
            self.points
                .iter()
                .filter(|point| point[0] >= minimum_time)
                .map(|point| [point[0], point[1] / divisor])
                .collect::<Vec<_>>(),
        )
    }

    fn last(&self) -> Option<f64> {
        self.points.back().map(|point| point[1])
    }
}

struct App {
    source: Box<dyn SessionSource>,
    descriptors: BTreeMap<MetricId, MetricDescriptor>,
    series: BTreeMap<SeriesKey, Series>,
    first_timestamp: Option<u64>,
    latest_time: f64,
    sample_count: usize,
    sample_period: Duration,
    last_poll: Instant,
    started: Instant,
    display_window_seconds: f64,
}

impl App {
    fn new() -> Self {
        let source = default_live_source();
        let descriptors = source
            .descriptors()
            .iter()
            .cloned()
            .map(|descriptor| (descriptor.metric_id.clone(), descriptor))
            .collect();
        let sample_period = Duration::from_secs(1);
        Self {
            source: Box::new(source),
            descriptors,
            series: BTreeMap::new(),
            first_timestamp: None,
            latest_time: 0.0,
            sample_count: 0,
            sample_period,
            last_poll: Instant::now()
                .checked_sub(sample_period)
                .unwrap_or_else(Instant::now),
            started: Instant::now(),
            display_window_seconds: 120.0,
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
        self.sample_count += 1;
        if sample.status != SampleStatus::Ok {
            return;
        }
        let Some(value) = sample.value.as_ref().and_then(|value| value.as_f64()) else {
            return;
        };
        let Some(descriptor) = self.descriptors.get(&sample.metric_id).cloned() else {
            return;
        };
        let origin = *self
            .first_timestamp
            .get_or_insert(sample.observation_time.0);
        let time = sample.observation_time.0.saturating_sub(origin) as f64 / 1_000_000_000.0;
        self.latest_time = self.latest_time.max(time);
        let key = SeriesKey {
            metric_id: sample.metric_id,
            entity_id: sample.entity_id.clone(),
        };
        self.series
            .entry(key)
            .or_insert_with(|| Series::new(descriptor, sample.entity_id, 5 * 60))
            .push(time, value);
    }

    fn latest_metric(&self, metric: &str) -> Option<f64> {
        self.series
            .values()
            .find(|series| series.descriptor.metric_id.0 == metric)
            .and_then(Series::last)
    }

    fn visible_range(&self) -> (f64, f64) {
        let maximum = self.latest_time.max(self.display_window_seconds);
        ((maximum - self.display_window_seconds).max(0.0), maximum)
    }

    fn utilization_series(&self) -> Vec<&Series> {
        self.series
            .values()
            .filter(|series| {
                matches!(series.descriptor.unit, CanonicalUnit::Percent)
                    && matches!(
                        series.descriptor.metric_id.0.as_str(),
                        "system.cpu.utilization"
                            | "system.memory.utilization"
                            | "gpu.nvidia.compute_utilization"
                            | "gpu.nvidia.vram_occupancy"
                    )
            })
            .collect()
    }

    fn temperature_series(&self) -> Vec<&Series> {
        self.series
            .values()
            .filter(|series| matches!(series.descriptor.unit, CanonicalUnit::Celsius))
            .collect()
    }

    fn frequency_series(&self) -> Vec<&Series> {
        self.series
            .values()
            .filter(|series| matches!(series.descriptor.unit, CanonicalUnit::Hertz))
            .collect()
    }

    fn series_label(series: &Series) -> String {
        format!("{} ({})", series.descriptor.display_name, series.entity_id.0)
    }
}

impl eframe::App for App {
    fn update(&mut self, context: &egui::Context, _frame: &mut eframe::Frame) {
        if self.last_poll.elapsed() >= self.sample_period {
            self.poll();
            self.last_poll = Instant::now();
        }
        context.request_repaint_after(
            self.sample_period
                .saturating_sub(self.last_poll.elapsed()),
        );

        egui::TopBottomPanel::top("top").show(context, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("SIA - System Information Analyzer - © David Crawley 2025");
                ui.separator();
                ui.label(format!("Uptime: {}s", self.started.elapsed().as_secs()));
                ui.separator();
                ui.label(format!("Samples: {}", self.sample_count));
                ui.separator();
                ui.label(format!(
                    "CPU: {:.0}%",
                    self.latest_metric("system.cpu.utilization").unwrap_or(0.0)
                ));
                ui.separator();
                ui.label(format!(
                    "RAM: {:.0}%",
                    self.latest_metric("system.memory.utilization").unwrap_or(0.0)
                ));
            });
        });

        egui::CentralPanel::default().show(context, |ui| {
            let (minimum_time, maximum_time) = self.visible_range();
            ui.horizontal(|ui| {
                ui.label("Window (seconds):");
                ui.add(egui::Slider::new(
                    &mut self.display_window_seconds,
                    30.0..=900.0,
                ));
                ui.separator();
                ui.label(format!(
                    "Clock: {}",
                    self.source.metadata().clock_domain
                ));
            });

            ui.heading("Utilization");
            Plot::new("utilization")
                .height(220.0)
                .legend(Legend::default().position(Corner::LeftTop))
                .show(ui, |plot_ui| {
                    plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                        [minimum_time, 0.0],
                        [maximum_time, 100.0],
                    ));
                    for (index, series) in self.utilization_series().into_iter().enumerate() {
                        plot_ui.line(
                            Line::new(series.points_after(minimum_time, 1.0))
                                .name(Self::series_label(series))
                                .color(series_color(&series.descriptor.metric_id.0, index)),
                        );
                    }
                    for value in [0.0, 25.0, 50.0, 75.0, 100.0] {
                        plot_ui.text(
                            Text::new([maximum_time, value].into(), format!("{value:.0}%"))
                                .anchor(Align2::RIGHT_CENTER),
                        );
                    }
                });

            ui.separator();
            ui.heading("Temperatures (°C)");
            let temperatures = self.temperature_series();
            let (temperature_min, temperature_max) = value_bounds(&temperatures, minimum_time, 1.0)
                .map(|(minimum, maximum)| padded_bounds(minimum, maximum, 0.0, 130.0))
                .unwrap_or((0.0, 120.0));
            Plot::new("temperatures")
                .height(260.0)
                .legend(Legend::default().position(Corner::LeftTop))
                .show(ui, |plot_ui| {
                    plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                        [minimum_time, temperature_min],
                        [maximum_time, temperature_max],
                    ));
                    for (index, series) in temperatures.into_iter().enumerate() {
                        plot_ui.line(
                            Line::new(series.points_after(minimum_time, 1.0))
                                .name(Self::series_label(series))
                                .color(series_color(&series.descriptor.metric_id.0, index)),
                        );
                    }
                });

            ui.separator();
            ui.heading("Frequencies (GHz)");
            let frequencies = self.frequency_series();
            let (frequency_min, frequency_max) = value_bounds(&frequencies, minimum_time, 1e9)
                .map(|(minimum, maximum)| padded_bounds(minimum, maximum, 0.0, 12.0))
                .unwrap_or((0.0, 6.0));
            Plot::new("frequencies")
                .height(240.0)
                .legend(Legend::default().position(Corner::LeftTop))
                .show(ui, |plot_ui| {
                    plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                        [minimum_time, frequency_min],
                        [maximum_time, frequency_max],
                    ));
                    for (index, series) in frequencies.into_iter().enumerate() {
                        plot_ui.line(
                            Line::new(series.points_after(minimum_time, 1e9))
                                .name(Self::series_label(series))
                                .color(series_color(&series.descriptor.metric_id.0, index)),
                        );
                    }
                });

            ui.separator();
            ui.label(
                RichText::new("Unavailable capabilities are omitted; missing observations appear as gaps.")
                    .small(),
            );
        });
    }
}

fn value_bounds(series: &[&Series], minimum_time: f64, divisor: f64) -> Option<(f64, f64)> {
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for series in series {
        for point in &series.points {
            if point[0] >= minimum_time {
                minimum = minimum.min(point[1] / divisor);
                maximum = maximum.max(point[1] / divisor);
            }
        }
    }
    (minimum.is_finite() && maximum.is_finite()).then_some((minimum, maximum))
}

fn padded_bounds(minimum: f64, maximum: f64, floor: f64, ceiling: f64) -> (f64, f64) {
    if (maximum - minimum).abs() < f64::EPSILON {
        return ((minimum - 1.0).max(floor), (maximum + 1.0).min(ceiling));
    }
    let padding = ((maximum - minimum) * 0.1).max(0.05);
    ((minimum - padding).max(floor), (maximum + padding).min(ceiling))
}

fn series_color(metric_id: &str, index: usize) -> Color32 {
    let base = if metric_id.contains("cpu") {
        Color32::from_rgb(244, 67, 54)
    } else if metric_id.contains("gpu") {
        Color32::from_rgb(33, 150, 243)
    } else if metric_id.contains("memory") {
        Color32::from_rgb(76, 175, 80)
    } else if metric_id.contains("thermal") {
        Color32::from_rgb(255, 152, 0)
    } else {
        Color32::from_rgb(156, 39, 176)
    };
    let factor = ((index % 5) as f32) * 0.1;
    let lighten = |component: u8| {
        (component as f32 + (255.0 - component as f32) * factor)
            .min(255.0) as u8
    };
    Color32::from_rgb(lighten(base.r()), lighten(base.g()), lighten(base.b()))
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
        Box::new(|_creation_context| Ok(Box::new(App::new()))),
    )
}