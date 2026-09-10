use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

use eframe::egui;
use egui::{Color32, RichText};
use egui_plot::{Legend, Line, Plot, PlotBounds, PlotPoints};
use sia::telemetry::{
    production_collector, CapabilityStatus, MetricDescriptor, SampleStatus, SeriesKey,
    SessionSource,
};

const HISTORY_CAPACITY: usize = 900;

struct App {
    source: Box<dyn SessionSource>,
    descriptors: Vec<MetricDescriptor>,
    history: BTreeMap<SeriesKey, VecDeque<(f64, f64)>>,
    first_mono_ns: Option<u64>,
    elapsed_seconds: f64,
    samples: u64,
    last_tick: Instant,
}

impl App {
    fn new() -> Self {
        let source: Box<dyn SessionSource> = Box::new(production_collector());
        let descriptors = source.session_metadata().descriptors;
        Self {
            source,
            descriptors,
            history: BTreeMap::new(),
            first_mono_ns: None,
            elapsed_seconds: 0.0,
            samples: 0,
            last_tick: Instant::now(),
        }
    }

    fn sample(&mut self) {
        let collection = self.source.next_collection();
        let first = *self
            .first_mono_ns
            .get_or_insert(collection.observation_mono_ns);
        self.elapsed_seconds =
            collection.observation_mono_ns.saturating_sub(first) as f64 / 1_000_000_000.0;
        self.descriptors = collection.descriptors;
        self.samples += 1;

        for sample in collection.samples {
            if sample.status != SampleStatus::Ok {
                continue;
            }
            let Some(value) = sample.value.as_f64() else {
                continue;
            };
            let series = self.history.entry(sample.series_key()).or_default();
            if series.len() == HISTORY_CAPACITY {
                series.pop_front();
            }
            series.push_back((self.elapsed_seconds, value));
        }
    }

    fn points(&self, descriptor: &MetricDescriptor, divisor: f64, start: f64) -> PlotPoints {
        let points = self
            .history
            .get(&descriptor.series_key())
            .into_iter()
            .flatten()
            .filter(|(time, _)| *time >= start)
            .map(|(time, value)| [*time, *value / divisor])
            .collect::<Vec<_>>();
        PlotPoints::from(points)
    }

    fn last_value(&self, metric_id: &str) -> Option<f64> {
        self.descriptors
            .iter()
            .filter(|descriptor| descriptor.metric_id == metric_id)
            .find_map(|descriptor| {
                self.history
                    .get(&descriptor.series_key())
                    .and_then(|series| series.back())
                    .map(|(_, value)| *value)
            })
    }

    fn descriptors_for<'a>(
        &'a self,
        predicate: impl Fn(&MetricDescriptor) -> bool + 'a,
    ) -> impl Iterator<Item = &'a MetricDescriptor> + 'a {
        self.descriptors.iter().filter(move |descriptor| {
            descriptor.capability_status == CapabilityStatus::Available && predicate(descriptor)
        })
    }

    fn series_color(descriptor: &MetricDescriptor) -> Color32 {
        let identity = format!("{}:{}", descriptor.metric_id, descriptor.entity_id);
        let hash = identity.bytes().fold(0_u32, |value, byte| {
            value.wrapping_mul(16777619) ^ byte as u32
        });
        Color32::from_rgb(
            72 + (hash & 127) as u8,
            72 + ((hash >> 8) & 127) as u8,
            72 + ((hash >> 16) & 127) as u8,
        )
    }

    fn show_plot(
        &self,
        ui: &mut egui::Ui,
        id: &str,
        title: &str,
        descriptors: Vec<&MetricDescriptor>,
        divisor: impl Fn(&MetricDescriptor) -> f64,
        y_bounds: Option<(f64, f64)>,
    ) {
        ui.heading(title);
        let start = (self.elapsed_seconds - 120.0).max(0.0);
        Plot::new(id)
            .height(230.0)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                if let Some((minimum, maximum)) = y_bounds {
                    plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                        [start, minimum],
                        [self.elapsed_seconds.max(120.0), maximum],
                    ));
                }
                for descriptor in descriptors {
                    let points = self.points(descriptor, divisor(descriptor), start);
                    plot_ui.line(
                        Line::new(points)
                            .name(descriptor.display_name.clone())
                            .color(Self::series_color(descriptor)),
                    );
                }
            });
    }
}

impl eframe::App for App {
    fn update(&mut self, context: &egui::Context, _frame: &mut eframe::Frame) {
        if self.last_tick.elapsed() >= Duration::from_secs(1) {
            self.sample();
            self.last_tick = Instant::now();
        }
        context.request_repaint_after(Duration::from_millis(50));

        egui::TopBottomPanel::top("summary").show(context, |ui| {
            ui.horizontal(|ui| {
                ui.heading("SIA - System Information Analyzer");
                ui.separator();
                ui.label(format!("Uptime: {:.0}s", self.elapsed_seconds));
                ui.separator();
                ui.label(format!("Samples: {}", self.samples));
                ui.separator();
                ui.label(format!(
                    "CPU: {:.0}%",
                    self.last_value("cpu.utilization").unwrap_or(0.0)
                ));
                ui.separator();
                ui.label(format!(
                    "RAM: {:.0}%",
                    self.last_value("memory.used_ratio").unwrap_or(0.0)
                ));
            });
        });

        egui::CentralPanel::default().show(context, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.show_plot(
                    ui,
                    "utilization",
                    "Utilization",
                    self.descriptors_for(|descriptor| {
                        matches!(
                            descriptor.metric_id.as_str(),
                            "cpu.utilization"
                                | "memory.used_ratio"
                                | "gpu.utilization"
                                | "gpu.vram_occupancy"
                        )
                    })
                    .collect(),
                    |_| 1.0,
                    Some((0.0, 100.0)),
                );
                ui.separator();
                self.show_plot(
                    ui,
                    "temperatures",
                    "Temperatures (°C)",
                    self.descriptors_for(|descriptor| {
                        descriptor.metric_id == "temperature.celsius"
                    })
                    .collect(),
                    |_| 1.0,
                    None,
                );
                ui.separator();
                self.show_plot(
                    ui,
                    "frequencies",
                    "Frequencies (GHz)",
                    self.descriptors_for(|descriptor| {
                        descriptor.metric_id == "cpu.frequency"
                            || descriptor.metric_id.starts_with("gpu.clock.")
                    })
                    .collect(),
                    |descriptor| {
                        if descriptor.unit == "kilohertz" {
                            1_000_000.0
                        } else {
                            1_000.0
                        }
                    },
                    None,
                );
                ui.separator();
                ui.label(
                    RichText::new(
                        "Unavailable capabilities are omitted; temporary failures create gaps.",
                    )
                    .italics(),
                );
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1230.0, 900.0])
            .with_min_inner_size([900.0, 650.0])
            .with_title("SIA - System Information Analyzer"),
        ..Default::default()
    };
    eframe::run_native(
        "SIA - System Information Analyzer",
        options,
        Box::new(|_creation_context| Ok(Box::new(App::new()))),
    )
}
