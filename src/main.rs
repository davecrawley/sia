use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use sia::collector::{Collector, LinuxMonotonicClock};
use sia::presentation::project_visible_traces;
use sia::system::{LinuxSystemBackend, SystemProvider};
use std::time::{Duration, Instant};

struct App {
    collector: Collector<LinuxMonotonicClock>,
    last_sample: Instant,
}

impl App {
    fn new() -> Self {
        let mut collector = Collector::new(LinuxMonotonicClock);
        collector.add_provider(SystemProvider::new(LinuxSystemBackend::discover()));

        #[cfg(feature = "nvidia")]
        if let Ok(backend) = sia::nvidia::NvmlBackend::initialize() {
            collector.add_provider(sia::nvidia::NvidiaProvider::new(backend));
        }

        collector.collect_once();
        Self {
            collector,
            last_sample: Instant::now(),
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, context: &egui::Context, _frame: &mut eframe::Frame) {
        if self.last_sample.elapsed() >= Duration::from_secs(1) {
            self.collector.collect_once();
            self.last_sample = Instant::now();
        }
        context.request_repaint_after(Duration::from_secs(1));

        egui::TopBottomPanel::top("header").show(context, |ui| {
            ui.heading("SIA - System Information Analyzer");
            ui.label("Live five-minute monitor");
        });

        egui::CentralPanel::default().show(context, |ui| {
            let traces = project_visible_traces(self.collector.model());
            if traces.is_empty() {
                ui.label("Waiting for available system metrics…");
                return;
            }

            for unit in ["%", "Cel", "kHz", "MHz"] {
                let matching: Vec<_> = traces.iter().filter(|trace| trace.unit == unit).collect();
                if matching.is_empty() {
                    continue;
                }
                let title = match unit {
                    "%" => "Utilization",
                    "Cel" => "Temperatures (°C)",
                    "kHz" | "MHz" => "Frequencies",
                    _ => unit,
                };
                ui.heading(title);
                Plot::new(format!("plot-{unit}"))
                    .height(220.0)
                    .legend(Legend::default())
                    .show(ui, |plot_ui| {
                        for trace in matching {
                            let scale = match unit {
                                "kHz" => 1_000_000.0,
                                "MHz" => 1_000.0,
                                _ => 1.0,
                            };
                            let points: Vec<[f64; 2]> = trace
                                .points
                                .iter()
                                .map(|(time, value)| {
                                    [*time as f64 / 1_000_000_000.0, *value / scale]
                                })
                                .collect();
                            plot_ui.line(
                                Line::new(PlotPoints::from(points))
                                    .name(format!("{} ({})", trace.display_name, trace.entity_id)),
                            );
                        }
                    });
                ui.separator();
            }

            ui.collapsing("Available metrics", |ui| {
                for trace in traces {
                    ui.label(format!(
                        "{} · {} · {} samples",
                        trace.display_name,
                        trace.entity_id,
                        trace.points.len()
                    ));
                }
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1230.0, 900.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("SIA - System Information Analyzer"),
        ..Default::default()
    };
    eframe::run_native(
        "SIA - System Information Analyzer",
        options,
        Box::new(|_creation_context| Ok(Box::new(App::new()))),
    )
}
