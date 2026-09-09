use eframe::egui;
use egui::{Color32, RichText};
use egui_plot::{Corner, HLine, Legend, Line, Plot, PlotPoints};
use sia::{
    production_coordinator, CapabilityState, Coordinator, MonotonicClock, PresentationSnapshot,
    SeriesKey, Unit, VisibleTrace, LIVE_RETENTION,
};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LegendPlacement {
    Footer,
    Side,
}

struct App {
    coordinator: Coordinator<MonotonicClock>,
    last_sample: Instant,
    display_window_secs: f64,
    legend_placement: LegendPlacement,
    trace_visibility: BTreeMap<SeriesKey, bool>,
    group_visibility: BTreeMap<String, bool>,
    font_size: f32,
    font_color: Color32,
}

impl App {
    fn new() -> Self {
        let mut coordinator = production_coordinator();
        coordinator.collect_once();
        Self {
            coordinator,
            last_sample: Instant::now(),
            display_window_secs: LIVE_RETENTION.as_secs_f64(),
            legend_placement: LegendPlacement::Footer,
            trace_visibility: BTreeMap::new(),
            group_visibility: BTreeMap::new(),
            font_size: 14.0,
            font_color: Color32::from_gray(220),
        }
    }

    fn apply_font(&self, context: &egui::Context) {
        let mut style = (*context.style()).clone();
        for font in style.text_styles.values_mut() {
            font.size = self.font_size;
        }
        style.visuals.override_text_color = Some(self.font_color);
        context.set_style(style);
    }

    fn trace_is_visible(&self, trace: &VisibleTrace, snapshot: &PresentationSnapshot) -> bool {
        let trace_visible = self
            .trace_visibility
            .get(&trace.key)
            .copied()
            .unwrap_or(true);
        let group_visible = snapshot
            .groups
            .iter()
            .find(|group| group.trace_keys.contains(&trace.key))
            .and_then(|group| self.group_visibility.get(&group.key))
            .copied()
            .unwrap_or(true);
        trace_visible && group_visible
    }

    fn draw_plot(
        &self,
        ui: &mut egui::Ui,
        snapshot: &PresentationSnapshot,
        title: &str,
        unit: Unit,
    ) {
        let traces: Vec<_> = snapshot
            .traces
            .iter()
            .filter(|trace| trace.unit == unit)
            .filter(|trace| self.trace_is_visible(trace, snapshot))
            .collect();
        if traces.is_empty() {
            return;
        }

        ui.heading(title);
        let corner = match self.legend_placement {
            LegendPlacement::Footer => Corner::LeftBottom,
            LegendPlacement::Side => Corner::RightTop,
        };
        let mut plot = Plot::new(format!("plot-{title}"))
            .height(220.0)
            .legend(Legend::default().position(corner))
            .include_y(0.0);
        if unit == Unit::Percent {
            plot = plot.include_y(100.0);
        }

        plot.show(ui, |plot_ui| {
            if unit == Unit::Celsius {
                plot_ui.hline(
                    HLine::new(85.0)
                        .color(Color32::YELLOW)
                        .name("Warning 85 °C"),
                );
                plot_ui.hline(HLine::new(95.0).color(Color32::RED).name("Hot 95 °C"));
            }
            for trace in traces {
                let color = trace_color(&trace.key);
                for (segment_index, segment) in trace.segments.iter().enumerate() {
                    let points: Vec<[f64; 2]> = segment
                        .iter()
                        .map(|(time_ns, value)| {
                            let value = if unit == Unit::Hertz {
                                *value / 1_000_000_000.0
                            } else {
                                *value
                            };
                            [*time_ns as f64 / 1_000_000_000.0, value]
                        })
                        .collect();
                    let label = if segment_index == 0 {
                        format!(
                            "{} · {} · {}",
                            trace.display_name, trace.key.entity_id, trace.source_name
                        )
                    } else {
                        String::new()
                    };
                    plot_ui.line(Line::new(PlotPoints::from(points)).color(color).name(label));
                }
            }
        });
        ui.separator();
    }

    fn sidebar(&mut self, ui: &mut egui::Ui, snapshot: &PresentationSnapshot) {
        ui.heading("Display");
        ui.add(
            egui::Slider::new(&mut self.display_window_secs, 10.0..=300.0)
                .text("window (seconds)")
                .logarithmic(true),
        );
        ui.horizontal(|ui| {
            ui.label("Legend");
            ui.radio_value(
                &mut self.legend_placement,
                LegendPlacement::Footer,
                "Footer",
            );
            ui.radio_value(&mut self.legend_placement, LegendPlacement::Side, "Side");
        });

        ui.collapsing("Font settings", |ui| {
            ui.add(egui::Slider::new(&mut self.font_size, 10.0..=28.0).text("size"));
            ui.color_edit_button_srgba(&mut self.font_color);
        });
        ui.separator();
        ui.heading("Sensors");

        for group in &snapshot.groups {
            let group_visible = self
                .group_visibility
                .entry(group.key.clone())
                .or_insert(true);
            ui.horizontal(|ui| {
                ui.checkbox(group_visible, group.display_name.as_str());
                if let Some(value) = latest_temperature(group, snapshot) {
                    let color = if value >= 95.0 {
                        Color32::RED
                    } else if value >= 85.0 {
                        Color32::YELLOW
                    } else {
                        Color32::LIGHT_GREEN
                    };
                    ui.colored_label(color, format!("{value:.1} °C"));
                }
            });
            ui.indent(format!("group-{}", group.key), |ui| {
                for key in &group.trace_keys {
                    if let Some(trace) = snapshot.traces.iter().find(|trace| &trace.key == key) {
                        let visible = self.trace_visibility.entry(key.clone()).or_insert(true);
                        ui.checkbox(
                            visible,
                            format!("{} ({})", trace.display_name, trace.key.entity_id),
                        );
                    }
                }
            });
        }

        ui.separator();
        ui.collapsing("Provider status", |ui| {
            for status in &snapshot.statuses {
                if status.capability.is_available() {
                    continue;
                }
                let reason = match &status.capability {
                    CapabilityState::Available => continue,
                    CapabilityState::Unsupported(reason)
                    | CapabilityState::PermissionDenied(reason)
                    | CapabilityState::TemporarilyUnavailable(reason) => reason,
                };
                ui.label(
                    RichText::new(format!(
                        "{} · {}: {}",
                        status.display_name, status.key.entity_id, reason
                    ))
                    .color(Color32::LIGHT_RED),
                );
            }
        });
    }
}

impl eframe::App for App {
    fn update(&mut self, context: &egui::Context, _frame: &mut eframe::Frame) {
        if self.last_sample.elapsed() >= Duration::from_secs(1) {
            self.coordinator.collect_once();
            self.last_sample = Instant::now();
        }
        context.request_repaint_after(Duration::from_secs(1));
        self.apply_font(context);

        let snapshot = PresentationSnapshot::from_model(
            self.coordinator.model(),
            Duration::from_secs_f64(self.display_window_secs),
        );

        egui::TopBottomPanel::top("header").show(context, |ui| {
            ui.heading("SIA - System Information Analyzer");
            ui.label("Live system, thermal, frequency, and optional NVIDIA monitoring");
        });

        egui::SidePanel::left("controls")
            .resizable(true)
            .default_width(280.0)
            .show(context, |ui| self.sidebar(ui, &snapshot));

        egui::CentralPanel::default().show(context, |ui| {
            if snapshot.traces.is_empty() {
                ui.label("Waiting for available system metrics…");
            }
            self.draw_plot(ui, &snapshot, "Utilization (%)", Unit::Percent);
            self.draw_plot(ui, &snapshot, "Temperatures (°C)", Unit::Celsius);
            self.draw_plot(ui, &snapshot, "Frequencies (GHz)", Unit::Hertz);
        });
    }
}

fn latest_temperature(group: &sia::SensorGroup, snapshot: &PresentationSnapshot) -> Option<f64> {
    group
        .trace_keys
        .iter()
        .filter_map(|key| snapshot.traces.iter().find(|trace| &trace.key == key))
        .filter(|trace| trace.unit == Unit::Celsius)
        .filter_map(|trace| trace.points.last().map(|(_, value)| *value))
        .max_by(|left, right| left.total_cmp(right))
}

fn trace_color(key: &SeriesKey) -> Color32 {
    const COLORS: [(u8, u8, u8); 12] = [
        (244, 67, 54),
        (33, 150, 243),
        (76, 175, 80),
        (255, 152, 0),
        (156, 39, 176),
        (0, 150, 136),
        (205, 220, 57),
        (233, 30, 99),
        (121, 85, 72),
        (63, 81, 181),
        (3, 169, 244),
        (255, 87, 34),
    ];
    let hash = key
        .metric_id
        .0
        .as_bytes()
        .iter()
        .copied()
        .chain(key.entity_id.bytes())
        .fold(0usize, |value, byte| {
            value.wrapping_mul(31).wrapping_add(byte as usize)
        });
    let (red, green, blue) = COLORS[hash % COLORS.len()];
    Color32::from_rgb(red, green, blue)
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
