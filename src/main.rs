use eframe::egui::{self, Color32, FontFamily, FontId, RichText, TextStyle};
use egui_plot::{Line, Plot, PlotBounds, PlotPoints};
use sia::{
    discover_hwmon, CanonicalUnit, CapabilityState, Collector, EntityId, EntityKind,
    LegendPlacement, LinuxMonotonicClock, MetricDescriptor, MetricId, MetricProvider, MetricStore,
    MetricValue, PresentationProjection, PresentedSeries, ProviderId, ProviderReading,
    SampleStatus, SeriesKey, SourceTiming, TemporalSemantics, ValueKind,
};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};
use sysinfo::{CpuExt, System, SystemExt};
fn descriptor(
    id: impl Into<MetricId>,
    name: impl Into<String>,
    kind: EntityKind,
    unit: CanonicalUnit,
    temporal: TemporalSemantics,
    provider: ProviderId,
    semantics: impl Into<String>,
    group: Option<String>,
) -> MetricDescriptor {
    MetricDescriptor {
        metric_id: id.into(),
        display_name: name.into(),
        entity_kind: kind,
        unit,
        value_kind: ValueKind::Gauge,
        temporal_semantics: temporal,
        provider,
        capability_state: CapabilityState::Available,
        source_resolution_ns: None,
        source_semantics: semantics.into(),
        comparability_group: group,
        semantics_version: 1,
    }
}
#[derive(Clone)]
struct PointSource {
    metric_id: MetricId,
    entity_id: EntityId,
    path: PathBuf,
    scale: f64,
}
#[cfg(feature = "nvidia")]
mod nvidia {
    use super::*;
    use nvml_wrapper::{
        enum_wrappers::device::{Clock as NvClock, TemperatureSensor},
        Nvml,
    };
    #[derive(Clone)]
    struct Device {
        index: u32,
        entity_id: EntityId,
    }
    pub struct Runtime {
        nvml: Nvml,
        devices: Vec<Device>,
        failures: Vec<(u32, String)>,
    }
    impl Runtime {
        pub fn discover() -> Result<Self, String> {
            let nvml = Nvml::init().map_err(|e| e.to_string())?;
            let count = nvml.device_count().map_err(|e| e.to_string())?;
            if count == 0 {
                return Err("NVML reported no NVIDIA devices".into());
            }
            let driver = nvml
                .sys_driver_version()
                .unwrap_or_else(|_| "unknown".into());
            let (mut devices, mut failures) = (Vec::new(), Vec::new());
            for index in 0..count {
                match nvml.device_by_index(index) {
                    Ok(device) => {
                        let identity = sia::NvidiaIdentity {
                            uuid: device.uuid().ok(),
                            pci_address: device
                                .pci_info()
                                .map(|p| p.bus_id)
                                .unwrap_or_else(|_| format!("unresolved-index-{index}")),
                            driver_identity: driver.clone(),
                        };
                        devices.push(Device {
                            index,
                            entity_id: identity.durable_entity_id(),
                        });
                    }
                    Err(e) => failures.push((index, e.to_string())),
                }
            }
            if devices.is_empty() {
                return Err(failures
                    .first()
                    .map(|x| x.1.clone())
                    .unwrap_or_else(|| "no NVIDIA device could be acquired".into()));
            }
            devices.sort_by(|a, b| a.entity_id.cmp(&b.entity_id));
            Ok(Self {
                nvml,
                devices,
                failures,
            })
        }
        pub fn descriptors(&self) -> Vec<MetricDescriptor> {
            let mut out: Vec<_> = [
                (
                    "gpu.utilization",
                    "GPU utilization",
                    CanonicalUnit::Percent,
                    "direct NVML utilization poll",
                ),
                (
                    "gpu.memory.utilization",
                    "VRAM utilization",
                    CanonicalUnit::Percent,
                    "used bytes divided by total bytes from a direct NVML memory poll",
                ),
                (
                    "gpu.temperature",
                    "GPU temperature",
                    CanonicalUnit::Celsius,
                    "direct NVML temperature poll",
                ),
                (
                    "gpu.clock.graphics",
                    "GPU Graphics",
                    CanonicalUnit::Hertz,
                    "direct NVML graphics-clock poll",
                ),
                (
                    "gpu.clock.sm",
                    "GPU SM",
                    CanonicalUnit::Hertz,
                    "direct NVML SM-clock poll",
                ),
                (
                    "gpu.clock.memory",
                    "GPU Memory",
                    CanonicalUnit::Hertz,
                    "direct NVML memory-clock poll",
                ),
                (
                    "gpu.clock.video",
                    "GPU Video",
                    CanonicalUnit::Hertz,
                    "direct NVML video-clock poll",
                ),
            ]
            .into_iter()
            .map(|(id, name, unit, semantics)| {
                descriptor(
                    id,
                    name,
                    EntityKind::Gpu,
                    unit,
                    TemporalSemantics::PointSample,
                    ProviderId::Nvml,
                    semantics,
                    Some(if id == "gpu.temperature" {
                        "gpu".into()
                    } else {
                        "nvidia".into()
                    }),
                )
            })
            .collect();
            out.push(MetricDescriptor {
                metric_id: "gpu.discovery".into(),
                display_name: "NVIDIA device discovery".into(),
                entity_kind: EntityKind::Gpu,
                unit: CanonicalUnit::State,
                value_kind: ValueKind::State,
                temporal_semantics: TemporalSemantics::PointSample,
                provider: ProviderId::Nvml,
                capability_state: CapabilityState::Available,
                source_resolution_ns: None,
                source_semantics: "per-index NVML device acquisition status".into(),
                comparability_group: Some("gpu".into()),
                semantics_version: 1,
            });
            out
        }
        fn ok(id: &str, e: &EntityId, v: f64, t: u64) -> ProviderReading {
            ProviderReading::ok(id, e.clone(), MetricValue::Float(v), SourceTiming::point(t))
        }
        fn err(id: &str, e: &EntityId, r: impl ToString, t: u64) -> ProviderReading {
            ProviderReading::error(id, e.clone(), r.to_string(), SourceTiming::point(t))
        }
        pub fn collect(&self, t: u64) -> Vec<ProviderReading> {
            let mut out = Vec::new();
            for (index, reason) in &self.failures {
                out.push(ProviderReading::error(
                    "gpu.discovery",
                    EntityId::new(format!("nvidia:acquisition-index:{index}")),
                    reason.clone(),
                    SourceTiming::point(t),
                ));
            }
            for d in &self.devices {
                let dev = match self.nvml.device_by_index(d.index) {
                    Ok(v) => v,
                    Err(e) => {
                        for id in [
                            "gpu.utilization",
                            "gpu.memory.utilization",
                            "gpu.temperature",
                            "gpu.clock.graphics",
                            "gpu.clock.sm",
                            "gpu.clock.memory",
                            "gpu.clock.video",
                        ] {
                            out.push(Self::err(id, &d.entity_id, &e, t));
                        }
                        continue;
                    }
                };
                out.push(match dev.utilization_rates() {
                    Ok(v) => Self::ok("gpu.utilization", &d.entity_id, v.gpu as f64, t),
                    Err(e) => Self::err("gpu.utilization", &d.entity_id, e, t),
                });
                out.push(match dev.memory_info() {
                    Ok(v) if v.total > 0 => Self::ok(
                        "gpu.memory.utilization",
                        &d.entity_id,
                        v.used as f64 * 100.0 / v.total as f64,
                        t,
                    ),
                    Ok(_) => Self::err(
                        "gpu.memory.utilization",
                        &d.entity_id,
                        "NVML reported zero total memory",
                        t,
                    ),
                    Err(e) => Self::err("gpu.memory.utilization", &d.entity_id, e, t),
                });
                out.push(match dev.temperature(TemperatureSensor::Gpu) {
                    Ok(v) => Self::ok("gpu.temperature", &d.entity_id, v as f64, t),
                    Err(e) => Self::err("gpu.temperature", &d.entity_id, e, t),
                });
                for (id, clock) in [
                    ("gpu.clock.graphics", NvClock::Graphics),
                    ("gpu.clock.sm", NvClock::SM),
                    ("gpu.clock.memory", NvClock::Memory),
                    ("gpu.clock.video", NvClock::Video),
                ] {
                    out.push(match dev.clock_info(clock) {
                        Ok(v) => Self::ok(id, &d.entity_id, v as f64 * 1_000_000.0, t),
                        Err(e) => Self::err(id, &d.entity_id, e, t),
                    });
                }
            }
            out
        }
    }
}
struct SystemProvider {
    system: System,
    descriptors: Vec<MetricDescriptor>,
    temperatures: Vec<PointSource>,
    frequencies: Vec<PointSource>,
    previous_cpu: Option<u64>,
    #[cfg(feature = "nvidia")]
    nvidia: Option<nvidia::Runtime>,
}
impl SystemProvider {
    fn new() -> Self {
        let mut descriptors = vec![
            descriptor(
                "cpu.utilization",
                "CPU utilization",
                EntityKind::Cpu,
                CanonicalUnit::Percent,
                TemporalSemantics::IntervalAverage,
                ProviderId::Sysinfo,
                "busy time averaged between successive CLOCK_MONOTONIC refresh observations",
                Some("cpu".into()),
            ),
            descriptor(
                "memory.utilization",
                "RAM utilization",
                EntityKind::System,
                CanonicalUnit::Percent,
                TemporalSemantics::PointSample,
                ProviderId::Sysinfo,
                "used bytes divided by total bytes at observation time",
                Some("memory".into()),
            ),
        ];
        let temperatures = discover_hwmon("/sys/class/hwmon")
            .into_iter()
            .map(|sensor| {
                let metric_id = MetricId::new(format!(
                    "{}.{}",
                    sensor.metric_id.0,
                    slug(&sensor.stable_parent)
                ));
                let (group, warn, hot) = taxonomy(&sensor.device_name);
                descriptors.push(descriptor(
                    metric_id.clone(),
                    format!("{}|{}|{}|{}", sensor.label, group, warn, hot),
                    EntityKind::Other(group.clone()),
                    CanonicalUnit::Celsius,
                    TemporalSemantics::PointSample,
                    ProviderId::Sysfs,
                    "direct Linux hwmon input read",
                    Some(group),
                ));
                PointSource {
                    metric_id,
                    entity_id: sensor.entity_id,
                    path: sensor.input_path,
                    scale: 0.001,
                }
            })
            .collect();
        let frequencies = frequencies();
        for s in &frequencies {
            descriptors.push(descriptor(
                s.metric_id.clone(),
                format!("CPU Core {}", s.entity_id.0.trim_start_matches("cpu:")),
                EntityKind::Cpu,
                CanonicalUnit::Hertz,
                TemporalSemantics::PointSample,
                ProviderId::Sysfs,
                "direct cpufreq sysfs read",
                Some("cpu-frequency".into()),
            ))
        }
        #[cfg(feature = "nvidia")]
        let nvidia = match nvidia::Runtime::discover() {
            Ok(v) => {
                descriptors.extend(v.descriptors());
                Some(v)
            }
            Err(reason) => {
                descriptors.push(MetricDescriptor {
                    metric_id: "gpu.provider".into(),
                    display_name: "NVIDIA provider".into(),
                    entity_kind: EntityKind::Gpu,
                    unit: CanonicalUnit::State,
                    value_kind: ValueKind::State,
                    temporal_semantics: TemporalSemantics::PointSample,
                    provider: ProviderId::Nvml,
                    capability_state: CapabilityState::Unsupported { reason },
                    source_resolution_ns: None,
                    source_semantics: "NVML provider capability".into(),
                    comparability_group: Some("gpu".into()),
                    semantics_version: 1,
                });
                None
            }
        };
        Self {
            system: System::new_all(),
            descriptors,
            temperatures,
            frequencies,
            previous_cpu: None,
            #[cfg(feature = "nvidia")]
            nvidia,
        }
    }
}
impl MetricProvider for SystemProvider {
    fn descriptors(&self) -> Vec<MetricDescriptor> {
        self.descriptors.clone()
    }
    fn collect(&mut self, t: u64) -> Vec<ProviderReading> {
        let mut out = Vec::new();
        self.system.refresh_cpu();
        if let Some(start) = self.previous_cpu.replace(t) {
            for (index, cpu) in self.system.cpus().iter().enumerate() {
                out.push(ProviderReading::ok(
                    "cpu.utilization",
                    EntityId::new(format!("cpu:{index}")),
                    MetricValue::Float(cpu.cpu_usage() as f64),
                    SourceTiming::window(TemporalSemantics::IntervalAverage, start, t),
                ));
            }
        }
        self.system.refresh_memory();
        let total = self.system.total_memory();
        if total == 0 {
            out.push(ProviderReading::error(
                "memory.utilization",
                "system",
                "sysinfo reported zero total memory",
                SourceTiming::point(t),
            ))
        } else {
            out.push(ProviderReading::ok(
                "memory.utilization",
                "system",
                MetricValue::Float(self.system.used_memory() as f64 * 100.0 / total as f64),
                SourceTiming::point(t),
            ))
        }
        for s in self.temperatures.iter().chain(&self.frequencies) {
            out.push(match read(s) {
                Ok(v) => ProviderReading::ok(
                    s.metric_id.clone(),
                    s.entity_id.clone(),
                    MetricValue::Float(v),
                    SourceTiming::point(t),
                ),
                Err(e) => ProviderReading::error(
                    s.metric_id.clone(),
                    s.entity_id.clone(),
                    e,
                    SourceTiming::point(t),
                ),
            });
        }
        #[cfg(feature = "nvidia")]
        if let Some(nvidia) = &self.nvidia {
            out.extend(nvidia.collect(t));
        }
        out
    }
}
fn slug(v: &str) -> String {
    v.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}
fn read(s: &PointSource) -> Result<f64, String> {
    fs::read_to_string(&s.path)
        .map_err(|e| e.to_string())?
        .trim()
        .parse::<f64>()
        .map(|v| v * s.scale)
        .map_err(|e| e.to_string())
}
fn frequencies() -> Vec<PointSource> {
    let Ok(entries) = fs::read_dir("/sys/devices/system/cpu") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for e in entries.flatten() {
        let p = e.path();
        let Some(name) = p.file_name().and_then(|x| x.to_str()) else {
            continue;
        };
        let Ok(index) = name.trim_start_matches("cpu").parse::<usize>() else {
            continue;
        };
        let root = p.join("cpufreq");
        let a = root.join("scaling_cur_freq");
        let b = root.join("cpuinfo_cur_freq");
        let path = if a.exists() {
            a
        } else if b.exists() {
            b
        } else {
            continue;
        };
        out.push(PointSource {
            metric_id: MetricId::new(format!("cpu.frequency.{index}")),
            entity_id: EntityId::new(format!("cpu:{index}")),
            path,
            scale: 1_000.0,
        });
    }
    out.sort_by(|a, b| a.entity_id.cmp(&b.entity_id));
    out
}
fn taxonomy(v: &str) -> (String, f64, f64) {
    let n = v.to_ascii_lowercase();
    if n.contains("coretemp") || n.contains("k10temp") || n.contains("cpu") {
        ("cpu".into(), 90.0, 100.0)
    } else if n.contains("gpu") || n.contains("nvidia") || n.contains("amdgpu") {
        ("gpu".into(), 85.0, 95.0)
    } else if n.contains("nvme") {
        ("nvme".into(), 70.0, 80.0)
    } else if n.contains("spd") {
        ("memory".into(), 70.0, 85.0)
    } else if n.contains("wifi") || n.contains("iwlwifi") {
        ("wi-fi".into(), 80.0, 90.0)
    } else if n.contains("eth") || n.contains("r8169") || n.contains("igc") || n.contains("e1000") {
        ("ethernet".into(), 80.0, 90.0)
    } else {
        (v.into(), 90.0, 100.0)
    }
}
struct App {
    collector: Collector<LinuxMonotonicClock, SystemProvider>,
    store: MetricStore,
    last: Instant,
    period: Duration,
    window: f64,
    legend: LegendPlacement,
    font: f32,
    pending_font: f32,
    font_color: Color32,
    pending_color: Color32,
    live: bool,
    visible: BTreeMap<SeriesKey, bool>,
    initialized: bool,
    effective_memory: bool,
    last_error: Option<String>,
}
impl App {
    fn new() -> Self {
        Self {
            collector: Collector::new(LinuxMonotonicClock, SystemProvider::new()),
            store: MetricStore::default(),
            last: Instant::now() - Duration::from_secs(2),
            period: Duration::from_secs(1),
            window: 120.0,
            legend: LegendPlacement::Footer,
            font: 14.0,
            pending_font: 14.0,
            font_color: Color32::WHITE,
            pending_color: Color32::WHITE,
            live: false,
            visible: BTreeMap::new(),
            initialized: false,
            effective_memory: false,
            last_error: None,
        }
    }
    fn collect(&mut self) {
        if self.last.elapsed() < self.period {
            return;
        }
        self.last = Instant::now();
        match self.collector.collect() {
            Ok(b) => {
                if let Err(e) = self.store.ingest(b) {
                    self.last_error = Some(e.to_string())
                }
            }
            Err(e) => self.last_error = Some(e.to_string()),
        }
    }
    fn initialize(&mut self, p: &PresentationProjection) {
        if self.initialized {
            return;
        }
        let mut preferred: BTreeMap<String, (u8, SeriesKey)> = BTreeMap::new();
        for s in &p.series {
            let temp = s.descriptor.unit == CanonicalUnit::Celsius;
            self.visible.insert(
                s.key.clone(),
                !temp && s.key.metric_id.0 != "gpu.clock.video",
            );
            if temp {
                let group = s
                    .descriptor
                    .comparability_group
                    .clone()
                    .unwrap_or_else(|| "other".into());
                let candidate = (preference(&s.descriptor.display_name), s.key.clone());
                if preferred.get(&group).map_or(true, |old| candidate < *old) {
                    preferred.insert(group, candidate);
                }
            }
        }
        for (_, (_, key)) in preferred {
            self.visible.insert(key, true);
        }
        self.initialized = true
    }
    fn typography(&self, ctx: &egui::Context) {
        let mut style = (*ctx.style()).clone();
        for text in [
            TextStyle::Body,
            TextStyle::Button,
            TextStyle::Monospace,
            TextStyle::Small,
        ] {
            style
                .text_styles
                .insert(text, FontId::new(self.font, FontFamily::Proportional));
        }
        style.visuals.override_text_color = Some(self.font_color);
        ctx.set_style(style)
    }
    fn shown(&self, s: &PresentedSeries) -> bool {
        self.visible.get(&s.key).copied().unwrap_or(false)
    }
    fn legend(&mut self, ui: &mut egui::Ui, p: &PresentationProjection) {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Legend:").strong());
            for (index, s) in ordered(p) {
                if !self.shown(s) {
                    continue;
                }
                let mut label = name(s);
                if let Some(r) = s.records.last() {
                    if let Some(reason) = r.status.reason() {
                        label.push_str(&format!(" (unavailable: {reason})"));
                    } else if s.descriptor.unit == CanonicalUnit::Celsius {
                        if let Some(v) = r.value.as_ref().and_then(MetricValue::as_f64) {
                            let (_, warn, hot) = temp_meta(&s.descriptor.display_name);
                            if v >= hot {
                                label.push_str(" 🔥")
                            } else if v >= warn {
                                label.push_str(" 🥵")
                            }
                        }
                    }
                }
                ui.colored_label(color(index), "●");
                ui.label(label);
            }
        })
    }
    fn summary(&self, ui: &mut egui::Ui, p: &PresentationProjection) {
        ui.horizontal_wrapped(|ui| {
            for id in [
                "cpu.utilization",
                "memory.utilization",
                "gpu.utilization",
                "gpu.memory.utilization",
            ] {
                let v: Vec<_> = p
                    .series
                    .iter()
                    .filter(|s| s.key.metric_id.0 == id)
                    .filter_map(|s| {
                        let r = s.records.last()?;
                        matches!(r.status, SampleStatus::Ok)
                            .then(|| r.value.as_ref()?.as_f64())
                            .flatten()
                    })
                    .collect();
                if !v.is_empty() {
                    ui.label(format!(
                        "{}: {:.1}%",
                        summary_name(id),
                        v.iter().sum::<f64>() / v.len() as f64
                    ));
                }
            }
        })
    }
    fn plot(
        &self,
        ui: &mut egui::Ui,
        p: &PresentationProjection,
        title: &str,
        min: f64,
        max: f64,
        accept: impl Fn(&PresentedSeries) -> bool,
    ) {
        let selected: Vec<_> = p
            .series
            .iter()
            .enumerate()
            .filter(|(_, s)| self.shown(s) && accept(s))
            .collect();
        if selected.is_empty() {
            return;
        }
        ui.heading(title);
        Plot::new(title)
            .height(190.0)
            .allow_drag(false)
            .allow_scroll(false)
            .allow_zoom(false)
            .show(ui, |plot| {
                plot.set_plot_bounds(PlotBounds::from_min_max(
                    [min, f64::NEG_INFINITY],
                    [max, f64::INFINITY],
                ));
                for (index, s) in &selected {
                    for (segment_index, segment) in s.segments.iter().enumerate() {
                        let points: Vec<_> = segment
                            .iter()
                            .filter_map(|point| {
                                let x = point.mono_ns as f64 / 1_000_000_000.0;
                                (x >= min).then_some([
                                    x,
                                    display_value(
                                        &s.key.metric_id.0,
                                        point.value,
                                        self.effective_memory,
                                    ),
                                ])
                            })
                            .collect();
                        if points.is_empty() {
                            continue;
                        }
                        let mut line = Line::new(PlotPoints::from(points)).color(color(*index));
                        if segment_index == 0 {
                            line = line.name(name(s));
                        }
                        plot.line(line);
                    }
                }
            });
    }
    fn settings(&mut self, ui: &mut egui::Ui, p: &PresentationProjection) {
        ui.heading("Display");
        ui.horizontal_wrapped(|ui| {
            ui.label("Window (seconds):");
            ui.add(egui::Slider::new(&mut self.window, 30.0..=900.0));
            egui::ComboBox::from_label("Legend")
                .selected_text(match self.legend {
                    LegendPlacement::Footer => "Footer",
                    LegendPlacement::Side => "Side",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.legend, LegendPlacement::Footer, "Footer");
                    ui.selectable_value(&mut self.legend, LegendPlacement::Side, "Side");
                });
            ui.label("Font size");
            ui.add(egui::Slider::new(&mut self.pending_font, 10.0..=22.0));
            ui.color_edit_button_srgba(&mut self.pending_color);
            if self.live {
                self.font = self.pending_font;
                self.font_color = self.pending_color;
            }
            if ui.button("Apply font").clicked() {
                self.font = self.pending_font;
                self.font_color = self.pending_color;
            }
            ui.toggle_value(&mut self.live, "Live preview");
            ui.toggle_value(
                &mut self.effective_memory,
                "Effective GPU memory clock (x2)",
            );
        });
        ui.separator();
        ui.heading("Traces");
        egui::Grid::new("traces")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                for (_, s) in ordered(p) {
                    let visible = self.visible.entry(s.key.clone()).or_default();
                    ui.checkbox(visible, name(s));
                    if let Some(r) = s.records.last() {
                        ui.label(match &r.status {
                            SampleStatus::Ok => "available".into(),
                            status => status
                                .reason()
                                .map(|x| format!("unavailable: {x}"))
                                .unwrap_or_else(|| "unavailable".into()),
                        });
                    } else {
                        ui.label("awaiting sample");
                    }
                    ui.end_row();
                }
            })
    }
}
impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.collect();
        self.typography(ctx);
        let p = self.store.project();
        self.initialize(&p);
        if self.legend == LegendPlacement::Side {
            egui::SidePanel::right("side-legend").show(ctx, |ui| self.legend(ui, &p));
        }
        if self.legend == LegendPlacement::Footer {
            egui::TopBottomPanel::bottom("footer-legend").show(ctx, |ui| self.legend(ui, &p));
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("SIA - System Information Analyzer");
            self.summary(ui, &p);
            if let Some(e) = &self.last_error {
                ui.colored_label(Color32::LIGHT_RED, e);
            }
            let max = p
                .series
                .iter()
                .flat_map(|s| s.records.iter())
                .map(|r| r.mono_ns as f64 / 1_000_000_000.0)
                .fold(0.0, f64::max);
            let min = (max - self.window).max(0.0);
            self.plot(ui, &p, "Utilization", min, max, |s| {
                s.descriptor.unit == CanonicalUnit::Percent
            });
            self.plot(ui, &p, "Temperatures", min, max, |s| {
                s.descriptor.unit == CanonicalUnit::Celsius
            });
            self.plot(ui, &p, "Frequencies", min, max, |s| {
                s.descriptor.unit == CanonicalUnit::Hertz
            });
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| self.settings(ui, &p));
        });
        ctx.request_repaint_after(Duration::from_millis(100));
    }
}
fn ordered(p: &PresentationProjection) -> Vec<(usize, &PresentedSeries)> {
    let mut out: Vec<_> = p.series.iter().enumerate().collect();
    out.sort_by_key(|(_, s)| {
        (
            rank(
                s.descriptor
                    .comparability_group
                    .as_deref()
                    .unwrap_or("other"),
            ),
            name(s),
            s.key.entity_id.clone(),
        )
    });
    out
}
fn rank(v: &str) -> u8 {
    match v.to_ascii_lowercase().as_str() {
        "cpu" => 0,
        "gpu" | "nvidia" => 1,
        "nvme" => 2,
        "memory" => 3,
        "wi-fi" => 4,
        "ethernet" => 5,
        _ => 6,
    }
}
fn preference(v: &str) -> u8 {
    let v = v.to_ascii_lowercase();
    if v.contains("package") {
        0
    } else if v.contains("composite") {
        1
    } else if v.contains("system") {
        2
    } else {
        3
    }
}
fn temp_meta(v: &str) -> (&str, f64, f64) {
    let mut p = v.split('|');
    let label = p.next().unwrap_or(v);
    let _ = p.next();
    let warn = p.next().and_then(|x| x.parse().ok()).unwrap_or(90.0);
    let hot = p.next().and_then(|x| x.parse().ok()).unwrap_or(100.0);
    (label, warn, hot)
}
fn name(s: &PresentedSeries) -> String {
    if s.descriptor.unit == CanonicalUnit::Celsius {
        temp_meta(&s.descriptor.display_name).0.into()
    } else if matches!(s.descriptor.entity_kind, EntityKind::Gpu)
        && s.key.entity_id.0.starts_with("nvidia:")
    {
        format!("{} ({})", s.descriptor.display_name, s.key.entity_id)
    } else {
        s.descriptor.display_name.clone()
    }
}
fn display_value(id: &str, v: f64, effective: bool) -> f64 {
    if id == "gpu.clock.memory" && effective {
        v * 2.0
    } else {
        v
    }
}
fn summary_name(id: &str) -> &'static str {
    match id {
        "cpu.utilization" => "CPU",
        "memory.utilization" => "RAM",
        "gpu.utilization" => "GPU",
        "gpu.memory.utilization" => "VRAM",
        _ => "Metric",
    }
}
fn color(i: usize) -> Color32 {
    let colors = [
        Color32::from_rgb(244, 67, 54),
        Color32::from_rgb(33, 150, 243),
        Color32::from_rgb(76, 175, 80),
        Color32::from_rgb(255, 152, 0),
        Color32::from_rgb(156, 39, 176),
        Color32::from_rgb(0, 150, 136),
        Color32::from_rgb(205, 220, 57),
        Color32::from_rgb(233, 30, 99),
    ];
    colors[i % colors.len()]
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
        Box::new(|_| Ok(Box::new(App::new()))),
    )
}
