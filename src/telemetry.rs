use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use sysinfo::{CpuExt, System, SystemExt};

pub const LINUX_CLOCK_MONOTONIC: &str = "linux_clock_monotonic";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeriesKey {
    pub metric_id: String,
    pub entity_id: String,
}

impl SeriesKey {
    pub fn new(metric_id: impl Into<String>, entity_id: impl Into<String>) -> Self {
        Self {
            metric_id: metric_id.into(),
            entity_id: entity_id.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntityKind {
    System,
    Cpu,
    Gpu,
    Disk,
    Net,
    Process,
    Thread,
    Application,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueKind {
    Gauge,
    Counter,
    Rate,
    State,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemporalSemantics {
    PointSample,
    IntervalAverage,
    IntervalDelta,
    CumulativeCounter,
    VendorSampled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapabilityStatus {
    Available,
    Unsupported,
    PermissionDenied,
    TemporarilyUnavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleStatus {
    Ok,
    Stale,
    TemporarilyUnavailable,
    Error,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SampleValue {
    F64(f64),
    I64(i64),
    U64(u64),
    State(String),
}

impl SampleValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F64(value) => Some(*value),
            Self::I64(value) => Some(*value as f64),
            Self::U64(value) => Some(*value as f64),
            Self::State(_) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceResolution {
    pub nanoseconds: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricDescriptor {
    pub metric_id: String,
    pub display_name: String,
    pub entity_kind: EntityKind,
    pub unit: String,
    pub value_kind: ValueKind,
    pub temporal_semantics: TemporalSemantics,
    pub provider: String,
    pub capability_status: CapabilityStatus,
    pub source_resolution: Option<SourceResolution>,
    pub source_semantics: String,
    pub comparability_group: Option<String>,
    pub semantics_version: u32,
    pub entity_id: String,
}

impl MetricDescriptor {
    pub fn series_key(&self) -> SeriesKey {
        SeriesKey::new(self.metric_id.clone(), self.entity_id.clone())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricSample {
    pub mono_ns: u64,
    pub window_start_mono_ns: Option<u64>,
    pub metric_id: String,
    pub entity_id: String,
    pub value: SampleValue,
    pub status: SampleStatus,
}

impl MetricSample {
    pub fn series_key(&self) -> SeriesKey {
        SeriesKey::new(self.metric_id.clone(), self.entity_id.clone())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionMetadata {
    pub clock_domain: String,
    pub descriptors: Vec<MetricDescriptor>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Collection {
    pub clock_domain: String,
    pub observation_mono_ns: u64,
    pub descriptors: Vec<MetricDescriptor>,
    pub samples: Vec<MetricSample>,
}

pub trait Clock {
    fn domain(&self) -> &'static str;
    fn now_mono_ns(&mut self) -> u64;
}

#[derive(Default)]
pub struct NativeClock;

#[cfg(target_os = "linux")]
impl Clock for NativeClock {
    fn domain(&self) -> &'static str {
        LINUX_CLOCK_MONOTONIC
    }

    fn now_mono_ns(&mut self) -> u64 {
        #[repr(C)]
        struct Timespec {
            tv_sec: i64,
            tv_nsec: i64,
        }

        extern "C" {
            fn clock_gettime(clock_id: i32, timespec: *mut Timespec) -> i32;
        }

        const CLOCK_MONOTONIC: i32 = 1;
        let mut value = Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        let result = unsafe { clock_gettime(CLOCK_MONOTONIC, &mut value) };
        assert_eq!(result, 0, "clock_gettime(CLOCK_MONOTONIC) failed");
        (value.tv_sec as u64)
            .saturating_mul(1_000_000_000)
            .saturating_add(value.tv_nsec as u64)
    }
}

#[cfg(not(target_os = "linux"))]
compile_error!("SIA's native clock contract currently requires Linux");

pub trait CapabilityDiscovery {
    fn descriptors(&self) -> &[MetricDescriptor];
}

pub trait MetricProvider: CapabilityDiscovery {
    fn collect(
        &mut self,
        observation_mono_ns: u64,
        previous_observation_mono_ns: Option<u64>,
    ) -> Vec<MetricSample>;
}

pub trait SessionSource {
    fn session_metadata(&self) -> SessionMetadata;
    fn next_collection(&mut self) -> Collection;
}

pub struct Collector<C> {
    clock: C,
    providers: Vec<Box<dyn MetricProvider>>,
    descriptors: Vec<MetricDescriptor>,
    previous_observation_mono_ns: Option<u64>,
}

impl<C: Clock> Collector<C> {
    pub fn new(clock: C, providers: Vec<Box<dyn MetricProvider>>) -> Self {
        let descriptors = providers
            .iter()
            .flat_map(|provider| provider.descriptors().iter().cloned())
            .collect();
        Self {
            clock,
            providers,
            descriptors,
            previous_observation_mono_ns: None,
        }
    }

    pub fn descriptors(&self) -> &[MetricDescriptor] {
        &self.descriptors
    }

    pub fn session_metadata(&self) -> SessionMetadata {
        SessionMetadata {
            clock_domain: self.clock.domain().to_owned(),
            descriptors: self.descriptors.clone(),
        }
    }

    pub fn collect(&mut self) -> Collection {
        let observation_mono_ns = self.clock.now_mono_ns();
        let previous = self.previous_observation_mono_ns;
        let samples = self
            .providers
            .iter_mut()
            .flat_map(|provider| provider.collect(observation_mono_ns, previous))
            .collect();
        self.descriptors = self
            .providers
            .iter()
            .flat_map(|provider| provider.descriptors().iter().cloned())
            .collect();
        self.previous_observation_mono_ns = Some(observation_mono_ns);
        Collection {
            clock_domain: self.clock.domain().to_owned(),
            observation_mono_ns,
            descriptors: self.descriptors.clone(),
            samples,
        }
    }
}

impl<C: Clock> SessionSource for Collector<C> {
    fn session_metadata(&self) -> SessionMetadata {
        Collector::session_metadata(self)
    }

    fn next_collection(&mut self) -> Collection {
        self.collect()
    }
}

fn descriptor(
    metric_id: &str,
    display_name: String,
    entity_kind: EntityKind,
    unit: &str,
    value_kind: ValueKind,
    temporal_semantics: TemporalSemantics,
    provider: &str,
    capability_status: CapabilityStatus,
    source_resolution: Option<SourceResolution>,
    source_semantics: &str,
    comparability_group: Option<&str>,
    entity_id: String,
) -> MetricDescriptor {
    MetricDescriptor {
        metric_id: metric_id.to_owned(),
        display_name,
        entity_kind,
        unit: unit.to_owned(),
        value_kind,
        temporal_semantics,
        provider: provider.to_owned(),
        capability_status,
        source_resolution,
        source_semantics: source_semantics.to_owned(),
        comparability_group: comparability_group.map(str::to_owned),
        semantics_version: 1,
        entity_id,
    }
}

pub struct SystemProvider {
    system: System,
    descriptors: Vec<MetricDescriptor>,
}

impl SystemProvider {
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        let descriptors = vec![
            descriptor(
                "cpu.utilization",
                "CPU utilization".to_owned(),
                EntityKind::Cpu,
                "percent",
                ValueKind::Gauge,
                TemporalSemantics::IntervalAverage,
                "sysinfo",
                CapabilityStatus::Available,
                None,
                "Mean logical-CPU utilization over the interval between sysinfo CPU refreshes.",
                Some("host_cpu_busy_percent"),
                "cpu:aggregate".to_owned(),
            ),
            descriptor(
                "memory.used_ratio",
                "RAM utilization".to_owned(),
                EntityKind::System,
                "percent",
                ValueKind::Gauge,
                TemporalSemantics::PointSample,
                "sysinfo",
                CapabilityStatus::Available,
                None,
                "Used physical memory divided by total physical memory at observation.",
                Some("host_memory_occupancy_percent"),
                "system:memory".to_owned(),
            ),
        ];
        Self {
            system,
            descriptors,
        }
    }
}

impl Default for SystemProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl CapabilityDiscovery for SystemProvider {
    fn descriptors(&self) -> &[MetricDescriptor] {
        &self.descriptors
    }
}

impl MetricProvider for SystemProvider {
    fn collect(
        &mut self,
        observation_mono_ns: u64,
        previous_observation_mono_ns: Option<u64>,
    ) -> Vec<MetricSample> {
        self.system.refresh_cpu();
        self.system.refresh_memory();
        let mut samples = Vec::new();
        if let Some(window_start_mono_ns) = previous_observation_mono_ns {
            if self.system.cpus().is_empty() {
                samples.push(unavailable_sample(
                    &self.descriptors[0],
                    observation_mono_ns,
                    SampleStatus::TemporarilyUnavailable,
                    "sysinfo returned no logical CPUs".to_owned(),
                ));
            } else {
                let cpu = self
                    .system
                    .cpus()
                    .iter()
                    .map(|cpu| cpu.cpu_usage())
                    .sum::<f32>()
                    / self.system.cpus().len() as f32;
                samples.push(MetricSample {
                    mono_ns: observation_mono_ns,
                    window_start_mono_ns: Some(window_start_mono_ns),
                    metric_id: "cpu.utilization".to_owned(),
                    entity_id: "cpu:aggregate".to_owned(),
                    value: SampleValue::F64(cpu as f64),
                    status: SampleStatus::Ok,
                });
            }
        }
        let total = self.system.total_memory() as f64;
        if total > 0.0 {
            samples.push(MetricSample {
                mono_ns: observation_mono_ns,
                window_start_mono_ns: None,
                metric_id: "memory.used_ratio".to_owned(),
                entity_id: "system:memory".to_owned(),
                value: SampleValue::F64(self.system.used_memory() as f64 / total * 100.0),
                status: SampleStatus::Ok,
            });
        } else {
            samples.push(unavailable_sample(
                &self.descriptors[1],
                observation_mono_ns,
                SampleStatus::TemporarilyUnavailable,
                "sysinfo returned zero total memory".to_owned(),
            ));
        }
        samples
    }
}

#[derive(Clone, Debug)]
struct FileSensor {
    path: PathBuf,
    descriptor: MetricDescriptor,
    scale: f64,
}

pub struct LinuxSensorsProvider {
    descriptors: Vec<MetricDescriptor>,
    sensors: Vec<FileSensor>,
}

impl LinuxSensorsProvider {
    pub fn discover() -> Self {
        Self::discover_at(
            Path::new("/sys/class/hwmon"),
            Path::new("/sys/devices/system/cpu"),
        )
    }

    pub fn discover_at(hwmon_root: &Path, cpu_root: &Path) -> Self {
        let mut sensors = Vec::new();
        discover_hwmon(hwmon_root, &mut sensors);
        discover_cpu_frequencies(cpu_root, &mut sensors);
        sensors.sort_by(|left, right| {
            left.descriptor
                .series_key()
                .cmp(&right.descriptor.series_key())
        });
        let descriptors = sensors
            .iter()
            .map(|sensor| sensor.descriptor.clone())
            .collect();
        Self {
            descriptors,
            sensors,
        }
    }
}

impl CapabilityDiscovery for LinuxSensorsProvider {
    fn descriptors(&self) -> &[MetricDescriptor] {
        &self.descriptors
    }
}

impl MetricProvider for LinuxSensorsProvider {
    fn collect(
        &mut self,
        observation_mono_ns: u64,
        _previous_observation_mono_ns: Option<u64>,
    ) -> Vec<MetricSample> {
        self.sensors
            .iter()
            .map(|sensor| match fs::read_to_string(&sensor.path) {
                Ok(raw) => match raw.trim().parse::<f64>() {
                    Ok(value) => MetricSample {
                        mono_ns: observation_mono_ns,
                        window_start_mono_ns: None,
                        metric_id: sensor.descriptor.metric_id.clone(),
                        entity_id: sensor.descriptor.entity_id.clone(),
                        value: SampleValue::F64(value / sensor.scale),
                        status: SampleStatus::Ok,
                    },
                    Err(error) => unavailable_sample(
                        &sensor.descriptor,
                        observation_mono_ns,
                        SampleStatus::Error,
                        format!("invalid sensor value: {error}"),
                    ),
                },
                Err(error) => unavailable_sample(
                    &sensor.descriptor,
                    observation_mono_ns,
                    status_for_io(&error),
                    error.to_string(),
                ),
            })
            .collect()
    }
}

fn status_for_io(error: &io::Error) -> SampleStatus {
    match error.kind() {
        io::ErrorKind::PermissionDenied => SampleStatus::Error,
        io::ErrorKind::WouldBlock
        | io::ErrorKind::Interrupted
        | io::ErrorKind::TimedOut
        | io::ErrorKind::NotFound => SampleStatus::TemporarilyUnavailable,
        _ => SampleStatus::Error,
    }
}

fn unavailable_sample(
    descriptor: &MetricDescriptor,
    mono_ns: u64,
    status: SampleStatus,
    reason: String,
) -> MetricSample {
    MetricSample {
        mono_ns,
        window_start_mono_ns: None,
        metric_id: descriptor.metric_id.clone(),
        entity_id: descriptor.entity_id.clone(),
        value: SampleValue::State(reason),
        status,
    }
}

fn path_identity(path: &Path) -> String {
    fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .replace(':', "%3A")
}

fn hwmon_entity_kind(name: &str) -> EntityKind {
    let name = name.to_lowercase();
    if name.contains("cpu") || name.contains("coretemp") || name.contains("k10temp") {
        EntityKind::Cpu
    } else if name.contains("gpu") {
        EntityKind::Gpu
    } else if name.contains("nvme") || name.contains("disk") {
        EntityKind::Disk
    } else if name.contains("wifi") || name.contains("iwl") || name.contains("eth") {
        EntityKind::Net
    } else {
        EntityKind::System
    }
}

fn discover_hwmon(root: &Path, sensors: &mut Vec<FileSensor>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let base = entry.path();
        let device_name = fs::read_to_string(base.join("name"))
            .unwrap_or_else(|_| "hwmon".to_owned())
            .trim()
            .to_owned();
        let device_path = if base.join("device").exists() {
            base.join("device")
        } else {
            base.clone()
        };
        let stable_device = path_identity(&device_path);
        let Ok(files) = fs::read_dir(&base) else {
            continue;
        };
        for file in files.flatten() {
            let path = file.path();
            let filename = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            if !filename.starts_with("temp") || !filename.ends_with("_input") {
                continue;
            }
            let label_path = base.join(filename.replace("_input", "_label"));
            let label = fs::read_to_string(label_path)
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| device_name.clone());
            let entity_id = format!("hwmon:{stable_device}:{device_name}:{filename}");
            sensors.push(FileSensor {
                path,
                descriptor: descriptor(
                    "temperature.celsius",
                    label,
                    hwmon_entity_kind(&device_name),
                    "degree_celsius",
                    ValueKind::Gauge,
                    TemporalSemantics::PointSample,
                    "sysfs",
                    capability_for_path(&file.path()),
                    None,
                    "Linux hwmon temperature input sampled at read time.",
                    Some("device_temperature_celsius"),
                    entity_id,
                ),
                scale: 1_000.0,
            });
        }
    }
}

fn discover_cpu_frequencies(root: &Path, sensors: &mut Vec<FileSensor>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        let Ok(core) = name.strip_prefix("cpu").unwrap_or("").parse::<usize>() else {
            continue;
        };
        let cpufreq = path.join("cpufreq");
        let primary = cpufreq.join("scaling_cur_freq");
        let fallback = cpufreq.join("cpuinfo_cur_freq");
        let sensor_path = if primary.exists() {
            primary
        } else if fallback.exists() {
            fallback
        } else {
            continue;
        };
        sensors.push(FileSensor {
            descriptor: descriptor(
                "cpu.frequency",
                format!("CPU Core {core}"),
                EntityKind::Cpu,
                "kilohertz",
                ValueKind::Gauge,
                TemporalSemantics::PointSample,
                "sysfs",
                capability_for_path(&sensor_path),
                None,
                "Current logical CPU frequency reported by Linux cpufreq.",
                Some("logical_cpu_frequency"),
                format!("cpu:logical:{core}"),
            ),
            path: sensor_path,
            scale: 1.0,
        });
    }
}

fn capability_for_path(path: &Path) -> CapabilityStatus {
    match fs::File::open(path) {
        Ok(_) => CapabilityStatus::Available,
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => {
            CapabilityStatus::PermissionDenied
        }
        Err(_) => CapabilityStatus::TemporarilyUnavailable,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NvidiaDeviceIdentity {
    pub index: u32,
    pub uuid: Option<String>,
    pub pci_address: Option<String>,
    pub driver_identity: String,
    pub display_name: String,
}

impl NvidiaDeviceIdentity {
    pub fn durable_entity_id(&self) -> Option<String> {
        self.uuid
            .as_ref()
            .filter(|uuid| !uuid.trim().is_empty())
            .map(|uuid| format!("gpu:nvidia:uuid:{uuid}"))
            .or_else(|| {
                self.pci_address
                    .as_ref()
                    .filter(|pci| !pci.trim().is_empty())
                    .map(|pci| format!("gpu:nvidia:pci:{pci}:driver:{}", self.driver_identity))
            })
    }
}

#[derive(Clone, Debug)]
pub struct NvidiaDeviceReading {
    pub gpu_utilization_percent: Result<f64, CapabilityStatus>,
    pub memory_controller_utilization_percent: Result<f64, CapabilityStatus>,
    pub vram_occupancy_percent: Result<f64, CapabilityStatus>,
    pub temperature_celsius: Result<f64, CapabilityStatus>,
    pub graphics_clock_mhz: Result<f64, CapabilityStatus>,
    pub sm_clock_mhz: Result<f64, CapabilityStatus>,
    pub memory_clock_mhz: Result<f64, CapabilityStatus>,
    pub video_clock_mhz: Result<f64, CapabilityStatus>,
}

pub trait NvidiaBackend {
    fn device_count(&self) -> Result<u32, String>;
    fn device_identity(&self, index: u32) -> Result<NvidiaDeviceIdentity, String>;
    fn read_device(&self, index: u32) -> Result<NvidiaDeviceReading, String>;
}

#[derive(Clone, Copy)]
struct NvidiaMetric {
    id: &'static str,
    display: &'static str,
    unit: &'static str,
    semantics: &'static str,
    comparability: Option<&'static str>,
    select: fn(&NvidiaDeviceReading) -> &Result<f64, CapabilityStatus>,
}

const NVIDIA_METRICS: [NvidiaMetric; 8] = [
    NvidiaMetric {
        id: "gpu.utilization",
        display: "GPU utilization",
        unit: "percent",
        semantics: "NVML GPU busy percentage over the vendor-defined sample period.",
        comparability: Some("nvidia_gpu_busy_vendor_sample"),
        select: |r| &r.gpu_utilization_percent,
    },
    NvidiaMetric {
        id: "gpu.memory_controller_utilization",
        display: "GPU memory-controller utilization",
        unit: "percent",
        semantics: "NVML memory-controller busy percentage over the vendor-defined sample period.",
        comparability: Some("nvidia_memory_controller_busy_vendor_sample"),
        select: |r| &r.memory_controller_utilization_percent,
    },
    NvidiaMetric {
        id: "gpu.vram_occupancy",
        display: "VRAM utilization",
        unit: "percent",
        semantics:
            "Allocated framebuffer memory divided by total framebuffer memory at observation.",
        comparability: Some("gpu_vram_occupancy_percent"),
        select: |r| &r.vram_occupancy_percent,
    },
    NvidiaMetric {
        id: "temperature.celsius",
        display: "GPU Core",
        unit: "degree_celsius",
        semantics: "NVML GPU temperature sampled by the vendor API.",
        comparability: Some("device_temperature_celsius"),
        select: |r| &r.temperature_celsius,
    },
    NvidiaMetric {
        id: "gpu.clock.graphics",
        display: "GPU Graphics",
        unit: "megahertz",
        semantics: "NVML graphics clock sampled by the vendor API.",
        comparability: Some("nvidia_graphics_clock"),
        select: |r| &r.graphics_clock_mhz,
    },
    NvidiaMetric {
        id: "gpu.clock.sm",
        display: "GPU SM",
        unit: "megahertz",
        semantics: "NVML streaming-multiprocessor clock sampled by the vendor API.",
        comparability: Some("nvidia_sm_clock"),
        select: |r| &r.sm_clock_mhz,
    },
    NvidiaMetric {
        id: "gpu.clock.memory",
        display: "GPU Memory",
        unit: "megahertz",
        semantics: "NVML memory clock sampled by the vendor API.",
        comparability: Some("nvidia_memory_clock"),
        select: |r| &r.memory_clock_mhz,
    },
    NvidiaMetric {
        id: "gpu.clock.video",
        display: "GPU Video",
        unit: "megahertz",
        semantics: "NVML video clock sampled by the vendor API.",
        comparability: Some("nvidia_video_clock"),
        select: |r| &r.video_clock_mhz,
    },
];

struct NvidiaDevice {
    identity: NvidiaDeviceIdentity,
    entity_id: String,
}

pub struct NvidiaProvider<B> {
    backend: B,
    devices: Vec<NvidiaDevice>,
    descriptors: Vec<MetricDescriptor>,
    descriptor_lookup: BTreeMap<SeriesKey, usize>,
}

impl<B: NvidiaBackend> NvidiaProvider<B> {
    pub fn new(backend: B) -> Self {
        let count = backend.device_count().unwrap_or(0);
        let mut devices = Vec::new();
        let mut descriptors = Vec::new();
        for index in 0..count {
            let Ok(identity) = backend.device_identity(index) else {
                continue;
            };
            let Some(entity_id) = identity.durable_entity_id() else {
                continue;
            };
            let probe = backend.read_device(index).ok();
            for metric in NVIDIA_METRICS {
                let capability = probe
                    .as_ref()
                    .map(|reading| match (metric.select)(reading) {
                        Ok(_) => CapabilityStatus::Available,
                        Err(status) => *status,
                    })
                    .unwrap_or(CapabilityStatus::TemporarilyUnavailable);
                descriptors.push(descriptor(
                    metric.id,
                    format!("{} {}", identity.display_name, metric.display),
                    EntityKind::Gpu,
                    metric.unit,
                    ValueKind::Gauge,
                    TemporalSemantics::VendorSampled,
                    "nvml",
                    capability,
                    None,
                    metric.semantics,
                    metric.comparability,
                    entity_id.clone(),
                ));
            }
            devices.push(NvidiaDevice {
                identity,
                entity_id,
            });
        }
        let descriptor_lookup = descriptors
            .iter()
            .enumerate()
            .map(|(index, descriptor)| (descriptor.series_key(), index))
            .collect();
        Self {
            backend,
            devices,
            descriptors,
            descriptor_lookup,
        }
    }
}

impl<B> CapabilityDiscovery for NvidiaProvider<B> {
    fn descriptors(&self) -> &[MetricDescriptor] {
        &self.descriptors
    }
}

impl<B: NvidiaBackend> MetricProvider for NvidiaProvider<B> {
    fn collect(
        &mut self,
        observation_mono_ns: u64,
        _previous_observation_mono_ns: Option<u64>,
    ) -> Vec<MetricSample> {
        let mut samples = Vec::new();
        for device in &self.devices {
            let reading = self.backend.read_device(device.identity.index);
            for metric in NVIDIA_METRICS {
                let key = SeriesKey::new(metric.id, device.entity_id.clone());
                let Some(descriptor_index) = self.descriptor_lookup.get(&key).copied() else {
                    continue;
                };
                let descriptor = &mut self.descriptors[descriptor_index];
                match &reading {
                    Ok(reading) => match (metric.select)(reading) {
                        Ok(value) => {
                            descriptor.capability_status = CapabilityStatus::Available;
                            samples.push(MetricSample {
                                mono_ns: observation_mono_ns,
                                window_start_mono_ns: None,
                                metric_id: metric.id.to_owned(),
                                entity_id: device.entity_id.clone(),
                                value: SampleValue::F64(*value),
                                status: SampleStatus::Ok,
                            });
                        }
                        Err(CapabilityStatus::Unsupported) => {
                            descriptor.capability_status = CapabilityStatus::Unsupported;
                        }
                        Err(status) => {
                            descriptor.capability_status = *status;
                            samples.push(unavailable_sample(
                                descriptor,
                                observation_mono_ns,
                                SampleStatus::TemporarilyUnavailable,
                                format!("{status:?}"),
                            ));
                        }
                    },
                    Err(reason) => {
                        descriptor.capability_status = CapabilityStatus::TemporarilyUnavailable;
                        samples.push(unavailable_sample(
                            descriptor,
                            observation_mono_ns,
                            SampleStatus::TemporarilyUnavailable,
                            reason.clone(),
                        ));
                    }
                }
            }
        }
        samples
    }
}

#[cfg(feature = "nvidia")]
pub struct NvmlBackend {
    nvml: nvml_wrapper::Nvml,
}

#[cfg(feature = "nvidia")]
impl NvmlBackend {
    pub fn try_new() -> Result<Self, String> {
        nvml_wrapper::Nvml::init()
            .map(|nvml| Self { nvml })
            .map_err(|error| error.to_string())
    }
}

#[cfg(feature = "nvidia")]
fn nvml_capability<T>(
    result: Result<T, nvml_wrapper::error::NvmlError>,
) -> Result<T, CapabilityStatus> {
    result.map_err(|error| {
        let message = error.to_string().to_lowercase();
        if message.contains("not supported") {
            CapabilityStatus::Unsupported
        } else if message.contains("permission") || message.contains("no permission") {
            CapabilityStatus::PermissionDenied
        } else {
            CapabilityStatus::TemporarilyUnavailable
        }
    })
}

#[cfg(feature = "nvidia")]
impl NvidiaBackend for NvmlBackend {
    fn device_count(&self) -> Result<u32, String> {
        self.nvml.device_count().map_err(|error| error.to_string())
    }

    fn device_identity(&self, index: u32) -> Result<NvidiaDeviceIdentity, String> {
        let device = self
            .nvml
            .device_by_index(index)
            .map_err(|error| error.to_string())?;
        let uuid = device.uuid().ok();
        let pci_address = device.pci_info().ok().map(|pci| pci.bus_id);
        let driver_identity = self
            .nvml
            .sys_driver_version()
            .unwrap_or_else(|_| "unknown".to_owned());
        let display_name = device.name().unwrap_or_else(|_| "NVIDIA GPU".to_owned());
        Ok(NvidiaDeviceIdentity {
            index,
            uuid,
            pci_address,
            driver_identity,
            display_name,
        })
    }

    fn read_device(&self, index: u32) -> Result<NvidiaDeviceReading, String> {
        use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
        let device = self
            .nvml
            .device_by_index(index)
            .map_err(|error| error.to_string())?;
        let gpu_utilization_percent =
            nvml_capability(device.utilization_rates().map(|value| value.gpu as f64));
        let memory_controller_utilization_percent =
            nvml_capability(device.utilization_rates().map(|value| value.memory as f64));
        let memory = device.memory_info();
        let vram_occupancy_percent = nvml_capability(memory.map(|value| {
            if value.total == 0 {
                0.0
            } else {
                value.used as f64 / value.total as f64 * 100.0
            }
        }));
        Ok(NvidiaDeviceReading {
            gpu_utilization_percent,
            memory_controller_utilization_percent,
            vram_occupancy_percent,
            temperature_celsius: nvml_capability(
                device
                    .temperature(TemperatureSensor::Gpu)
                    .map(|value| value as f64),
            ),
            graphics_clock_mhz: nvml_capability(
                device.clock_info(Clock::Graphics).map(|value| value as f64),
            ),
            sm_clock_mhz: nvml_capability(device.clock_info(Clock::SM).map(|value| value as f64)),
            memory_clock_mhz: nvml_capability(
                device.clock_info(Clock::Memory).map(|value| value as f64),
            ),
            video_clock_mhz: nvml_capability(
                device.clock_info(Clock::Video).map(|value| value as f64),
            ),
        })
    }
}

pub fn production_collector() -> Collector<NativeClock> {
    let mut providers: Vec<Box<dyn MetricProvider>> = vec![
        Box::new(SystemProvider::new()),
        Box::new(LinuxSensorsProvider::discover()),
    ];
    #[cfg(feature = "nvidia")]
    if let Ok(backend) = NvmlBackend::try_new() {
        providers.push(Box::new(NvidiaProvider::new(backend)));
    }
    Collector::new(NativeClock, providers)
}
