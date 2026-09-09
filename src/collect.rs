use crate::model::{
    CapabilityState, EntityKind, MetricDescriptor, MetricId, MetricValue, Observation,
    SampleStatus, SeriesKey, TelemetryModel, TemporalSemantics, Unit, ValueKind,
};
use std::fs;
use std::path::{Path, PathBuf};
use sysinfo::{CpuExt, System, SystemExt};

pub trait Clock {
    fn now_ns(&mut self) -> u64;
}

#[repr(C)]
struct LinuxTimespec {
    tv_sec: i64,
    tv_nsec: i64,
}

const LINUX_CLOCK_MONOTONIC: i32 = 1;

extern "C" {
    fn clock_gettime(clock_id: i32, value: *mut LinuxTimespec) -> i32;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MonotonicClock;

impl Clock for MonotonicClock {
    fn now_ns(&mut self) -> u64 {
        let mut value = LinuxTimespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: value is a valid writable timespec and CLOCK_MONOTONIC has no
        // ownership or lifetime requirements.
        let result = unsafe { clock_gettime(LINUX_CLOCK_MONOTONIC, &mut value) };
        if result != 0 {
            return 0;
        }
        (value.tv_sec as u64)
            .saturating_mul(1_000_000_000)
            .saturating_add(value.tv_nsec as u64)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SourceReading<T> {
    pub value: T,
    pub observation_ns: Option<u64>,
    pub window_start_ns: Option<u64>,
}

impl<T> SourceReading<T> {
    pub fn fallback_timestamp(value: T) -> Self {
        Self {
            value,
            observation_ns: None,
            window_start_ns: None,
        }
    }

    pub fn timestamped(value: T, observation_ns: u64, window_start_ns: Option<u64>) -> Self {
        Self {
            value,
            observation_ns: Some(observation_ns),
            window_start_ns,
        }
    }

    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> SourceReading<U> {
        SourceReading {
            value: map(self.value),
            observation_ns: self.observation_ns,
            window_start_ns: self.window_start_ns,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum BackendOutcome<T> {
    Available(T),
    Unsupported(String),
    PermissionDenied(String),
    TemporarilyUnavailable(String),
    Stale(String),
    Error(String),
}

impl<T> BackendOutcome<T> {
    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> BackendOutcome<U> {
        match self {
            Self::Available(value) => BackendOutcome::Available(map(value)),
            Self::Unsupported(reason) => BackendOutcome::Unsupported(reason),
            Self::PermissionDenied(reason) => BackendOutcome::PermissionDenied(reason),
            Self::TemporarilyUnavailable(reason) => BackendOutcome::TemporarilyUnavailable(reason),
            Self::Stale(reason) => BackendOutcome::Stale(reason),
            Self::Error(reason) => BackendOutcome::Error(reason),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricReading {
    pub descriptor: MetricDescriptor,
    pub outcome: BackendOutcome<SourceReading<MetricValue>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProviderBatch {
    pub readings: Vec<MetricReading>,
}

pub trait Provider {
    fn collect(&mut self) -> ProviderBatch;
}

pub struct Coordinator<C: Clock> {
    clock: C,
    providers: Vec<Box<dyn Provider>>,
    model: TelemetryModel,
}

impl<C: Clock> Coordinator<C> {
    pub fn new(clock: C) -> Self {
        Self::from_providers(clock, Vec::new())
    }

    pub fn from_providers(clock: C, providers: Vec<Box<dyn Provider>>) -> Self {
        Self {
            clock,
            providers,
            model: TelemetryModel::default(),
        }
    }

    pub fn add_provider<P: Provider + 'static>(&mut self, provider: P) {
        self.providers.push(Box::new(provider));
    }

    pub fn collect_once(&mut self) {
        for provider in &mut self.providers {
            let batch = provider.collect();
            let needs_fallback = batch.readings.iter().any(|reading| match &reading.outcome {
                BackendOutcome::Available(source) => source.observation_ns.is_none(),
                _ => true,
            });
            let fallback_observation_ns = if needs_fallback {
                self.clock.now_ns()
            } else {
                0
            };
            for reading in batch.readings {
                ingest_reading(&mut self.model, reading, fallback_observation_ns);
            }
        }
    }

    pub fn model(&self) -> &TelemetryModel {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut TelemetryModel {
        &mut self.model
    }

    pub fn into_model(self) -> TelemetryModel {
        self.model
    }

    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }
}

fn ingest_reading(
    model: &mut TelemetryModel,
    reading: MetricReading,
    fallback_observation_ns: u64,
) {
    let mut descriptor = reading.descriptor;
    let (source, status, capability) = match reading.outcome {
        BackendOutcome::Available(source) => (
            Some(source),
            SampleStatus::Available,
            CapabilityState::Available,
        ),
        BackendOutcome::Unsupported(reason) => (
            None,
            SampleStatus::Error(reason.clone()),
            CapabilityState::Unsupported(reason),
        ),
        BackendOutcome::PermissionDenied(reason) => (
            None,
            SampleStatus::Error(reason.clone()),
            CapabilityState::PermissionDenied(reason),
        ),
        BackendOutcome::TemporarilyUnavailable(reason) => (
            None,
            SampleStatus::TemporarilyUnavailable(reason.clone()),
            CapabilityState::TemporarilyUnavailable(reason),
        ),
        BackendOutcome::Stale(reason) => (
            None,
            SampleStatus::Stale(reason),
            CapabilityState::Available,
        ),
        BackendOutcome::Error(reason) => (
            None,
            SampleStatus::Error(reason.clone()),
            CapabilityState::TemporarilyUnavailable(reason),
        ),
    };

    let (observation_ns, window_start_ns, value) = match source {
        Some(source) => (
            source.observation_ns.unwrap_or(fallback_observation_ns),
            source.window_start_ns,
            Some(source.value),
        ),
        None => (fallback_observation_ns, None, None),
    };
    descriptor.capability = capability;
    let observation = Observation {
        key: descriptor.key.clone(),
        observation_ns,
        window_start_ns,
        value,
        status,
    };
    model.ingest(descriptor, observation);
}

#[derive(Clone, Debug, PartialEq)]
pub struct NamedMeasurement {
    pub entity_id: String,
    pub display_name: String,
    pub outcome: BackendOutcome<SourceReading<f64>>,
}

impl NamedMeasurement {
    pub fn available(
        entity_id: impl Into<String>,
        display_name: impl Into<String>,
        value: f64,
    ) -> Self {
        Self {
            entity_id: entity_id.into(),
            display_name: display_name.into(),
            outcome: BackendOutcome::Available(SourceReading::fallback_timestamp(value)),
        }
    }
}

pub trait SystemBackend {
    fn cpu_percent(&mut self) -> BackendOutcome<SourceReading<f64>>;
    fn ram_percent(&mut self) -> BackendOutcome<SourceReading<f64>>;
    fn temperatures_c(&mut self) -> Vec<NamedMeasurement>;
    fn frequencies_hz(&mut self) -> Vec<NamedMeasurement>;
}

pub struct SystemProvider<B: SystemBackend> {
    backend: B,
}

impl<B: SystemBackend> SystemProvider<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

fn system_descriptor(
    metric_id: &str,
    entity_id: &str,
    display_name: &str,
    entity_kind: EntityKind,
    unit: Unit,
    semantics: TemporalSemantics,
    provider: &str,
    definition: &str,
    comparability_group: &str,
) -> MetricDescriptor {
    MetricDescriptor {
        key: SeriesKey::new(MetricId::from(metric_id), entity_id, 0),
        display_name: display_name.to_owned(),
        entity_kind,
        unit,
        value_kind: ValueKind::Gauge,
        temporal_semantics: semantics,
        provider: provider.to_owned(),
        capability: CapabilityState::Available,
        source_resolution_ns: None,
        source_definition: definition.to_owned(),
        comparability_group: Some(comparability_group.to_owned()),
        semantics_version: 1,
    }
}

fn system_reading(
    descriptor: MetricDescriptor,
    outcome: BackendOutcome<SourceReading<f64>>,
) -> MetricReading {
    MetricReading {
        descriptor,
        outcome: outcome.map(|reading| reading.map(MetricValue::F64)),
    }
}

impl<B: SystemBackend> Provider for SystemProvider<B> {
    fn collect(&mut self) -> ProviderBatch {
        let mut readings = Vec::new();
        readings.push(system_reading(
            system_descriptor(
                "system.cpu.utilization",
                "cpu:all",
                "CPU utilization",
                EntityKind::Cpu,
                Unit::Percent,
                TemporalSemantics::IntervalAverage,
                "sysinfo",
                "Aggregate CPU busy time across the interval between sysinfo refresh observations.",
                "sysinfo_cpu_busy_interval_percent",
            ),
            self.backend.cpu_percent(),
        ));
        readings.push(system_reading(
            system_descriptor(
                "system.ram.occupancy",
                "system:memory",
                "RAM utilization",
                EntityKind::System,
                Unit::Percent,
                TemporalSemantics::PointSample,
                "sysinfo",
                "Used physical memory divided by total physical memory at observation time.",
                "physical_memory_occupancy_percent",
            ),
            self.backend.ram_percent(),
        ));

        for measurement in self.backend.temperatures_c() {
            let descriptor = system_descriptor(
                "system.temperature",
                &measurement.entity_id,
                &measurement.display_name,
                EntityKind::Thermal,
                Unit::Celsius,
                TemporalSemantics::PointSample,
                "sysfs",
                "Point temperature reported by the named Linux hwmon sensor.",
                "temperature_celsius",
            );
            readings.push(system_reading(descriptor, measurement.outcome));
        }

        for measurement in self.backend.frequencies_hz() {
            let descriptor = system_descriptor(
                "cpu.frequency",
                &measurement.entity_id,
                &measurement.display_name,
                EntityKind::Cpu,
                Unit::Hertz,
                TemporalSemantics::PointSample,
                "sysfs",
                "Current logical CPU frequency reported by cpufreq.",
                "cpu_frequency_hz",
            );
            readings.push(system_reading(descriptor, measurement.outcome));
        }

        ProviderBatch { readings }
    }
}

#[derive(Clone, Debug)]
struct SensorPath {
    entity_id: String,
    display_name: String,
    path: PathBuf,
}

pub struct LinuxSystemBackend {
    system: System,
    temperatures: Vec<SensorPath>,
    frequencies: Vec<SensorPath>,
}

impl LinuxSystemBackend {
    pub fn discover() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        Self {
            system,
            temperatures: discover_temperatures(),
            frequencies: discover_frequencies(),
        }
    }
}

impl Default for LinuxSystemBackend {
    fn default() -> Self {
        Self::discover()
    }
}

fn read_number(path: &Path) -> BackendOutcome<f64> {
    match fs::read_to_string(path) {
        Ok(text) => match text.trim().parse::<f64>() {
            Ok(value) => BackendOutcome::Available(value),
            Err(error) => BackendOutcome::Error(error.to_string()),
        },
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {
            BackendOutcome::PermissionDenied(error.to_string())
        }
        Err(error) => BackendOutcome::TemporarilyUnavailable(error.to_string()),
    }
}

impl SystemBackend for LinuxSystemBackend {
    fn cpu_percent(&mut self) -> BackendOutcome<SourceReading<f64>> {
        self.system.refresh_cpu();
        BackendOutcome::Available(SourceReading::fallback_timestamp(
            self.system.global_cpu_info().cpu_usage() as f64,
        ))
    }

    fn ram_percent(&mut self) -> BackendOutcome<SourceReading<f64>> {
        self.system.refresh_memory();
        let total = self.system.total_memory();
        if total == 0 {
            return BackendOutcome::TemporarilyUnavailable(
                "total physical memory is unavailable".to_owned(),
            );
        }
        BackendOutcome::Available(SourceReading::fallback_timestamp(
            self.system.used_memory() as f64 * 100.0 / total as f64,
        ))
    }

    fn temperatures_c(&mut self) -> Vec<NamedMeasurement> {
        self.temperatures
            .iter()
            .map(|sensor| NamedMeasurement {
                entity_id: sensor.entity_id.clone(),
                display_name: sensor.display_name.clone(),
                outcome: read_number(&sensor.path).map(|value| {
                    SourceReading::fallback_timestamp(if value.abs() > 1000.0 {
                        value / 1000.0
                    } else {
                        value
                    })
                }),
            })
            .collect()
    }

    fn frequencies_hz(&mut self) -> Vec<NamedMeasurement> {
        self.frequencies
            .iter()
            .map(|sensor| NamedMeasurement {
                entity_id: sensor.entity_id.clone(),
                display_name: sensor.display_name.clone(),
                outcome: read_number(&sensor.path)
                    .map(|khz| SourceReading::fallback_timestamp(khz * 1_000.0)),
            })
            .collect()
    }
}

fn discover_temperatures() -> Vec<SensorPath> {
    let mut sensors = Vec::new();
    let Ok(devices) = fs::read_dir("/sys/class/hwmon") else {
        return sensors;
    };
    for device in devices.flatten() {
        let base = device.path();
        let device_name = fs::read_to_string(base.join("name"))
            .unwrap_or_else(|_| "hwmon".to_owned())
            .trim()
            .to_owned();
        let Ok(entries) = fs::read_dir(&base) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if !file_name.starts_with("temp") || !file_name.ends_with("_input") {
                continue;
            }
            let label_path = base.join(file_name.replace("_input", "_label"));
            let label = fs::read_to_string(label_path)
                .unwrap_or_else(|_| file_name.to_owned())
                .trim()
                .to_owned();
            sensors.push(SensorPath {
                entity_id: format!("thermal:{device_name}:{file_name}"),
                display_name: format!("{device_name} {label}"),
                path,
            });
        }
    }
    sensors.sort_by(|left, right| left.entity_id.cmp(&right.entity_id));
    sensors
}

fn discover_frequencies() -> Vec<SensorPath> {
    let mut sensors = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/devices/system/cpu") else {
        return sensors;
    };
    for entry in entries.flatten() {
        let base = entry.path();
        let Some(name) = base.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Ok(index) = name.strip_prefix("cpu").unwrap_or("").parse::<usize>() else {
            continue;
        };
        let cpufreq = base.join("cpufreq");
        let scaling = cpufreq.join("scaling_cur_freq");
        let hardware = cpufreq.join("cpuinfo_cur_freq");
        let path = if scaling.exists() {
            scaling
        } else if hardware.exists() {
            hardware
        } else {
            continue;
        };
        sensors.push(SensorPath {
            entity_id: format!("cpu:{index}"),
            display_name: format!("CPU Core {index}"),
            path,
        });
    }
    sensors.sort_by(|left, right| left.entity_id.cmp(&right.entity_id));
    sensors
}
