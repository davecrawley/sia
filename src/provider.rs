use crate::collector::{MetricProvider, ProviderResult};
use crate::model::{
    CanonicalUnit, CapabilityState, EntityId, EntityKind, MetricDescriptor, MetricId,
    MetricValue, MonotonicTimestamp, SampleStatus, TemporalSemantics, ValueKind,
};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use sysinfo::{CpuExt, System, SystemExt};

pub enum Discovery<T> {
    Available(T),
    Unavailable(CapabilityState),
}

#[derive(Clone, Debug)]
struct FileSensor {
    metric_id: MetricId,
    entity_id: EntityId,
    path: PathBuf,
}

pub struct SystemProvider {
    system: System,
    descriptors: Vec<MetricDescriptor>,
    temperatures: Vec<FileSensor>,
    frequencies: Vec<FileSensor>,
    previous_observation: Option<MonotonicTimestamp>,
}

impl SystemProvider {
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        let temperatures = discover_temperatures();
        let frequencies = discover_frequencies();
        let mut descriptors = vec![
            descriptor(
                "system.cpu.utilization",
                "CPU utilization",
                EntityKind::Cpu,
                CanonicalUnit::Percent,
                ValueKind::Gauge,
                TemporalSemantics::IntervalAverage,
                "sysinfo",
                "mean logical-CPU utilization over the sysinfo refresh interval",
                Some("host_cpu_time_busy"),
            ),
            descriptor(
                "system.memory.utilization",
                "RAM utilization",
                EntityKind::System,
                CanonicalUnit::Percent,
                ValueKind::Gauge,
                TemporalSemantics::PointSampled,
                "sysinfo",
                "used physical memory divided by total physical memory",
                Some("physical_memory_occupancy"),
            ),
        ];

        descriptors.extend(temperatures.iter().map(|sensor| {
            descriptor_owned(
                sensor.metric_id.clone(),
                format!("{} temperature", sensor.entity_id.0),
                EntityKind::System,
                CanonicalUnit::Celsius,
                ValueKind::Gauge,
                TemporalSemantics::PointSampled,
                "sysfs",
                "Linux hwmon temperature input",
                Some("hwmon_temperature_celsius"),
            )
        }));
        descriptors.extend(frequencies.iter().map(|sensor| {
            descriptor_owned(
                sensor.metric_id.clone(),
                format!("{} frequency", sensor.entity_id.0),
                EntityKind::Cpu,
                CanonicalUnit::Hertz,
                ValueKind::Gauge,
                TemporalSemantics::PointSampled,
                "sysfs",
                "Linux cpufreq current frequency",
                Some("cpu_clock_hertz"),
            )
        }));

        Self {
            system,
            descriptors,
            temperatures,
            frequencies,
            previous_observation: None,
        }
    }
}

impl Default for SystemProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricProvider for SystemProvider {
    fn descriptors(&self) -> &[MetricDescriptor] {
        &self.descriptors
    }

    fn observe(&mut self, observation_time: MonotonicTimestamp) -> Vec<ProviderResult> {
        self.system.refresh_cpu();
        self.system.refresh_memory();
        let mut results = Vec::new();

        let cpu_count = self.system.cpus().len();
        let cpu = if cpu_count == 0 {
            None
        } else {
            Some(
                self.system
                    .cpus()
                    .iter()
                    .map(|cpu| cpu.cpu_usage() as f64)
                    .sum::<f64>()
                    / cpu_count as f64,
            )
        };
        results.push(observation(
            "system.cpu.utilization",
            "system:cpu",
            observation_time,
            self.previous_observation,
            cpu.map(MetricValue::F64),
        ));

        let total_memory = self.system.total_memory();
        let memory = (total_memory > 0).then(|| {
            MetricValue::F64(self.system.used_memory() as f64 * 100.0 / total_memory as f64)
        });
        results.push(observation(
            "system.memory.utilization",
            "system:memory",
            observation_time,
            None,
            memory,
        ));

        for sensor in &self.temperatures {
            results.push(observation_owned(
                sensor.metric_id.clone(),
                sensor.entity_id.clone(),
                observation_time,
                None,
                read_number(&sensor.path).map(|value| {
                    MetricValue::F64(if value.abs() >= 1000.0 {
                        value / 1000.0
                    } else {
                        value
                    })
                }),
            ));
        }

        for sensor in &self.frequencies {
            results.push(observation_owned(
                sensor.metric_id.clone(),
                sensor.entity_id.clone(),
                observation_time,
                None,
                read_number(&sensor.path).map(|kilohertz| MetricValue::F64(kilohertz * 1000.0)),
            ));
        }

        self.previous_observation = Some(observation_time);
        results
    }
}

fn observation(
    metric_id: &str,
    entity_id: &str,
    observation_time: MonotonicTimestamp,
    interval_start: Option<MonotonicTimestamp>,
    value: Option<MetricValue>,
) -> ProviderResult {
    observation_owned(
        metric_id.into(),
        entity_id.into(),
        observation_time,
        interval_start,
        value,
    )
}

fn observation_owned(
    metric_id: MetricId,
    entity_id: EntityId,
    observation_time: MonotonicTimestamp,
    interval_start: Option<MonotonicTimestamp>,
    value: Option<MetricValue>,
) -> ProviderResult {
    let status = if value.is_some() {
        SampleStatus::Ok
    } else {
        SampleStatus::TemporarilyUnavailable {
            reason: Some("source did not return a value".into()),
        }
    };
    ProviderResult::Observation {
        metric_id,
        entity_id,
        observation_time: Some(observation_time),
        interval_start,
        value,
        status,
    }
}

fn descriptor(
    id: &str,
    name: &str,
    entity_kind: EntityKind,
    unit: CanonicalUnit,
    value_kind: ValueKind,
    temporal_semantics: TemporalSemantics,
    provider: &str,
    source_semantics: &str,
    comparability_group: Option<&str>,
) -> MetricDescriptor {
    descriptor_owned(
        id.into(),
        name.into(),
        entity_kind,
        unit,
        value_kind,
        temporal_semantics,
        provider,
        source_semantics,
        comparability_group,
    )
}

fn descriptor_owned(
    metric_id: MetricId,
    display_name: String,
    entity_kind: EntityKind,
    unit: CanonicalUnit,
    value_kind: ValueKind,
    temporal_semantics: TemporalSemantics,
    provider: &str,
    source_semantics: &str,
    comparability_group: Option<&str>,
) -> MetricDescriptor {
    MetricDescriptor {
        metric_id,
        display_name,
        entity_kind,
        unit,
        value_kind,
        temporal_semantics,
        provider: provider.into(),
        capability_status: CapabilityState::Available,
        source_resolution: None,
        source_semantics: source_semantics.into(),
        comparability_group: comparability_group.map(str::to_owned),
        semantics_version: 1,
    }
}

fn discover_temperatures() -> Vec<FileSensor> {
    let mut sensors = Vec::new();
    let Ok(hwmon_entries) = fs::read_dir("/sys/class/hwmon") else {
        return sensors;
    };

    for hwmon in hwmon_entries.flatten() {
        let base = hwmon.path();
        let device_name = fs::read_to_string(base.join("name"))
            .unwrap_or_else(|_| "hwmon".into())
            .trim()
            .to_owned();
        let Ok(files) = fs::read_dir(&base) else {
            continue;
        };
        for file in files.flatten() {
            let path = file.path();
            let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if !file_name.starts_with("temp") || !file_name.ends_with("_input") {
                continue;
            }
            let label_path = base.join(file_name.replace("_input", "_label"));
            let label = fs::read_to_string(label_path)
                .ok()
                .map(|label| label.trim().to_owned())
                .filter(|label| !label.is_empty())
                .unwrap_or_else(|| file_name.trim_end_matches("_input").to_owned());
            let index = sensors.len();
            sensors.push(FileSensor {
                metric_id: format!("thermal.temperature.{index}").into(),
                entity_id: format!("sensor:{}:{}", clean_id(&device_name), clean_id(&label)).into(),
                path,
            });
        }
    }
    sensors
}

fn discover_frequencies() -> Vec<FileSensor> {
    let mut sensors = Vec::new();
    let Ok(cpu_entries) = fs::read_dir("/sys/devices/system/cpu") else {
        return sensors;
    };

    for cpu in cpu_entries.flatten() {
        let path = cpu.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Ok(index) = name.strip_prefix("cpu").unwrap_or("").parse::<usize>() else {
            continue;
        };
        let cpufreq = path.join("cpufreq");
        let scaling = cpufreq.join("scaling_cur_freq");
        let hardware = cpufreq.join("cpuinfo_cur_freq");
        let source = if scaling.exists() {
            scaling
        } else if hardware.exists() {
            hardware
        } else {
            continue;
        };
        sensors.push(FileSensor {
            metric_id: format!("cpu.frequency.{index}").into(),
            entity_id: format!("cpu:{index}").into(),
            path: source,
        });
    }
    sensors.sort_by(|left, right| left.entity_id.cmp(&right.entity_id));
    sensors
}

fn read_number(path: &Path) -> Option<f64> {
    let mut text = String::new();
    fs::File::open(path)
        .ok()?
        .read_to_string(&mut text)
        .ok()?;
    text.trim().parse().ok()
}

fn clean_id(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(feature = "nvidia")]
pub struct NvidiaProvider {
    nvml: nvml_wrapper::Nvml,
    devices: Vec<NvidiaDevice>,
    descriptors: Vec<MetricDescriptor>,
}

#[cfg(feature = "nvidia")]
#[derive(Clone, Debug)]
struct NvidiaDevice {
    index: u32,
    entity_id: EntityId,
}

#[cfg(feature = "nvidia")]
impl NvidiaProvider {
    pub fn discover() -> Discovery<Self> {
        use nvml_wrapper::Nvml;

        let nvml = match Nvml::init() {
            Ok(nvml) => nvml,
            Err(error) => {
                let reason = format!("{error:?}");
                let state = if reason.to_ascii_lowercase().contains("permission") {
                    CapabilityState::PermissionDenied {
                        reason: Some(reason),
                    }
                } else {
                    CapabilityState::Unsupported {
                        reason: Some(reason),
                    }
                };
                return Discovery::Unavailable(state);
            }
        };
        let count = match nvml.device_count() {
            Ok(count) if count > 0 => count,
            Ok(_) => {
                return Discovery::Unavailable(CapabilityState::Unsupported {
                    reason: Some("NVML reported no visible devices".into()),
                })
            }
            Err(error) => {
                return Discovery::Unavailable(CapabilityState::TemporarilyUnavailable {
                    reason: Some(format!("{error:?}")),
                })
            }
        };

        let mut devices = Vec::new();
        for index in 0..count {
            if let Ok(device) = nvml.device_by_index(index) {
                let identity = device
                    .uuid()
                    .map(|uuid| format!("gpu:uuid:{uuid}"))
                    .unwrap_or_else(|_| format!("gpu:nvidia:{index}"));
                devices.push(NvidiaDevice {
                    index,
                    entity_id: identity.into(),
                });
            }
        }
        if devices.is_empty() {
            return Discovery::Unavailable(CapabilityState::TemporarilyUnavailable {
                reason: Some("NVML devices could not be opened".into()),
            });
        }

        Discovery::Available(Self {
            nvml,
            devices,
            descriptors: nvidia_descriptors(),
        })
    }
}

#[cfg(feature = "nvidia")]
impl MetricProvider for NvidiaProvider {
    fn descriptors(&self) -> &[MetricDescriptor] {
        &self.descriptors
    }

    fn observe(&mut self, observation_time: MonotonicTimestamp) -> Vec<ProviderResult> {
        use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};

        let mut results = Vec::new();
        for identity in &self.devices {
            let Ok(device) = self.nvml.device_by_index(identity.index) else {
                continue;
            };
            match device.utilization_rates() {
                Ok(utilization) => {
                    results.push(observation_owned(
                        "gpu.nvidia.compute_utilization".into(),
                        identity.entity_id.clone(),
                        observation_time,
                        None,
                        Some(MetricValue::F64(utilization.gpu as f64)),
                    ));
                    results.push(observation_owned(
                        "gpu.nvidia.memory_activity".into(),
                        identity.entity_id.clone(),
                        observation_time,
                        None,
                        Some(MetricValue::F64(utilization.memory as f64)),
                    ));
                }
                Err(_) => {
                    results.push(observation_owned(
                        "gpu.nvidia.compute_utilization".into(),
                        identity.entity_id.clone(),
                        observation_time,
                        None,
                        None,
                    ));
                    results.push(observation_owned(
                        "gpu.nvidia.memory_activity".into(),
                        identity.entity_id.clone(),
                        observation_time,
                        None,
                        None,
                    ));
                }
            }
            let occupancy = device.memory_info().ok().and_then(|memory| {
                (memory.total > 0).then(|| {
                    MetricValue::F64(memory.used as f64 * 100.0 / memory.total as f64)
                })
            });
            results.push(observation_owned(
                "gpu.nvidia.vram_occupancy".into(),
                identity.entity_id.clone(),
                observation_time,
                None,
                occupancy,
            ));
            results.push(observation_owned(
                "gpu.nvidia.temperature".into(),
                identity.entity_id.clone(),
                observation_time,
                None,
                device
                    .temperature(TemperatureSensor::Gpu)
                    .ok()
                    .map(|value| MetricValue::F64(value as f64)),
            ));

            for (metric, clock) in [
                ("gpu.nvidia.clock.graphics", Clock::Graphics),
                ("gpu.nvidia.clock.sm", Clock::SM),
                ("gpu.nvidia.clock.memory", Clock::Memory),
                ("gpu.nvidia.clock.video", Clock::Video),
            ] {
                results.push(observation_owned(
                    metric.into(),
                    identity.entity_id.clone(),
                    observation_time,
                    None,
                    device
                        .clock_info(clock)
                        .ok()
                        .map(|megahertz| MetricValue::F64(megahertz as f64 * 1_000_000.0)),
                ));
            }
        }
        results
    }
}

#[cfg(feature = "nvidia")]
fn nvidia_descriptors() -> Vec<MetricDescriptor> {
    vec![
        descriptor(
            "gpu.nvidia.compute_utilization",
            "GPU utilization",
            EntityKind::Gpu,
            CanonicalUnit::Percent,
            ValueKind::Gauge,
            TemporalSemantics::VendorSampled,
            "nvml",
            "percent of the vendor sample period during which one or more kernels executed",
            Some("nvml_gpu_time_busy"),
        ),
        descriptor(
            "gpu.nvidia.memory_activity",
            "GPU memory activity",
            EntityKind::Gpu,
            CanonicalUnit::Percent,
            ValueKind::Gauge,
            TemporalSemantics::VendorSampled,
            "nvml",
            "percent of the vendor sample period during which global device memory was accessed",
            Some("nvml_memory_time_busy"),
        ),
        descriptor(
            "gpu.nvidia.vram_occupancy",
            "VRAM occupancy",
            EntityKind::Gpu,
            CanonicalUnit::Percent,
            ValueKind::Gauge,
            TemporalSemantics::PointSampled,
            "nvml",
            "allocated device memory divided by total device memory",
            Some("device_memory_occupancy"),
        ),
        descriptor(
            "gpu.nvidia.temperature",
            "GPU temperature",
            EntityKind::Gpu,
            CanonicalUnit::Celsius,
            ValueKind::Gauge,
            TemporalSemantics::PointSampled,
            "nvml",
            "NVML GPU temperature sensor",
            Some("gpu_temperature_celsius"),
        ),
        descriptor(
            "gpu.nvidia.clock.graphics",
            "GPU graphics clock",
            EntityKind::Gpu,
            CanonicalUnit::Hertz,
            ValueKind::Gauge,
            TemporalSemantics::PointSampled,
            "nvml",
            "current NVML graphics clock",
            Some("gpu_graphics_clock_hertz"),
        ),
        descriptor(
            "gpu.nvidia.clock.sm",
            "GPU SM clock",
            EntityKind::Gpu,
            CanonicalUnit::Hertz,
            ValueKind::Gauge,
            TemporalSemantics::PointSampled,
            "nvml",
            "current NVML streaming-multiprocessor clock",
            Some("gpu_sm_clock_hertz"),
        ),
        descriptor(
            "gpu.nvidia.clock.memory",
            "GPU memory clock",
            EntityKind::Gpu,
            CanonicalUnit::Hertz,
            ValueKind::Gauge,
            TemporalSemantics::PointSampled,
            "nvml",
            "current physical NVML memory clock without an inferred effective multiplier",
            Some("gpu_memory_clock_hertz"),
        ),
        descriptor(
            "gpu.nvidia.clock.video",
            "GPU video clock",
            EntityKind::Gpu,
            CanonicalUnit::Hertz,
            ValueKind::Gauge,
            TemporalSemantics::PointSampled,
            "nvml",
            "current NVML video clock",
            Some("gpu_video_clock_hertz"),
        ),
    ]
}

#[cfg(not(feature = "nvidia"))]
pub struct NvidiaProvider;

#[cfg(not(feature = "nvidia"))]
impl NvidiaProvider {
    pub fn discover() -> Discovery<Self> {
        Discovery::Unavailable(CapabilityState::Unsupported {
            reason: Some("sia was built without the nvidia feature".into()),
        })
    }
}