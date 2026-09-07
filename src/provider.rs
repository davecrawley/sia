use crate::collector::{MetricProvider, ProviderResult};
use crate::model::{
    CanonicalUnit, CapabilityState, EntityId, EntityKind, MetricDescriptor, MetricId, MetricValue,
    MonotonicTimestamp, SampleStatus, TemporalSemantics, ValueKind,
};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};
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
        descriptors.extend(temperatures.iter().map(|s| {
            descriptor_owned(
                s.metric_id.clone(),
                format!("{} temperature", s.entity_id.0),
                EntityKind::System,
                CanonicalUnit::Celsius,
                ValueKind::Gauge,
                TemporalSemantics::PointSampled,
                "sysfs",
                "Linux hwmon temperature input",
                Some("hwmon_temperature_celsius"),
            )
        }));
        descriptors.extend(frequencies.iter().map(|s| {
            descriptor_owned(
                s.metric_id.clone(),
                format!("{} frequency", s.entity_id.0),
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
    fn observe(&mut self, time: MonotonicTimestamp) -> Vec<ProviderResult> {
        self.system.refresh_cpu();
        self.system.refresh_memory();
        let mut out = Vec::new();
        let cpu = (!self.system.cpus().is_empty()).then(|| {
            self.system
                .cpus()
                .iter()
                .map(|c| c.cpu_usage() as f64)
                .sum::<f64>()
                / self.system.cpus().len() as f64
        });
        out.push(observation(
            "system.cpu.utilization",
            "system:cpu",
            time,
            self.previous_observation,
            cpu.map(MetricValue::F64),
        ));
        let total = self.system.total_memory();
        let ram = (total > 0)
            .then(|| MetricValue::F64(self.system.used_memory() as f64 * 100.0 / total as f64));
        out.push(observation(
            "system.memory.utilization",
            "system:memory",
            time,
            None,
            ram,
        ));
        for s in &self.temperatures {
            out.push(observation_owned(
                s.metric_id.clone(),
                s.entity_id.clone(),
                time,
                None,
                read_number(&s.path)
                    .map(|v| MetricValue::F64(if v.abs() >= 1000.0 { v / 1000.0 } else { v })),
            ));
        }
        for s in &self.frequencies {
            out.push(observation_owned(
                s.metric_id.clone(),
                s.entity_id.clone(),
                time,
                None,
                read_number(&s.path).map(|v| MetricValue::F64(v * 1000.0)),
            ));
        }
        self.previous_observation = Some(time);
        out
    }
}

fn observation(
    metric: &str,
    entity: &str,
    time: MonotonicTimestamp,
    start: Option<MonotonicTimestamp>,
    value: Option<MetricValue>,
) -> ProviderResult {
    observation_owned(metric.into(), entity.into(), time, start, value)
}
fn observation_owned(
    metric_id: MetricId,
    entity_id: EntityId,
    time: MonotonicTimestamp,
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
        observation_time: Some(time),
        interval_start,
        value,
        status,
    }
}
fn descriptor(
    id: &str,
    name: &str,
    entity: EntityKind,
    unit: CanonicalUnit,
    kind: ValueKind,
    temporal: TemporalSemantics,
    provider: &str,
    semantics: &str,
    group: Option<&str>,
) -> MetricDescriptor {
    descriptor_owned(
        id.into(),
        name.into(),
        entity,
        unit,
        kind,
        temporal,
        provider,
        semantics,
        group,
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
    group: Option<&str>,
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
        comparability_group: group.map(str::to_owned),
        semantics_version: 1,
    }
}

fn discover_temperatures() -> Vec<FileSensor> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/class/hwmon") else {
        return out;
    };
    for entry in entries.flatten() {
        let base = entry.path();
        let device = fs::read_to_string(base.join("name"))
            .unwrap_or_else(|_| "hwmon".into())
            .trim()
            .to_owned();
        let Ok(files) = fs::read_dir(&base) else {
            continue;
        };
        for file in files.flatten() {
            let path = file.path();
            let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
                continue;
            };
            if !name.starts_with("temp") || !name.ends_with("_input") {
                continue;
            }
            let label = fs::read_to_string(base.join(name.replace("_input", "_label")))
                .ok()
                .map(|v| v.trim().to_owned())
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| name.trim_end_matches("_input").into());
            let index = out.len();
            out.push(FileSensor {
                metric_id: format!("thermal.temperature.{index}").into(),
                entity_id: format!("sensor:{}:{}", clean_id(&device), clean_id(&label)).into(),
                path,
            });
        }
    }
    out
}

fn discover_frequencies() -> Vec<FileSensor> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/devices/system/cpu") else {
        return out;
    };
    for entry in entries.flatten() {
        let base = entry.path();
        let Some(name) = base.file_name().and_then(|v| v.to_str()) else {
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
        out.push(FileSensor {
            metric_id: format!("cpu.frequency.{index}").into(),
            entity_id: format!("cpu:{index}").into(),
            path,
        });
    }
    out.sort_by(|a, b| a.entity_id.cmp(&b.entity_id));
    out
}
fn read_number(path: &Path) -> Option<f64> {
    let mut text = String::new();
    fs::File::open(path).ok()?.read_to_string(&mut text).ok()?;
    text.trim().parse().ok()
}
fn clean_id(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(feature = "nvidia")]
pub struct NvidiaProvider {
    nvml: nvml_wrapper::Nvml,
    devices: Vec<(u32, EntityId)>,
    descriptors: Vec<MetricDescriptor>,
}
#[cfg(feature = "nvidia")]
impl NvidiaProvider {
    pub fn discover() -> Discovery<Self> {
        let nvml = match nvml_wrapper::Nvml::init() {
            Ok(v) => v,
            Err(e) => {
                return Discovery::Unavailable(CapabilityState::Unsupported {
                    reason: Some(format!("{e:?}")),
                })
            }
        };
        let count = match nvml.device_count() {
            Ok(v) if v > 0 => v,
            Ok(_) => {
                return Discovery::Unavailable(CapabilityState::Unsupported {
                    reason: Some("NVML reported no visible devices".into()),
                })
            }
            Err(e) => {
                return Discovery::Unavailable(CapabilityState::TemporarilyUnavailable {
                    reason: Some(format!("{e:?}")),
                })
            }
        };
        let devices: Vec<_> = (0..count)
            .filter_map(|index| {
                nvml.device_by_index(index).ok().map(|d| {
                    (
                        index,
                        d.uuid()
                            .map(|u| format!("gpu:uuid:{u}"))
                            .unwrap_or_else(|_| format!("gpu:nvidia:{index}"))
                            .into(),
                    )
                })
            })
            .collect();
        if devices.is_empty() {
            return Discovery::Unavailable(CapabilityState::TemporarilyUnavailable {
                reason: Some("NVML devices could not be opened".into()),
            });
        }
        let descriptors = vec![
            descriptor(
                "gpu.nvidia.compute_utilization",
                "GPU utilization",
                EntityKind::Gpu,
                CanonicalUnit::Percent,
                ValueKind::Gauge,
                TemporalSemantics::VendorSampled,
                "nvml",
                "percent of the vendor sample period during which kernels executed",
                Some("nvml_gpu_time_busy"),
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
        ];
        Discovery::Available(Self {
            nvml,
            devices,
            descriptors,
        })
    }
}
#[cfg(feature = "nvidia")]
impl MetricProvider for NvidiaProvider {
    fn descriptors(&self) -> &[MetricDescriptor] {
        &self.descriptors
    }
    fn observe(&mut self, time: MonotonicTimestamp) -> Vec<ProviderResult> {
        use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
        let mut out = Vec::new();
        for (index, entity) in &self.devices {
            let Ok(device) = self.nvml.device_by_index(*index) else {
                continue;
            };
            out.push(observation_owned(
                "gpu.nvidia.compute_utilization".into(),
                entity.clone(),
                time,
                None,
                device
                    .utilization_rates()
                    .ok()
                    .map(|v| MetricValue::F64(v.gpu as f64)),
            ));
            out.push(observation_owned(
                "gpu.nvidia.vram_occupancy".into(),
                entity.clone(),
                time,
                None,
                device.memory_info().ok().and_then(|v| {
                    (v.total > 0).then(|| MetricValue::F64(v.used as f64 * 100.0 / v.total as f64))
                }),
            ));
            out.push(observation_owned(
                "gpu.nvidia.temperature".into(),
                entity.clone(),
                time,
                None,
                device
                    .temperature(TemperatureSensor::Gpu)
                    .ok()
                    .map(|v| MetricValue::F64(v as f64)),
            ));
            out.push(observation_owned(
                "gpu.nvidia.clock.graphics".into(),
                entity.clone(),
                time,
                None,
                device
                    .clock_info(Clock::Graphics)
                    .ok()
                    .map(|v| MetricValue::F64(v as f64 * 1_000_000.0)),
            ));
        }
        out
    }
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
