use crate::collection::{MetricProvider, ProviderReading, ReadingOutcome};
use crate::model::{
    CanonicalUnit, CapabilityStatus, EntityId, EntityKind, MetricDescriptor, MetricId,
    ObservationTime, SampleValue, TemporalSemantics, ValueKind,
};
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use sysinfo::{CpuExt, System, SystemExt};

pub const CPU_UTILIZATION: &str = "system.cpu.utilization";
pub const RAM_UTILIZATION: &str = "system.memory.utilization";
pub const GPU_UTILIZATION: &str = "gpu.utilization";
pub const GPU_MEMORY_UTILIZATION: &str = "gpu.memory.utilization";

#[derive(Clone)]
struct FileMetric {
    descriptor: MetricDescriptor,
    entity: EntityId,
    path: PathBuf,
    scale: f64,
}

pub struct HostProvider {
    system: System,
    file_metrics: Vec<FileMetric>,
    descriptors: Vec<MetricDescriptor>,
    #[cfg(feature = "nvidia")]
    nvidia: NvidiaProvider,
}

impl HostProvider {
    pub fn new() -> Self {
        let mut descriptors = vec![
            descriptor(
                CPU_UTILIZATION,
                "CPU",
                EntityKind::Cpu,
                CanonicalUnit::Percent,
                "sysinfo",
                "mean logical CPU utilization",
            ),
            descriptor(
                RAM_UTILIZATION,
                "RAM",
                EntityKind::Memory,
                CanonicalUnit::Percent,
                "sysinfo",
                "used physical memory divided by total physical memory",
            ),
        ];
        let mut file_metrics = discover_frequencies();
        file_metrics.extend(discover_temperatures());
        descriptors.extend(file_metrics.iter().map(|metric| metric.descriptor.clone()));
        #[cfg(feature = "nvidia")]
        let nvidia = NvidiaProvider::new();
        #[cfg(feature = "nvidia")]
        descriptors.extend(nvidia.descriptors());
        Self {
            system: System::new_all(),
            file_metrics,
            descriptors,
            #[cfg(feature = "nvidia")]
            nvidia,
        }
    }
}

impl Default for HostProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricProvider for HostProvider {
    fn descriptors(&self) -> Vec<MetricDescriptor> {
        self.descriptors.clone()
    }

    fn collect(&mut self, requested_at: ObservationTime) -> Vec<ProviderReading> {
        self.system.refresh_cpu();
        self.system.refresh_memory();
        let mut readings = Vec::new();
        if self.system.cpus().is_empty() {
            readings.push(unavailable(CPU_UTILIZATION, "system", "no CPU records"));
        } else {
            let value = self
                .system
                .cpus()
                .iter()
                .map(|cpu| cpu.cpu_usage() as f64)
                .sum::<f64>()
                / self.system.cpus().len() as f64;
            readings.push(ProviderReading::numeric(CPU_UTILIZATION, "system", value));
        }
        let total = self.system.total_memory() as f64;
        if total > 0.0 {
            readings.push(ProviderReading::numeric(
                RAM_UTILIZATION,
                "system",
                self.system.used_memory() as f64 / total * 100.0,
            ));
        } else {
            readings.push(unavailable(
                RAM_UTILIZATION,
                "system",
                "total memory is unavailable",
            ));
        }
        for metric in &self.file_metrics {
            let outcome = read_number(&metric.path)
                .map(|value| ReadingOutcome::Value(SampleValue::Numeric(value * metric.scale)))
                .unwrap_or_else(classify_io_error);
            readings.push(ProviderReading {
                metric_id: metric.descriptor.metric_id.clone(),
                entity_id: metric.entity.clone(),
                observation_time: None,
                interval_start: None,
                outcome,
            });
        }
        #[cfg(feature = "nvidia")]
        readings.extend(self.nvidia.collect(requested_at));
        readings
    }
}

fn descriptor(
    id: &str,
    name: &str,
    kind: EntityKind,
    unit: CanonicalUnit,
    provider: &str,
    semantics: &str,
) -> MetricDescriptor {
    MetricDescriptor {
        metric_id: MetricId::from(id),
        display_name: name.into(),
        entity_kind: kind,
        canonical_unit: unit,
        value_kind: ValueKind::Gauge,
        temporal_semantics: TemporalSemantics::PointSampled,
        provider: provider.into(),
        capability_status: CapabilityStatus::TemporarilyUnavailable,
        source_resolution_ns: None,
        source_semantics: semantics.into(),
        comparability_group: None,
        semantics_version: 1,
    }
}

fn unavailable(metric: &str, entity: &str, detail: &str) -> ProviderReading {
    ProviderReading {
        metric_id: metric.into(),
        entity_id: entity.into(),
        observation_time: None,
        interval_start: None,
        outcome: ReadingOutcome::TemporarilyUnavailable(detail.into()),
    }
}

fn read_number(path: &PathBuf) -> io::Result<f64> {
    let mut text = String::new();
    fs::File::open(path)?.read_to_string(&mut text)?;
    text.trim()
        .parse()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn classify_io_error(error: io::Error) -> ReadingOutcome {
    if error.kind() == io::ErrorKind::PermissionDenied {
        ReadingOutcome::PermissionDenied(error.to_string())
    } else if matches!(
        error.kind(),
        io::ErrorKind::NotFound | io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
    ) {
        ReadingOutcome::TemporarilyUnavailable(error.to_string())
    } else {
        ReadingOutcome::Error(error.to_string())
    }
}

fn discover_frequencies() -> Vec<FileMetric> {
    let mut metrics = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/devices/system/cpu") else {
        return metrics;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Ok(core) = name.strip_prefix("cpu").unwrap_or("").parse::<usize>() else {
            continue;
        };
        let base = path.join("cpufreq");
        let primary = base.join("scaling_cur_freq");
        let fallback = base.join("cpuinfo_cur_freq");
        let source = if primary.exists() {
            primary
        } else if fallback.exists() {
            fallback
        } else {
            continue;
        };
        let id = format!("cpu.frequency.core.{core}");
        metrics.push(FileMetric {
            descriptor: descriptor(
                &id,
                &format!("CPU Core {core}"),
                EntityKind::CpuCore,
                CanonicalUnit::Hertz,
                "linux_sysfs",
                "source-native CPU frequency",
            ),
            entity: EntityId(format!("cpu:{core}")),
            path: source,
            scale: 1_000.0,
        });
    }
    metrics.sort_by(|a, b| a.entity.cmp(&b.entity));
    metrics
}

fn discover_temperatures() -> Vec<FileMetric> {
    let mut metrics = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/class/hwmon") else {
        return metrics;
    };
    for entry in entries.flatten() {
        let base = entry.path();
        let chip = fs::read_to_string(base.join("name"))
            .unwrap_or_else(|_| "hwmon".into())
            .trim()
            .to_owned();
        let Ok(files) = fs::read_dir(&base) else {
            continue;
        };
        for file in files.flatten() {
            let path = file.path();
            let Some(filename) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if !filename.starts_with("temp") || !filename.ends_with("_input") {
                continue;
            }
            let label = fs::read_to_string(base.join(filename.replace("_input", "_label")))
                .unwrap_or_else(|_| chip.clone())
                .trim()
                .to_owned();
            let id = format!("thermal.{}.{}", chip, filename.trim_end_matches("_input"));
            let mut item = descriptor(
                &id,
                &label,
                EntityKind::ThermalSensor,
                CanonicalUnit::Celsius,
                "linux_hwmon",
                &chip,
            );
            item.source_resolution_ns = None;
            metrics.push(FileMetric {
                descriptor: item,
                entity: EntityId(format!("hwmon:{chip}:{filename}")),
                path,
                scale: 0.001,
            });
        }
    }
    metrics
}

#[cfg(feature = "nvidia")]
struct NvidiaProvider {
    backend: NvidiaBackend,
    descriptors: Vec<MetricDescriptor>,
}

#[cfg(feature = "nvidia")]
enum NvidiaBackend {
    Ready(Box<nvml_wrapper::Nvml>, Vec<(u32, EntityId)>),
    Unavailable(String),
}

#[cfg(feature = "nvidia")]
impl NvidiaProvider {
    fn new() -> Self {
        use nvml_wrapper::Nvml;
        let (backend, entities) = match Nvml::init() {
            Ok(nvml) => match nvml.device_count() {
                Ok(count) => {
                    let entities = (0..count)
                        .map(|index| {
                            let entity = nvml
                                .device_by_index(index)
                                .ok()
                                .and_then(|device| device.uuid().ok())
                                .map(|uuid| EntityId(format!("gpu:uuid:{uuid}")))
                                .unwrap_or_else(|| EntityId(format!("gpu:index:{index}")));
                            (index, entity)
                        })
                        .collect::<Vec<_>>();
                    (
                        NvidiaBackend::Ready(Box::new(nvml), entities.clone()),
                        entities,
                    )
                }
                Err(error) => (
                    NvidiaBackend::Unavailable(format!("NVML device discovery failed: {error}")),
                    vec![(0, EntityId("gpu:nvidia:discovery-error".into()))],
                ),
            },
            Err(error) => (
                NvidiaBackend::Unavailable(format!("NVML unavailable: {error}")),
                vec![(0, EntityId("gpu:nvidia:unavailable".into()))],
            ),
        };
        let mut descriptors = Vec::new();
        for (_, entity) in entities {
            descriptors.extend(nvidia_descriptors(&entity));
        }
        Self {
            backend,
            descriptors,
        }
    }

    fn descriptors(&self) -> Vec<MetricDescriptor> {
        self.descriptors.clone()
    }

    fn collect(&mut self, _requested_at: ObservationTime) -> Vec<ProviderReading> {
        use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
        let NvidiaBackend::Ready(nvml, entities) = &self.backend else {
            let NvidiaBackend::Unavailable(detail) = &self.backend else {
                unreachable!()
            };
            return self
                .descriptors
                .iter()
                .map(|descriptor| ProviderReading {
                    metric_id: descriptor.metric_id.clone(),
                    entity_id: EntityId("gpu:nvidia:unavailable".into()),
                    observation_time: None,
                    interval_start: None,
                    outcome: ReadingOutcome::Error(detail.clone()),
                })
                .collect();
        };
        let mut result = Vec::new();
        for (index, entity) in entities {
            let device = match nvml.device_by_index(*index) {
                Ok(device) => device,
                Err(error) => {
                    for descriptor in nvidia_descriptors(entity) {
                        result.push(ProviderReading {
                            metric_id: descriptor.metric_id,
                            entity_id: entity.clone(),
                            observation_time: None,
                            interval_start: None,
                            outcome: ReadingOutcome::Error(format!(
                                "NVML device query failed: {error}"
                            )),
                        });
                    }
                    continue;
                }
            };
            let prefix = entity.0.replace(':', ".");
            result.push(nv_read(
                format!("{prefix}.utilization"),
                entity.clone(),
                device.utilization_rates().map(|value| value.gpu as f64),
            ));
            result.push(nv_read(
                format!("{prefix}.memory_utilization"),
                entity.clone(),
                device.memory_info().map(|value| {
                    if value.total == 0 {
                        0.0
                    } else {
                        value.used as f64 / value.total as f64 * 100.0
                    }
                }),
            ));
            result.push(nv_read(
                format!("{prefix}.temperature"),
                entity.clone(),
                device
                    .temperature(TemperatureSensor::Gpu)
                    .map(|value| value as f64),
            ));
            result.push(nv_read(
                format!("{prefix}.clock.graphics"),
                entity.clone(),
                device
                    .clock_info(Clock::Graphics)
                    .map(|value| value as f64 * 1_000_000.0),
            ));
            result.push(nv_read(
                format!("{prefix}.clock.sm"),
                entity.clone(),
                device
                    .clock_info(Clock::SM)
                    .map(|value| value as f64 * 1_000_000.0),
            ));
            result.push(nv_read(
                format!("{prefix}.clock.memory"),
                entity.clone(),
                device
                    .clock_info(Clock::Memory)
                    .map(|value| value as f64 * 1_000_000.0),
            ));
            result.push(nv_read(
                format!("{prefix}.clock.video"),
                entity.clone(),
                device
                    .clock_info(Clock::Video)
                    .map(|value| value as f64 * 1_000_000.0),
            ));
        }
        result
    }
}

#[cfg(feature = "nvidia")]
fn nvidia_descriptors(entity: &EntityId) -> Vec<MetricDescriptor> {
    let prefix = entity.0.replace(':', ".");
    [
        ("utilization", "GPU %", CanonicalUnit::Percent),
        ("memory_utilization", "VRAM %", CanonicalUnit::Percent),
        ("temperature", "GPU Core", CanonicalUnit::Celsius),
        ("clock.graphics", "GPU Graphics", CanonicalUnit::Hertz),
        ("clock.sm", "GPU SM", CanonicalUnit::Hertz),
        ("clock.memory", "GPU Memory", CanonicalUnit::Hertz),
        ("clock.video", "GPU Video", CanonicalUnit::Hertz),
    ]
    .into_iter()
    .map(|(suffix, name, unit)| {
        let mut value = descriptor(
            &format!("{prefix}.{suffix}"),
            name,
            EntityKind::Gpu,
            unit,
            "nvml",
            "per-device NVML query result",
        );
        value.temporal_semantics = TemporalSemantics::VendorSampled;
        value.comparability_group = Some(format!("nvidia.{suffix}"));
        value
    })
    .collect()
}

#[cfg(feature = "nvidia")]
fn nv_read<T: std::fmt::Display>(
    metric: String,
    entity: EntityId,
    result: Result<f64, T>,
) -> ProviderReading {
    let outcome = match result {
        Ok(value) => ReadingOutcome::Value(SampleValue::Numeric(value)),
        Err(error) => {
            let detail = error.to_string();
            let lower = detail.to_lowercase();
            if lower.contains("not supported") {
                ReadingOutcome::Unsupported(detail)
            } else if lower.contains("permission") || lower.contains("no permission") {
                ReadingOutcome::PermissionDenied(detail)
            } else if lower.contains("temporar")
                || lower.contains("lost")
                || lower.contains("reset")
            {
                ReadingOutcome::TemporarilyUnavailable(detail)
            } else {
                ReadingOutcome::Error(detail)
            }
        }
    };
    ProviderReading {
        metric_id: MetricId(metric),
        entity_id: entity,
        observation_time: None,
        interval_start: None,
        outcome,
    }
}
