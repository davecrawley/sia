use crate::collection::{MetricProvider, MetricTarget, ProviderReading, ReadingOutcome};
use crate::model::{
    CanonicalUnit, CapabilityStatus, EntityId, EntityKind, MetricDescriptor, MetricId,
    ObservationTime, SampleValue, TemporalSemantics, ValueKind,
};
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use sysinfo::{CpuExt, System, SystemExt};

pub const CPU_UTILIZATION: &str = "system.cpu.utilization";
pub const RAM_UTILIZATION: &str = "system.memory.utilization";

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
    previous_collection: Option<ObservationTime>,
    #[cfg(feature = "nvidia")]
    nvidia: NvidiaProvider,
}

impl HostProvider {
    pub fn new() -> Self {
        let mut cpu = descriptor(
            CPU_UTILIZATION,
            "CPU",
            EntityKind::Cpu,
            CanonicalUnit::Percent,
            "sysinfo",
            "mean logical CPU utilization over the sysinfo refresh interval",
        );
        cpu.temporal_semantics = TemporalSemantics::IntervalAverage;
        let mut descriptors = vec![
            cpu,
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
            previous_collection: None,
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

    fn targets(&self) -> Vec<MetricTarget> {
        let mut targets = vec![
            MetricTarget {
                descriptor: self.descriptors[0].clone(),
                entity_id: "system".into(),
            },
            MetricTarget {
                descriptor: self.descriptors[1].clone(),
                entity_id: "system".into(),
            },
        ];
        targets.extend(self.file_metrics.iter().map(|metric| MetricTarget {
            descriptor: metric.descriptor.clone(),
            entity_id: metric.entity.clone(),
        }));
        #[cfg(feature = "nvidia")]
        targets.extend(self.nvidia.targets());
        targets
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
            let mut reading = ProviderReading::numeric(CPU_UTILIZATION, "system", value);
            reading.interval_start = self.previous_collection;
            readings.push(reading);
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
        self.previous_collection = Some(requested_at);
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
        metric_id: id.into(),
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

fn read_number(path: &Path) -> io::Result<f64> {
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
            entity: format!("cpu:{core}").into(),
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
        let instance = base
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("hwmon")
            .to_owned();
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
            let sensor = filename.trim_end_matches("_input");
            let id = format!("thermal.{instance}.{sensor}");
            metrics.push(FileMetric {
                descriptor: descriptor(
                    &id,
                    &label,
                    EntityKind::ThermalSensor,
                    CanonicalUnit::Celsius,
                    "linux_hwmon",
                    &format!("{chip} at {instance}/{filename}"),
                ),
                entity: format!("hwmon:{instance}:{sensor}").into(),
                path,
                scale: 0.001,
            });
        }
    }
    metrics.sort_by(|a, b| a.entity.cmp(&b.entity));
    metrics
}

#[cfg(feature = "nvidia")]
struct NvidiaProvider {
    backend: NvidiaBackend,
    targets: Vec<MetricTarget>,
}

#[cfg(feature = "nvidia")]
enum NvidiaBackend {
    Ready(Box<nvml_wrapper::Nvml>, Vec<(u32, EntityId)>),
    Unavailable {
        outcome: ReadingOutcome,
        entity: EntityId,
    },
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
                Err(error) => {
                    let entity = EntityId("gpu:nvidia:discovery-error".into());
                    (
                        NvidiaBackend::Unavailable {
                            outcome: nv_error(format!("NVML device discovery failed: {error}")),
                            entity: entity.clone(),
                        },
                        vec![(0, entity)],
                    )
                }
            },
            Err(error) => {
                let entity = EntityId("gpu:nvidia:unavailable".into());
                (
                    NvidiaBackend::Unavailable {
                        outcome: nv_error(format!("NVML unavailable: {error}")),
                        entity: entity.clone(),
                    },
                    vec![(0, entity)],
                )
            }
        };
        let targets = entities
            .into_iter()
            .flat_map(|(_, entity)| nvidia_descriptors(&entity))
            .collect();
        Self { backend, targets }
    }

    fn descriptors(&self) -> Vec<MetricDescriptor> {
        self.targets
            .iter()
            .map(|target| target.descriptor.clone())
            .collect()
    }

    fn targets(&self) -> Vec<MetricTarget> {
        self.targets.clone()
    }

    fn collect(&mut self, _requested_at: ObservationTime) -> Vec<ProviderReading> {
        use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
        let NvidiaBackend::Ready(nvml, entities) = &self.backend else {
            let NvidiaBackend::Unavailable { outcome, entity } = &self.backend else {
                unreachable!()
            };
            return self
                .targets
                .iter()
                .map(|target| ProviderReading {
                    metric_id: target.descriptor.metric_id.clone(),
                    entity_id: entity.clone(),
                    observation_time: None,
                    interval_start: None,
                    outcome: outcome.clone(),
                })
                .collect();
        };
        let mut readings = Vec::new();
        for (index, entity) in entities {
            let device = match nvml.device_by_index(*index) {
                Ok(device) => device,
                Err(error) => {
                    readings.extend(nvidia_descriptors(entity).into_iter().map(|target| {
                        ProviderReading {
                            metric_id: target.descriptor.metric_id,
                            entity_id: entity.clone(),
                            observation_time: None,
                            interval_start: None,
                            outcome: nv_error(format!("NVML device query failed: {error}")),
                        }
                    }));
                    continue;
                }
            };
            readings.push(nv_read(
                nvidia_metric_id(entity, "utilization"),
                entity.clone(),
                device.utilization_rates().map(|value| value.gpu as f64),
            ));
            let memory_outcome = match device.memory_info() {
                Ok(memory) if memory.total > 0 => ReadingOutcome::Value(SampleValue::Numeric(
                    memory.used as f64 / memory.total as f64 * 100.0,
                )),
                Ok(_) => ReadingOutcome::Error("NVML reported zero total device memory".into()),
                Err(error) => nv_error(error.to_string()),
            };
            readings.push(ProviderReading {
                metric_id: nvidia_metric_id(entity, "memory_utilization"),
                entity_id: entity.clone(),
                observation_time: None,
                interval_start: None,
                outcome: memory_outcome,
            });
            readings.push(nv_read(
                nvidia_metric_id(entity, "temperature"),
                entity.clone(),
                device
                    .temperature(TemperatureSensor::Gpu)
                    .map(|value| value as f64),
            ));
            for (suffix, clock) in [
                ("clock.graphics", Clock::Graphics),
                ("clock.sm", Clock::SM),
                ("clock.memory", Clock::Memory),
                ("clock.video", Clock::Video),
            ] {
                readings.push(nv_read(
                    nvidia_metric_id(entity, suffix),
                    entity.clone(),
                    device
                        .clock_info(clock)
                        .map(|value| value as f64 * 1_000_000.0),
                ));
            }
        }
        readings
    }
}

#[cfg(feature = "nvidia")]
fn nvidia_metric_id(entity: &EntityId, suffix: &str) -> MetricId {
    format!("{}.{}", entity.0.replace(':', "."), suffix).into()
}

#[cfg(feature = "nvidia")]
fn nvidia_descriptors(entity: &EntityId) -> Vec<MetricTarget> {
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
        let mut item = descriptor(
            &nvidia_metric_id(entity, suffix).0,
            name,
            EntityKind::Gpu,
            unit,
            "nvml",
            "per-device point query from NVML",
        );
        item.comparability_group = Some(format!("nvidia.{suffix}"));
        if suffix == "utilization" {
            item.temporal_semantics = TemporalSemantics::VendorSampled;
            item.source_semantics =
                "GPU utilization over an NVML vendor-defined sampling window; NVML does not expose the exact window for this query"
                    .into();
            item.source_resolution_ns = None;
        }
        MetricTarget {
            descriptor: item,
            entity_id: entity.clone(),
        }
    })
    .collect()
}

#[cfg(feature = "nvidia")]
fn nv_read<T: std::fmt::Display>(
    metric: MetricId,
    entity: EntityId,
    result: Result<f64, T>,
) -> ProviderReading {
    ProviderReading {
        metric_id: metric,
        entity_id: entity,
        observation_time: None,
        interval_start: None,
        outcome: match result {
            Ok(value) => ReadingOutcome::Value(SampleValue::Numeric(value)),
            Err(error) => nv_error(error.to_string()),
        },
    }
}

#[cfg(feature = "nvidia")]
fn nv_error(detail: String) -> ReadingOutcome {
    let lower = detail.to_lowercase();
    if lower.contains("not supported") || lower.contains("unsupported") {
        ReadingOutcome::Unsupported(detail)
    } else if lower.contains("permission") || lower.contains("not authorized") {
        ReadingOutcome::PermissionDenied(detail)
    } else if lower.contains("temporar") || lower.contains("lost") || lower.contains("reset") {
        ReadingOutcome::TemporarilyUnavailable(detail)
    } else {
        ReadingOutcome::Error(detail)
    }
}
