//! All system refreshes, sensor discovery/reads, and NVML queries live here.

use crate::{
    Capability, EntityKind, MetricDescriptor, MonitorRole, Provider, Reading, SourceValue,
    TemporalSemantics, Unit, ValueKind,
};
use std::fs;
use std::path::{Path, PathBuf};
use sysinfo::{CpuExt, System, SystemExt};

#[cfg(feature = "nvidia")]
use crate::GpuClock;
#[cfg(feature = "nvidia")]
use nvml_wrapper::{
    enum_wrappers::device::{Clock as NvClock, TemperatureSensor},
    error::NvmlError,
    Nvml,
};

#[derive(Clone, Copy)]
enum Conversion {
    Temperature,
    Kilohertz,
}

enum Source {
    Cpu,
    Ram,
    File(PathBuf, Conversion),
    #[cfg(feature = "nvidia")]
    Nvidia(NvidiaMetric),
    #[cfg(not(feature = "nvidia"))]
    Unsupported,
}

#[cfg(feature = "nvidia")]
#[derive(Clone, Copy)]
enum NvidiaMetric {
    Utilization,
    Vram,
    Temperature,
    Clock(GpuClock),
}

struct Entry {
    descriptor: MetricDescriptor,
    source: Source,
}

pub struct HostProvider {
    system: System,
    entries: Vec<Entry>,
    #[cfg(feature = "nvidia")]
    nvml: Result<Nvml, Capability>,
}

fn descriptor(
    metric_id: &str,
    entity_id: &str,
    role: MonitorRole,
    provider: &str,
    semantics: &str,
) -> MetricDescriptor {
    let (entity_kind, unit, temporal_semantics) = match &role {
        MonitorRole::CpuUtilization => (
            EntityKind::Cpu,
            Unit::Percent,
            TemporalSemantics::IntervalAverage,
        ),
        MonitorRole::RamUtilization => (
            EntityKind::System,
            Unit::Percent,
            TemporalSemantics::PointSample,
        ),
        MonitorRole::GpuUtilization => (
            EntityKind::Gpu,
            Unit::Percent,
            TemporalSemantics::VendorSampled,
        ),
        MonitorRole::VramUtilization => (
            EntityKind::Gpu,
            Unit::Percent,
            TemporalSemantics::PointSample,
        ),
        MonitorRole::Temperature { .. } => (
            EntityKind::Sensor,
            Unit::Celsius,
            TemporalSemantics::PointSample,
        ),
        MonitorRole::CpuFrequency { .. } => {
            (EntityKind::Cpu, Unit::Hertz, TemporalSemantics::PointSample)
        }
        MonitorRole::GpuFrequency(_) => {
            (EntityKind::Gpu, Unit::Hertz, TemporalSemantics::PointSample)
        }
    };
    MetricDescriptor {
        metric_id: metric_id.into(),
        entity_id: entity_id.into(),
        entity_kind,
        display_name: metric_id.into(),
        unit,
        value_kind: ValueKind::Gauge,
        temporal_semantics,
        provider: provider.into(),
        source_semantics: semantics.into(),
        semantics_version: 1,
        capability: Capability::Available,
        role,
    }
}

impl Default for HostProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl HostProvider {
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        let mut provider = Self {
            system,
            entries: vec![
                Entry {
                    descriptor: descriptor(
                        "cpu.utilization",
                        "cpu:all",
                        MonitorRole::CpuUtilization,
                        "sysinfo",
                        "Mean logical-CPU utilization between sysinfo refreshes",
                    ),
                    source: Source::Cpu,
                },
                Entry {
                    descriptor: descriptor(
                        "memory.occupancy",
                        "system:memory",
                        MonitorRole::RamUtilization,
                        "sysinfo",
                        "Used RAM divided by total RAM, multiplied by 100",
                    ),
                    source: Source::Ram,
                },
            ],
            #[cfg(feature = "nvidia")]
            nvml: translate_nvml_result(Nvml::init()),
        };
        provider.add_gpu_metrics();
        provider.discover_temperatures();
        provider.discover_frequencies();
        provider
    }

    fn add_gpu_metrics(&mut self) {
        for (id, role, semantics) in [
            (
                "gpu.utilization",
                MonitorRole::GpuUtilization,
                "NVML GPU busy percentage over the vendor's internal sample period",
            ),
            (
                "gpu.memory.occupancy",
                MonitorRole::VramUtilization,
                "NVML used device-memory bytes divided by total bytes, multiplied by 100",
            ),
        ] {
            #[cfg(feature = "nvidia")]
            let source = Source::Nvidia(if role == MonitorRole::GpuUtilization {
                NvidiaMetric::Utilization
            } else {
                NvidiaMetric::Vram
            });
            #[cfg(not(feature = "nvidia"))]
            let source = Source::Unsupported;
            self.entries.push(Entry {
                descriptor: descriptor(id, "gpu:0", role, "nvml", semantics),
                source,
            });
        }
        #[cfg(feature = "nvidia")]
        {
            self.entries.push(Entry {
                descriptor: descriptor(
                    "gpu.temperature",
                    "gpu:0",
                    MonitorRole::Temperature {
                        source_name: "nvidia".into(),
                        label: "GPU (Core)".into(),
                    },
                    "nvml",
                    "NVML GPU temperature in degrees Celsius",
                ),
                source: Source::Nvidia(NvidiaMetric::Temperature),
            });
            for (id, clock) in [
                ("gpu.clock.graphics", GpuClock::Graphics),
                ("gpu.clock.sm", GpuClock::Sm),
                ("gpu.clock.memory", GpuClock::Memory),
                ("gpu.clock.video", GpuClock::Video),
            ] {
                self.entries.push(Entry {
                    descriptor: descriptor(
                        id,
                        "gpu:0",
                        MonitorRole::GpuFrequency(clock),
                        "nvml",
                        "NVML current clock, converted from MHz to Hz",
                    ),
                    source: Source::Nvidia(NvidiaMetric::Clock(clock)),
                });
            }
        }
    }

    fn discover_temperatures(&mut self) {
        let Ok(entries) = fs::read_dir("/sys/class/hwmon") else {
            return;
        };
        for entry in entries.flatten() {
            let base = entry.path();
            let name = fs::read_to_string(base.join("name"))
                .unwrap_or_default()
                .trim()
                .to_string();
            let Ok(files) = fs::read_dir(&base) else {
                continue;
            };
            for file in files.flatten() {
                let path = file.path();
                let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                if !filename.starts_with("temp") || !filename.ends_with("_input") {
                    continue;
                }
                let mut label = name.clone();
                if let Ok(text) =
                    fs::read_to_string(base.join(filename.replace("_input", "_label")))
                {
                    if !text.trim().is_empty() {
                        label = text.trim().to_string();
                    }
                }
                let entity = path.to_string_lossy().into_owned();
                self.entries.push(Entry {
                    descriptor: descriptor(
                        "sensor.temperature",
                        &entity,
                        MonitorRole::Temperature {
                            source_name: name.clone(),
                            label,
                        },
                        "sysfs",
                        "hwmon temperature using the existing monitor's millidegree conversion",
                    ),
                    source: Source::File(path, Conversion::Temperature),
                });
            }
        }
    }

    fn discover_frequencies(&mut self) {
        let Ok(entries) = fs::read_dir("/sys/devices/system/cpu") else {
            return;
        };
        let mut frequencies = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let Some(suffix) = name.strip_prefix("cpu") else {
                continue;
            };
            let Ok(core) = suffix.parse::<usize>() else {
                continue;
            };
            let first = path.join("cpufreq/scaling_cur_freq");
            let second = path.join("cpufreq/cpuinfo_cur_freq");
            let path = if first.exists() {
                first
            } else if second.exists() {
                second
            } else {
                continue;
            };
            frequencies.push((core, path));
        }
        frequencies.sort_by_key(|(core, _)| *core);
        for (core, path) in frequencies {
            self.entries.push(Entry {
                descriptor: descriptor(
                    "cpu.frequency",
                    &format!("cpu:{core}"),
                    MonitorRole::CpuFrequency { core },
                    "sysfs",
                    "cpufreq scaling_cur_freq or cpuinfo_cur_freq, converted from kHz to Hz",
                ),
                source: Source::File(path, Conversion::Kilohertz),
            });
        }
    }

    fn read_source(&self, source: &Source) -> Result<SourceValue, Capability> {
        let value = match source {
            Source::Cpu => {
                let cpus = self.system.cpus();
                if cpus.is_empty() {
                    return Err(Capability::TemporarilyUnavailable(
                        "No CPU observations".into(),
                    ));
                }
                (cpus.iter().map(|cpu| cpu.cpu_usage()).sum::<f32>() / cpus.len() as f32) as f64
            }
            Source::Ram => {
                let total = self.system.total_memory();
                if total == 0 {
                    return Err(Capability::TemporarilyUnavailable(
                        "Total memory is unavailable".into(),
                    ));
                }
                self.system.used_memory() as f64 / total as f64 * 100.0
            }
            Source::File(path, conversion) => {
                let raw = read_number(path)?;
                match conversion {
                    Conversion::Temperature => {
                        if raw > 1000.0 {
                            raw / 1000.0
                        } else {
                            raw
                        }
                    }
                    Conversion::Kilohertz => raw * 1000.0,
                }
            }
            #[cfg(feature = "nvidia")]
            Source::Nvidia(metric) => return self.read_nvidia(*metric),
            #[cfg(not(feature = "nvidia"))]
            Source::Unsupported => {
                return Err(Capability::Unsupported(
                    "NVIDIA support was not compiled in".into(),
                ));
            }
        };
        Ok(SourceValue::gauge(value))
    }

    #[cfg(feature = "nvidia")]
    fn read_nvidia(&self, metric: NvidiaMetric) -> Result<SourceValue, Capability> {
        let nvml = self.nvml.as_ref().map_err(Clone::clone)?;
        let device = translate_nvml_result(nvml.device_by_index(0))?;
        let value = match metric {
            NvidiaMetric::Utilization => {
                translate_nvml_result(device.utilization_rates())?.gpu as f64
            }
            NvidiaMetric::Vram => {
                let memory = translate_nvml_result(device.memory_info())?;
                if memory.total == 0 {
                    return Err(Capability::TemporarilyUnavailable(
                        "Total device memory is unavailable".into(),
                    ));
                }
                memory.used as f64 / memory.total as f64 * 100.0
            }
            NvidiaMetric::Temperature => {
                translate_nvml_result(device.temperature(TemperatureSensor::Gpu))? as f64
            }
            NvidiaMetric::Clock(clock) => {
                let clock = match clock {
                    GpuClock::Graphics => NvClock::Graphics,
                    GpuClock::Sm => NvClock::SM,
                    GpuClock::Memory => NvClock::Memory,
                    GpuClock::Video => NvClock::Video,
                };
                translate_nvml_result(device.clock_info(clock))? as f64 * 1_000_000.0
            }
        };
        // These NVML operations return no native timestamp or interval. Do not
        // invent either, and do not substitute another clock on query failure.
        Ok(SourceValue::gauge(value))
    }
}

impl Provider for HostProvider {
    fn discover(&mut self) -> Vec<MetricDescriptor> {
        for index in 0..self.entries.len() {
            let capability = match self.read_source(&self.entries[index].source) {
                Ok(_) => Capability::Available,
                Err(capability) => capability,
            };
            self.entries[index].descriptor.capability = capability;
        }
        self.entries
            .iter()
            .map(|entry| entry.descriptor.clone())
            .collect()
    }

    fn read(&mut self) -> Vec<Reading> {
        self.system.refresh_cpu();
        self.system.refresh_memory();
        self.entries
            .iter()
            .map(|entry| Reading {
                metric_id: entry.descriptor.metric_id.clone(),
                entity_id: entry.descriptor.entity_id.clone(),
                result: match &entry.descriptor.capability {
                    Capability::Unsupported(_) | Capability::PermissionDenied(_) => {
                        Err(entry.descriptor.capability.clone())
                    }
                    _ => self.read_source(&entry.source),
                },
            })
            .collect()
    }
}

fn read_number(path: &Path) -> Result<f64, Capability> {
    let text = fs::read_to_string(path).map_err(|error| {
        let reason = format!("{}: {error}", path.display());
        if error.kind() == std::io::ErrorKind::PermissionDenied {
            Capability::PermissionDenied(reason)
        } else {
            Capability::TemporarilyUnavailable(reason)
        }
    })?;
    text.trim()
        .parse()
        .map_err(|error| Capability::TemporarilyUnavailable(format!("{}: {error}", path.display())))
}

/// Shared by every NVML query and available to adapter-level test doubles.
/// Successful values, including any native timing carried by T, pass unchanged.
#[cfg(feature = "nvidia")]
pub fn translate_nvml_result<T>(result: Result<T, NvmlError>) -> Result<T, Capability> {
    result.map_err(|error| match error {
        NvmlError::NotSupported => Capability::Unsupported(error.to_string()),
        NvmlError::NoPermission => Capability::PermissionDenied(error.to_string()),
        _ => Capability::TemporarilyUnavailable(error.to_string()),
    })
}

#[cfg(all(test, feature = "nvidia"))]
mod tests {
    use super::*;
    use crate::{Interval, Timestamp};

    #[test]
    fn nvml_errors_are_capabilities_and_native_timing_is_unchanged() {
        assert!(matches!(
            translate_nvml_result::<SourceValue>(Err(NvmlError::NotSupported)),
            Err(Capability::Unsupported(_))
        ));
        assert!(matches!(
            translate_nvml_result::<SourceValue>(Err(NvmlError::NoPermission)),
            Err(Capability::PermissionDenied(_))
        ));
        let native = SourceValue {
            value: 0.0,
            timestamp: Some(Timestamp {
                ns: 750,
                clock_domain: "nvml_native".into(),
            }),
            interval: Some(Interval {
                start_ns: 250,
                end_ns: 750,
                clock_domain: "nvml_native".into(),
            }),
        };
        assert_eq!(translate_nvml_result(Ok(native.clone())), Ok(native));
    }
}
