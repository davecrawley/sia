use crate::collector::{Provider, ProviderBatch, Reading};
use crate::model::{
    CapabilityStatus, EntityKind, MetricDescriptor, MetricValue, SampleStatus, TemporalSemantics,
    ValueKind,
};
use std::fs;
use std::path::{Path, PathBuf};
use sysinfo::{CpuExt, System, SystemExt};

#[derive(Clone, Debug, PartialEq)]
pub struct NamedMeasurement {
    pub entity_id: String,
    pub display_name: String,
    pub value: Result<f64, String>,
}

pub trait SystemBackend {
    fn cpu_percent(&mut self) -> Result<f64, String>;
    fn ram_percent(&mut self) -> Result<f64, String>;
    fn temperatures_c(&mut self) -> Vec<NamedMeasurement>;
    fn frequencies_khz(&mut self) -> Vec<NamedMeasurement>;
}

pub struct SystemProvider<B: SystemBackend> {
    backend: B,
}

impl<B: SystemBackend> SystemProvider<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

fn descriptor(
    metric_id: &str,
    display_name: &str,
    entity_kind: EntityKind,
    unit: &str,
    provider: &str,
    definition: &str,
    comparability_group: &str,
) -> MetricDescriptor {
    MetricDescriptor {
        metric_id: metric_id.into(),
        display_name: display_name.into(),
        entity_kind,
        unit: unit.into(),
        value_kind: ValueKind::Gauge,
        temporal_semantics: TemporalSemantics::PointSample,
        provider: provider.into(),
        capability_status: CapabilityStatus::Available,
        source_resolution_ns: None,
        source_definition: definition.into(),
        comparability_group: Some(comparability_group.into()),
        semantics_version: 1,
    }
}

fn reading(metric: &str, measurement: NamedMeasurement) -> Reading {
    match measurement.value {
        Ok(value) => Reading::available(metric, measurement.entity_id, MetricValue::F64(value)),
        Err(reason) => Reading {
            metric_id: metric.into(),
            entity_id: measurement.entity_id,
            value: None,
            status: SampleStatus::Error(reason),
            source_timestamp_ns: None,
            window_start_ns: None,
        },
    }
}

impl<B: SystemBackend> Provider for SystemProvider<B> {
    fn collect(&mut self) -> ProviderBatch {
        let mut descriptors = vec![
            descriptor(
                "system.cpu.utilization",
                "CPU utilization",
                EntityKind::Cpu,
                "%",
                "sysinfo",
                "Aggregate CPU busy percentage reported by sysinfo.",
                "system_cpu_busy_percent",
            ),
            descriptor(
                "system.ram.utilization",
                "RAM utilization",
                EntityKind::System,
                "%",
                "sysinfo",
                "Used physical memory divided by total physical memory.",
                "physical_memory_occupancy_percent",
            ),
        ];
        let mut readings = Vec::new();
        let cpu = NamedMeasurement {
            entity_id: "cpu:all".into(),
            display_name: "CPU".into(),
            value: self.backend.cpu_percent(),
        };
        readings.push(reading("system.cpu.utilization", cpu));
        let ram = NamedMeasurement {
            entity_id: "system:memory".into(),
            display_name: "RAM".into(),
            value: self.backend.ram_percent(),
        };
        readings.push(reading("system.ram.utilization", ram));

        let temperatures = self.backend.temperatures_c();
        if !temperatures.is_empty() {
            descriptors.push(descriptor(
                "system.temperature",
                "Temperature",
                EntityKind::Thermal,
                "Cel",
                "sysfs",
                "Point temperature reported by a discovered Linux hwmon sensor.",
                "temperature_celsius",
            ));
            readings.extend(
                temperatures
                    .into_iter()
                    .map(|value| reading("system.temperature", value)),
            );
        }

        let frequencies = self.backend.frequencies_khz();
        if !frequencies.is_empty() {
            descriptors.push(descriptor(
                "cpu.frequency",
                "CPU frequency",
                EntityKind::Cpu,
                "kHz",
                "sysfs",
                "Current logical CPU frequency reported by cpufreq.",
                "cpu_frequency_khz",
            ));
            readings.extend(
                frequencies
                    .into_iter()
                    .map(|value| reading("cpu.frequency", value)),
            );
        }

        ProviderBatch {
            descriptors,
            readings,
        }
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
        Self {
            system: System::new_all(),
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

fn read_number(path: &Path) -> Result<f64, String> {
    fs::read_to_string(path)
        .map_err(|error| error.to_string())?
        .trim()
        .parse::<f64>()
        .map_err(|error| error.to_string())
}

impl SystemBackend for LinuxSystemBackend {
    fn cpu_percent(&mut self) -> Result<f64, String> {
        self.system.refresh_cpu();
        Ok(self.system.global_cpu_info().cpu_usage() as f64)
    }

    fn ram_percent(&mut self) -> Result<f64, String> {
        self.system.refresh_memory();
        let total = self.system.total_memory();
        if total == 0 {
            return Err("total memory is unavailable".into());
        }
        Ok(self.system.used_memory() as f64 * 100.0 / total as f64)
    }

    fn temperatures_c(&mut self) -> Vec<NamedMeasurement> {
        self.temperatures
            .iter()
            .map(|sensor| {
                let value = read_number(&sensor.path).map(|value| {
                    if value.abs() > 1000.0 {
                        value / 1000.0
                    } else {
                        value
                    }
                });
                NamedMeasurement {
                    entity_id: sensor.entity_id.clone(),
                    display_name: sensor.display_name.clone(),
                    value,
                }
            })
            .collect()
    }

    fn frequencies_khz(&mut self) -> Vec<NamedMeasurement> {
        self.frequencies
            .iter()
            .map(|sensor| NamedMeasurement {
                entity_id: sensor.entity_id.clone(),
                display_name: sensor.display_name.clone(),
                value: read_number(&sensor.path),
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
            .unwrap_or_else(|_| "hwmon".into())
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
                display_name: label,
                path,
            });
        }
    }
    sensors.sort_by(|a, b| a.entity_id.cmp(&b.entity_id));
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
    sensors.sort_by(|a, b| a.entity_id.cmp(&b.entity_id));
    sensors
}
