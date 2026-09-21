use crate::collection::Provider;
use crate::model::{
    Capability, CapabilityStatus, MetricDescriptor, MetricKind, ObservationWindow, Reading,
    TemporalSemantics, Unavailable, Unit, ValueKind,
};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use sysinfo::{CpuExt, System, SystemExt};

struct Sensor {
    descriptor: MetricDescriptor,
    path: PathBuf,
}

pub struct SystemProvider {
    system: System,
    descriptors: Vec<MetricDescriptor>,
    sensors: Vec<Sensor>,
    cpu_window: Option<ObservationWindow>,
}

fn descriptor(
    metric_id: &str,
    entity_id: String,
    display_name: String,
    kind: MetricKind,
    unit: Unit,
    source_semantics: &str,
) -> MetricDescriptor {
    MetricDescriptor {
        metric_id: metric_id.into(),
        entity_id,
        display_name,
        kind,
        unit,
        value_kind: ValueKind::Gauge,
        temporal_semantics: TemporalSemantics::PointSample,
        provider: "sysfs".into(),
        source_semantics: source_semantics.into(),
        semantics_version: 1,
    }
}

fn io_unavailable(error: io::Error) -> Unavailable {
    Unavailable {
        status: if error.kind() == io::ErrorKind::PermissionDenied {
            CapabilityStatus::PermissionDenied
        } else {
            CapabilityStatus::TemporarilyUnavailable
        },
        reason: error.to_string(),
    }
}

fn read_number(path: &Path) -> Result<f64, Unavailable> {
    fs::read_to_string(path)
        .map_err(io_unavailable)?
        .trim()
        .parse::<f64>()
        .map_err(|error| Unavailable::temporary(error.to_string()))
}

impl Default for SystemProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemProvider {
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        let mut cpu = descriptor(
            "cpu.utilization_pct",
            "system:cpu".into(),
            "CPU %".into(),
            MetricKind::CpuUtilization,
            Unit::Percent,
            "arithmetic mean of sysinfo logical CPU usage since the previous refresh",
        );
        cpu.provider = "sysinfo".into();
        cpu.temporal_semantics = TemporalSemantics::IntervalAverage;
        let mut ram = descriptor(
            "memory.ram_occupancy_pct",
            "system:ram".into(),
            "RAM %".into(),
            MetricKind::RamOccupancy,
            Unit::Percent,
            "sysinfo used memory divided by total memory, multiplied by 100",
        );
        ram.provider = "sysinfo".into();
        let mut sensors = discover_temperatures();
        sensors.extend(discover_frequencies());
        let mut descriptors = vec![cpu, ram];
        descriptors.extend(sensors.iter().map(|sensor| sensor.descriptor.clone()));
        Self {
            system,
            descriptors,
            sensors,
            cpu_window: None,
        }
    }
}

impl Provider for SystemProvider {
    fn discover(&mut self) -> Vec<Capability> {
        self.descriptors
            .clone()
            .into_iter()
            .map(|descriptor| {
                let result = self.read(&descriptor);
                Capability::from_reading(descriptor, &result)
            })
            .collect()
    }

    fn prepare(&mut self, observation_ns: u64, previous_ns: Option<u64>, clock_domain: &str) {
        self.system.refresh_cpu();
        self.system.refresh_memory();
        self.cpu_window = previous_ns.map(|start_ns| ObservationWindow {
            start_ns,
            end_ns: observation_ns,
            clock_domain: clock_domain.into(),
        });
    }

    fn read(&mut self, metric: &MetricDescriptor) -> Result<Reading, Unavailable> {
        match &metric.kind {
            MetricKind::CpuUtilization => {
                let cpus = self.system.cpus();
                let value =
                    cpus.iter().map(|cpu| cpu.cpu_usage()).sum::<f32>() / cpus.len().max(1) as f32;
                let mut reading = Reading::gauge(value as f64);
                reading.window = self.cpu_window.clone();
                Ok(reading)
            }
            MetricKind::RamOccupancy => {
                let total = self.system.total_memory();
                if total == 0 {
                    return Err(Unavailable::temporary("total RAM is unavailable"));
                }
                Ok(Reading::gauge(
                    self.system.used_memory() as f64 / total as f64 * 100.0,
                ))
            }
            MetricKind::Temperature { .. } | MetricKind::CpuFrequency { .. } => {
                let sensor = self
                    .sensors
                    .iter()
                    .find(|sensor| sensor.descriptor.same_series(metric))
                    .ok_or_else(|| Unavailable::temporary("sensor no longer exists"))?;
                let raw = read_number(&sensor.path)?;
                let value = match &metric.kind {
                    MetricKind::CpuFrequency { .. } => raw * 1_000.0,
                    // Preserve the baseline hwmon conversion convention.
                    _ if raw > 1_000.0 => raw / 1_000.0,
                    _ => raw,
                };
                Ok(Reading::gauge(value))
            }
            _ => Err(Unavailable {
                status: CapabilityStatus::Unsupported,
                reason: "metric is not provided by the system provider".into(),
            }),
        }
    }
}

fn discover_temperatures() -> Vec<Sensor> {
    let mut sensors = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/class/hwmon") {
        for entry in entries.flatten() {
            let base = entry.path();
            let name = fs::read_to_string(base.join("name"))
                .unwrap_or_default()
                .trim()
                .to_owned();
            if let Ok(files) = fs::read_dir(&base) {
                for file in files.flatten() {
                    let path = file.path();
                    let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                    if !filename.starts_with("temp") || !filename.ends_with("_input") {
                        continue;
                    }
                    let mut label = name.clone();
                    let label_path = base.join(filename.replace("_input", "_label"));
                    if let Ok(text) = fs::read_to_string(label_path) {
                        if !text.trim().is_empty() {
                            label = text.trim().to_owned();
                        }
                    }
                    let identity_path = fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
                    sensors.push(Sensor {
                        descriptor: descriptor(
                            "sensor.temperature_c",
                            format!("hwmon:{}", identity_path.display()),
                            label.clone(),
                            MetricKind::Temperature {
                                sensor_name: name.clone(),
                                sensor_label: label,
                            },
                            Unit::Celsius,
                            "hwmon temperature input using the baseline Celsius conversion",
                        ),
                        path,
                    });
                }
            }
        }
    }
    sensors
}

fn discover_frequencies() -> Vec<Sensor> {
    let mut sensors = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/devices/system/cpu") {
        for entry in entries.flatten() {
            let base = entry.path();
            let name = base.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if !name.starts_with("cpu") {
                continue;
            }
            let core = match name.trim_start_matches("cpu").parse::<usize>() {
                Ok(core) => core,
                Err(_) => continue,
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
            let source = format!("{} in kHz, converted to Hz", path.display());
            sensors.push(Sensor {
                descriptor: descriptor(
                    "cpu.frequency_hz",
                    format!("cpu:{core}"),
                    format!("CPU Core {core}"),
                    MetricKind::CpuFrequency { core },
                    Unit::Hertz,
                    &source,
                ),
                path,
            });
        }
    }
    sensors.sort_by_key(|sensor| match sensor.descriptor.kind {
        MetricKind::CpuFrequency { core } => core,
        _ => 0,
    });
    sensors
}
