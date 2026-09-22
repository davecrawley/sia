use super::{descriptor, emit};
use crate::collection::Provider;
use crate::model::{
    EntityKind, MetricDescriptor, MetricValue, ObservationWindow, ProviderOutput,
    TemporalSemantics, Timestamp, Unavailable, UnavailableKind, Unit,
};
use std::fs;
use std::path::{Path, PathBuf};
use sysinfo::{CpuExt, System, SystemExt};

struct Sensor {
    descriptor: MetricDescriptor,
    path: PathBuf,
    scale: f64,
}

/// The authoritative sysinfo and Linux sensor path. Discovery is performed once;
/// individual read failures remain samples and can recover on the next poll.
pub struct NativeProvider {
    system: System,
    sensors: Vec<Sensor>,
}

impl Default for NativeProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeProvider {
    pub fn new() -> Self {
        let mut sensors = discover_temperatures();
        sensors.extend(discover_frequencies());
        Self {
            system: System::new(),
            sensors,
        }
    }
}

impl Provider for NativeProvider {
    fn collect(&mut self, now: &Timestamp, previous: Option<&Timestamp>) -> ProviderOutput {
        if previous.is_none() {
            // sysinfo retains CPU counter baselines internally. Do not let one
            // survive a collector clock discontinuity or failed collection.
            self.system = System::new();
        }
        self.system.refresh_cpu();
        self.system.refresh_memory();
        let mut output = ProviderOutput::default();
        let mut cpu = descriptor(
            "cpu.utilization",
            "system:cpu",
            EntityKind::System,
            "CPU %",
            Unit::Percent,
            "sysinfo",
            "Mean logical CPU busy percentage from successive sysinfo CPU counter refreshes",
        );
        cpu.temporal_semantics = TemporalSemantics::IntervalAverage;
        let cpu_value = if previous.is_none() {
            Err(Unavailable::temporary(
                "CPU counter baseline is being established",
            ))
        } else if self.system.cpus().is_empty() {
            Err(Unavailable::temporary("no CPU counters returned"))
        } else {
            let sum: f64 = self
                .system
                .cpus()
                .iter()
                .map(|cpu| f64::from(cpu.cpu_usage()))
                .sum();
            Ok(MetricValue::Float(sum / self.system.cpus().len() as f64))
        };
        let window = previous.map(|previous| ObservationWindow {
            start: previous.clone(),
            end: now.clone(),
        });
        emit(&mut output, cpu, now, window, cpu_value);

        let ram = descriptor(
            "memory.occupancy",
            "system:memory",
            EntityKind::System,
            "RAM %",
            Unit::Percent,
            "sysinfo",
            "Used system memory divided by total system memory, multiplied by 100",
        );
        let total = self.system.total_memory();
        let ram_value = if total == 0 {
            Err(Unavailable::temporary("total system memory is unavailable"))
        } else {
            Ok(MetricValue::Float(
                self.system.used_memory() as f64 / total as f64 * 100.0,
            ))
        };
        emit(&mut output, ram, now, None, ram_value);
        for sensor in &self.sensors {
            let value =
                read_number(&sensor.path).map(|value| MetricValue::Float(value * sensor.scale));
            emit(&mut output, sensor.descriptor.clone(), now, None, value);
        }
        output
    }
}

fn io_unavailable(error: std::io::Error) -> Unavailable {
    let kind = if error.kind() == std::io::ErrorKind::PermissionDenied {
        UnavailableKind::PermissionDenied
    } else {
        UnavailableKind::TemporarilyUnavailable
    };
    Unavailable::new(kind, error.to_string())
}

fn read_number(path: &Path) -> Result<f64, Unavailable> {
    let text = fs::read_to_string(path).map_err(io_unavailable)?;
    let value = text
        .trim()
        .parse::<f64>()
        .map_err(|error| Unavailable::temporary(error.to_string()))?;
    if !value.is_finite() {
        return Err(Unavailable::temporary("sensor returned a non-finite value"));
    }
    Ok(value)
}

fn discover_temperatures() -> Vec<Sensor> {
    let mut sensors = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/class/hwmon") else {
        return sensors;
    };
    for entry in entries.flatten() {
        let base = entry.path();
        let name = fs::read_to_string(base.join("name"))
            .unwrap_or_default()
            .trim()
            .to_owned();
        let Ok(files) = fs::read_dir(&base) else {
            continue;
        };
        for file in files.flatten() {
            let path = file.path();
            let filename = file.file_name().to_string_lossy().into_owned();
            if !filename.starts_with("temp") || !filename.ends_with("_input") {
                continue;
            }
            let label = fs::read_to_string(base.join(filename.replace("_input", "_label")))
                .ok()
                .map(|label| label.trim().to_owned())
                .filter(|label| !label.is_empty())
                .unwrap_or_else(|| name.clone());
            let canonical = fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
            let entity_id = format!("sensor:{}", canonical.display());
            let mut metric = descriptor(
                "sensor.temperature",
                &entity_id,
                EntityKind::Sensor,
                &label,
                Unit::Celsius,
                "sysfs",
                "Linux hwmon temperature input in millidegrees Celsius, converted to Celsius",
            );
            metric.entity_display_name = name.clone();
            sensors.push(Sensor {
                descriptor: metric,
                path,
                scale: 0.001,
            });
        }
    }
    sensors.sort_by(|a, b| a.descriptor.entity_id.cmp(&b.descriptor.entity_id));
    sensors
}

fn discover_frequencies() -> Vec<Sensor> {
    let mut sensors = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/devices/system/cpu") else {
        return sensors;
    };
    let mut cores = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(core) = name
            .strip_prefix("cpu")
            .and_then(|suffix| suffix.parse::<usize>().ok())
        else {
            continue;
        };
        cores.push((core, entry.path()));
    }
    cores.sort_by_key(|(core, _)| *core);
    for (core, base) in cores {
        let scaling = base.join("cpufreq/scaling_cur_freq");
        let hardware = base.join("cpufreq/cpuinfo_cur_freq");
        let (path, semantics) = if scaling.exists() {
            (
                scaling,
                "Linux cpufreq scaling_cur_freq in kHz, converted to Hz",
            )
        } else if hardware.exists() {
            (
                hardware,
                "Linux cpufreq cpuinfo_cur_freq in kHz, converted to Hz",
            )
        } else {
            continue;
        };
        sensors.push(Sensor {
            descriptor: descriptor(
                "cpu.frequency",
                &format!("cpu:logical:{core}"),
                EntityKind::Cpu,
                &format!("CPU Core {core}"),
                Unit::Hertz,
                "sysfs",
                semantics,
            ),
            path,
            scale: 1000.0,
        });
    }
    sensors
}
