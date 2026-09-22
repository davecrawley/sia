use super::{descriptor, emit};
use crate::collection::Provider;
use crate::model::{
    EntityKind, MetricDescriptor, MetricValue, ObservationWindow, ProviderOutput,
    TemporalSemantics, Timestamp, Unavailable, UnavailableKind, Unit,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

struct Sensor {
    descriptor: MetricDescriptor,
    path: PathBuf,
    scale: f64,
}

/// Linux counters are read directly so each baseline and its actual observation
/// time belong to this provider, independently of GUI cadence or refresh caches.
pub struct NativeProvider {
    sensors: Vec<Sensor>,
    cpu_baselines: BTreeMap<String, ([u64; 8], Timestamp)>,
    cpu_names: BTreeSet<String>,
    pressure_baselines: BTreeMap<String, (u64, Timestamp)>,
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
            sensors,
            cpu_baselines: BTreeMap::new(),
            cpu_names: BTreeSet::from(["cpu".into()]),
            pressure_baselines: BTreeMap::new(),
        }
    }

    fn cpu(&mut self, now: &Timestamp, output: &mut ProviderOutput) {
        let counters = fs::read_to_string("/proc/stat")
            .map_err(io_unavailable)
            .map(|text| {
                let mut rows = BTreeMap::new();
                for line in text.lines() {
                    let mut fields = line.split_whitespace();
                    let Some(name) = fields.next() else { continue };
                    if name != "cpu"
                        && !name.strip_prefix("cpu").is_some_and(|id| {
                            !id.is_empty() && id.bytes().all(|byte| byte.is_ascii_digit())
                        })
                    {
                        continue;
                    }
                    let values: Result<Vec<u64>, _> =
                        fields.take(8).map(str::parse::<u64>).collect();
                    let row = values
                        .map_err(|_| Unavailable::temporary("malformed CPU counters"))
                        .and_then(|values| {
                            if values.len() < 4 {
                                return Err(Unavailable::temporary("incomplete CPU counters"));
                            }
                            let mut row = [0; 8];
                            row[..values.len()].copy_from_slice(&values);
                            Ok(row)
                        });
                    rows.insert(name.to_owned(), row);
                }
                rows
            });
        if let Ok(rows) = &counters {
            self.cpu_names.extend(rows.keys().cloned());
        }
        for name in &self.cpu_names {
            let current = counters.as_ref().map_err(Clone::clone).and_then(|rows| {
                rows.get(name)
                    .cloned()
                    .unwrap_or_else(|| Err(Unavailable::temporary("CPU counters absent")))
            });
            let old = self.cpu_baselines.remove(name);
            let mut window = None;
            let value = current.and_then(|counts| {
                self.cpu_baselines
                    .insert(name.clone(), (counts, now.clone()));
                let Some((before, start)) = old else {
                    return Err(Unavailable::temporary(
                        "CPU counter baseline is being established",
                    ));
                };
                let elapsed = now
                    .elapsed_since(&start)
                    .filter(|elapsed| !elapsed.is_zero());
                if elapsed.is_none() {
                    return Err(Unavailable::temporary(
                        "CPU observation interval is invalid",
                    ));
                }
                let deltas: Option<Vec<u64>> = counts
                    .iter()
                    .zip(before)
                    .map(|(current, previous)| current.checked_sub(previous))
                    .collect();
                let deltas = deltas.ok_or_else(|| {
                    Unavailable::temporary("CPU counters reset; new baseline established")
                })?;
                let total = deltas
                    .iter()
                    .try_fold(0u64, |sum, value| sum.checked_add(*value))
                    .filter(|total| *total > 0)
                    .ok_or_else(|| {
                        Unavailable::temporary("CPU counter interval has no valid ticks")
                    })?;
                let idle = deltas[3]
                    .checked_add(deltas[4])
                    .ok_or_else(|| Unavailable::temporary("CPU idle counter overflow"))?;
                window = Some(ObservationWindow {
                    start,
                    end: now.clone(),
                });
                Ok(MetricValue::Float(
                    (total - idle) as f64 / total as f64 * 100.0,
                ))
            });
            let aggregate = name == "cpu";
            let entity = if aggregate {
                "system:cpu".to_owned()
            } else {
                format!("cpu:logical:{}", &name[3..])
            };
            let label = if aggregate {
                "CPU %".to_owned()
            } else {
                format!("CPU {} %", &name[3..])
            };
            let mut metric = descriptor(
                "cpu.utilization", &entity,
                if aggregate { EntityKind::System } else { EntityKind::Cpu },
                &label, Unit::Percent, "procfs",
                "100 * delta(user+nice+system+irq+softirq+steal) / delta(user+nice+system+idle+iowait+irq+softirq+steal); guest counters are not counted twice",
            );
            metric.temporal_semantics = TemporalSemantics::IntervalAverage;
            metric.comparability_group = Some("linux_cpu_nonidle_time".into());
            emit(output, metric, now, window, value);
        }
    }

    fn memory(&self, now: &Timestamp, output: &mut ProviderOutput) {
        let memory = fs::read_to_string("/proc/meminfo")
            .map_err(io_unavailable)
            .and_then(|text| {
                let read = |key: &str| -> Result<u64, Unavailable> {
                    let line = text
                        .lines()
                        .find_map(|line| line.strip_prefix(key))
                        .ok_or_else(|| Unavailable::temporary(format!("{key} absent")))?;
                    let mut fields = line.split_whitespace();
                    let value = fields
                        .next()
                        .and_then(|value| value.parse::<u64>().ok())
                        .ok_or_else(|| Unavailable::temporary(format!("malformed {key}")))?;
                    if fields.next() != Some("kB") {
                        return Err(Unavailable::temporary(format!("unexpected unit for {key}")));
                    }
                    value
                        .checked_mul(1024)
                        .ok_or_else(|| Unavailable::temporary("memory capacity overflow"))
                };
                let total = read("MemTotal:")?;
                let available = read("MemAvailable:")?;
                let used = total
                    .checked_sub(available)
                    .filter(|_| total > 0)
                    .ok_or_else(|| Unavailable::temporary("inconsistent memory capacities"))?;
                Ok((used, available, total))
            });
        for (id, label, semantics, index) in [
            (
                "memory.used",
                "RAM used",
                "MemTotal minus MemAvailable, in bytes",
                0,
            ),
            (
                "memory.available",
                "RAM available",
                "Linux MemAvailable estimate, in bytes",
                1,
            ),
            ("memory.total", "RAM total", "Linux MemTotal, in bytes", 2),
        ] {
            let mut metric = descriptor(
                id,
                "system:memory",
                EntityKind::System,
                label,
                Unit::Bytes,
                "procfs",
                semantics,
            );
            metric.comparability_group = Some("memory_capacity_bytes".into());
            let value = memory
                .as_ref()
                .map_err(Clone::clone)
                .map(|&(used, available, total)| {
                    MetricValue::Unsigned([used, available, total][index])
                });
            emit(output, metric, now, None, value);
        }
        let mut metric = descriptor(
            "memory.occupancy",
            "system:memory",
            EntityKind::System,
            "RAM %",
            Unit::Percent,
            "procfs",
            "100 * (MemTotal - MemAvailable) / MemTotal; capacity occupancy, not memory activity",
        );
        metric.comparability_group = Some("memory_capacity_ratio".into());
        let value =
            memory.map(|(used, _, total)| MetricValue::Float(used as f64 / total as f64 * 100.0));
        emit(output, metric, now, None, value);
    }

    fn pressure(&mut self, now: &Timestamp, output: &mut ProviderOutput) {
        for resource in ["cpu", "memory", "io"] {
            let text =
                fs::read_to_string(format!("/proc/pressure/{resource}")).map_err(io_unavailable);
            for scope in ["some", "full"] {
                let id = format!("pressure.{resource}.{scope}");
                let total = text.as_ref().map_err(Clone::clone).and_then(|text| {
                    let mut matching = text
                        .lines()
                        .filter(|line| line.split_whitespace().next() == Some(scope));
                    let row = matching.next().ok_or_else(|| {
                        if text.lines().any(|line| {
                            matches!(line.split_whitespace().next(), Some("some" | "full"))
                        }) {
                            Unavailable::new(
                                UnavailableKind::Unsupported,
                                format!("PSI {scope} not exposed"),
                            )
                        } else {
                            Unavailable::temporary("malformed PSI file")
                        }
                    })?;
                    if matching.next().is_some() {
                        return Err(Unavailable::temporary("duplicate PSI row"));
                    }
                    let totals: Vec<_> = row
                        .split_whitespace()
                        .filter_map(|field| field.strip_prefix("total="))
                        .collect();
                    if totals.len() != 1 {
                        return Err(Unavailable::temporary("missing or duplicate PSI total"));
                    }
                    totals[0]
                        .parse::<u64>()
                        .map_err(|_| Unavailable::temporary("malformed PSI total"))
                });
                let old = self.pressure_baselines.remove(&id);
                let mut window = None;
                let value = total.and_then(|total| {
                    self.pressure_baselines
                        .insert(id.clone(), (total, now.clone()));
                    let Some((before, start)) = old else {
                        return Err(Unavailable::temporary(
                            "PSI counter baseline is being established",
                        ));
                    };
                    let elapsed = now
                        .elapsed_since(&start)
                        .filter(|elapsed| !elapsed.is_zero())
                        .ok_or_else(|| {
                            Unavailable::temporary("invalid PSI observation interval")
                        })?;
                    let delta = total.checked_sub(before).ok_or_else(|| {
                        Unavailable::temporary("PSI counter reset; new baseline established")
                    })?;
                    let share = delta as f64 / (elapsed.as_secs_f64() * 1_000_000.0) * 100.0;
                    if !share.is_finite() || share > 100.0 {
                        return Err(Unavailable::temporary(
                            "PSI delta exceeds elapsed interval; new baseline established",
                        ));
                    }
                    window = Some(ObservationWindow {
                        start,
                        end: now.clone(),
                    });
                    Ok(MetricValue::Float(share))
                });
                let qualification = match (resource, scope) {
                    ("cpu", "some") => "Principal system CPU-pressure signal: at least one runnable task stalled for CPU",
                    ("cpu", "full") => "Kernel-specific CPU full-pressure signal; system-level meaning depends on kernel version and may be undefined or reported as zero",
                    (_, "some") => "At least one task stalled on this resource",
                    _ => "All non-idle tasks stalled on this resource",
                };
                let mut metric = descriptor(&id, &format!("system:pressure:{resource}"), EntityKind::System,
                    &format!("{} PSI {scope}", resource.to_uppercase()), Unit::Percent, "procfs",
                    &format!("{qualification}; 100 * delta(cumulative total microseconds) / elapsed observation-window microseconds; avg10/60/300 are not used"));
                metric.temporal_semantics = TemporalSemantics::IntervalAverage;
                metric.comparability_group =
                    Some(format!("linux_psi_{resource}_{scope}_stall_time"));
                emit(output, metric, now, window, value);
            }
        }
    }
}

impl Provider for NativeProvider {
    fn collect(&mut self, now: &Timestamp, previous: Option<&Timestamp>) -> ProviderOutput {
        if previous.is_none() {
            self.cpu_baselines.clear();
            self.pressure_baselines.clear();
        }
        let mut output = ProviderOutput::default();
        self.cpu(now, &mut output);
        self.memory(now, &mut output);
        self.pressure(now, &mut output);
        for sensor in &self.sensors {
            let value =
                read_number(&sensor.path).map(|value| MetricValue::Float(value * sensor.scale));
            emit(&mut output, sensor.descriptor.clone(), now, None, value);
        }
        output
    }
}

fn io_unavailable(error: std::io::Error) -> Unavailable {
    let kind = match error.kind() {
        std::io::ErrorKind::PermissionDenied => UnavailableKind::PermissionDenied,
        std::io::ErrorKind::NotFound => UnavailableKind::Unsupported,
        _ => UnavailableKind::TemporarilyUnavailable,
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
