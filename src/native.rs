//! The executable's only hardware access path. The view consumes model data.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use sysinfo::{CpuExt, System, SystemExt};

use crate::clock::LinuxClock;
use crate::collection::{CollectionContext, Collector, Provider};
use crate::model::{
    Capability, MetricDescriptor, ObservationWindow, ProviderBatch, Reading, Sample,
    TemporalSemantics, Unavailable, Unit, Value, ValueKind,
};

pub fn collector() -> Collector<LinuxClock> {
    Collector::new(
        LinuxClock,
        vec![
            Box::new(SystemProvider::new()),
            Box::new(NvidiaProvider::new(NativeNvidiaInitializer)),
        ],
    )
}

fn temporary(reason: impl Into<String>) -> Unavailable {
    Unavailable::TemporarilyUnavailable(reason.into())
}

fn io_unavailable(error: io::Error) -> Unavailable {
    if error.kind() == io::ErrorKind::PermissionDenied {
        Unavailable::PermissionDenied(error.to_string())
    } else {
        temporary(error.to_string())
    }
}

fn descriptor(
    metric: &str,
    entity: &str,
    entity_name: &str,
    name: &str,
    unit: Unit,
    provider: &str,
    semantics: &str,
) -> MetricDescriptor {
    MetricDescriptor {
        metric_id: metric.to_string(),
        entity_id: entity.to_string(),
        entity_name: entity_name.to_string(),
        display_name: name.to_string(),
        unit,
        value_kind: ValueKind::Gauge,
        temporal_semantics: TemporalSemantics::Instantaneous,
        provider: provider.to_string(),
        source_semantics: semantics.to_string(),
        semantics_version: 1,
        capability: Capability::Available,
    }
}

fn emit(
    batch: &mut ProviderBatch,
    mut descriptor: MetricDescriptor,
    result: Result<f64, Unavailable>,
    context: &CollectionContext,
) {
    let reading = match result {
        Ok(value) if value.is_finite() => Reading::Available(Value::Float(value)),
        Ok(_) => Reading::Unavailable(temporary("source returned a non-finite value")),
        Err(reason) => Reading::Unavailable(reason),
    };
    descriptor.capability = reading.capability();
    let observation_window = match descriptor.temporal_semantics {
        TemporalSemantics::Instantaneous => Some(ObservationWindow {
            start: context.observed_at.clone(),
            end: context.observed_at.clone(),
        }),
        TemporalSemantics::Interval => {
            context.previous.as_ref().map(|previous| ObservationWindow {
                start: previous.clone(),
                end: context.observed_at.clone(),
            })
        }
        TemporalSemantics::VendorSampled => None,
    };
    batch.samples.push(Sample {
        metric_id: descriptor.metric_id.clone(),
        entity_id: descriptor.entity_id.clone(),
        observed_at: context.observed_at.clone(),
        source_timestamp: None,
        observation_window,
        reading,
    });
    batch.descriptors.push(descriptor);
}

struct FileSensor {
    descriptor: MetricDescriptor,
    path: PathBuf,
    scale: f64,
}

pub struct SystemProvider {
    system: System,
    sensors: Vec<FileSensor>,
}

impl Default for SystemProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemProvider {
    pub fn new() -> Self {
        let mut sensors = Vec::new();
        if let Ok(entries) = fs::read_dir("/sys/class/hwmon") {
            for entry in entries.flatten() {
                let base = entry.path();
                let name = fs::read_to_string(base.join("name"))
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                let device = fs::canonicalize(base.join("device"))
                    .or_else(|_| fs::canonicalize(&base))
                    .unwrap_or_else(|_| base.clone());
                if let Ok(files) = fs::read_dir(&base) {
                    for file in files.flatten() {
                        let path = file.path();
                        let filename = file.file_name().to_string_lossy().into_owned();
                        if !filename.starts_with("temp") || !filename.ends_with("_input") {
                            continue;
                        }
                        let label_path = base.join(filename.replace("_input", "_label"));
                        let label = fs::read_to_string(label_path)
                            .ok()
                            .map(|value| value.trim().to_string())
                            .filter(|value| !value.is_empty())
                            .unwrap_or_else(|| name.clone());
                        let entity = format!("hwmon:{}:{}:{}", device.display(), name, filename);
                        sensors.push(FileSensor {
                            descriptor: descriptor(
                                "temperature",
                                &entity,
                                &name,
                                &label,
                                Unit::Celsius,
                                "linux_hwmon",
                                "hwmon temperature input in millidegrees Celsius",
                            ),
                            path,
                            scale: 0.001,
                        });
                    }
                }
            }
        }
        let mut frequencies = Vec::new();
        if let Ok(entries) = fs::read_dir("/sys/devices/system/cpu") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                let Some(core) = name
                    .strip_prefix("cpu")
                    .and_then(|number| number.parse::<usize>().ok())
                else {
                    continue;
                };
                let base = entry.path().join("cpufreq");
                let scaling = base.join("scaling_cur_freq");
                let hardware = base.join("cpuinfo_cur_freq");
                let path = if scaling.exists() {
                    scaling
                } else if hardware.exists() {
                    hardware
                } else {
                    continue;
                };
                let semantics = format!("{} in kHz, converted to Hz", path.display());
                frequencies.push((
                    core,
                    FileSensor {
                        descriptor: descriptor(
                            "cpu.frequency",
                            &format!("cpu:{core}"),
                            "cpu",
                            &format!("CPU Core {core}"),
                            Unit::Hertz,
                            "linux_cpufreq",
                            &semantics,
                        ),
                        path,
                        scale: 1000.0,
                    },
                ));
            }
        }
        sensors.sort_by(|a, b| a.descriptor.entity_id.cmp(&b.descriptor.entity_id));
        frequencies.sort_by_key(|(core, _)| *core);
        sensors.extend(frequencies.into_iter().map(|(_, sensor)| sensor));
        Self {
            system: System::new(),
            sensors,
        }
    }
}

fn read_number(path: &Path) -> Result<f64, Unavailable> {
    fs::read_to_string(path)
        .map_err(io_unavailable)?
        .trim()
        .parse::<f64>()
        .map_err(|error| temporary(format!("{}: {error}", path.display())))
}

impl Provider for SystemProvider {
    fn collect(&mut self, context: &CollectionContext) -> ProviderBatch {
        self.system.refresh_cpu();
        self.system.refresh_memory();
        let mut batch = ProviderBatch::default();
        let mut cpu = descriptor(
            "cpu.utilization",
            "system",
            "cpu",
            "CPU %",
            Unit::Percent,
            "sysinfo",
            "mean logical-CPU busy percentage between CPU refreshes",
        );
        cpu.temporal_semantics = TemporalSemantics::Interval;
        let usage = if context.previous.is_none() {
            Err(temporary("CPU interval baseline reset"))
        } else if self.system.cpus().is_empty() {
            Err(temporary("no CPU counters available"))
        } else {
            Ok(self
                .system
                .cpus()
                .iter()
                .map(|cpu| cpu.cpu_usage() as f64)
                .sum::<f64>()
                / self.system.cpus().len() as f64)
        };
        emit(&mut batch, cpu, usage, context);
        let total = self.system.total_memory();
        let ram = if total == 0 {
            Err(temporary("total memory unavailable"))
        } else {
            Ok(self.system.used_memory() as f64 / total as f64 * 100.0)
        };
        emit(
            &mut batch,
            descriptor(
                "memory.occupancy",
                "system",
                "memory",
                "RAM %",
                Unit::Percent,
                "sysinfo",
                "used physical memory divided by total physical memory",
            ),
            ram,
            context,
        );
        for sensor in &self.sensors {
            emit(
                &mut batch,
                sensor.descriptor.clone(),
                read_number(&sensor.path).map(|value| value * sensor.scale),
                context,
            );
        }
        batch
    }
}

/// A hardware-free seam for initialization failures and later recovery. A
/// successfully initialized source remains owned by the same provider.
pub trait NvidiaInitializer {
    fn initialize(&mut self) -> Result<Box<dyn Provider>, Unavailable>;
}

pub struct NvidiaProvider<I> {
    initializer: I,
    source: Option<Box<dyn Provider>>,
    permanent_failure: Option<Unavailable>,
}

impl<I: NvidiaInitializer> NvidiaProvider<I> {
    pub fn new(initializer: I) -> Self {
        Self {
            initializer,
            source: None,
            permanent_failure: None,
        }
    }
}

const GPU_METRICS: [(&str, &str, Unit, &str); 7] = [
    (
        "gpu.utilization",
        "GPU %",
        Unit::Percent,
        "NVML percent of vendor sample period with one or more kernels executing",
    ),
    (
        "gpu.memory.occupancy",
        "VRAM %",
        Unit::Percent,
        "NVML used device memory divided by total device memory; not memory activity",
    ),
    (
        "gpu.temperature",
        "GPU (Core)",
        Unit::Celsius,
        "NVML GPU temperature in Celsius",
    ),
    (
        "gpu.clock.graphics",
        "GPU Graphics",
        Unit::Hertz,
        "NVML graphics clock in MHz, converted to Hz",
    ),
    (
        "gpu.clock.sm",
        "GPU SM",
        Unit::Hertz,
        "NVML SM clock in MHz, converted to Hz",
    ),
    (
        "gpu.clock.memory",
        "GPU Memory",
        Unit::Hertz,
        "NVML memory clock in MHz, converted to Hz; not effective transfer rate",
    ),
    (
        "gpu.clock.video",
        "GPU Video",
        Unit::Hertz,
        "NVML video clock in MHz, converted to Hz",
    ),
];

fn gpu_descriptor(index: usize, entity: &str) -> MetricDescriptor {
    let (metric, name, unit, semantics) = GPU_METRICS[index];
    let mut output = descriptor(metric, entity, "nvidia", name, unit, "nvml", semantics);
    if index == 0 {
        output.temporal_semantics = TemporalSemantics::VendorSampled;
    }
    output
}

impl<I: NvidiaInitializer> Provider for NvidiaProvider<I> {
    fn collect(&mut self, context: &CollectionContext) -> ProviderBatch {
        if let Some(source) = &mut self.source {
            return source.collect(context);
        }
        let reason = if let Some(reason) = &self.permanent_failure {
            reason.clone()
        } else {
            match self.initializer.initialize() {
                Ok(mut source) => {
                    let batch = source.collect(context);
                    self.source = Some(source);
                    return batch;
                }
                Err(reason) => {
                    if matches!(reason, Unavailable::Unsupported(_)) {
                        self.permanent_failure = Some(reason.clone());
                    }
                    // Permission changes and temporary failures can recover on
                    // the next collection. Neither is cached as unsupported.
                    reason
                }
            }
        };
        let mut batch = ProviderBatch::default();
        for index in 0..GPU_METRICS.len() {
            emit(
                &mut batch,
                gpu_descriptor(index, "provider:nvidia"),
                Err(reason.clone()),
                context,
            );
        }
        batch
    }
}

pub struct NativeNvidiaInitializer;

#[cfg(not(feature = "nvidia"))]
impl NvidiaInitializer for NativeNvidiaInitializer {
    fn initialize(&mut self) -> Result<Box<dyn Provider>, Unavailable> {
        Err(Unavailable::Unsupported(
            "NVIDIA support was disabled at build time".to_string(),
        ))
    }
}

#[cfg(feature = "nvidia")]
mod nvidia {
    use nvml_wrapper::{
        enum_wrappers::device::{Clock, TemperatureSensor},
        error::NvmlError,
        Nvml,
    };

    use super::*;

    fn unavailable(error: NvmlError) -> Unavailable {
        match error {
            NvmlError::NoPermission => Unavailable::PermissionDenied(error.to_string()),
            NvmlError::NotSupported => Unavailable::Unsupported(error.to_string()),
            _ => temporary(error.to_string()),
        }
    }

    struct Source {
        nvml: Nvml,
        uuids: Vec<String>,
    }

    impl NvidiaInitializer for NativeNvidiaInitializer {
        fn initialize(&mut self) -> Result<Box<dyn Provider>, Unavailable> {
            let nvml = Nvml::init().map_err(unavailable)?;
            let count = nvml.device_count().map_err(unavailable)?;
            if count == 0 {
                return Err(temporary("NVML currently reports no devices"));
            }
            let mut uuids = Vec::new();
            for index in 0..count {
                let device = nvml.device_by_index(index).map_err(unavailable)?;
                uuids.push(device.uuid().map_err(unavailable)?);
            }
            Ok(Box::new(Source { nvml, uuids }))
        }
    }

    impl Provider for Source {
        fn collect(&mut self, context: &CollectionContext) -> ProviderBatch {
            let mut batch = ProviderBatch::default();
            for uuid in &self.uuids {
                let entity = format!("gpu:uuid:{uuid}");
                let device = match self.nvml.device_by_uuid(uuid) {
                    Ok(device) => device,
                    Err(error) => {
                        let reason = unavailable(error);
                        for index in 0..GPU_METRICS.len() {
                            emit(
                                &mut batch,
                                gpu_descriptor(index, &entity),
                                Err(reason.clone()),
                                context,
                            );
                        }
                        continue;
                    }
                };
                let usage = device
                    .utilization_rates()
                    .map(|rates| rates.gpu as f64)
                    .map_err(unavailable);
                let memory = device
                    .memory_info()
                    .map_err(unavailable)
                    .and_then(|memory| {
                        if memory.total == 0 {
                            Err(temporary("NVML total memory unavailable"))
                        } else {
                            Ok(memory.used as f64 / memory.total as f64 * 100.0)
                        }
                    });
                let temperature = device
                    .temperature(TemperatureSensor::Gpu)
                    .map(|value| value as f64)
                    .map_err(unavailable);
                for (index, result) in [usage, memory, temperature].into_iter().enumerate() {
                    emit(&mut batch, gpu_descriptor(index, &entity), result, context);
                }
                for (offset, clock) in [Clock::Graphics, Clock::SM, Clock::Memory, Clock::Video]
                    .into_iter()
                    .enumerate()
                {
                    let result = device
                        .clock_info(clock)
                        .map(|mhz| mhz as f64 * 1_000_000.0)
                        .map_err(unavailable);
                    emit(
                        &mut batch,
                        gpu_descriptor(offset + 3, &entity),
                        result,
                        context,
                    );
                }
            }
            batch
        }
    }
}
