use super::{descriptor, emit};
use crate::collection::Provider;
use crate::model::{
    Capability, EntityKind, MetricDescriptor, MetricSample, MetricValue, ProviderOutput,
    TemporalSemantics, Timestamp, Unavailable, UnavailableKind, Unit,
};

/// Hardware-free initialization seam, including recoverable initialization errors.
pub trait NvidiaInitializer {
    fn initialize(&mut self) -> Result<Box<dyn NvidiaSource>, Unavailable>;
}

impl<F> NvidiaInitializer for F
where
    F: FnMut() -> Result<Box<dyn NvidiaSource>, Unavailable>,
{
    fn initialize(&mut self) -> Result<Box<dyn NvidiaSource>, Unavailable> {
        self()
    }
}

/// A successfully initialized source retains durable device identity. Each metric
/// has its own result; a failed call must not discard unrelated readings.
pub trait NvidiaSource {
    fn descriptors(&self) -> Vec<MetricDescriptor>;
    fn read(&mut self, now: &Timestamp) -> Vec<MetricSample>;
}

pub struct NvidiaProvider {
    initializer: Box<dyn NvidiaInitializer>,
    source: Option<Box<dyn NvidiaSource>>,
    initialization_failure: Option<Unavailable>,
}

impl NvidiaProvider {
    pub fn new(initializer: impl NvidiaInitializer + 'static) -> Self {
        Self {
            initializer: Box::new(initializer),
            source: None,
            initialization_failure: None,
        }
    }

    pub fn native() -> Self {
        Self::new(NativeInitializer)
    }
}

impl Provider for NvidiaProvider {
    fn collect(&mut self, now: &Timestamp, _previous: Option<&Timestamp>) -> ProviderOutput {
        let retry = self
            .initialization_failure
            .as_ref()
            .map(|failure| failure.kind == UnavailableKind::TemporarilyUnavailable)
            .unwrap_or(true);
        if self.source.is_none() && retry {
            match self.initializer.initialize() {
                Ok(source) => {
                    self.source = Some(source);
                    self.initialization_failure = None;
                }
                Err(failure) => self.initialization_failure = Some(failure),
            }
        }
        if let Some(source) = &mut self.source {
            let mut descriptors = source.descriptors();
            let samples = source.read(now);
            for descriptor in &mut descriptors {
                if let Some(sample) = samples.iter().find(|sample| {
                    sample.metric_id == descriptor.metric_id
                        && sample.entity_id == descriptor.entity_id
                }) {
                    descriptor.capability = match &sample.value {
                        Ok(_) => Capability::Available,
                        Err(reason) => Capability::from(reason),
                    };
                }
            }
            return ProviderOutput {
                descriptors,
                samples,
            };
        }
        let failure = self
            .initialization_failure
            .clone()
            .unwrap_or_else(|| Unavailable::temporary("NVIDIA initialization pending"));
        let mut output = ProviderOutput::default();
        for descriptor in descriptors("gpu:nvidia:unresolved") {
            emit(&mut output, descriptor, now, None, Err(failure.clone()));
        }
        output
    }
}

fn descriptors(entity_id: &str) -> Vec<MetricDescriptor> {
    let metrics = [
        (
            "gpu.utilization",
            "GPU %",
            Unit::Percent,
            "NVML percentage of vendor sample period during which one or more kernels executed",
        ),
        (
            "gpu.memory.occupancy",
            "VRAM %",
            Unit::Percent,
            "NVML used device memory divided by total device memory, multiplied by 100",
        ),
        (
            "gpu.temperature",
            "GPU (Core)",
            Unit::Celsius,
            "NVML GPU temperature in degrees Celsius",
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
            "NVML physical memory clock in MHz, converted to Hz",
        ),
        (
            "gpu.clock.video",
            "GPU Video",
            Unit::Hertz,
            "NVML video clock in MHz, converted to Hz",
        ),
    ];
    metrics
        .into_iter()
        .map(|(id, name, unit, semantics)| {
            let mut metric = descriptor(
                id,
                entity_id,
                EntityKind::Gpu,
                name,
                unit,
                "nvml",
                semantics,
            );
            metric.entity_display_name = "nvidia".into();
            if id == "gpu.utilization" {
                metric.temporal_semantics = TemporalSemantics::VendorSampled;
            }
            metric
        })
        .collect()
}

struct NativeInitializer;

#[cfg(not(feature = "nvidia"))]
impl NvidiaInitializer for NativeInitializer {
    fn initialize(&mut self) -> Result<Box<dyn NvidiaSource>, Unavailable> {
        Err(Unavailable::new(
            UnavailableKind::Unsupported,
            "NVIDIA support was disabled at build time",
        ))
    }
}

#[cfg(feature = "nvidia")]
mod nvml {
    use super::*;
    use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
    use nvml_wrapper::error::NvmlError;
    use nvml_wrapper::{Device, Nvml};

    fn unavailable(error: NvmlError) -> Unavailable {
        let kind = match &error {
            NvmlError::NotSupported => UnavailableKind::Unsupported,
            NvmlError::NoPermission => UnavailableKind::PermissionDenied,
            _ => UnavailableKind::TemporarilyUnavailable,
        };
        Unavailable::new(kind, error.to_string())
    }

    fn identity(device: &Device<'_>) -> Result<String, Unavailable> {
        if let Ok(uuid) = device.uuid() {
            return Ok(format!("gpu:uuid:{uuid}"));
        }
        let pci = device.pci_info().map_err(unavailable)?;
        Ok(format!("gpu:pci:{}:nvidia", pci.bus_id))
    }

    struct Source {
        nvml: Nvml,
        entity_id: String,
    }

    impl NvidiaInitializer for NativeInitializer {
        fn initialize(&mut self) -> Result<Box<dyn NvidiaSource>, Unavailable> {
            let nvml = Nvml::init().map_err(unavailable)?;
            if nvml.device_count().map_err(unavailable)? == 0 {
                return Err(Unavailable::new(
                    UnavailableKind::Unsupported,
                    "NVML reports no NVIDIA devices",
                ));
            }
            // Preserve the baseline monitor's first-device selection, but never
            // use that enumeration index as the emitted device identity.
            let entity_id = identity(&nvml.device_by_index(0).map_err(unavailable)?)?;
            Ok(Box::new(Source { nvml, entity_id }))
        }
    }

    impl NvidiaSource for Source {
        fn descriptors(&self) -> Vec<MetricDescriptor> {
            descriptors(&self.entity_id)
        }

        fn read(&mut self, now: &Timestamp) -> Vec<MetricSample> {
            let device = self.nvml.device_by_index(0).map_err(unavailable);
            let device = device.and_then(|device| {
                if identity(&device)? != self.entity_id {
                    Err(Unavailable::temporary("NVIDIA device identity changed"))
                } else {
                    Ok(device)
                }
            });
            let mut output = ProviderOutput::default();
            for descriptor in self.descriptors() {
                let value = match &device {
                    Err(reason) => Err(reason.clone()),
                    Ok(device) => read_metric(device, &descriptor.metric_id),
                };
                emit(&mut output, descriptor, now, None, value);
            }
            output.samples
        }
    }

    fn read_metric(device: &Device<'_>, id: &str) -> Result<MetricValue, Unavailable> {
        let value = match id {
            "gpu.utilization" => f64::from(device.utilization_rates().map_err(unavailable)?.gpu),
            "gpu.memory.occupancy" => {
                let memory = device.memory_info().map_err(unavailable)?;
                if memory.total == 0 {
                    return Err(Unavailable::temporary("total NVIDIA memory is unavailable"));
                }
                memory.used as f64 / memory.total as f64 * 100.0
            }
            "gpu.temperature" => f64::from(
                device
                    .temperature(TemperatureSensor::Gpu)
                    .map_err(unavailable)?,
            ),
            _ => {
                let clock = match id {
                    "gpu.clock.graphics" => Clock::Graphics,
                    "gpu.clock.sm" => Clock::SM,
                    "gpu.clock.memory" => Clock::Memory,
                    "gpu.clock.video" => Clock::Video,
                    _ => {
                        return Err(Unavailable::new(
                            UnavailableKind::Unsupported,
                            "unknown NVIDIA metric",
                        ));
                    }
                };
                f64::from(device.clock_info(clock).map_err(unavailable)?) * 1_000_000.0
            }
        };
        Ok(MetricValue::Float(value))
    }
}
