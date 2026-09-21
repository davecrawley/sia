//! Optional NVIDIA adapter. Every query fails independently.

use crate::collection::Provider;
use crate::model::{
    Capability, CapabilityStatus, GpuClock, MetricDescriptor, MetricKind, Reading,
    TemporalSemantics, Unavailable, Unit, ValueKind,
};
use nvml_wrapper::{
    enum_wrappers::device::{Clock as NvClock, TemperatureSensor},
    error::NvmlError,
    Nvml,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuMetric {
    Utilization,
    VramOccupancy,
    Temperature,
    Clock(GpuClock),
}

const METRICS: [GpuMetric; 7] = [
    GpuMetric::Utilization,
    GpuMetric::VramOccupancy,
    GpuMetric::Temperature,
    GpuMetric::Clock(GpuClock::Graphics),
    GpuMetric::Clock(GpuClock::Sm),
    GpuMetric::Clock(GpuClock::Memory),
    GpuMetric::Clock(GpuClock::Video),
];

/// Adapter-level substitution point; tests need neither a driver nor hardware.
pub trait NvmlSource {
    fn read(&mut self, metric: GpuMetric) -> Result<Reading, NvmlError>;
}

pub fn translate_error(error: NvmlError) -> Unavailable {
    let status = match &error {
        NvmlError::NotSupported => CapabilityStatus::Unsupported,
        NvmlError::NoPermission => CapabilityStatus::PermissionDenied,
        _ => CapabilityStatus::TemporarilyUnavailable,
    };
    Unavailable {
        status,
        reason: error.to_string(),
    }
}

pub struct NvmlProvider<S> {
    source: Option<S>,
    initial_failure: Option<Unavailable>,
    descriptors: Vec<(GpuMetric, MetricDescriptor)>,
}

impl<S: NvmlSource> NvmlProvider<S> {
    pub fn new(source: S, entity_id: String) -> Self {
        Self {
            source: Some(source),
            initial_failure: None,
            descriptors: descriptors(&entity_id),
        }
    }

    pub fn unavailable(error: NvmlError) -> Self {
        Self {
            source: None,
            initial_failure: Some(translate_error(error)),
            descriptors: descriptors("gpu:nvml:undiscovered"),
        }
    }
}

impl<S: NvmlSource> Provider for NvmlProvider<S> {
    fn discover(&mut self) -> Vec<Capability> {
        self.descriptors
            .clone()
            .into_iter()
            .map(|(_, descriptor)| {
                let result = self.read(&descriptor);
                Capability::from_reading(descriptor, &result)
            })
            .collect()
    }

    fn read(&mut self, descriptor: &MetricDescriptor) -> Result<Reading, Unavailable> {
        let metric = self
            .descriptors
            .iter()
            .find(|(_, candidate)| candidate.same_series(descriptor))
            .map(|(metric, _)| *metric)
            .ok_or_else(|| Unavailable {
                status: CapabilityStatus::Unsupported,
                reason: "unknown NVIDIA metric".into(),
            })?;
        match self.source.as_mut() {
            // Successful readings, including source windows and timestamps,
            // pass through unchanged. In particular zero is not absence.
            Some(source) => source.read(metric).map_err(translate_error),
            None => Err(self
                .initial_failure
                .clone()
                .unwrap_or_else(|| Unavailable::temporary("NVML could not be initialized"))),
        }
    }
}

fn descriptors(entity_id: &str) -> Vec<(GpuMetric, MetricDescriptor)> {
    METRICS
        .into_iter()
        .map(|metric| {
            let (id, name, kind, unit, temporal, semantics) = match metric {
                GpuMetric::Utilization => (
                    "gpu.nvidia.utilization_pct",
                    "GPU %",
                    MetricKind::GpuUtilization,
                    Unit::Percent,
                    TemporalSemantics::VendorSampled,
                    "percent of the vendor sample period with one or more kernels executing",
                ),
                GpuMetric::VramOccupancy => (
                    "gpu.nvidia.vram_occupancy_pct",
                    "VRAM %",
                    MetricKind::VramOccupancy,
                    Unit::Percent,
                    TemporalSemantics::PointSample,
                    "NVML used device memory divided by total device memory, multiplied by 100",
                ),
                GpuMetric::Temperature => (
                    "gpu.nvidia.temperature_c",
                    "GPU (Core)",
                    MetricKind::Temperature {
                        sensor_name: "nvidia".into(),
                        sensor_label: "GPU (Core)".into(),
                    },
                    Unit::Celsius,
                    TemporalSemantics::PointSample,
                    "NVML GPU temperature in degrees Celsius",
                ),
                GpuMetric::Clock(clock) => {
                    let (id, name) = match clock {
                        GpuClock::Graphics => ("gpu.nvidia.graphics_hz", "GPU Graphics"),
                        GpuClock::Sm => ("gpu.nvidia.sm_hz", "GPU SM"),
                        GpuClock::Memory => ("gpu.nvidia.memory_hz", "GPU Memory"),
                        GpuClock::Video => ("gpu.nvidia.video_hz", "GPU Video"),
                    };
                    (
                        id,
                        name,
                        MetricKind::GpuFrequency(clock),
                        Unit::Hertz,
                        TemporalSemantics::PointSample,
                        "NVML current clock in MHz, converted to Hz; memory clock is not doubled",
                    )
                }
            };
            (
                metric,
                MetricDescriptor {
                    metric_id: id.into(),
                    entity_id: entity_id.into(),
                    display_name: name.into(),
                    kind,
                    unit,
                    value_kind: ValueKind::Gauge,
                    temporal_semantics: temporal,
                    provider: "nvml".into(),
                    source_semantics: semantics.into(),
                    semantics_version: 1,
                },
            )
        })
        .collect()
}

struct NativeNvmlSource {
    nvml: Nvml,
    entity_id: String,
}

impl NativeNvmlSource {
    fn new() -> Result<Self, NvmlError> {
        let nvml = Nvml::init()?;
        if nvml.device_count()? == 0 {
            return Err(NvmlError::NotSupported);
        }
        // Keep the baseline monitor's first-device selection. This identity is
        // session-local if the device does not expose a UUID.
        let entity_id = {
            let device = nvml.device_by_index(0)?;
            device
                .uuid()
                .map(|uuid| format!("gpu:uuid:{uuid}"))
                .unwrap_or_else(|_| "gpu:nvml:session-index:0".into())
        };
        Ok(Self { nvml, entity_id })
    }
}

impl NvmlSource for NativeNvmlSource {
    fn read(&mut self, metric: GpuMetric) -> Result<Reading, NvmlError> {
        let device = self.nvml.device_by_index(0)?;
        let value = match metric {
            GpuMetric::Utilization => device.utilization_rates()?.gpu as f64,
            GpuMetric::VramOccupancy => {
                let memory = device.memory_info()?;
                if memory.total == 0 {
                    return Err(NvmlError::NotSupported);
                }
                memory.used as f64 / memory.total as f64 * 100.0
            }
            GpuMetric::Temperature => device.temperature(TemperatureSensor::Gpu)? as f64,
            GpuMetric::Clock(clock) => {
                let clock = match clock {
                    GpuClock::Graphics => NvClock::Graphics,
                    GpuClock::Sm => NvClock::SM,
                    GpuClock::Memory => NvClock::Memory,
                    GpuClock::Video => NvClock::Video,
                };
                device.clock_info(clock)? as f64 * 1_000_000.0
            }
        };
        // These polling operations do not expose a native timestamp/window.
        Ok(Reading::gauge(value))
    }
}

pub fn local_provider() -> Box<dyn Provider> {
    match NativeNvmlSource::new() {
        Ok(source) => {
            let entity_id = source.entity_id.clone();
            Box::new(NvmlProvider::new(source, entity_id))
        }
        Err(error) => Box::new(NvmlProvider::<NativeNvmlSource>::unavailable(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ObservationWindow, SourceTimestamp};

    struct FakeNvml;

    impl NvmlSource for FakeNvml {
        fn read(&mut self, metric: GpuMetric) -> Result<Reading, NvmlError> {
            match metric {
                GpuMetric::Temperature => Err(NvmlError::NotSupported),
                GpuMetric::VramOccupancy => Err(NvmlError::NoPermission),
                _ => Ok(Reading {
                    value: 0.0,
                    mono_ns: Some(800),
                    source_timestamp: Some(SourceTimestamp {
                        ns: 777,
                        clock_domain: "vendor".into(),
                    }),
                    window: Some(ObservationWindow {
                        start_ns: 600,
                        end_ns: 777,
                        clock_domain: "vendor".into(),
                    }),
                }),
            }
        }
    }

    #[test]
    fn translates_capabilities_and_preserves_source_timing_without_hardware() {
        let mut provider = NvmlProvider::new(FakeNvml, "gpu:fixture".into());
        let capabilities = provider.discover();
        assert_eq!(capabilities[0].status, CapabilityStatus::Available);
        assert_eq!(capabilities[1].status, CapabilityStatus::PermissionDenied);
        assert_eq!(capabilities[2].status, CapabilityStatus::Unsupported);
        let reading = provider.read(&capabilities[0].descriptor).unwrap();
        assert_eq!(reading.value, 0.0);
        assert_eq!(reading.mono_ns, Some(800));
        assert_eq!(reading.source_timestamp.unwrap().ns, 777);
        let window = reading.window.unwrap();
        assert_eq!((window.start_ns, window.end_ns), (600, 777));
        assert_eq!(window.clock_domain, "vendor");
    }
}
