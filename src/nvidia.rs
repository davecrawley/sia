use crate::collect::{BackendOutcome, MetricReading, Provider, ProviderBatch, SourceReading};
use crate::model::{
    CapabilityState, EntityKind, MetricDescriptor, MetricId, MetricValue, SeriesKey,
    TemporalSemantics, Unit, ValueKind,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NvidiaDevice {
    pub backend_id: String,
    pub uuid: Option<String>,
    pub pci_address: Option<String>,
    pub driver_identity: Option<String>,
}

impl NvidiaDevice {
    pub fn durable_id(&self) -> Option<String> {
        if let Some(uuid) = self
            .uuid
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return Some(format!("gpu:uuid:{uuid}"));
        }
        let pci = self
            .pci_address
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())?;
        let driver = self
            .driver_identity
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())?;
        Some(format!("gpu:pci:{pci}:driver:{driver}"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NvidiaMetric {
    GpuUtilization,
    MemoryActivity,
    VramOccupancy,
    Temperature,
    GraphicsClock,
    SmClock,
    MemoryClock,
    VideoClock,
}

impl NvidiaMetric {
    pub const ALL: [Self; 8] = [
        Self::GpuUtilization,
        Self::MemoryActivity,
        Self::VramOccupancy,
        Self::Temperature,
        Self::GraphicsClock,
        Self::SmClock,
        Self::MemoryClock,
        Self::VideoClock,
    ];
}

pub trait NvidiaBackend {
    fn initialize(&mut self) -> BackendOutcome<()>;
    fn discover(&mut self) -> BackendOutcome<Vec<NvidiaDevice>>;
    fn read_metric(
        &mut self,
        backend_id: &str,
        metric: NvidiaMetric,
    ) -> BackendOutcome<SourceReading<f64>>;
}

#[derive(Clone, Debug)]
struct DeviceState {
    device: NvidiaDevice,
    generation: u64,
    present: bool,
    terminally_unsupported: BTreeMap<NvidiaMetric, String>,
}

pub struct NvidiaProvider<B: NvidiaBackend> {
    backend: B,
    initialized: bool,
    terminal_initialization: Option<BackendOutcome<()>>,
    terminal_discovery: Option<String>,
    devices: BTreeMap<String, DeviceState>,
}

impl<B: NvidiaBackend> NvidiaProvider<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            initialized: false,
            terminal_initialization: None,
            terminal_discovery: None,
            devices: BTreeMap::new(),
        }
    }

    pub fn device_generation(&self, durable_id: &str) -> Option<u64> {
        self.devices.get(durable_id).map(|state| state.generation)
    }

    fn status_batch(&self, outcome: BackendOutcome<()>, label: &str) -> ProviderBatch {
        ProviderBatch {
            readings: vec![MetricReading {
                descriptor: status_descriptor(label),
                outcome: outcome.map(|()| {
                    SourceReading::fallback_timestamp(MetricValue::State("available".to_owned()))
                }),
            }],
        }
    }

    fn mark_devices_absent(&mut self, reason: &str) -> Vec<MetricReading> {
        let mut readings = Vec::new();
        for (identity, state) in &mut self.devices {
            if !state.present {
                continue;
            }
            state.present = false;
            for metric in NvidiaMetric::ALL {
                readings.push(metric_reading(
                    metric,
                    identity,
                    state.generation,
                    BackendOutcome::TemporarilyUnavailable(reason.to_owned()),
                ));
            }
        }
        readings
    }
}

impl<B: NvidiaBackend> Provider for NvidiaProvider<B> {
    fn collect(&mut self) -> ProviderBatch {
        if let Some(outcome) = self.terminal_initialization.clone() {
            return self.status_batch(outcome, "NVIDIA initialization");
        }
        if !self.initialized {
            match self.backend.initialize() {
                BackendOutcome::Available(()) => self.initialized = true,
                BackendOutcome::Unsupported(reason) => {
                    let outcome = BackendOutcome::Unsupported(reason);
                    self.terminal_initialization = Some(outcome.clone());
                    return self.status_batch(outcome, "NVIDIA initialization");
                }
                BackendOutcome::PermissionDenied(reason) => {
                    let outcome = BackendOutcome::PermissionDenied(reason);
                    self.terminal_initialization = Some(outcome.clone());
                    return self.status_batch(outcome, "NVIDIA initialization");
                }
                BackendOutcome::TemporarilyUnavailable(reason) => {
                    return self.status_batch(
                        BackendOutcome::TemporarilyUnavailable(reason),
                        "NVIDIA initialization",
                    );
                }
                BackendOutcome::Stale(reason) | BackendOutcome::Error(reason) => {
                    return self
                        .status_batch(BackendOutcome::Error(reason), "NVIDIA initialization");
                }
            }
        }

        if let Some(reason) = self.terminal_discovery.clone() {
            return self.status_batch(BackendOutcome::Unsupported(reason), "NVIDIA discovery");
        }

        let discovered = match self.backend.discover() {
            BackendOutcome::Available(devices) => devices,
            BackendOutcome::Unsupported(reason) => {
                self.terminal_discovery = Some(reason.clone());
                let mut readings = self.mark_devices_absent(&reason);
                readings.extend(
                    self.status_batch(BackendOutcome::Unsupported(reason), "NVIDIA discovery")
                        .readings,
                );
                return ProviderBatch { readings };
            }
            BackendOutcome::PermissionDenied(reason) => {
                let mut readings = self.mark_devices_absent(&reason);
                readings.extend(
                    self.status_batch(BackendOutcome::PermissionDenied(reason), "NVIDIA discovery")
                        .readings,
                );
                return ProviderBatch { readings };
            }
            BackendOutcome::TemporarilyUnavailable(reason)
            | BackendOutcome::Stale(reason)
            | BackendOutcome::Error(reason) => {
                let mut readings = self.mark_devices_absent(&reason);
                readings.extend(
                    self.status_batch(
                        BackendOutcome::TemporarilyUnavailable(reason),
                        "NVIDIA discovery",
                    )
                    .readings,
                );
                return ProviderBatch { readings };
            }
        };

        if discovered.is_empty() {
            let reason = "no NVIDIA devices are currently visible".to_owned();
            let mut readings = self.mark_devices_absent(&reason);
            readings.extend(
                self.status_batch(
                    BackendOutcome::TemporarilyUnavailable(reason),
                    "NVIDIA discovery",
                )
                .readings,
            );
            return ProviderBatch { readings };
        }

        let mut seen = BTreeSet::new();
        let mut readings = Vec::new();
        for device in discovered {
            let Some(identity) = device.durable_id() else {
                readings.push(MetricReading {
                    descriptor: status_descriptor("NVIDIA device identity"),
                    outcome: BackendOutcome::TemporarilyUnavailable(
                        "device has neither a UUID nor PCI address plus driver identity".to_owned(),
                    ),
                });
                continue;
            };
            seen.insert(identity.clone());

            let state = self
                .devices
                .entry(identity.clone())
                .or_insert_with(|| DeviceState {
                    device: device.clone(),
                    generation: 0,
                    present: true,
                    terminally_unsupported: BTreeMap::new(),
                });
            if !state.present {
                state.generation = state.generation.saturating_add(1);
                state.terminally_unsupported.clear();
            }
            state.present = true;
            state.device = device.clone();
            let generation = state.generation;
            let terminal = state.terminally_unsupported.clone();
            let mut attempted = 0usize;
            let mut retryable_failures = 0usize;

            for metric in NvidiaMetric::ALL {
                if let Some(reason) = terminal.get(&metric) {
                    readings.push(metric_reading(
                        metric,
                        &identity,
                        generation,
                        BackendOutcome::Unsupported(reason.clone()),
                    ));
                    continue;
                }
                attempted += 1;
                let outcome = self.backend.read_metric(&device.backend_id, metric);
                if matches!(
                    &outcome,
                    BackendOutcome::TemporarilyUnavailable(_) | BackendOutcome::Error(_)
                ) {
                    retryable_failures += 1;
                }
                if let BackendOutcome::Unsupported(reason) = &outcome {
                    if let Some(current) = self.devices.get_mut(&identity) {
                        current
                            .terminally_unsupported
                            .insert(metric, reason.clone());
                    }
                }
                readings.push(metric_reading(metric, &identity, generation, outcome));
            }
            if attempted > 0 && retryable_failures == attempted {
                if let Some(current) = self.devices.get_mut(&identity) {
                    current.present = false;
                }
            }
        }

        let missing: Vec<_> = self
            .devices
            .iter()
            .filter(|(identity, state)| state.present && !seen.contains(*identity))
            .map(|(identity, _)| identity.clone())
            .collect();
        for identity in missing {
            if let Some(state) = self.devices.get_mut(&identity) {
                state.present = false;
                let generation = state.generation;
                for metric in NvidiaMetric::ALL {
                    readings.push(metric_reading(
                        metric,
                        &identity,
                        generation,
                        BackendOutcome::TemporarilyUnavailable(
                            "NVIDIA device disappeared".to_owned(),
                        ),
                    ));
                }
            }
        }

        readings.extend(
            self.status_batch(BackendOutcome::Available(()), "NVIDIA provider")
                .readings,
        );
        ProviderBatch { readings }
    }
}

fn capability<T>(outcome: &BackendOutcome<T>) -> CapabilityState {
    match outcome {
        BackendOutcome::Available(_) | BackendOutcome::Stale(_) => CapabilityState::Available,
        BackendOutcome::Unsupported(reason) => CapabilityState::Unsupported(reason.clone()),
        BackendOutcome::PermissionDenied(reason) => {
            CapabilityState::PermissionDenied(reason.clone())
        }
        BackendOutcome::TemporarilyUnavailable(reason) | BackendOutcome::Error(reason) => {
            CapabilityState::TemporarilyUnavailable(reason.clone())
        }
    }
}

fn metric_reading(
    metric: NvidiaMetric,
    entity_id: &str,
    generation: u64,
    outcome: BackendOutcome<SourceReading<f64>>,
) -> MetricReading {
    let descriptor = nvidia_descriptor(metric, entity_id, generation, capability(&outcome));
    MetricReading {
        descriptor,
        outcome: outcome.map(|reading| reading.map(MetricValue::F64)),
    }
}

pub fn nvidia_descriptor(
    metric: NvidiaMetric,
    entity_id: &str,
    generation: u64,
    capability: CapabilityState,
) -> MetricDescriptor {
    let (id, name, unit, semantics, definition, group) = match metric {
        NvidiaMetric::GpuUtilization => (
            "gpu.nvidia.utilization",
            "GPU utilization",
            Unit::Percent,
            TemporalSemantics::VendorSampled,
            "Percentage of the vendor sampling period during which one or more kernels executed.",
            "nvml_gpu_time_busy",
        ),
        NvidiaMetric::MemoryActivity => (
            "gpu.nvidia.memory_activity",
            "GPU memory-subsystem activity",
            Unit::Percent,
            TemporalSemantics::VendorSampled,
            "Percentage of the vendor sampling period during which global device memory was read or written.",
            "nvml_memory_time_busy",
        ),
        NvidiaMetric::VramOccupancy => (
            "gpu.nvidia.vram_occupancy",
            "VRAM occupancy",
            Unit::Percent,
            TemporalSemantics::PointSample,
            "Used device memory divided by total device memory at observation time.",
            "device_memory_occupancy_percent",
        ),
        NvidiaMetric::Temperature => (
            "gpu.nvidia.temperature",
            "GPU temperature",
            Unit::Celsius,
            TemporalSemantics::PointSample,
            "Current NVML GPU temperature sensor reading.",
            "gpu_temperature_celsius",
        ),
        NvidiaMetric::GraphicsClock => (
            "gpu.nvidia.clock.graphics",
            "GPU graphics clock",
            Unit::Hertz,
            TemporalSemantics::PointSample,
            "Current NVML graphics clock frequency.",
            "nvml_graphics_clock_hz",
        ),
        NvidiaMetric::SmClock => (
            "gpu.nvidia.clock.sm",
            "GPU SM clock",
            Unit::Hertz,
            TemporalSemantics::PointSample,
            "Current NVML streaming-multiprocessor clock frequency.",
            "nvml_sm_clock_hz",
        ),
        NvidiaMetric::MemoryClock => (
            "gpu.nvidia.clock.memory",
            "GPU memory clock",
            Unit::Hertz,
            TemporalSemantics::PointSample,
            "Current NVML memory clock without effective-rate scaling.",
            "nvml_memory_clock_hz",
        ),
        NvidiaMetric::VideoClock => (
            "gpu.nvidia.clock.video",
            "GPU video clock",
            Unit::Hertz,
            TemporalSemantics::PointSample,
            "Current NVML video clock frequency.",
            "nvml_video_clock_hz",
        ),
    };
    MetricDescriptor {
        key: SeriesKey::new(MetricId::from(id), entity_id, generation),
        display_name: name.to_owned(),
        entity_kind: EntityKind::Gpu,
        unit,
        value_kind: ValueKind::Gauge,
        temporal_semantics: semantics,
        provider: "nvml".to_owned(),
        capability,
        source_resolution_ns: None,
        source_definition: definition.to_owned(),
        comparability_group: Some(group.to_owned()),
        semantics_version: 2,
    }
}

fn status_descriptor(label: &str) -> MetricDescriptor {
    MetricDescriptor {
        key: SeriesKey::new(
            MetricId::from("gpu.nvidia.provider_status"),
            "gpu:nvidia:provider",
            0,
        ),
        display_name: label.to_owned(),
        entity_kind: EntityKind::Gpu,
        unit: Unit::Other("status".to_owned()),
        value_kind: ValueKind::State,
        temporal_semantics: TemporalSemantics::PointSample,
        provider: "nvml".to_owned(),
        capability: CapabilityState::TemporarilyUnavailable(String::new()),
        source_resolution_ns: None,
        source_definition: "NVIDIA provider discovery and identity state.".to_owned(),
        comparability_group: None,
        semantics_version: 1,
    }
}

use nvml_wrapper::enum_wrappers::device::{Clock as NvClock, TemperatureSensor};
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::Nvml;

#[derive(Default)]
pub struct NvmlBackend {
    nvml: Option<Nvml>,
    driver_identity: Option<String>,
}

impl NvidiaBackend for NvmlBackend {
    fn initialize(&mut self) -> BackendOutcome<()> {
        if self.nvml.is_some() {
            return BackendOutcome::Available(());
        }
        let nvml = match Nvml::init() {
            Ok(value) => value,
            Err(error) => return classify_nvml(error),
        };
        self.driver_identity = nvml.sys_driver_version().ok();
        self.nvml = Some(nvml);
        BackendOutcome::Available(())
    }

    fn discover(&mut self) -> BackendOutcome<Vec<NvidiaDevice>> {
        let Some(nvml) = self.nvml.as_ref() else {
            return BackendOutcome::TemporarilyUnavailable("NVML is not initialized".to_owned());
        };
        let count = match nvml.device_count() {
            Ok(value) => value,
            Err(error) => return classify_nvml(error),
        };
        let mut devices = Vec::new();
        for index in 0..count {
            let device = match nvml.device_by_index(index) {
                Ok(value) => value,
                Err(error) => return classify_nvml(error),
            };
            devices.push(NvidiaDevice {
                backend_id: index.to_string(),
                uuid: device.uuid().ok(),
                pci_address: device.pci_info().ok().map(|value| value.bus_id),
                driver_identity: self.driver_identity.clone(),
            });
        }
        BackendOutcome::Available(devices)
    }

    fn read_metric(
        &mut self,
        backend_id: &str,
        metric: NvidiaMetric,
    ) -> BackendOutcome<SourceReading<f64>> {
        let index = match backend_id.parse::<u32>() {
            Ok(value) => value,
            Err(error) => return BackendOutcome::Error(error.to_string()),
        };
        let Some(nvml) = self.nvml.as_ref() else {
            return BackendOutcome::TemporarilyUnavailable("NVML is not initialized".to_owned());
        };
        let device = match nvml.device_by_index(index) {
            Ok(value) => value,
            Err(error) => return classify_nvml(error),
        };
        let value = match metric {
            NvidiaMetric::GpuUtilization => {
                device.utilization_rates().map(|value| value.gpu as f64)
            }
            NvidiaMetric::MemoryActivity => {
                device.utilization_rates().map(|value| value.memory as f64)
            }
            NvidiaMetric::VramOccupancy => device.memory_info().and_then(|value| {
                if value.total == 0 {
                    Err(NvmlError::NoData)
                } else {
                    Ok(value.used as f64 * 100.0 / value.total as f64)
                }
            }),
            NvidiaMetric::Temperature => device
                .temperature(TemperatureSensor::Gpu)
                .map(|value| value as f64),
            NvidiaMetric::GraphicsClock => device
                .clock_info(NvClock::Graphics)
                .map(|value| value as f64 * 1_000_000.0),
            NvidiaMetric::SmClock => device
                .clock_info(NvClock::SM)
                .map(|value| value as f64 * 1_000_000.0),
            NvidiaMetric::MemoryClock => device
                .clock_info(NvClock::Memory)
                .map(|value| value as f64 * 1_000_000.0),
            NvidiaMetric::VideoClock => device
                .clock_info(NvClock::Video)
                .map(|value| value as f64 * 1_000_000.0),
        };
        match value {
            Ok(value) => BackendOutcome::Available(SourceReading::fallback_timestamp(value)),
            Err(error) => classify_nvml(error),
        }
    }
}

pub fn classify_nvml<T>(error: NvmlError) -> BackendOutcome<T> {
    use NvmlError::*;
    let reason = error.to_string();
    match error {
        NotSupported | VgpuEccNotSupported => BackendOutcome::Unsupported(reason),
        NoPermission => BackendOutcome::PermissionDenied(reason),
        LibraryNotFound | DriverNotLoaded | FunctionNotFound => BackendOutcome::Unsupported(reason),
        Timeout | InUse | NoData | GpuLost | ResetRequired | Uninitialized | NotFound
        | LibRmVersionMismatch | OperatingSystem => BackendOutcome::TemporarilyUnavailable(reason),
        _ => BackendOutcome::Error(reason),
    }
}
