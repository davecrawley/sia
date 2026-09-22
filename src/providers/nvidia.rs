use super::{descriptor, emit};
use crate::collection::{DeviceEvent, DeviceMetadata, Provider};
use crate::model::{
    Capability, EntityKind, MetricDescriptor, MetricSample, MetricValue, ProviderOutput,
    TemporalSemantics, Timestamp, Unavailable, UnavailableKind, Unit,
};

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

pub trait NvidiaSource {
    fn descriptors(&self) -> Vec<MetricDescriptor>;
    fn read(&mut self, now: &Timestamp) -> Vec<MetricSample>;
    fn take_device_events(&mut self) -> Vec<DeviceEvent> {
        Vec::new()
    }
    fn devices(&self) -> Vec<DeviceMetadata> {
        Vec::new()
    }
}

/// Initialization failures are retried on each collection, except Unsupported.
/// Native topology is refreshed on every collection, using fresh handles and
/// durable identities. Neither policy relies on wall time or GUI frame rate.
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
            .map(|failure| failure.kind != UnavailableKind::Unsupported)
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
            // Reading can discover devices; obtain descriptors afterwards.
            let samples = source.read(now);
            let mut descriptors = source.descriptors();
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
        for descriptor in descriptors("gpu:nvidia:unresolved", "NVIDIA unavailable") {
            emit(&mut output, descriptor, now, None, Err(failure.clone()));
        }
        output
    }

    fn take_device_events(&mut self) -> Vec<DeviceEvent> {
        self.source
            .as_mut()
            .map(|source| source.take_device_events())
            .unwrap_or_default()
    }

    fn devices(&self) -> Vec<DeviceMetadata> {
        self.source
            .as_ref()
            .map(|source| source.devices())
            .unwrap_or_default()
    }
}

fn descriptors(entity_id: &str, device_name: &str) -> Vec<MetricDescriptor> {
    let metrics = [
        ("gpu.utilization", "GPU kernel busy", Unit::Percent,
            "Percent of the NVML vendor sample period with one or more kernels executing; not percent of peak FLOPS",
            Some("nvml_kernel_execution_time")),
        ("gpu.memory.activity", "GPU global-memory activity", Unit::Percent,
            "Percent of the NVML vendor sample period with global device memory reads or writes; not percent of peak bandwidth",
            Some("nvml_global_memory_active_time")),
        ("gpu.memory.used", "VRAM used", Unit::Bytes,
            "NVML allocated device-memory bytes", Some("device_memory_capacity_bytes")),
        ("gpu.memory.free", "VRAM free", Unit::Bytes,
            "NVML available device-memory bytes", Some("device_memory_capacity_bytes")),
        ("gpu.memory.total", "VRAM total", Unit::Bytes,
            "NVML total device-memory bytes", Some("device_memory_capacity_bytes")),
        ("gpu.memory.occupancy", "VRAM occupancy", Unit::Percent,
            "100 * NVML used device-memory bytes / total device-memory bytes; capacity occupancy, not activity",
            Some("device_memory_capacity_ratio")),
        ("gpu.temperature", "GPU (Core)", Unit::Celsius,
            "NVML GPU temperature in degrees Celsius", None),
        ("gpu.clock.graphics", "GPU Graphics", Unit::Hertz,
            "NVML graphics clock in MHz, converted to Hz", None),
        ("gpu.clock.sm", "GPU SM", Unit::Hertz,
            "NVML SM clock in MHz, converted to Hz", None),
        ("gpu.clock.memory", "GPU Memory", Unit::Hertz,
            "NVML physical memory clock in MHz, converted to Hz", None),
        ("gpu.clock.video", "GPU Video", Unit::Hertz,
            "NVML video clock in MHz, converted to Hz", None),
        ("device.generation", "Device generation", Unit::Count,
            "Provider generation applying to this entity at this observation; increments after loss or removal and recovery", None),
    ];
    metrics
        .into_iter()
        .map(|(id, name, unit, semantics, group)| {
            let mut metric = descriptor(
                id,
                entity_id,
                EntityKind::Gpu,
                &format!("{name} — {device_name} [{entity_id}]"),
                unit,
                "nvml",
                semantics,
            );
            metric.entity_display_name = format!("nvidia {device_name} [{entity_id}]");
            if matches!(id, "gpu.utilization" | "gpu.memory.activity") {
                metric.temporal_semantics = TemporalSemantics::VendorSampled;
            }
            metric.comparability_group = group.map(str::to_owned);
            metric.semantics_version = 2;
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
    use crate::collection::DeviceState;
    use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
    use nvml_wrapper::error::NvmlError;
    use nvml_wrapper::{Device, Nvml};
    use std::collections::{BTreeMap, BTreeSet};

    fn unavailable(error: NvmlError) -> Unavailable {
        let kind = match &error {
            NvmlError::NotSupported => UnavailableKind::Unsupported,
            NvmlError::NoPermission => UnavailableKind::PermissionDenied,
            _ => UnavailableKind::TemporarilyUnavailable,
        };
        Unavailable::new(kind, error.to_string())
    }

    fn result<T>(value: Result<T, NvmlError>, lost: &mut bool) -> Result<T, Unavailable> {
        if matches!(&value, Err(NvmlError::GpuLost)) {
            *lost = true;
        }
        value.map_err(unavailable)
    }

    fn identity(device: &Device<'_>, version: Option<String>) -> Result<DeviceMetadata, NvmlError> {
        let uuid = device.uuid().ok().filter(|uuid| !uuid.is_empty());
        let pci = device.pci_info();
        let pci_address = match pci {
            Ok(pci) => Some(pci.bus_id).filter(|address| !address.is_empty()),
            Err(error) if uuid.is_none() => return Err(error),
            Err(_) => None,
        };
        let entity_id = if let Some(uuid) = &uuid {
            format!("gpu:uuid:{uuid}")
        } else if let Some(pci) = &pci_address {
            format!("gpu:pci:{pci}:nvidia")
        } else {
            return Err(NvmlError::NotSupported);
        };
        Ok(DeviceMetadata {
            entity_id,
            display_name: device.name().unwrap_or_else(|_| "NVIDIA GPU".into()),
            uuid,
            pci_address,
            driver: "nvidia".into(),
            driver_version: version,
            generation: 0,
            state: DeviceState::Discovered,
        })
    }

    struct Source {
        nvml: Nvml,
        devices: BTreeMap<String, DeviceMetadata>,
        events: Vec<DeviceEvent>,
        enumerated: bool,
        unresolved: Option<Unavailable>,
    }

    fn event(
        events: &mut Vec<DeviceEvent>,
        device: &mut DeviceMetadata,
        now: &Timestamp,
        state: DeviceState,
        reason: &str,
        environment_changed: bool,
    ) {
        device.state = state;
        events.push(DeviceEvent {
            observed_at: now.clone(),
            entity_id: device.entity_id.clone(),
            generation: device.generation,
            state,
            reason: reason.into(),
            environment_changed,
        });
    }

    impl NvidiaInitializer for NativeInitializer {
        fn initialize(&mut self) -> Result<Box<dyn NvidiaSource>, Unavailable> {
            let nvml = Nvml::init().map_err(unavailable)?;
            Ok(Box::new(Source {
                nvml,
                devices: BTreeMap::new(),
                events: Vec::new(),
                enumerated: false,
                unresolved: None,
            }))
        }
    }

    impl NvidiaSource for Source {
        fn descriptors(&self) -> Vec<MetricDescriptor> {
            let mut metrics: Vec<_> = self
                .devices
                .values()
                .flat_map(|device| descriptors(&device.entity_id, &device.display_name))
                .collect();
            if self.unresolved.is_some() {
                metrics.extend(descriptors("gpu:nvidia:unresolved", "NVIDIA unavailable"));
            }
            metrics
        }

        fn devices(&self) -> Vec<DeviceMetadata> {
            self.devices.values().cloned().collect()
        }
        fn take_device_events(&mut self) -> Vec<DeviceEvent> {
            std::mem::take(&mut self.events)
        }

        fn read(&mut self, now: &Timestamp) -> Vec<MetricSample> {
            let mut output = ProviderOutput::default();
            let mut seen = BTreeSet::new();
            let mut incomplete = None;
            let mut enumeration_lost = false;
            self.unresolved = None;
            let count = result(self.nvml.device_count(), &mut enumeration_lost);
            let driver_version = self.nvml.sys_driver_version().ok();
            match count {
                Err(reason) => incomplete = Some(reason),
                Ok(count) => {
                    for index in 0..count {
                        let device =
                            match result(self.nvml.device_by_index(index), &mut enumeration_lost) {
                                Ok(device) => device,
                                Err(reason) => {
                                    incomplete = Some(reason);
                                    continue;
                                }
                            };
                        let mut metadata = match result(
                            identity(&device, driver_version.clone()),
                            &mut enumeration_lost,
                        ) {
                            Ok(metadata) => metadata,
                            Err(reason) => {
                                incomplete = Some(reason);
                                continue;
                            }
                        };
                        let id = metadata.entity_id.clone();
                        if !seen.insert(id.clone()) {
                            incomplete =
                                Some(Unavailable::temporary("duplicate NVIDIA durable identity"));
                            continue;
                        }
                        let (mut values, lost) = read_device(&device);
                        if let Some(old) = self.devices.get(&id) {
                            metadata.generation = old.generation;
                            metadata.state = old.state;
                            if metadata.uuid.is_none() {
                                metadata.uuid = old.uuid.clone();
                            }
                            if metadata.pci_address.is_none() {
                                metadata.pci_address = old.pci_address.clone();
                            }
                            if metadata.driver_version.is_none() {
                                metadata.driver_version = old.driver_version.clone();
                            }
                        } else {
                            event(
                                &mut self.events,
                                &mut metadata,
                                now,
                                DeviceState::Discovered,
                                "NVIDIA durable identity discovered",
                                self.enumerated,
                            );
                        }
                        if lost {
                            if metadata.state != DeviceState::Lost {
                                event(&mut self.events, &mut metadata, now, DeviceState::Lost,
                                    "NVML GPU lost/reset; all telemetry for this observation is unavailable", true);
                            }
                            for value in values.values_mut() {
                                *value = Err(Unavailable::temporary("NVML GPU lost/reset"));
                            }
                        } else {
                            let has_value = values.values().any(Result::is_ok);
                            let recovering =
                                matches!(metadata.state, DeviceState::Lost | DeviceState::Removed);
                            if recovering && has_value {
                                metadata.generation += 1;
                                event(
                                    &mut self.events,
                                    &mut metadata,
                                    now,
                                    DeviceState::Reappeared,
                                    "NVIDIA durable identity recovered; new provider generation",
                                    true,
                                );
                            }
                            if !recovering || has_value {
                                let state = if values.values().any(Result::is_err) {
                                    DeviceState::TemporarilyUnavailable
                                } else {
                                    DeviceState::Available
                                };
                                if metadata.state != state {
                                    event(
                                        &mut self.events,
                                        &mut metadata,
                                        now,
                                        state,
                                        if state == DeviceState::Available {
                                            "NVIDIA telemetry available"
                                        } else {
                                            "One or more fields unavailable; see per-field capability reasons"
                                        },
                                        false,
                                    );
                                }
                            }
                        }
                        for metric in descriptors(&id, &metadata.display_name) {
                            let value = if metric.metric_id == "device.generation" {
                                Ok(MetricValue::Unsigned(metadata.generation))
                            } else {
                                values.remove(&metric.metric_id).unwrap_or_else(|| {
                                    Err(Unavailable::temporary("NVIDIA field absent"))
                                })
                            };
                            emit(&mut output, metric, now, None, value);
                        }
                        self.devices.insert(id, metadata);
                    }
                    if count == 0 && self.devices.is_empty() {
                        self.unresolved = Some(Unavailable::temporary("NVML reports no visible NVIDIA devices; discovery retries each collection"));
                    }
                }
            }
            for (id, metadata) in &mut self.devices {
                if seen.contains(id) {
                    continue;
                }
                let (state, reason) = if enumeration_lost {
                    (
                        DeviceState::Lost,
                        Unavailable::temporary("NVML GPU lost during enumeration"),
                    )
                } else if let Some(reason) = &incomplete {
                    (DeviceState::TemporarilyUnavailable, reason.clone())
                } else {
                    (
                        DeviceState::Removed,
                        Unavailable::temporary("NVIDIA device removed from visible topology"),
                    )
                };
                // An inconclusive enumeration must not erase a pending recovery
                // generation or claim removal of a device it could not identify.
                let preserve = state == DeviceState::TemporarilyUnavailable
                    && matches!(metadata.state, DeviceState::Lost | DeviceState::Removed);
                if metadata.state != state && !preserve {
                    event(
                        &mut self.events,
                        metadata,
                        now,
                        state,
                        &reason.reason,
                        matches!(state, DeviceState::Lost | DeviceState::Removed),
                    );
                }
                for metric in descriptors(id, &metadata.display_name) {
                    let value = if metric.metric_id == "device.generation" {
                        Ok(MetricValue::Unsigned(metadata.generation))
                    } else {
                        Err(reason.clone())
                    };
                    emit(&mut output, metric, now, None, value);
                }
            }
            if let Some(reason) = incomplete {
                self.unresolved = Some(reason);
            } else {
                self.enumerated = true;
            }
            if let Some(reason) = &self.unresolved {
                for metric in descriptors("gpu:nvidia:unresolved", "NVIDIA unavailable") {
                    emit(&mut output, metric, now, None, Err(reason.clone()));
                }
            }
            output.samples
        }
    }

    fn read_device(
        device: &Device<'_>,
    ) -> (BTreeMap<String, Result<MetricValue, Unavailable>>, bool) {
        let mut values = BTreeMap::new();
        let mut lost = false;
        let utilization = result(device.utilization_rates(), &mut lost);
        for (id, gpu) in [("gpu.utilization", true), ("gpu.memory.activity", false)] {
            let value = utilization.as_ref().map_err(Clone::clone).map(|util| {
                MetricValue::Float(f64::from(if gpu { util.gpu } else { util.memory }))
            });
            values.insert(id.into(), value);
        }
        let memory = result(device.memory_info(), &mut lost);
        for id in [
            "gpu.memory.used",
            "gpu.memory.free",
            "gpu.memory.total",
            "gpu.memory.occupancy",
        ] {
            let value = memory.as_ref().map_err(Clone::clone).and_then(|memory| {
                Ok(match id {
                    "gpu.memory.used" => MetricValue::Unsigned(memory.used),
                    "gpu.memory.free" => MetricValue::Unsigned(memory.free),
                    "gpu.memory.total" => MetricValue::Unsigned(memory.total),
                    _ => {
                        if memory.total == 0 {
                            return Err(Unavailable::temporary(
                                "total NVIDIA memory is unavailable",
                            ));
                        }
                        MetricValue::Float(memory.used as f64 / memory.total as f64 * 100.0)
                    }
                })
            });
            values.insert(id.into(), value);
        }
        values.insert(
            "gpu.temperature".into(),
            result(device.temperature(TemperatureSensor::Gpu), &mut lost)
                .map(|value| MetricValue::Float(f64::from(value))),
        );
        for (id, clock) in [
            ("gpu.clock.graphics", Clock::Graphics),
            ("gpu.clock.sm", Clock::SM),
            ("gpu.clock.memory", Clock::Memory),
            ("gpu.clock.video", Clock::Video),
        ] {
            values.insert(
                id.into(),
                result(device.clock_info(clock), &mut lost)
                    .map(|value| MetricValue::Float(f64::from(value) * 1_000_000.0)),
            );
        }
        (values, lost)
    }
}
