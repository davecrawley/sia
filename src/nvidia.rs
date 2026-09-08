use crate::collector::{Provider, ProviderBatch, Reading};
use crate::model::{
    CapabilityStatus, EntityKind, MetricDescriptor, MetricValue, SampleStatus, TemporalSemantics,
    ValueKind,
};
use nvml_wrapper::{
    enum_wrappers::device::{Clock as NvClock, TemperatureSensor},
    Nvml,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NvidiaError {
    Unsupported(String),
    PermissionDenied(String),
    TemporarilyUnavailable(String),
    Error(String),
}

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
            .filter(|value| !value.trim().is_empty())
        {
            return Some(format!("gpu:uuid:{}", uuid.trim()));
        }
        let pci = self
            .pci_address
            .as_deref()
            .filter(|value| !value.trim().is_empty())?;
        let driver = self
            .driver_identity
            .as_deref()
            .filter(|value| !value.trim().is_empty())?;
        Some(format!("gpu:pci:{}:driver:{}", pci.trim(), driver.trim()))
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NvidiaMeasurements {
    pub gpu_util_percent: Option<f64>,
    pub vram_used_bytes: Option<u64>,
    pub vram_total_bytes: Option<u64>,
    pub temperature_c: Option<f64>,
    pub graphics_clock_mhz: Option<f64>,
    pub sm_clock_mhz: Option<f64>,
    pub memory_clock_mhz: Option<f64>,
    pub video_clock_mhz: Option<f64>,
    pub source_timestamp_ns: Option<u64>,
    pub window_start_ns: Option<u64>,
}

pub trait NvidiaBackend {
    fn discover(&mut self) -> Result<Vec<NvidiaDevice>, NvidiaError>;
    fn read(&mut self, backend_id: &str) -> Result<NvidiaMeasurements, NvidiaError>;
}

pub struct NvidiaProvider<B: NvidiaBackend> {
    backend: B,
    devices: BTreeMap<String, NvidiaDevice>,
    terminally_unsupported: BTreeSet<String>,
    discovery_unsupported: bool,
}

impl<B: NvidiaBackend> NvidiaProvider<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            devices: BTreeMap::new(),
            terminally_unsupported: BTreeSet::new(),
            discovery_unsupported: false,
        }
    }

    fn discover(&mut self) -> Result<(), NvidiaError> {
        if self.discovery_unsupported {
            return Err(NvidiaError::Unsupported(
                "NVIDIA discovery is unsupported".into(),
            ));
        }
        let discovered = self.backend.discover()?;
        self.devices.clear();
        for device in discovered {
            if let Some(identity) = device.durable_id() {
                self.devices.insert(identity, device);
            }
        }
        Ok(())
    }
}

fn capability(error: &NvidiaError) -> CapabilityStatus {
    match error {
        NvidiaError::Unsupported(_) => CapabilityStatus::Unsupported,
        NvidiaError::PermissionDenied(_) => CapabilityStatus::PermissionDenied,
        NvidiaError::TemporarilyUnavailable(_) | NvidiaError::Error(_) => {
            CapabilityStatus::TemporarilyUnavailable
        }
    }
}

fn sample_status(error: &NvidiaError) -> SampleStatus {
    match error {
        NvidiaError::TemporarilyUnavailable(reason) => {
            SampleStatus::TemporarilyUnavailable(reason.clone())
        }
        NvidiaError::Unsupported(reason)
        | NvidiaError::PermissionDenied(reason)
        | NvidiaError::Error(reason) => SampleStatus::Error(reason.clone()),
    }
}

fn metric_descriptor(
    id: &str,
    name: &str,
    unit: &str,
    definition: &str,
    group: &str,
    status: CapabilityStatus,
) -> MetricDescriptor {
    MetricDescriptor {
        metric_id: id.into(),
        display_name: name.into(),
        entity_kind: EntityKind::Gpu,
        unit: unit.into(),
        value_kind: ValueKind::Gauge,
        temporal_semantics: TemporalSemantics::VendorSampled,
        provider: "nvml".into(),
        capability_status: status,
        source_resolution_ns: None,
        source_definition: definition.into(),
        comparability_group: Some(group.into()),
        semantics_version: 1,
    }
}

fn descriptors(status: CapabilityStatus) -> Vec<MetricDescriptor> {
    vec![
        metric_descriptor(
            "nvidia.gpu.utilization",
            "GPU utilization",
            "%",
            "NVML percentage of time one or more kernels executed during the vendor sample period.",
            "nvml_gpu_time_busy",
            status.clone(),
        ),
        metric_descriptor(
            "nvidia.vram.occupancy",
            "VRAM occupancy",
            "%",
            "NVML used device memory divided by total device memory.",
            "device_memory_occupancy_percent",
            status.clone(),
        ),
        metric_descriptor(
            "nvidia.temperature",
            "GPU temperature",
            "Cel",
            "NVML GPU temperature sensor point reading.",
            "gpu_temperature_celsius",
            status.clone(),
        ),
        metric_descriptor(
            "nvidia.clock.graphics",
            "GPU graphics clock",
            "MHz",
            "NVML current graphics clock.",
            "nvml_graphics_clock_mhz",
            status.clone(),
        ),
        metric_descriptor(
            "nvidia.clock.sm",
            "GPU SM clock",
            "MHz",
            "NVML current streaming-multiprocessor clock.",
            "nvml_sm_clock_mhz",
            status.clone(),
        ),
        metric_descriptor(
            "nvidia.clock.memory",
            "GPU memory clock",
            "MHz",
            "NVML current memory clock without hidden effective-rate scaling.",
            "nvml_memory_clock_mhz",
            status.clone(),
        ),
        metric_descriptor(
            "nvidia.clock.video",
            "GPU video clock",
            "MHz",
            "NVML current video clock.",
            "nvml_video_clock_mhz",
            status,
        ),
    ]
}

fn unavailable_readings(entity: &str, error: &NvidiaError) -> Vec<Reading> {
    descriptors(capability(error))
        .into_iter()
        .map(|descriptor| Reading {
            metric_id: descriptor.metric_id,
            entity_id: entity.into(),
            value: None,
            status: sample_status(error),
            source_timestamp_ns: None,
            window_start_ns: None,
        })
        .collect()
}

fn available_readings(entity: &str, values: NvidiaMeasurements) -> Vec<Reading> {
    let timestamp = values.source_timestamp_ns;
    let window = values.window_start_ns;
    let mut readings = Vec::new();
    let mut push = |metric: &str, value: MetricValue| {
        readings.push(Reading {
            metric_id: metric.into(),
            entity_id: entity.into(),
            value: Some(value),
            status: SampleStatus::Ok,
            source_timestamp_ns: timestamp,
            window_start_ns: window,
        });
    };
    if let Some(value) = values.gpu_util_percent {
        push("nvidia.gpu.utilization", MetricValue::F64(value));
    }
    if let (Some(used), Some(total)) = (values.vram_used_bytes, values.vram_total_bytes) {
        if total > 0 {
            push(
                "nvidia.vram.occupancy",
                MetricValue::F64(used as f64 * 100.0 / total as f64),
            );
        }
    }
    if let Some(value) = values.temperature_c {
        push("nvidia.temperature", MetricValue::F64(value));
    }
    if let Some(value) = values.graphics_clock_mhz {
        push("nvidia.clock.graphics", MetricValue::F64(value));
    }
    if let Some(value) = values.sm_clock_mhz {
        push("nvidia.clock.sm", MetricValue::F64(value));
    }
    if let Some(value) = values.memory_clock_mhz {
        push("nvidia.clock.memory", MetricValue::F64(value));
    }
    if let Some(value) = values.video_clock_mhz {
        push("nvidia.clock.video", MetricValue::F64(value));
    }
    readings
}

impl<B: NvidiaBackend> Provider for NvidiaProvider<B> {
    fn collect(&mut self) -> ProviderBatch {
        if let Err(error) = self.discover() {
            if matches!(error, NvidiaError::Unsupported(_)) {
                self.discovery_unsupported = true;
            }
            return ProviderBatch {
                descriptors: descriptors(capability(&error)),
                readings: unavailable_readings("gpu:nvidia:discovery", &error),
            };
        }

        let mut batch = ProviderBatch::default();
        for (identity, device) in self.devices.clone() {
            if self.terminally_unsupported.contains(&identity) {
                batch
                    .descriptors
                    .extend(descriptors(CapabilityStatus::Unsupported));
                continue;
            }
            let first = self.backend.read(&device.backend_id);
            let result = match first {
                Err(NvidiaError::TemporarilyUnavailable(_)) | Err(NvidiaError::Error(_)) => {
                    self.backend.read(&device.backend_id)
                }
                other => other,
            };
            match result {
                Ok(values) => {
                    batch
                        .descriptors
                        .extend(descriptors(CapabilityStatus::Available));
                    batch.readings.extend(available_readings(&identity, values));
                }
                Err(error) => {
                    if matches!(error, NvidiaError::Unsupported(_)) {
                        self.terminally_unsupported.insert(identity.clone());
                    }
                    batch.descriptors.extend(descriptors(capability(&error)));
                    batch
                        .readings
                        .extend(unavailable_readings(&identity, &error));
                }
            }
        }
        if self.devices.is_empty() {
            let error = NvidiaError::TemporarilyUnavailable(
                "no NVIDIA device with UUID or PCI plus driver identity was discovered".into(),
            );
            batch.descriptors = descriptors(capability(&error));
            batch.readings = unavailable_readings("gpu:nvidia:discovery", &error);
        }
        batch
    }
}

pub struct NvmlBackend {
    nvml: Nvml,
    driver_identity: String,
}

impl NvmlBackend {
    pub fn initialize() -> Result<Self, NvidiaError> {
        let nvml = Nvml::init().map_err(classify_nvml)?;
        let driver_identity = nvml.sys_driver_version().map_err(classify_nvml)?;
        Ok(Self {
            nvml,
            driver_identity,
        })
    }
}

fn classify_nvml(error: nvml_wrapper::error::NvmlError) -> NvidiaError {
    let text = error.to_string();
    let debug = format!("{error:?}");
    if debug.contains("NotSupported") {
        NvidiaError::Unsupported(text)
    } else if debug.contains("NoPermission") {
        NvidiaError::PermissionDenied(text)
    } else if debug.contains("GpuIsLost")
        || debug.contains("Uninitialized")
        || debug.contains("Unknown")
    {
        NvidiaError::TemporarilyUnavailable(text)
    } else {
        NvidiaError::Error(text)
    }
}

impl NvidiaBackend for NvmlBackend {
    fn discover(&mut self) -> Result<Vec<NvidiaDevice>, NvidiaError> {
        let count = self.nvml.device_count().map_err(classify_nvml)?;
        let mut devices = Vec::new();
        for index in 0..count {
            let device = self.nvml.device_by_index(index).map_err(classify_nvml)?;
            let uuid = device.uuid().ok();
            let pci_address = device.pci_info().ok().map(|pci| pci.bus_id);
            devices.push(NvidiaDevice {
                backend_id: index.to_string(),
                uuid,
                pci_address,
                driver_identity: Some(self.driver_identity.clone()),
            });
        }
        Ok(devices)
    }

    fn read(&mut self, backend_id: &str) -> Result<NvidiaMeasurements, NvidiaError> {
        let index = backend_id
            .parse::<u32>()
            .map_err(|error| NvidiaError::Error(error.to_string()))?;
        let device = self.nvml.device_by_index(index).map_err(classify_nvml)?;
        let utilization = device.utilization_rates().map_err(classify_nvml)?;
        let memory = device.memory_info().map_err(classify_nvml)?;
        Ok(NvidiaMeasurements {
            gpu_util_percent: Some(utilization.gpu as f64),
            vram_used_bytes: Some(memory.used),
            vram_total_bytes: Some(memory.total),
            temperature_c: device
                .temperature(TemperatureSensor::Gpu)
                .ok()
                .map(|value| value as f64),
            graphics_clock_mhz: device
                .clock_info(NvClock::Graphics)
                .ok()
                .map(|value| value as f64),
            sm_clock_mhz: device
                .clock_info(NvClock::SM)
                .ok()
                .map(|value| value as f64),
            memory_clock_mhz: device
                .clock_info(NvClock::Memory)
                .ok()
                .map(|value| value as f64),
            video_clock_mhz: device
                .clock_info(NvClock::Video)
                .ok()
                .map(|value| value as f64),
            source_timestamp_ns: None,
            window_start_ns: None,
        })
    }
}
