//! Hardware-independent telemetry contracts for SIA.
//!
//! Collection, storage, and presentation depend on these types rather than on
//! GUI state or concrete operating-system and vendor providers.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::ffi::{c_int, c_long};
use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};

pub const NATIVE_CLOCK_DOMAIN: &str = "linux_clock_monotonic";
pub const BASELINE_REVISION: &str = "4479bcc19db72f6ad243a87e4b7271496d60d0b7";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MetricId(pub String);

impl MetricId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

impl From<&str> for MetricId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for MetricId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl fmt::Display for MetricId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EntityId(pub String);

impl EntityId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

impl From<&str> for EntityId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for EntityId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeriesKey {
    pub metric_id: MetricId,
    pub entity_id: EntityId,
}

impl SeriesKey {
    pub fn new(metric_id: impl Into<MetricId>, entity_id: impl Into<EntityId>) -> Self {
        Self {
            metric_id: metric_id.into(),
            entity_id: entity_id.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EntityKind {
    System,
    Cpu,
    Gpu,
    Disk,
    Net,
    Process,
    Thread,
    Application,
    Other(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CanonicalUnit {
    Percent,
    Celsius,
    Hertz,
    Bytes,
    BytesPerSecond,
    Seconds,
    Nanoseconds,
    Watts,
    Joules,
    Count,
    Boolean,
    State,
    Custom(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueKind {
    Gauge,
    Counter,
    Rate,
    State,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemporalSemantics {
    PointSample,
    IntervalAverage,
    IntervalDelta,
    CumulativeCounter,
    VendorSampled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderId {
    Sysinfo,
    Procfs,
    Sysfs,
    Nvml,
    Drm,
    Imported,
    Other(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CapabilityState {
    Available,
    Unsupported { reason: String },
    PermissionDenied { reason: String },
    TemporarilyUnavailable { reason: String },
}

impl CapabilityState {
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Available => None,
            Self::Unsupported { reason }
            | Self::PermissionDenied { reason }
            | Self::TemporarilyUnavailable { reason } => Some(reason),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricDescriptor {
    pub metric_id: MetricId,
    pub display_name: String,
    pub entity_kind: EntityKind,
    pub unit: CanonicalUnit,
    pub value_kind: ValueKind,
    pub temporal_semantics: TemporalSemantics,
    pub provider: ProviderId,
    pub capability_state: CapabilityState,
    pub source_resolution_ns: Option<u64>,
    pub source_semantics: String,
    pub comparability_group: Option<String>,
    pub semantics_version: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MetricValue {
    Float(f64),
    Signed(i64),
    Unsigned(u64),
    State(String),
}

impl MetricValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(value) => Some(*value),
            Self::Signed(value) => Some(*value as f64),
            Self::Unsigned(value) => Some(*value as f64),
            Self::State(_) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SampleStatus {
    Ok,
    Stale { reason: String },
    TemporarilyUnavailable { reason: String },
    Error { reason: String },
}

impl SampleStatus {
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Ok => None,
            Self::Stale { reason }
            | Self::TemporarilyUnavailable { reason }
            | Self::Error { reason } => Some(reason),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceTiming {
    pub temporal_semantics: TemporalSemantics,
    pub end_mono_ns: Option<u64>,
    pub window_start_mono_ns: Option<u64>,
    pub source_resolution_ns: Option<u64>,
}

impl SourceTiming {
    pub fn observed(temporal_semantics: TemporalSemantics) -> Self {
        Self {
            temporal_semantics,
            end_mono_ns: None,
            window_start_mono_ns: None,
            source_resolution_ns: None,
        }
    }

    pub fn point(end_mono_ns: u64) -> Self {
        Self {
            temporal_semantics: TemporalSemantics::PointSample,
            end_mono_ns: Some(end_mono_ns),
            window_start_mono_ns: None,
            source_resolution_ns: None,
        }
    }

    pub fn window(
        temporal_semantics: TemporalSemantics,
        start_mono_ns: u64,
        end_mono_ns: u64,
    ) -> Self {
        Self {
            temporal_semantics,
            end_mono_ns: Some(end_mono_ns),
            window_start_mono_ns: Some(start_mono_ns),
            source_resolution_ns: Some(end_mono_ns.saturating_sub(start_mono_ns)),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProviderReading {
    pub metric_id: MetricId,
    pub entity_id: EntityId,
    pub value: Option<MetricValue>,
    pub status: SampleStatus,
    pub timing: SourceTiming,
}

impl ProviderReading {
    pub fn ok(
        metric_id: impl Into<MetricId>,
        entity_id: impl Into<EntityId>,
        value: MetricValue,
        timing: SourceTiming,
    ) -> Self {
        Self {
            metric_id: metric_id.into(),
            entity_id: entity_id.into(),
            value: Some(value),
            status: SampleStatus::Ok,
            timing,
        }
    }

    pub fn unavailable(
        metric_id: impl Into<MetricId>,
        entity_id: impl Into<EntityId>,
        reason: impl Into<String>,
        timing: SourceTiming,
    ) -> Self {
        Self {
            metric_id: metric_id.into(),
            entity_id: entity_id.into(),
            value: None,
            status: SampleStatus::TemporarilyUnavailable {
                reason: reason.into(),
            },
            timing,
        }
    }

    pub fn error(
        metric_id: impl Into<MetricId>,
        entity_id: impl Into<EntityId>,
        reason: impl Into<String>,
        timing: SourceTiming,
    ) -> Self {
        Self {
            metric_id: metric_id.into(),
            entity_id: entity_id.into(),
            value: None,
            status: SampleStatus::Error {
                reason: reason.into(),
            },
            timing,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricSample {
    pub mono_ns: u64,
    pub window_start_mono_ns: Option<u64>,
    pub metric_id: MetricId,
    pub entity_id: EntityId,
    pub value: Option<MetricValue>,
    pub status: SampleStatus,
    pub temporal_semantics: TemporalSemantics,
    pub source_resolution_ns: Option<u64>,
}

impl MetricSample {
    pub fn series_key(&self) -> SeriesKey {
        SeriesKey {
            metric_id: self.metric_id.clone(),
            entity_id: self.entity_id.clone(),
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if matches!(self.status, SampleStatus::Ok) && self.value.is_none() {
            return Err(ModelError::OkSampleWithoutValue);
        }
        if !matches!(self.status, SampleStatus::Ok) && self.value.is_some() {
            return Err(ModelError::UnavailableSampleWithValue);
        }
        if self
            .window_start_mono_ns
            .is_some_and(|start| start > self.mono_ns)
        {
            return Err(ModelError::WindowEndsBeforeItStarts);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelError {
    OkSampleWithoutValue,
    UnavailableSampleWithValue,
    WindowEndsBeforeItStarts,
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::OkSampleWithoutValue => "an ok sample must contain a value",
            Self::UnavailableSampleWithValue => {
                "a non-ok sample must not contain a numeric or state value"
            }
            Self::WindowEndsBeforeItStarts => "sample window starts after its end",
        })
    }
}

impl Error for ModelError {}

pub trait Clock {
    fn domain(&self) -> &'static str;
    fn now_ns(&mut self) -> Result<u64, ClockError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LinuxMonotonicClock;

#[repr(C)]
struct Timespec {
    tv_sec: c_long,
    tv_nsec: c_long,
}

unsafe extern "C" {
    fn clock_gettime(clock_id: c_int, timestamp: *mut Timespec) -> c_int;
}

impl Clock for LinuxMonotonicClock {
    fn domain(&self) -> &'static str {
        NATIVE_CLOCK_DOMAIN
    }

    fn now_ns(&mut self) -> Result<u64, ClockError> {
        const CLOCK_MONOTONIC: c_int = 1;
        let mut timestamp = Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        let result = unsafe { clock_gettime(CLOCK_MONOTONIC, &mut timestamp) };
        if result != 0 || timestamp.tv_sec < 0 || timestamp.tv_nsec < 0 {
            return Err(ClockError);
        }
        Ok((timestamp.tv_sec as u64)
            .saturating_mul(1_000_000_000)
            .saturating_add(timestamp.tv_nsec as u64))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClockError;

impl fmt::Display for ClockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CLOCK_MONOTONIC could not be read")
    }
}

impl Error for ClockError {}

#[derive(Clone, Debug)]
pub struct DeterministicClock {
    readings: VecDeque<u64>,
}

impl DeterministicClock {
    pub fn new(readings: impl IntoIterator<Item = u64>) -> Self {
        Self {
            readings: readings.into_iter().collect(),
        }
    }
}

impl Clock for DeterministicClock {
    fn domain(&self) -> &'static str {
        NATIVE_CLOCK_DOMAIN
    }

    fn now_ns(&mut self) -> Result<u64, ClockError> {
        self.readings.pop_front().ok_or(ClockError)
    }
}

pub trait MetricProvider {
    fn descriptors(&self) -> Vec<MetricDescriptor>;
    fn collect(&mut self, observation_mono_ns: u64) -> Vec<ProviderReading>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct CollectionBatch {
    pub clock_domain: &'static str,
    pub observation_mono_ns: u64,
    pub descriptors: Vec<MetricDescriptor>,
    pub samples: Vec<MetricSample>,
}

pub struct Collector<C, P> {
    clock: C,
    provider: P,
}

impl<C: Clock, P: MetricProvider> Collector<C, P> {
    pub fn new(clock: C, provider: P) -> Self {
        Self { clock, provider }
    }

    pub fn clock(&self) -> &C {
        &self.clock
    }

    pub fn provider(&self) -> &P {
        &self.provider
    }

    pub fn provider_mut(&mut self) -> &mut P {
        &mut self.provider
    }

    pub fn collect(&mut self) -> Result<CollectionBatch, ClockError> {
        let observation_mono_ns = self.clock.now_ns()?;
        let descriptors = self.provider.descriptors();
        let samples = self
            .provider
            .collect(observation_mono_ns)
            .into_iter()
            .map(|reading| MetricSample {
                mono_ns: reading.timing.end_mono_ns.unwrap_or(observation_mono_ns),
                window_start_mono_ns: reading.timing.window_start_mono_ns,
                metric_id: reading.metric_id,
                entity_id: reading.entity_id,
                value: reading.value,
                status: reading.status,
                temporal_semantics: reading.timing.temporal_semantics,
                source_resolution_ns: reading.timing.source_resolution_ns,
            })
            .collect();
        Ok(CollectionBatch {
            clock_domain: self.clock.domain(),
            observation_mono_ns,
            descriptors,
            samples,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct MetricStore {
    descriptors: BTreeMap<MetricId, MetricDescriptor>,
    samples: BTreeMap<SeriesKey, Vec<MetricSample>>,
}

impl MetricStore {
    pub fn ingest(&mut self, batch: CollectionBatch) -> Result<(), ModelError> {
        for descriptor in batch.descriptors {
            self.descriptors
                .insert(descriptor.metric_id.clone(), descriptor);
        }
        for sample in batch.samples {
            sample.validate()?;
            self.samples
                .entry(sample.series_key())
                .or_default()
                .push(sample);
        }
        Ok(())
    }

    pub fn descriptor(&self, metric_id: &MetricId) -> Option<&MetricDescriptor> {
        self.descriptors.get(metric_id)
    }

    pub fn samples(&self, key: &SeriesKey) -> &[MetricSample] {
        self.samples.get(key).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn project(&self) -> PresentationProjection {
        let mut series = Vec::new();
        for (key, samples) in &self.samples {
            let Some(descriptor) = self.descriptors.get(&key.metric_id) else {
                continue;
            };
            if matches!(
                descriptor.capability_state,
                CapabilityState::Unsupported { .. }
            ) {
                continue;
            }
            let records = samples
                .iter()
                .map(|sample| PresentedSample {
                    mono_ns: sample.mono_ns,
                    window_start_mono_ns: sample.window_start_mono_ns,
                    value: sample.value.clone(),
                    status: sample.status.clone(),
                    temporal_semantics: sample.temporal_semantics,
                    source_resolution_ns: sample.source_resolution_ns,
                })
                .collect::<Vec<_>>();
            series.push(PresentedSeries {
                key: key.clone(),
                descriptor: descriptor.clone(),
                segments: numeric_segments(&records),
                records,
            });
        }
        PresentationProjection { series }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PresentedSample {
    pub mono_ns: u64,
    pub window_start_mono_ns: Option<u64>,
    pub value: Option<MetricValue>,
    pub status: SampleStatus,
    pub temporal_semantics: TemporalSemantics,
    pub source_resolution_ns: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PresentedPoint {
    pub mono_ns: u64,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PresentedSeries {
    pub key: SeriesKey,
    pub descriptor: MetricDescriptor,
    pub records: Vec<PresentedSample>,
    pub segments: Vec<Vec<PresentedPoint>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PresentationProjection {
    pub series: Vec<PresentedSeries>,
}

fn numeric_segments(records: &[PresentedSample]) -> Vec<Vec<PresentedPoint>> {
    let mut segments = Vec::new();
    let mut current = Vec::new();
    for record in records {
        let numeric = if matches!(record.status, SampleStatus::Ok) {
            record.value.as_ref().and_then(MetricValue::as_f64)
        } else {
            None
        };
        if let Some(value) = numeric {
            current.push(PresentedPoint {
                mono_ns: record.mono_ns,
                value,
            });
        } else if !current.is_empty() {
            segments.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

pub trait CpuSnapshotSource {
    fn refresh_cpu_utilization(&mut self) -> Result<Vec<(EntityId, f64)>, String>;
}

pub struct CpuUtilizationCollector<C, S> {
    clock: C,
    source: S,
    previous_observation_ns: Option<u64>,
    metric_id: MetricId,
}

impl<C: Clock, S: CpuSnapshotSource> CpuUtilizationCollector<C, S> {
    pub fn new(clock: C, source: S) -> Self {
        Self {
            clock,
            source,
            previous_observation_ns: None,
            metric_id: MetricId::from("cpu.utilization"),
        }
    }

    pub fn descriptor(&self) -> MetricDescriptor {
        MetricDescriptor {
            metric_id: self.metric_id.clone(),
            display_name: "CPU utilization".to_owned(),
            entity_kind: EntityKind::Cpu,
            unit: CanonicalUnit::Percent,
            value_kind: ValueKind::Gauge,
            temporal_semantics: TemporalSemantics::IntervalAverage,
            provider: ProviderId::Sysinfo,
            capability_state: CapabilityState::Available,
            source_resolution_ns: None,
            source_semantics: "CPU busy time averaged between successive refresh observations"
                .to_owned(),
            comparability_group: Some("linux.cpu.utilization".to_owned()),
            semantics_version: 1,
        }
    }

    pub fn refresh(&mut self) -> Result<Vec<MetricSample>, ClockError> {
        let observation_ns = self.clock.now_ns()?;
        let values = self.source.refresh_cpu_utilization();
        let start = self.previous_observation_ns;
        self.previous_observation_ns = Some(observation_ns);
        let Some(window_start_mono_ns) = start else {
            return Ok(Vec::new());
        };
        let resolution = observation_ns.saturating_sub(window_start_mono_ns);
        Ok(match values {
            Ok(values) => values
                .into_iter()
                .map(|(entity_id, value)| MetricSample {
                    mono_ns: observation_ns,
                    window_start_mono_ns: Some(window_start_mono_ns),
                    metric_id: self.metric_id.clone(),
                    entity_id,
                    value: Some(MetricValue::Float(value)),
                    status: SampleStatus::Ok,
                    temporal_semantics: TemporalSemantics::IntervalAverage,
                    source_resolution_ns: Some(resolution),
                })
                .collect(),
            Err(reason) => vec![MetricSample {
                mono_ns: observation_ns,
                window_start_mono_ns: Some(window_start_mono_ns),
                metric_id: self.metric_id.clone(),
                entity_id: EntityId::from("system"),
                value: None,
                status: SampleStatus::Error { reason },
                temporal_semantics: TemporalSemantics::IntervalAverage,
                source_resolution_ns: Some(resolution),
            }],
        })
    }

    pub fn into_parts(self) -> (C, S) {
        (self.clock, self.source)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HwmonSensor {
    pub entity_id: EntityId,
    pub metric_id: MetricId,
    pub device_name: String,
    pub label: String,
    pub input_path: PathBuf,
    pub stable_parent: String,
}

pub fn hwmon_entity_id(stable_parent: &str) -> EntityId {
    EntityId::new(format!("hwmon:{}", normalize_stable_path(stable_parent)))
}

pub fn hwmon_series_key(stable_parent: &str, input_filename: &str) -> SeriesKey {
    SeriesKey::new(
        MetricId::new(format!("temperature.{input_filename}")),
        hwmon_entity_id(stable_parent),
    )
}

pub fn discover_hwmon(root: impl AsRef<Path>) -> Vec<HwmonSensor> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut sensors = Vec::new();
    for entry in entries.flatten() {
        let base = entry.path();
        if !base.is_dir() {
            continue;
        }
        let device_name = fs::read_to_string(base.join("name"))
            .unwrap_or_default()
            .trim()
            .to_owned();
        let stable_parent = stable_hwmon_parent(&base);
        let Ok(files) = fs::read_dir(&base) else {
            continue;
        };
        for file in files.flatten() {
            let input_path = file.path();
            let Some(filename) = input_path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if !filename.starts_with("temp") || !filename.ends_with("_input") {
                continue;
            }
            let label = fs::read_to_string(base.join(filename.replace("_input", "_label")))
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| device_name.clone());
            sensors.push(HwmonSensor {
                entity_id: hwmon_entity_id(&stable_parent),
                metric_id: MetricId::new(format!("temperature.{filename}")),
                device_name: device_name.clone(),
                label,
                input_path,
                stable_parent: stable_parent.clone(),
            });
        }
    }
    sensors.sort_by(|left, right| {
        (&left.entity_id, &left.metric_id).cmp(&(&right.entity_id, &right.metric_id))
    });
    sensors
}

fn stable_hwmon_parent(base: &Path) -> String {
    let device = base.join("device");
    let resolved = fs::canonicalize(&device)
        .or_else(|_| fs::read_link(&device).map(|target| base.join(target)))
        .unwrap_or_else(|_| base.to_path_buf());
    stable_path_suffix(&resolved)
}

fn stable_path_suffix(path: &Path) -> String {
    let components = path.components().collect::<Vec<_>>();
    let start = components.iter().position(|component| match component {
        Component::Normal(value) => {
            let value = value.to_string_lossy();
            value == "devices"
                || value.starts_with("pci")
                || value.starts_with("platform")
                || value.starts_with("virtual")
        }
        _ => false,
    });
    let selected = start
        .map(|index| &components[index..])
        .unwrap_or(&components);
    normalize_stable_path(
        &selected
            .iter()
            .filter_map(|component| match component {
                Component::Normal(value) => Some(value.to_string_lossy()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("/"),
    )
}

fn normalize_stable_path(value: &str) -> String {
    value
        .trim()
        .trim_matches('/')
        .replace('\\', "/")
        .to_ascii_lowercase()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NvidiaIdentity {
    pub uuid: Option<String>,
    pub pci_address: String,
    pub driver_identity: String,
}

impl NvidiaIdentity {
    pub fn durable_entity_id(&self) -> EntityId {
        match self
            .uuid
            .as_deref()
            .map(str::trim)
            .filter(|uuid| !uuid.is_empty())
        {
            Some(uuid) => EntityId::new(format!("nvidia:uuid:{}", uuid.to_ascii_lowercase())),
            None => EntityId::new(format!(
                "nvidia:pci:{}:driver:{}",
                self.pci_address.to_ascii_lowercase(),
                self.driver_identity.to_ascii_lowercase()
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NvidiaDevice {
    pub acquisition_index: u32,
    pub entity_id: EntityId,
    pub identity: NvidiaIdentity,
    pub metrics: Vec<MetricDescriptor>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NvidiaDiscoveryFailure {
    pub acquisition_index: u32,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NvidiaInventory {
    pub devices: Vec<NvidiaDevice>,
    pub failures: Vec<NvidiaDiscoveryFailure>,
}

pub trait NvidiaBackend {
    fn device_indices(&self) -> Vec<u32>;
    fn identity(&self, acquisition_index: u32) -> Result<NvidiaIdentity, String>;
    fn descriptors(&self, acquisition_index: u32) -> Result<Vec<MetricDescriptor>, String>;

    fn begin_device_sample(&mut self, _acquisition_index: u32) -> Result<(), String> {
        Ok(())
    }

    fn read_metric(
        &mut self,
        acquisition_index: u32,
        metric_id: &MetricId,
        observation_mono_ns: u64,
    ) -> Result<ProviderReading, String>;
}

pub fn discover_nvidia<B: NvidiaBackend>(backend: &B) -> NvidiaInventory {
    let mut inventory = NvidiaInventory::default();
    for index in backend.device_indices() {
        let identity = match backend.identity(index) {
            Ok(identity) => identity,
            Err(reason) => {
                inventory.failures.push(NvidiaDiscoveryFailure {
                    acquisition_index: index,
                    reason,
                });
                continue;
            }
        };
        let metrics = match backend.descriptors(index) {
            Ok(metrics) => metrics,
            Err(reason) => {
                inventory.failures.push(NvidiaDiscoveryFailure {
                    acquisition_index: index,
                    reason,
                });
                continue;
            }
        };
        inventory.devices.push(NvidiaDevice {
            acquisition_index: index,
            entity_id: identity.durable_entity_id(),
            identity,
            metrics,
        });
    }
    inventory
        .devices
        .sort_by(|left, right| left.entity_id.cmp(&right.entity_id));
    inventory
}

pub struct NvidiaCollector<B> {
    backend: B,
    inventory: NvidiaInventory,
}

impl<B: NvidiaBackend> NvidiaCollector<B> {
    pub fn new(backend: B) -> Self {
        let inventory = discover_nvidia(&backend);
        Self { backend, inventory }
    }

    pub fn inventory(&self) -> &NvidiaInventory {
        &self.inventory
    }

    pub fn rediscover(&mut self) {
        self.inventory = discover_nvidia(&self.backend);
    }

    pub fn sample(&mut self, observation_mono_ns: u64) -> Vec<ProviderReading> {
        let mut output = Vec::new();
        for device in &self.inventory.devices {
            if let Err(reason) = self.backend.begin_device_sample(device.acquisition_index) {
                for descriptor in &device.metrics {
                    output.push(ProviderReading::error(
                        descriptor.metric_id.clone(),
                        device.entity_id.clone(),
                        reason.clone(),
                        SourceTiming::observed(descriptor.temporal_semantics),
                    ));
                }
                continue;
            }
            for descriptor in &device.metrics {
                match self.backend.read_metric(
                    device.acquisition_index,
                    &descriptor.metric_id,
                    observation_mono_ns,
                ) {
                    Ok(mut reading) => {
                        reading.entity_id = device.entity_id.clone();
                        output.push(reading);
                    }
                    Err(reason) => output.push(ProviderReading::error(
                        descriptor.metric_id.clone(),
                        device.entity_id.clone(),
                        reason,
                        SourceTiming::observed(descriptor.temporal_semantics),
                    )),
                }
            }
        }
        output
    }

    pub fn into_backend(self) -> B {
        self.backend
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegendPlacement {
    Footer,
    Side,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThermalIndicator {
    Normal,
    Warning,
    Hot,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TemperatureTrace {
    pub id: String,
    pub group: String,
    pub label: String,
    pub visible: bool,
    pub warning_c: f64,
    pub hot_c: f64,
}

impl TemperatureTrace {
    pub fn indicator(&self, reading_c: f64) -> ThermalIndicator {
        if reading_c >= self.hot_c {
            ThermalIndicator::Hot
        } else if reading_c >= self.warning_c {
            ThermalIndicator::Warning
        } else {
            ThermalIndicator::Normal
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MonitorRegressionProjection {
    pub attributed_baseline: &'static str,
    pub display_window_seconds: f64,
    pub display_window_range: (f64, f64),
    pub legend_placement: LegendPlacement,
    pub applied_font_size: f32,
    pub pending_font_size: f32,
    pub font_is_white: bool,
    pub live_font_preview: bool,
    pub cpu_frequency_visibility: BTreeMap<String, bool>,
    pub temperatures: Vec<TemperatureTrace>,
    pub nvidia_graphics_clock_visible: bool,
    pub nvidia_sm_clock_visible: bool,
    pub nvidia_memory_clock_visible: bool,
    pub nvidia_video_clock_visible: bool,
    pub nvidia_effective_memory_clock: bool,
}

impl MonitorRegressionProjection {
    pub fn baseline(
        cpu_frequency_ids: impl IntoIterator<Item = String>,
        mut temperatures: Vec<TemperatureTrace>,
        nvidia_enabled: bool,
    ) -> Self {
        sort_temperature_groups(&mut temperatures);
        choose_preferred_temperature_per_group(&mut temperatures);
        Self {
            attributed_baseline: BASELINE_REVISION,
            display_window_seconds: 120.0,
            display_window_range: (30.0, 900.0),
            legend_placement: LegendPlacement::Footer,
            applied_font_size: 14.0,
            pending_font_size: 14.0,
            font_is_white: true,
            live_font_preview: false,
            cpu_frequency_visibility: cpu_frequency_ids.into_iter().map(|id| (id, true)).collect(),
            temperatures,
            nvidia_graphics_clock_visible: nvidia_enabled,
            nvidia_sm_clock_visible: nvidia_enabled,
            nvidia_memory_clock_visible: nvidia_enabled,
            nvidia_video_clock_visible: false,
            nvidia_effective_memory_clock: false,
        }
    }

    pub fn x_bounds(&self, current_seconds: f64) -> (f64, f64) {
        (
            (current_seconds - self.display_window_seconds).max(0.0),
            current_seconds,
        )
    }

    pub fn set_display_window(&mut self, seconds: f64) {
        self.display_window_seconds =
            seconds.clamp(self.display_window_range.0, self.display_window_range.1);
    }

    pub fn set_cpu_frequency_visible(&mut self, id: &str, visible: bool) {
        if let Some(value) = self.cpu_frequency_visibility.get_mut(id) {
            *value = visible;
        }
    }

    pub fn set_temperature_visible(&mut self, id: &str, visible: bool) {
        if let Some(trace) = self.temperatures.iter_mut().find(|trace| trace.id == id) {
            trace.visible = visible;
        }
    }

    pub fn visible_legend_entries(&self) -> Vec<String> {
        let mut entries = self
            .temperatures
            .iter()
            .filter(|trace| trace.visible)
            .map(|trace| trace.label.clone())
            .collect::<Vec<_>>();
        entries.extend(
            self.cpu_frequency_visibility
                .iter()
                .filter(|(_, visible)| **visible)
                .map(|(id, _)| id.clone()),
        );
        entries
    }

    pub fn set_pending_font_size(&mut self, value: f32) {
        self.pending_font_size = value;
        if self.live_font_preview {
            self.applied_font_size = value;
        }
    }

    pub fn set_live_font_preview(&mut self, enabled: bool) {
        self.live_font_preview = enabled;
        if enabled {
            self.applied_font_size = self.pending_font_size;
        }
    }

    pub fn apply_font(&mut self) {
        self.applied_font_size = self.pending_font_size;
    }

    pub fn displayed_memory_clock_mhz(&self, source_mhz: f64) -> f64 {
        if self.nvidia_effective_memory_clock {
            source_mhz * 2.0
        } else {
            source_mhz
        }
    }
}

fn sort_temperature_groups(temperatures: &mut [TemperatureTrace]) {
    fn rank(group: &str) -> u8 {
        match group.to_ascii_lowercase().as_str() {
            "cpu" => 0,
            "gpu" => 1,
            "nvme" | "nvme ssd" => 2,
            "memory" | "ram" | "ramspd" => 3,
            "wi-fi" | "wifi" => 4,
            "ethernet" | "eth" => 5,
            _ => 6,
        }
    }
    temperatures.sort_by(|left, right| {
        rank(&left.group)
            .cmp(&rank(&right.group))
            .then(left.group.cmp(&right.group))
            .then(preference_rank(&left.label).cmp(&preference_rank(&right.label)))
            .then(left.label.cmp(&right.label))
            .then(left.id.cmp(&right.id))
    });
}

fn preference_rank(label: &str) -> u8 {
    let lower = label.to_ascii_lowercase();
    if lower.contains("package") {
        0
    } else if lower.contains("composite") {
        1
    } else if lower.contains("system") {
        2
    } else {
        3
    }
}

fn choose_preferred_temperature_per_group(temperatures: &mut [TemperatureTrace]) {
    let mut selected = BTreeSet::new();
    for trace in temperatures {
        trace.visible = selected.insert(trace.group.to_ascii_lowercase());
    }
}
