use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    error::Error,
    ffi::{c_int, c_long},
    fmt, fs,
    path::{Component, Path, PathBuf},
};
pub const NATIVE_CLOCK_DOMAIN: &str = "linux_clock_monotonic";
pub const BASELINE_REVISION: &str = "4479bcc19db72f6ad243a87e4b7271496d60d0b7";
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MetricId(pub String);
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EntityId(pub String);
macro_rules! id_impl {
    ($t:ty) => {
        impl $t {
            pub fn new(v: impl Into<String>) -> Self {
                Self(v.into())
            }
        }
        impl From<&str> for $t {
            fn from(v: &str) -> Self {
                Self::new(v)
            }
        }
        impl From<String> for $t {
            fn from(v: String) -> Self {
                Self(v)
            }
        }
        impl fmt::Display for $t {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}
id_impl!(MetricId);
id_impl!(EntityId);
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeriesKey {
    pub metric_id: MetricId,
    pub entity_id: EntityId,
}
impl SeriesKey {
    pub fn new(m: impl Into<MetricId>, e: impl Into<EntityId>) -> Self {
        Self {
            metric_id: m.into(),
            entity_id: e.into(),
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
            Self::Float(v) => Some(*v),
            Self::Signed(v) => Some(*v as f64),
            Self::Unsigned(v) => Some(*v as f64),
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
    pub fn observed(s: TemporalSemantics) -> Self {
        Self {
            temporal_semantics: s,
            end_mono_ns: None,
            window_start_mono_ns: None,
            source_resolution_ns: None,
        }
    }
    pub fn at(s: TemporalSemantics, end: u64) -> Self {
        Self {
            temporal_semantics: s,
            end_mono_ns: Some(end),
            window_start_mono_ns: None,
            source_resolution_ns: None,
        }
    }
    pub fn point(end: u64) -> Self {
        Self::at(TemporalSemantics::PointSample, end)
    }
    pub fn window(s: TemporalSemantics, start: u64, end: u64) -> Self {
        Self {
            temporal_semantics: s,
            end_mono_ns: Some(end),
            window_start_mono_ns: Some(start),
            source_resolution_ns: None,
        }
    }
    pub fn with_source_resolution(mut self, v: u64) -> Self {
        self.source_resolution_ns = Some(v);
        self
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
        m: impl Into<MetricId>,
        e: impl Into<EntityId>,
        v: MetricValue,
        t: SourceTiming,
    ) -> Self {
        Self {
            metric_id: m.into(),
            entity_id: e.into(),
            value: Some(v),
            status: SampleStatus::Ok,
            timing: t,
        }
    }
    pub fn unavailable(
        m: impl Into<MetricId>,
        e: impl Into<EntityId>,
        r: impl Into<String>,
        t: SourceTiming,
    ) -> Self {
        Self {
            metric_id: m.into(),
            entity_id: e.into(),
            value: None,
            status: SampleStatus::TemporarilyUnavailable { reason: r.into() },
            timing: t,
        }
    }
    pub fn error(
        m: impl Into<MetricId>,
        e: impl Into<EntityId>,
        r: impl Into<String>,
        t: SourceTiming,
    ) -> Self {
        Self {
            metric_id: m.into(),
            entity_id: e.into(),
            value: None,
            status: SampleStatus::Error { reason: r.into() },
            timing: t,
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
        SeriesKey::new(self.metric_id.clone(), self.entity_id.clone())
    }
    pub fn validate(&self) -> Result<(), ModelError> {
        if matches!(self.status, SampleStatus::Ok) && self.value.is_none() {
            return Err(ModelError::OkSampleWithoutValue);
        }
        if !matches!(self.status, SampleStatus::Ok) && self.value.is_some() {
            return Err(ModelError::UnavailableSampleWithValue);
        }
        if self
            .value
            .as_ref()
            .and_then(MetricValue::as_f64)
            .is_some_and(|v| !v.is_finite())
        {
            return Err(ModelError::NonFiniteNumericValue);
        }
        if self.window_start_mono_ns.is_some_and(|s| s > self.mono_ns) {
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
    NonFiniteNumericValue,
}
impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::OkSampleWithoutValue => "an ok sample must contain a value",
            Self::UnavailableSampleWithValue => "a non-ok sample must not contain a value",
            Self::WindowEndsBeforeItStarts => "sample window starts after its end",
            Self::NonFiniteNumericValue => "numeric samples must be finite",
        })
    }
}
impl Error for ModelError {}
pub trait Clock {
    fn domain(&self) -> &'static str;
    fn now_ns(&mut self) -> Result<u64, ClockError>;
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClockError;
impl fmt::Display for ClockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CLOCK_MONOTONIC could not be read")
    }
}
impl Error for ClockError {}
#[derive(Clone, Copy, Debug, Default)]
pub struct LinuxMonotonicClock;
#[repr(C)]
struct Timespec {
    tv_sec: c_long,
    tv_nsec: c_long,
}
extern "C" {
    fn clock_gettime(id: c_int, t: *mut Timespec) -> c_int;
}
impl Clock for LinuxMonotonicClock {
    fn domain(&self) -> &'static str {
        NATIVE_CLOCK_DOMAIN
    }
    fn now_ns(&mut self) -> Result<u64, ClockError> {
        let mut t = Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        if unsafe { clock_gettime(1, &mut t) } != 0 || t.tv_sec < 0 || t.tv_nsec < 0 {
            Err(ClockError)
        } else {
            Ok((t.tv_sec as u64)
                .saturating_mul(1_000_000_000)
                .saturating_add(t.tv_nsec as u64))
        }
    }
}
#[derive(Clone, Debug)]
pub struct DeterministicClock {
    readings: VecDeque<u64>,
}
impl DeterministicClock {
    pub fn new(v: impl IntoIterator<Item = u64>) -> Self {
        Self {
            readings: v.into_iter().collect(),
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
        let now = self.clock.now_ns()?;
        let samples = self
            .provider
            .collect(now)
            .into_iter()
            .map(|r| MetricSample {
                mono_ns: r.timing.end_mono_ns.unwrap_or(now),
                window_start_mono_ns: r.timing.window_start_mono_ns,
                metric_id: r.metric_id,
                entity_id: r.entity_id,
                value: r.value,
                status: r.status,
                temporal_semantics: r.timing.temporal_semantics,
                source_resolution_ns: r.timing.source_resolution_ns,
            })
            .collect();
        Ok(CollectionBatch {
            clock_domain: self.clock.domain(),
            observation_mono_ns: now,
            descriptors: self.provider.descriptors(),
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
    pub fn ingest(&mut self, b: CollectionBatch) -> Result<(), ModelError> {
        for d in b.descriptors {
            self.descriptors.insert(d.metric_id.clone(), d);
        }
        for s in b.samples {
            s.validate()?;
            self.samples.entry(s.series_key()).or_default().push(s);
        }
        Ok(())
    }
    pub fn descriptor(&self, id: &MetricId) -> Option<&MetricDescriptor> {
        self.descriptors.get(id)
    }
    pub fn samples(&self, k: &SeriesKey) -> &[MetricSample] {
        self.samples.get(k).map(Vec::as_slice).unwrap_or(&[])
    }
    pub fn project(&self) -> PresentationProjection {
        let series = self
            .samples
            .iter()
            .filter_map(|(k, samples)| {
                let d = self.descriptors.get(&k.metric_id)?;
                if matches!(d.capability_state, CapabilityState::Unsupported { .. }) {
                    return None;
                }
                let records: Vec<_> = samples
                    .iter()
                    .map(|s| PresentedSample {
                        mono_ns: s.mono_ns,
                        window_start_mono_ns: s.window_start_mono_ns,
                        value: s.value.clone(),
                        status: s.status.clone(),
                        temporal_semantics: s.temporal_semantics,
                        source_resolution_ns: s.source_resolution_ns,
                    })
                    .collect();
                Some(PresentedSeries {
                    key: k.clone(),
                    descriptor: d.clone(),
                    segments: numeric_segments(&records),
                    records,
                })
            })
            .collect();
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
fn numeric_segments(r: &[PresentedSample]) -> Vec<Vec<PresentedPoint>> {
    let (mut out, mut current) = (Vec::new(), Vec::new());
    for x in r {
        let v = if matches!(x.status, SampleStatus::Ok) {
            x.value
                .as_ref()
                .and_then(MetricValue::as_f64)
                .filter(|v| v.is_finite())
        } else {
            None
        };
        if let Some(value) = v {
            current.push(PresentedPoint {
                mono_ns: x.mono_ns,
                value,
            })
        } else if !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current)
    }
    out
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
            metric_id: "cpu.utilization".into(),
        }
    }
    pub fn descriptor(&self) -> MetricDescriptor {
        MetricDescriptor {
            metric_id: self.metric_id.clone(),
            display_name: "CPU utilization".into(),
            entity_kind: EntityKind::Cpu,
            unit: CanonicalUnit::Percent,
            value_kind: ValueKind::Gauge,
            temporal_semantics: TemporalSemantics::IntervalAverage,
            provider: ProviderId::Sysinfo,
            capability_state: CapabilityState::Available,
            source_resolution_ns: None,
            source_semantics:
                "CPU busy time averaged between successive CLOCK_MONOTONIC refresh observations"
                    .into(),
            comparability_group: Some("linux.cpu.utilization".into()),
            semantics_version: 1,
        }
    }
    pub fn refresh(&mut self) -> Result<Vec<MetricSample>, ClockError> {
        let end = self.clock.now_ns()?;
        let values = self.source.refresh_cpu_utilization();
        let Some(start) = self.previous_observation_ns.replace(end) else {
            return Ok(Vec::new());
        };
        Ok(match values {
            Ok(v) => v
                .into_iter()
                .map(|(entity_id, value)| MetricSample {
                    mono_ns: end,
                    window_start_mono_ns: Some(start),
                    metric_id: self.metric_id.clone(),
                    entity_id,
                    value: Some(MetricValue::Float(value)),
                    status: SampleStatus::Ok,
                    temporal_semantics: TemporalSemantics::IntervalAverage,
                    source_resolution_ns: None,
                })
                .collect(),
            Err(reason) => vec![MetricSample {
                mono_ns: end,
                window_start_mono_ns: Some(start),
                metric_id: self.metric_id.clone(),
                entity_id: "system".into(),
                value: None,
                status: SampleStatus::Error { reason },
                temporal_semantics: TemporalSemantics::IntervalAverage,
                source_resolution_ns: None,
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
pub fn hwmon_entity_id(p: &str) -> EntityId {
    EntityId::new(format!("hwmon:{}", normalize(p)))
}
pub fn hwmon_series_key(p: &str, f: &str) -> SeriesKey {
    SeriesKey::new(
        MetricId::new(format!("temperature.{f}")),
        hwmon_entity_id(p),
    )
}
fn normalize(v: &str) -> String {
    v.trim()
        .trim_matches('/')
        .replace('\\', "/")
        .to_ascii_lowercase()
}
fn stable_parent(base: &Path) -> String {
    let p = fs::canonicalize(base.join("device")).unwrap_or_else(|_| base.to_path_buf());
    let c: Vec<_> = p.components().collect();
    let i=c.iter().position(|x|matches!(x,Component::Normal(v)if{let s=v.to_string_lossy();s=="devices"||s.starts_with("pci")||s.starts_with("platform")||s.starts_with("virtual")})).unwrap_or(0);
    normalize(
        &c[i..]
            .iter()
            .filter_map(|x| match x {
                Component::Normal(v) => Some(v.to_string_lossy()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("/"),
    )
}
pub fn discover_hwmon(root: impl AsRef<Path>) -> Vec<HwmonSensor> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for e in entries.flatten() {
        let b = e.path();
        let name = fs::read_to_string(b.join("name"))
            .unwrap_or_default()
            .trim()
            .to_owned();
        let parent = stable_parent(&b);
        let Ok(files) = fs::read_dir(&b) else {
            continue;
        };
        for f in files.flatten() {
            let p = f.path();
            let Some(n) = p.file_name().and_then(|v| v.to_str()) else {
                continue;
            };
            if n.starts_with("temp") && n.ends_with("_input") {
                let label = fs::read_to_string(b.join(n.replace("_input", "_label")))
                    .ok()
                    .map(|v| v.trim().to_owned())
                    .filter(|v| !v.is_empty())
                    .unwrap_or_else(|| name.clone());
                out.push(HwmonSensor {
                    entity_id: hwmon_entity_id(&parent),
                    metric_id: MetricId::new(format!("temperature.{n}")),
                    device_name: name.clone(),
                    label,
                    input_path: p,
                    stable_parent: parent.clone(),
                });
            }
        }
    }
    out.sort_by(|a, b| (&a.entity_id, &a.metric_id).cmp(&(&b.entity_id, &b.metric_id)));
    out
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NvidiaIdentity {
    pub uuid: Option<String>,
    pub pci_address: String,
    pub driver_identity: String,
}
impl NvidiaIdentity {
    pub fn durable_entity_id(&self) -> EntityId {
        if let Some(u) = self
            .uuid
            .as_deref()
            .map(str::trim)
            .filter(|u| !u.is_empty())
        {
            EntityId::new(format!("nvidia:uuid:{}", u.to_ascii_lowercase()))
        } else {
            EntityId::new(format!(
                "nvidia:pci:{}:driver:{}",
                self.pci_address.to_ascii_lowercase(),
                self.driver_identity.to_ascii_lowercase()
            ))
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
    fn identity(&self, i: u32) -> Result<NvidiaIdentity, String>;
    fn descriptors(&self, i: u32) -> Result<Vec<MetricDescriptor>, String>;
    fn begin_device_sample(&mut self, _i: u32) -> Result<(), String> {
        Ok(())
    }
    fn read_metric(&mut self, i: u32, m: &MetricId, t: u64) -> Result<ProviderReading, String>;
}
pub fn discover_nvidia<B: NvidiaBackend>(b: &B) -> NvidiaInventory {
    let mut out = NvidiaInventory::default();
    for i in b.device_indices() {
        let identity = match b.identity(i) {
            Ok(v) => v,
            Err(reason) => {
                out.failures.push(NvidiaDiscoveryFailure {
                    acquisition_index: i,
                    reason,
                });
                continue;
            }
        };
        let metrics = match b.descriptors(i) {
            Ok(v) => v,
            Err(reason) => {
                out.failures.push(NvidiaDiscoveryFailure {
                    acquisition_index: i,
                    reason,
                });
                continue;
            }
        };
        out.devices.push(NvidiaDevice {
            acquisition_index: i,
            entity_id: identity.durable_entity_id(),
            identity,
            metrics,
        });
    }
    out.devices.sort_by(|a, b| a.entity_id.cmp(&b.entity_id));
    out
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
        self.inventory = discover_nvidia(&self.backend)
    }
    pub fn sample(&mut self, t: u64) -> Vec<ProviderReading> {
        let mut out = Vec::new();
        for d in &self.inventory.devices {
            if let Err(reason) = self.backend.begin_device_sample(d.acquisition_index) {
                for m in &d.metrics {
                    out.push(ProviderReading::error(
                        m.metric_id.clone(),
                        d.entity_id.clone(),
                        reason.clone(),
                        SourceTiming::at(m.temporal_semantics, t),
                    ));
                }
                continue;
            }
            for m in &d.metrics {
                match self
                    .backend
                    .read_metric(d.acquisition_index, &m.metric_id, t)
                {
                    Ok(mut r) => {
                        r.entity_id = d.entity_id.clone();
                        out.push(r)
                    }
                    Err(reason) => out.push(ProviderReading::error(
                        m.metric_id.clone(),
                        d.entity_id.clone(),
                        reason,
                        SourceTiming::at(m.temporal_semantics, t),
                    )),
                }
            }
        }
        out
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
    pub fn indicator(&self, v: f64) -> ThermalIndicator {
        if v >= self.hot_c {
            ThermalIndicator::Hot
        } else if v >= self.warning_c {
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
        ids: impl IntoIterator<Item = String>,
        mut temperatures: Vec<TemperatureTrace>,
        n: bool,
    ) -> Self {
        temperatures.sort_by_key(|t| {
            (
                group_rank(&t.group),
                preference_rank(&t.label),
                t.label.clone(),
                t.id.clone(),
            )
        });
        let mut groups = BTreeSet::new();
        for t in &mut temperatures {
            t.visible = groups.insert(t.group.to_ascii_lowercase());
        }
        Self {
            attributed_baseline: BASELINE_REVISION,
            display_window_seconds: 120.0,
            display_window_range: (30.0, 900.0),
            legend_placement: LegendPlacement::Footer,
            applied_font_size: 14.0,
            pending_font_size: 14.0,
            font_is_white: true,
            live_font_preview: false,
            cpu_frequency_visibility: ids.into_iter().map(|x| (x, true)).collect(),
            temperatures,
            nvidia_graphics_clock_visible: n,
            nvidia_sm_clock_visible: n,
            nvidia_memory_clock_visible: n,
            nvidia_video_clock_visible: false,
            nvidia_effective_memory_clock: false,
        }
    }
    pub fn x_bounds(&self, now: f64) -> (f64, f64) {
        ((now - self.display_window_seconds).max(0.0), now)
    }
    pub fn set_display_window(&mut self, v: f64) {
        self.display_window_seconds =
            v.clamp(self.display_window_range.0, self.display_window_range.1)
    }
    pub fn set_cpu_frequency_visible(&mut self, id: &str, v: bool) {
        if let Some(x) = self.cpu_frequency_visibility.get_mut(id) {
            *x = v
        }
    }
    pub fn set_temperature_visible(&mut self, id: &str, v: bool) {
        if let Some(x) = self.temperatures.iter_mut().find(|x| x.id == id) {
            x.visible = v
        }
    }
    pub fn visible_legend_entries(&self) -> Vec<String> {
        let mut v: Vec<_> = self
            .temperatures
            .iter()
            .filter(|x| x.visible)
            .map(|x| x.label.clone())
            .collect();
        v.extend(
            self.cpu_frequency_visibility
                .iter()
                .filter(|(_, x)| **x)
                .map(|(x, _)| x.clone()),
        );
        v
    }
    pub fn set_pending_font_size(&mut self, v: f32) {
        self.pending_font_size = v;
        if self.live_font_preview {
            self.applied_font_size = v
        }
    }
    pub fn set_live_font_preview(&mut self, v: bool) {
        self.live_font_preview = v;
        if v {
            self.applied_font_size = self.pending_font_size
        }
    }
    pub fn apply_font(&mut self) {
        self.applied_font_size = self.pending_font_size
    }
    pub fn displayed_memory_clock_mhz(&self, v: f64) -> f64 {
        if self.nvidia_effective_memory_clock {
            v * 2.0
        } else {
            v
        }
    }
}
fn group_rank(g: &str) -> u8 {
    match g.to_ascii_lowercase().as_str() {
        "cpu" => 0,
        "gpu" => 1,
        "nvme" | "nvme ssd" => 2,
        "memory" | "ram" | "ramspd" => 3,
        "wi-fi" | "wifi" => 4,
        "ethernet" | "eth" => 5,
        _ => 6,
    }
}
fn preference_rank(v: &str) -> u8 {
    let v = v.to_ascii_lowercase();
    if v.contains("package") {
        0
    } else if v.contains("composite") {
        1
    } else if v.contains("system") {
        2
    } else {
        3
    }
}
