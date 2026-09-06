#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MetricId(pub String);

impl From<&str> for MetricId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for MetricId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EntityId(pub String);

impl From<&str> for EntityId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for EntityId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntityKind {
    System,
    Cpu,
    Gpu,
    Disk,
    Network,
    Process,
    Thread,
    Application,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CanonicalUnit {
    Percent,
    Bytes,
    Celsius,
    Hertz,
    Nanoseconds,
    Seconds,
    Count,
    BytesPerSecond,
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
    PointSampled,
    IntervalAverage,
    IntervalDelta,
    CumulativeCounter,
    VendorSampled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CapabilityState {
    Available,
    Unsupported { reason: Option<String> },
    PermissionDenied { reason: Option<String> },
    TemporarilyUnavailable { reason: Option<String> },
}

impl CapabilityState {
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MonotonicTimestamp(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceResolution {
    pub nanoseconds: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricDescriptor {
    pub metric_id: MetricId,
    pub display_name: String,
    pub entity_kind: EntityKind,
    pub unit: CanonicalUnit,
    pub value_kind: ValueKind,
    pub temporal_semantics: TemporalSemantics,
    pub provider: String,
    pub capability_status: CapabilityState,
    pub source_resolution: Option<SourceResolution>,
    pub source_semantics: String,
    pub comparability_group: Option<String>,
    pub semantics_version: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MetricValue {
    F64(f64),
    I64(i64),
    U64(u64),
    State(String),
}

impl MetricValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F64(value) => Some(*value),
            Self::I64(value) => Some(*value as f64),
            Self::U64(value) => Some(*value as f64),
            Self::State(_) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SampleStatus {
    Ok,
    Stale { reason: Option<String> },
    TemporarilyUnavailable { reason: Option<String> },
    Error { message: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricSample {
    pub observation_time: MonotonicTimestamp,
    pub interval_start: Option<MonotonicTimestamp>,
    pub metric_id: MetricId,
    pub entity_id: EntityId,
    pub value: Option<MetricValue>,
    pub status: SampleStatus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionMetadata {
    pub schema_version: u32,
    pub clock_domain: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SessionSnapshot {
    pub metadata: SessionMetadata,
    pub descriptors: Vec<MetricDescriptor>,
    pub samples: Vec<MetricSample>,
}