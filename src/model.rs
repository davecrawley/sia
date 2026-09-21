use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClockIdentity {
    pub domain: String,
    pub boot_id: String,
    pub time_namespace_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Timestamp {
    pub identity: ClockIdentity,
    pub mono_ns: u64,
}

impl Timestamp {
    pub fn elapsed_since(&self, previous: &Self) -> Option<Duration> {
        if self.identity != previous.identity {
            return None;
        }
        self.mono_ns
            .checked_sub(previous.mono_ns)
            .map(Duration::from_nanos)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservationWindow {
    pub start: Timestamp,
    pub end: Timestamp,
}

impl ObservationWindow {
    pub fn elapsed(&self) -> Option<Duration> {
        self.end.elapsed_since(&self.start)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnavailableKind {
    Unsupported,
    PermissionDenied,
    TemporarilyUnavailable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unavailable {
    pub kind: UnavailableKind,
    pub reason: String,
}

impl Unavailable {
    pub fn new(kind: UnavailableKind, reason: impl Into<String>) -> Self {
        Self {
            kind,
            reason: reason.into(),
        }
    }

    pub fn temporary(reason: impl Into<String>) -> Self {
        Self::new(UnavailableKind::TemporarilyUnavailable, reason)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Capability {
    Available,
    Unsupported(String),
    PermissionDenied(String),
    TemporarilyUnavailable(String),
}

impl From<&Unavailable> for Capability {
    fn from(value: &Unavailable) -> Self {
        match value.kind {
            UnavailableKind::Unsupported => Self::Unsupported(value.reason.clone()),
            UnavailableKind::PermissionDenied => Self::PermissionDenied(value.reason.clone()),
            UnavailableKind::TemporarilyUnavailable => {
                Self::TemporarilyUnavailable(value.reason.clone())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntityKind {
    System,
    Cpu,
    Gpu,
    Sensor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Percent,
    Bytes,
    Celsius,
    Hertz,
    Count,
    State,
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
pub struct MetricDescriptor {
    pub metric_id: String,
    pub entity_id: String,
    pub entity_kind: EntityKind,
    pub entity_display_name: String,
    pub display_name: String,
    pub unit: Unit,
    pub value_kind: ValueKind,
    pub temporal_semantics: TemporalSemantics,
    pub provider: String,
    pub source_semantics: String,
    pub source_resolution_hint: Option<Duration>,
    pub comparability_group: Option<String>,
    pub semantics_version: u32,
    pub capability: Capability,
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

#[derive(Clone, Debug, PartialEq)]
pub struct MetricSample {
    pub metric_id: String,
    pub entity_id: String,
    pub observed_at: Timestamp,
    pub source_timestamp: Option<Timestamp>,
    pub observation_window: Option<ObservationWindow>,
    pub value: Result<MetricValue, Unavailable>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProviderOutput {
    pub descriptors: Vec<MetricDescriptor>,
    pub samples: Vec<MetricSample>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CollectionBatch {
    pub observed_at: Timestamp,
    pub elapsed: Option<Duration>,
    pub discontinuity: bool,
    pub descriptors: Vec<MetricDescriptor>,
    pub samples: Vec<MetricSample>,
}
