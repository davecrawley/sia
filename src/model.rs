use crate::clock::Timestamp;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unavailable {
    Unsupported(String),
    PermissionDenied(String),
    TemporarilyUnavailable(String),
}

impl Unavailable {
    pub fn reason(&self) -> &str {
        match self {
            Self::Unsupported(reason)
            | Self::PermissionDenied(reason)
            | Self::TemporarilyUnavailable(reason) => reason,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Capability {
    Available,
    Unavailable(Unavailable),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Percent,
    Celsius,
    Hertz,
    Bytes,
    Count,
    Seconds,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueKind {
    Gauge,
    Counter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemporalSemantics {
    Instantaneous,
    Interval,
    VendorSampled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricDescriptor {
    pub metric_id: String,
    pub entity_id: String,
    pub entity_name: String,
    pub display_name: String,
    pub unit: Unit,
    pub value_kind: ValueKind,
    pub temporal_semantics: TemporalSemantics,
    pub provider: String,
    pub source_semantics: String,
    pub semantics_version: u32,
    pub capability: Capability,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Float(f64),
    Unsigned(u64),
    Signed(i64),
}

impl Value {
    pub fn as_f64(&self) -> f64 {
        match self {
            Self::Float(value) => *value,
            Self::Unsigned(value) => *value as f64,
            Self::Signed(value) => *value as f64,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Reading {
    Available(Value),
    Unavailable(Unavailable),
}

impl Reading {
    pub fn capability(&self) -> Capability {
        match self {
            Self::Available(_) => Capability::Available,
            Self::Unavailable(reason) => Capability::Unavailable(reason.clone()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservationWindow {
    pub start: Timestamp,
    pub end: Timestamp,
}

impl ObservationWindow {
    pub fn elapsed_ns(&self) -> Option<u64> {
        self.end.elapsed_since(&self.start)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub metric_id: String,
    pub entity_id: String,
    pub observed_at: Timestamp,
    /// A timestamp supplied by the source, when the source actually provides one.
    pub source_timestamp: Option<Timestamp>,
    /// Unknown vendor averaging windows remain absent rather than being inferred.
    pub observation_window: Option<ObservationWindow>,
    pub reading: Reading,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProviderBatch {
    pub descriptors: Vec<MetricDescriptor>,
    pub samples: Vec<Sample>,
}
