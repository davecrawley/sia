//! Measurement values and metadata, independent of both hardware APIs and egui.

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unit {
    Percent,
    Celsius,
    Hertz,
    Bytes,
    Count,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueKind {
    Gauge,
    Counter,
    Rate,
    State,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemporalSemantics {
    PointSample,
    IntervalAverage,
    IntervalDelta,
    CumulativeCounter,
    VendorSampled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuClock {
    Graphics,
    Sm,
    Memory,
    Video,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetricKind {
    CpuUtilization,
    RamOccupancy,
    GpuUtilization,
    VramOccupancy,
    Temperature {
        sensor_name: String,
        sensor_label: String,
    },
    CpuFrequency {
        core: usize,
    },
    GpuFrequency(GpuClock),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricDescriptor {
    pub metric_id: String,
    pub entity_id: String,
    pub display_name: String,
    pub kind: MetricKind,
    pub unit: Unit,
    pub value_kind: ValueKind,
    pub temporal_semantics: TemporalSemantics,
    pub provider: String,
    pub source_semantics: String,
    pub semantics_version: u32,
}

impl MetricDescriptor {
    pub fn same_series(&self, other: &Self) -> bool {
        self.metric_id == other.metric_id
            && self.entity_id == other.entity_id
            && self.provider == other.provider
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapabilityStatus {
    Available,
    Unsupported,
    PermissionDenied,
    TemporarilyUnavailable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unavailable {
    pub status: CapabilityStatus,
    pub reason: String,
}

impl Unavailable {
    pub fn temporary(reason: impl Into<String>) -> Self {
        Self {
            status: CapabilityStatus::TemporarilyUnavailable,
            reason: reason.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capability {
    pub descriptor: MetricDescriptor,
    pub status: CapabilityStatus,
    pub reason: Option<String>,
}

impl Capability {
    pub fn from_reading(
        descriptor: MetricDescriptor,
        result: &Result<Reading, Unavailable>,
    ) -> Self {
        let (status, reason) = match result {
            Ok(_) => (CapabilityStatus::Available, None),
            Err(error) => (error.status, Some(error.reason.clone())),
        };
        Self {
            descriptor,
            status,
            reason,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClockIdentity {
    pub domain: String,
    pub boot_id: Option<String>,
    pub time_namespace: Option<String>,
}

/// A source window retains both endpoints and its clock domain verbatim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservationWindow {
    pub start_ns: u64,
    pub end_ns: u64,
    pub clock_domain: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceTimestamp {
    pub ns: u64,
    pub clock_domain: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Reading {
    pub value: f64,
    /// Optional source observation time already in the collection clock domain.
    pub mono_ns: Option<u64>,
    /// Native source timestamps need not be in the collection clock domain.
    pub source_timestamp: Option<SourceTimestamp>,
    pub window: Option<ObservationWindow>,
}

impl Reading {
    pub fn gauge(value: f64) -> Self {
        Self {
            value,
            mono_ns: None,
            source_timestamp: None,
            window: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SampleStatus {
    Ok,
    Unavailable(Unavailable),
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricSample {
    pub descriptor: MetricDescriptor,
    pub mono_ns: u64,
    pub source_timestamp: Option<SourceTimestamp>,
    pub window: Option<ObservationWindow>,
    pub value: Option<f64>,
    pub status: SampleStatus,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CollectionBatch {
    pub clock: ClockIdentity,
    pub observation_ns: u64,
    pub elapsed_ns: Option<u64>,
    pub samples: Vec<MetricSample>,
}
