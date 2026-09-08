use std::collections::BTreeMap;

pub const LINUX_MONOTONIC_CLOCK: &str = "linux_clock_monotonic";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObservationTime {
    pub monotonic_ns: u64,
    pub clock_domain: &'static str,
}

impl ObservationTime {
    pub fn elapsed_seconds_since(self, earlier: Self) -> f64 {
        self.monotonic_ns.saturating_sub(earlier.monotonic_ns) as f64 / 1_000_000_000.0
    }
}

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EntityKind {
    System,
    Cpu,
    CpuCore,
    Memory,
    ThermalSensor,
    Gpu,
    Other(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CanonicalUnit {
    Percent,
    Celsius,
    Hertz,
    Bytes,
    Count,
    State,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueKind {
    Gauge,
    Rate,
    CumulativeCounter,
    State,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemporalSemantics {
    PointSampled,
    IntervalAverage,
    IntervalDelta,
    CumulativeCounter,
    VendorSampled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CapabilityStatus {
    Available,
    Unsupported,
    PermissionDenied,
    TemporarilyUnavailable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capability {
    pub status: CapabilityStatus,
    pub detail: Option<String>,
}

impl Capability {
    pub fn available() -> Self {
        Self {
            status: CapabilityStatus::Available,
            detail: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricDescriptor {
    pub metric_id: MetricId,
    pub display_name: String,
    pub entity_kind: EntityKind,
    pub canonical_unit: CanonicalUnit,
    pub value_kind: ValueKind,
    pub temporal_semantics: TemporalSemantics,
    pub provider: String,
    pub capability_status: CapabilityStatus,
    pub source_resolution_ns: Option<u64>,
    pub source_semantics: String,
    pub comparability_group: Option<String>,
    pub semantics_version: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SampleValue {
    Numeric(f64),
    Integer(i64),
    Unsigned(u64),
    State(String),
}

impl SampleValue {
    pub fn numeric(&self) -> Option<f64> {
        match self {
            Self::Numeric(value) => Some(*value),
            Self::Integer(value) => Some(*value as f64),
            Self::Unsigned(value) => Some(*value as f64),
            Self::State(_) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SampleStatus {
    Ok,
    Stale,
    TemporarilyUnavailable(String),
    Error(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricSample {
    pub observation_time: ObservationTime,
    pub interval_start: Option<ObservationTime>,
    pub metric_id: MetricId,
    pub entity_id: EntityId,
    pub value: Option<SampleValue>,
    pub status: SampleStatus,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CapabilityRecord {
    pub metric_id: MetricId,
    pub entity_id: EntityId,
    pub capability: Capability,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CollectionBatch {
    pub observation_time: ObservationTime,
    pub capabilities: Vec<CapabilityRecord>,
    pub samples: Vec<MetricSample>,
}

#[derive(Default)]
pub struct SessionModel {
    descriptors: BTreeMap<MetricId, MetricDescriptor>,
    capabilities: BTreeMap<(MetricId, EntityId), Capability>,
    samples: BTreeMap<(MetricId, EntityId), Vec<MetricSample>>,
}

impl SessionModel {
    pub fn new(descriptors: impl IntoIterator<Item = MetricDescriptor>) -> Self {
        let mut model = Self::default();
        for descriptor in descriptors {
            model
                .descriptors
                .insert(descriptor.metric_id.clone(), descriptor);
        }
        model
    }

    pub fn ingest(&mut self, batch: CollectionBatch) {
        for record in batch.capabilities {
            self.capabilities
                .insert((record.metric_id, record.entity_id), record.capability);
        }
        for sample in batch.samples {
            self.samples
                .entry((sample.metric_id.clone(), sample.entity_id.clone()))
                .or_default()
                .push(sample);
        }
    }

    pub fn descriptor(&self, id: &MetricId) -> Option<&MetricDescriptor> {
        self.descriptors.get(id)
    }

    pub fn descriptors(&self) -> impl Iterator<Item = &MetricDescriptor> {
        self.descriptors.values()
    }

    pub fn capability(&self, metric: &MetricId, entity: &EntityId) -> Option<&Capability> {
        self.capabilities.get(&(metric.clone(), entity.clone()))
    }

    pub fn samples(&self, metric: &MetricId, entity: &EntityId) -> &[MetricSample] {
        self.samples
            .get(&(metric.clone(), entity.clone()))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn latest(&self, metric: &MetricId, entity: &EntityId) -> Option<&MetricSample> {
        self.samples(metric, entity).last()
    }
}
