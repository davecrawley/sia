use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

pub const LIVE_RETENTION: Duration = Duration::from_secs(300);
pub const LIVE_RETENTION_NS: u64 = 300_000_000_000;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MetricId(pub String);

impl MetricId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

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

impl AsRef<str> for MetricId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for MetricId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EntityKind {
    System,
    Cpu,
    Gpu,
    Thermal,
    Other(String),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Unit {
    Percent,
    Celsius,
    Hertz,
    Bytes,
    Count,
    Seconds,
    Other(String),
}

impl Unit {
    pub fn symbol(&self) -> &str {
        match self {
            Self::Percent => "%",
            Self::Celsius => "°C",
            Self::Hertz => "Hz",
            Self::Bytes => "B",
            Self::Count => "count",
            Self::Seconds => "s",
            Self::Other(value) => value,
        }
    }
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CapabilityState {
    Available,
    Unsupported(String),
    PermissionDenied(String),
    TemporarilyUnavailable(String),
}

impl CapabilityState {
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }
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
    Available,
    Stale(String),
    TemporarilyUnavailable(String),
    Error(String),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeriesKey {
    pub metric_id: MetricId,
    pub entity_id: String,
    pub provider_generation: u64,
}

impl SeriesKey {
    pub fn new(
        metric_id: impl Into<MetricId>,
        entity_id: impl Into<String>,
        provider_generation: u64,
    ) -> Self {
        Self {
            metric_id: metric_id.into(),
            entity_id: entity_id.into(),
            provider_generation,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricDescriptor {
    pub key: SeriesKey,
    pub display_name: String,
    pub entity_kind: EntityKind,
    pub unit: Unit,
    pub value_kind: ValueKind,
    pub temporal_semantics: TemporalSemantics,
    pub provider: String,
    pub capability: CapabilityState,
    pub source_resolution_ns: Option<u64>,
    pub source_definition: String,
    pub comparability_group: Option<String>,
    pub semantics_version: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    pub key: SeriesKey,
    pub observation_ns: u64,
    pub window_start_ns: Option<u64>,
    pub value: Option<MetricValue>,
    pub status: SampleStatus,
}

impl Observation {
    pub fn is_plottable(&self) -> bool {
        matches!(self.status, SampleStatus::Available)
            && self.value.as_ref().and_then(MetricValue::as_f64).is_some()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Series {
    pub descriptor: MetricDescriptor,
    pub observations: Vec<Observation>,
}

#[derive(Clone, Debug, Default)]
pub struct TelemetryModel {
    series: BTreeMap<SeriesKey, Series>,
    latest_observation_ns: Option<u64>,
}

impl TelemetryModel {
    pub fn register_descriptor(&mut self, descriptor: MetricDescriptor) {
        let key = descriptor.key.clone();
        match self.series.get_mut(&key) {
            Some(series) => series.descriptor = descriptor,
            None => {
                self.series.insert(
                    key,
                    Series {
                        descriptor,
                        observations: Vec::new(),
                    },
                );
            }
        }
    }

    pub fn ingest(&mut self, descriptor: MetricDescriptor, observation: Observation) {
        debug_assert_eq!(descriptor.key, observation.key);
        let observed = observation.observation_ns;
        let latest = self
            .latest_observation_ns
            .map_or(observed, |current| current.max(observed));
        self.latest_observation_ns = Some(latest);

        let key = descriptor.key.clone();
        self.register_descriptor(descriptor);
        if let Some(series) = self.series.get_mut(&key) {
            series.observations.push(observation);
        }
        self.prune(latest);
    }

    pub fn prune(&mut self, latest_observation_ns: u64) {
        let cutoff = latest_observation_ns.saturating_sub(LIVE_RETENTION_NS);
        for series in self.series.values_mut() {
            series
                .observations
                .retain(|sample| sample.observation_ns >= cutoff);
            series
                .observations
                .sort_by_key(|sample| sample.observation_ns);
        }
    }

    pub fn descriptor(&self, key: &SeriesKey) -> Option<&MetricDescriptor> {
        self.series.get(key).map(|series| &series.descriptor)
    }

    pub fn observations(&self, key: &SeriesKey) -> Option<&[Observation]> {
        self.series
            .get(key)
            .map(|series| series.observations.as_slice())
    }

    pub fn all_series(&self) -> impl Iterator<Item = (&SeriesKey, &Series)> {
        self.series.iter()
    }

    pub fn descriptors(&self) -> impl Iterator<Item = &MetricDescriptor> {
        self.series.values().map(|series| &series.descriptor)
    }

    pub fn latest_observation_ns(&self) -> Option<u64> {
        self.latest_observation_ns
    }

    pub fn total_observations(&self) -> usize {
        self.series
            .values()
            .map(|series| series.observations.len())
            .sum()
    }
}
