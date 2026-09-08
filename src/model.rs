use std::collections::{BTreeMap, VecDeque};

pub const LIVE_WINDOW_NS: u64 = 300_000_000_000;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EntityKind {
    System,
    Cpu,
    Gpu,
    Thermal,
    Other(String),
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
pub enum CapabilityStatus {
    Available,
    Unsupported,
    PermissionDenied,
    TemporarilyUnavailable,
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
    Stale,
    TemporarilyUnavailable(String),
    Error(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricDescriptor {
    pub metric_id: String,
    pub display_name: String,
    pub entity_kind: EntityKind,
    pub unit: String,
    pub value_kind: ValueKind,
    pub temporal_semantics: TemporalSemantics,
    pub provider: String,
    pub capability_status: CapabilityStatus,
    pub source_resolution_ns: Option<u64>,
    pub source_definition: String,
    pub comparability_group: Option<String>,
    pub semantics_version: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricSample {
    pub observation_ns: u64,
    pub window_start_ns: Option<u64>,
    pub metric_id: String,
    pub entity_id: String,
    pub value: Option<MetricValue>,
    pub status: SampleStatus,
}

impl MetricSample {
    pub fn is_plottable(&self) -> bool {
        self.status == SampleStatus::Ok
            && self.value.as_ref().and_then(MetricValue::as_f64).is_some()
    }
}

#[derive(Clone, Debug, Default)]
pub struct TelemetryModel {
    descriptors: BTreeMap<String, MetricDescriptor>,
    series: BTreeMap<(String, String), VecDeque<MetricSample>>,
    latest_observation_ns: Option<u64>,
    empty: VecDeque<MetricSample>,
}

impl TelemetryModel {
    pub fn register(&mut self, descriptor: MetricDescriptor) {
        self.descriptors
            .insert(descriptor.metric_id.clone(), descriptor);
    }

    pub fn ingest(&mut self, sample: MetricSample) {
        let latest = self
            .latest_observation_ns
            .map_or(sample.observation_ns, |old| old.max(sample.observation_ns));
        self.latest_observation_ns = Some(latest);
        self.series
            .entry((sample.metric_id.clone(), sample.entity_id.clone()))
            .or_default()
            .push_back(sample);
        self.prune(latest);
    }

    fn prune(&mut self, latest: u64) {
        let cutoff = latest.saturating_sub(LIVE_WINDOW_NS);
        for samples in self.series.values_mut() {
            while samples
                .front()
                .is_some_and(|sample| sample.observation_ns < cutoff)
            {
                samples.pop_front();
            }
        }
    }

    pub fn descriptor(&self, metric_id: &str) -> Option<&MetricDescriptor> {
        self.descriptors.get(metric_id)
    }

    pub fn descriptors(&self) -> impl Iterator<Item = &MetricDescriptor> {
        self.descriptors.values()
    }

    pub fn samples(&self, metric_id: &str, entity_id: &str) -> &VecDeque<MetricSample> {
        self.series
            .get(&(metric_id.to_owned(), entity_id.to_owned()))
            .unwrap_or(&self.empty)
    }

    pub fn all_series(&self) -> impl Iterator<Item = (&(String, String), &VecDeque<MetricSample>)> {
        self.series.iter()
    }

    pub fn latest_observation_ns(&self) -> Option<u64> {
        self.latest_observation_ns
    }
}
