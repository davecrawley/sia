use crate::clock::{Clock, ClockError};
use crate::model::{
    CapabilityState, EntityId, MetricDescriptor, MetricId, MetricSample, MetricValue,
    MonotonicTimestamp, SampleStatus,
};
use std::{collections::VecDeque, fmt};

#[derive(Clone, Debug, PartialEq)]
pub struct NativePoint {
    pub observation_time: MonotonicTimestamp,
    pub interval_start: Option<MonotonicTimestamp>,
    pub value: Option<MetricValue>,
    pub status: SampleStatus,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ProviderResult {
    Observation {
        metric_id: MetricId,
        entity_id: EntityId,
        observation_time: Option<MonotonicTimestamp>,
        interval_start: Option<MonotonicTimestamp>,
        value: Option<MetricValue>,
        status: SampleStatus,
    },
    NativeCounter {
        metric_id: MetricId,
        entity_id: EntityId,
        observation_time: Option<MonotonicTimestamp>,
        value: u64,
        status: SampleStatus,
    },
    TimestampedBuffer {
        metric_id: MetricId,
        entity_id: EntityId,
        points: Vec<NativePoint>,
    },
}

pub trait MetricProvider: Send {
    fn descriptors(&self) -> &[MetricDescriptor];
    fn observe(&mut self, observation_time: MonotonicTimestamp) -> Vec<ProviderResult>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CollectionError {
    Clock(ClockError),
}
impl fmt::Display for CollectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clock(e) => write!(f, "clock error: {e}"),
        }
    }
}
impl std::error::Error for CollectionError {}

pub trait MetricCollector: Send {
    fn clock_domain(&self) -> &'static str;
    fn descriptors(&self) -> &[MetricDescriptor];
    fn collect(&mut self) -> Result<Vec<MetricSample>, CollectionError>;
}

pub struct TypedCollector<P, C> {
    provider: P,
    clock: C,
}
impl<P, C> TypedCollector<P, C> {
    pub fn new(provider: P, clock: C) -> Self {
        Self { provider, clock }
    }
    pub fn provider(&self) -> &P {
        &self.provider
    }
    pub fn provider_mut(&mut self) -> &mut P {
        &mut self.provider
    }
}

impl<P: MetricProvider, C: Clock + Send> MetricCollector for TypedCollector<P, C> {
    fn clock_domain(&self) -> &'static str {
        self.clock.domain()
    }
    fn descriptors(&self) -> &[MetricDescriptor] {
        self.provider.descriptors()
    }
    fn collect(&mut self) -> Result<Vec<MetricSample>, CollectionError> {
        let actual = self.clock.now().map_err(CollectionError::Clock)?;
        let results = self.provider.observe(actual);
        let descriptors = self.provider.descriptors();
        let available = |id: &MetricId| {
            descriptors.iter().any(|d| {
                d.metric_id == *id && matches!(d.capability_status, CapabilityState::Available)
            })
        };
        let mut samples = Vec::new();
        for result in results {
            match result {
                ProviderResult::Observation {
                    metric_id,
                    entity_id,
                    observation_time,
                    interval_start,
                    value,
                    status,
                } if available(&metric_id) => samples.push(MetricSample {
                    observation_time: observation_time.unwrap_or(actual),
                    interval_start,
                    metric_id,
                    entity_id,
                    value,
                    status,
                }),
                ProviderResult::NativeCounter {
                    metric_id,
                    entity_id,
                    observation_time,
                    value,
                    status,
                } if available(&metric_id) => samples.push(MetricSample {
                    observation_time: observation_time.unwrap_or(actual),
                    interval_start: None,
                    metric_id,
                    entity_id,
                    value: Some(MetricValue::U64(value)),
                    status,
                }),
                ProviderResult::TimestampedBuffer {
                    metric_id,
                    entity_id,
                    points,
                } if available(&metric_id) => {
                    samples.extend(points.into_iter().map(|point| MetricSample {
                        observation_time: point.observation_time,
                        interval_start: point.interval_start,
                        metric_id: metric_id.clone(),
                        entity_id: entity_id.clone(),
                        value: point.value,
                        status: point.status,
                    }))
                }
                _ => {}
            }
        }
        Ok(samples)
    }
}

#[derive(Clone, Debug)]
pub struct FakeMetricProvider {
    descriptors: Vec<MetricDescriptor>,
    batches: VecDeque<Vec<ProviderResult>>,
}
impl FakeMetricProvider {
    pub fn new(
        descriptors: Vec<MetricDescriptor>,
        batches: impl IntoIterator<Item = Vec<ProviderResult>>,
    ) -> Self {
        Self {
            descriptors,
            batches: batches.into_iter().collect(),
        }
    }
}
impl MetricProvider for FakeMetricProvider {
    fn descriptors(&self) -> &[MetricDescriptor] {
        &self.descriptors
    }
    fn observe(&mut self, _: MonotonicTimestamp) -> Vec<ProviderResult> {
        self.batches.pop_front().unwrap_or_default()
    }
}
