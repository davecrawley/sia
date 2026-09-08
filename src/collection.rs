use crate::clock::Clock;
use crate::model::{
    Capability, CapabilityRecord, CapabilityStatus, CollectionBatch, EntityId, MetricDescriptor,
    MetricId, MetricSample, ObservationTime, SampleStatus, SampleValue,
};

#[derive(Clone, Debug, PartialEq)]
pub enum ReadingOutcome {
    Value(SampleValue),
    Stale,
    Unsupported(String),
    PermissionDenied(String),
    TemporarilyUnavailable(String),
    Error(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProviderReading {
    pub metric_id: MetricId,
    pub entity_id: EntityId,
    pub observation_time: Option<ObservationTime>,
    pub interval_start: Option<ObservationTime>,
    pub outcome: ReadingOutcome,
}

impl ProviderReading {
    pub fn numeric(metric: impl Into<MetricId>, entity: impl Into<EntityId>, value: f64) -> Self {
        Self {
            metric_id: metric.into(),
            entity_id: entity.into(),
            observation_time: None,
            interval_start: None,
            outcome: ReadingOutcome::Value(SampleValue::Numeric(value)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricTarget {
    pub descriptor: MetricDescriptor,
    pub entity_id: EntityId,
}

pub trait MetricProvider {
    fn descriptors(&self) -> Vec<MetricDescriptor>;

    fn targets(&self) -> Vec<MetricTarget> {
        self.descriptors()
            .into_iter()
            .map(|descriptor| MetricTarget {
                descriptor,
                entity_id: EntityId("system".into()),
            })
            .collect()
    }

    fn collect(&mut self, requested_at: ObservationTime) -> Vec<ProviderReading>;
}

pub struct Collector<C, P> {
    clock: C,
    provider: P,
}

impl<C: Clock, P: MetricProvider> Collector<C, P> {
    pub fn new(clock: C, provider: P) -> Self {
        Self { clock, provider }
    }

    pub fn provider(&self) -> &P {
        &self.provider
    }

    pub fn provider_mut(&mut self) -> &mut P {
        &mut self.provider
    }

    pub fn collect(&mut self) -> CollectionBatch {
        let requested_at = self.clock.now();
        let mut capabilities = Vec::new();
        let mut samples = Vec::new();
        for reading in self.provider.collect(requested_at) {
            let observation_time = reading.observation_time.unwrap_or(requested_at);
            let (capability, sample) = match reading.outcome {
                ReadingOutcome::Value(value) => (
                    Capability::available(),
                    Some((Some(value), SampleStatus::Ok)),
                ),
                ReadingOutcome::Stale => {
                    (Capability::available(), Some((None, SampleStatus::Stale)))
                }
                ReadingOutcome::Unsupported(detail) => (
                    Capability {
                        status: CapabilityStatus::Unsupported,
                        detail: Some(detail),
                    },
                    None,
                ),
                ReadingOutcome::PermissionDenied(detail) => (
                    Capability {
                        status: CapabilityStatus::PermissionDenied,
                        detail: Some(detail),
                    },
                    None,
                ),
                ReadingOutcome::TemporarilyUnavailable(detail) => (
                    Capability {
                        status: CapabilityStatus::TemporarilyUnavailable,
                        detail: Some(detail.clone()),
                    },
                    Some((None, SampleStatus::TemporarilyUnavailable(detail))),
                ),
                ReadingOutcome::Error(detail) => (
                    Capability {
                        status: CapabilityStatus::TemporarilyUnavailable,
                        detail: Some(detail.clone()),
                    },
                    Some((None, SampleStatus::Error(detail))),
                ),
            };
            capabilities.push(CapabilityRecord {
                metric_id: reading.metric_id.clone(),
                entity_id: reading.entity_id.clone(),
                capability,
            });
            if let Some((value, status)) = sample {
                samples.push(MetricSample {
                    observation_time,
                    interval_start: reading.interval_start,
                    metric_id: reading.metric_id,
                    entity_id: reading.entity_id,
                    value,
                    status,
                });
            }
        }
        CollectionBatch {
            observation_time: requested_at,
            capabilities,
            samples,
        }
    }
}
