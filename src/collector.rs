use crate::clock::{Clock, ClockError};
use crate::model::{
    CapabilityState, EntityId, MetricDescriptor, MetricId, MetricSample, MetricValue,
    MonotonicTimestamp, SampleStatus,
};
use std::collections::VecDeque;
use std::fmt;

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
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clock(error) => write!(formatter, "clock error: {error}"),
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

impl<P, C> MetricCollector for TypedCollector<P, C>
where
    P: MetricProvider,
    C: Clock + Send,
{
    fn clock_domain(&self) -> &'static str {
        self.clock.domain()
    }

    fn descriptors(&self) -> &[MetricDescriptor] {
        self.provider.descriptors()
    }

    fn collect(&mut self) -> Result<Vec<MetricSample>, CollectionError> {
        let actual_observation_time = self.clock.now().map_err(CollectionError::Clock)?;
        let results = self.provider.observe(actual_observation_time);
        let descriptors = self.provider.descriptors();
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
                } => {
                    if is_available(descriptors, &metric_id) {
                        samples.push(MetricSample {
                            observation_time: observation_time
                                .unwrap_or(actual_observation_time),
                            interval_start,
                            metric_id,
                            entity_id,
                            value,
                            status,
                        });
                    }
                }
                ProviderResult::NativeCounter {
                    metric_id,
                    entity_id,
                    observation_time,
                    value,
                    status,
                } => {
                    if is_available(descriptors, &metric_id) {
                        samples.push(MetricSample {
                            observation_time: observation_time
                                .unwrap_or(actual_observation_time),
                            interval_start: None,
                            metric_id,
                            entity_id,
                            value: Some(MetricValue::U64(value)),
                            status,
                        });
                    }
                }
                ProviderResult::TimestampedBuffer {
                    metric_id,
                    entity_id,
                    points,
                } => {
                    if is_available(descriptors, &metric_id) {
                        samples.extend(points.into_iter().map(|point| MetricSample {
                            observation_time: point.observation_time,
                            interval_start: point.interval_start,
                            metric_id: metric_id.clone(),
                            entity_id: entity_id.clone(),
                            value: point.value,
                            status: point.status,
                        }));
                    }
                }
            }
        }

        Ok(samples)
    }
}

fn is_available(descriptors: &[MetricDescriptor], metric_id: &MetricId) -> bool {
    descriptors.iter().any(|descriptor| {
        descriptor.metric_id == *metric_id
            && matches!(descriptor.capability_status, CapabilityState::Available)
    })
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

    fn observe(&mut self, _observation_time: MonotonicTimestamp) -> Vec<ProviderResult> {
        self.batches.pop_front().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::DeterministicClock;
    use crate::model::{
        CanonicalUnit, EntityKind, SourceResolution, TemporalSemantics, ValueKind,
    };

    fn descriptor(id: &str, temporal_semantics: TemporalSemantics) -> MetricDescriptor {
        MetricDescriptor {
            metric_id: id.into(),
            display_name: id.into(),
            entity_kind: EntityKind::System,
            unit: CanonicalUnit::Count,
            value_kind: ValueKind::Gauge,
            temporal_semantics,
            provider: "fake".into(),
            capability_status: CapabilityState::Available,
            source_resolution: Some(SourceResolution { nanoseconds: 10 }),
            source_semantics: "fixture observation".into(),
            comparability_group: Some("fixture".into()),
            semantics_version: 1,
        }
    }

    #[test]
    fn injected_clock_supplies_actual_observation_time_without_gui_or_hardware() {
        let descriptor = descriptor("fixture.metric", TemporalSemantics::PointSampled);
        let result = ProviderResult::Observation {
            metric_id: descriptor.metric_id.clone(),
            entity_id: "fixture:entity".into(),
            observation_time: None,
            interval_start: None,
            value: Some(MetricValue::F64(7.0)),
            status: SampleStatus::Ok,
        };
        let provider = FakeMetricProvider::new(vec![descriptor], [vec![result]]);
        let clock = DeterministicClock::new([MonotonicTimestamp(42)]);
        let mut collector = TypedCollector::new(provider, clock);

        let samples = collector.collect().unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].observation_time, MonotonicTimestamp(42));
        assert_eq!(samples[0].value, Some(MetricValue::F64(7.0)));
    }

    #[test]
    fn unavailable_capabilities_are_distinct_and_do_not_fabricate_zeroes() {
        let states = [
            CapabilityState::Unsupported {
                reason: Some("not implemented".into()),
            },
            CapabilityState::PermissionDenied {
                reason: Some("access denied".into()),
            },
            CapabilityState::TemporarilyUnavailable {
                reason: Some("device reset".into()),
            },
        ];

        for state in states {
            let mut unavailable = descriptor("fixture.metric", TemporalSemantics::PointSampled);
            unavailable.capability_status = state;
            let result = ProviderResult::Observation {
                metric_id: unavailable.metric_id.clone(),
                entity_id: "fixture:entity".into(),
                observation_time: None,
                interval_start: None,
                value: Some(MetricValue::F64(0.0)),
                status: SampleStatus::Ok,
            };
            let provider = FakeMetricProvider::new(vec![unavailable], [vec![result]]);
            let clock = DeterministicClock::new([MonotonicTimestamp(1)]);
            let mut collector = TypedCollector::new(provider, clock);
            assert!(collector.collect().unwrap().is_empty());
        }

        let available = descriptor("fixture.zero", TemporalSemantics::PointSampled);
        let result = ProviderResult::Observation {
            metric_id: available.metric_id.clone(),
            entity_id: "fixture:entity".into(),
            observation_time: None,
            interval_start: None,
            value: Some(MetricValue::F64(0.0)),
            status: SampleStatus::Ok,
        };
        let provider = FakeMetricProvider::new(vec![available], [vec![result]]);
        let clock = DeterministicClock::new([MonotonicTimestamp(2)]);
        let mut collector = TypedCollector::new(provider, clock);
        assert_eq!(collector.collect().unwrap()[0].value, Some(MetricValue::F64(0.0)));
    }

    #[test]
    fn temporal_semantics_and_native_windows_cross_the_boundary_unchanged() {
        let semantics = [
            TemporalSemantics::PointSampled,
            TemporalSemantics::IntervalDelta,
            TemporalSemantics::CumulativeCounter,
            TemporalSemantics::VendorSampled,
        ];
        let descriptors: Vec<_> = semantics
            .iter()
            .enumerate()
            .map(|(index, semantics)| descriptor(&format!("fixture.{index}"), *semantics))
            .collect();
        let results: Vec<_> = descriptors
            .iter()
            .map(|descriptor| ProviderResult::Observation {
                metric_id: descriptor.metric_id.clone(),
                entity_id: "fixture:entity".into(),
                observation_time: Some(MonotonicTimestamp(200)),
                interval_start: Some(MonotonicTimestamp(100)),
                value: Some(MetricValue::U64(9)),
                status: SampleStatus::Ok,
            })
            .collect();
        let provider = FakeMetricProvider::new(descriptors.clone(), [results]);
        let clock = DeterministicClock::new([MonotonicTimestamp(250)]);
        let mut collector = TypedCollector::new(provider, clock);
        let samples = collector.collect().unwrap();

        assert_eq!(collector.descriptors(), descriptors.as_slice());
        assert!(samples
            .iter()
            .all(|sample| sample.interval_start == Some(MonotonicTimestamp(100))));
        assert!(samples
            .iter()
            .all(|sample| sample.observation_time == MonotonicTimestamp(200)));
    }

    #[test]
    fn timestamped_native_buffer_is_not_collapsed_to_an_instantaneous_gauge() {
        let descriptor = descriptor(
            "fixture.native_counter",
            TemporalSemantics::CumulativeCounter,
        );
        let buffer = ProviderResult::TimestampedBuffer {
            metric_id: descriptor.metric_id.clone(),
            entity_id: "fixture:entity".into(),
            points: vec![
                NativePoint {
                    observation_time: MonotonicTimestamp(20),
                    interval_start: Some(MonotonicTimestamp(10)),
                    value: Some(MetricValue::U64(4)),
                    status: SampleStatus::Ok,
                },
                NativePoint {
                    observation_time: MonotonicTimestamp(30),
                    interval_start: Some(MonotonicTimestamp(20)),
                    value: Some(MetricValue::U64(8)),
                    status: SampleStatus::Ok,
                },
            ],
        };
        let provider = FakeMetricProvider::new(vec![descriptor], [vec![buffer]]);
        let clock = DeterministicClock::new([MonotonicTimestamp(40)]);
        let mut collector = TypedCollector::new(provider, clock);
        let samples = collector.collect().unwrap();

        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].observation_time, MonotonicTimestamp(20));
        assert_eq!(samples[1].interval_start, Some(MonotonicTimestamp(20)));
    }
}