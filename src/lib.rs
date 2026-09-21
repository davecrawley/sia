//! Typed collection boundaries shared by presentation and deterministic providers.

pub mod clock;
pub mod provider;

use std::collections::BTreeMap;
use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Capability {
    Available,
    Unsupported(String),
    PermissionDenied(String),
    TemporarilyUnavailable(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Percent,
    Celsius,
    Hertz,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntityKind {
    System,
    Cpu,
    Gpu,
    Sensor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuClock {
    Graphics,
    Sm,
    Memory,
    Video,
}

/// The existing monitor's presentation roles, independent of any GUI toolkit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MonitorRole {
    CpuUtilization,
    RamUtilization,
    GpuUtilization,
    VramUtilization,
    Temperature { source_name: String, label: String },
    CpuFrequency { core: usize },
    GpuFrequency(GpuClock),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricDescriptor {
    pub metric_id: String,
    pub entity_id: String,
    pub entity_kind: EntityKind,
    pub display_name: String,
    pub unit: Unit,
    pub value_kind: ValueKind,
    pub temporal_semantics: TemporalSemantics,
    pub provider: String,
    pub source_semantics: String,
    pub semantics_version: u32,
    pub capability: Capability,
    pub role: MonitorRole,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Timestamp {
    pub ns: u64,
    pub clock_domain: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interval {
    pub start_ns: u64,
    pub end_ns: u64,
    pub clock_domain: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SourceValue {
    pub value: f64,
    /// A source-native timestamp, including its own clock domain when applicable.
    pub timestamp: Option<Timestamp>,
    /// A source-reported window; the collector never estimates or rewrites it.
    pub interval: Option<Interval>,
}

impl SourceValue {
    pub fn gauge(value: f64) -> Self {
        Self {
            value,
            timestamp: None,
            interval: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Reading {
    pub metric_id: String,
    pub entity_id: String,
    pub result: Result<SourceValue, Capability>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SampleStatus {
    Ok,
    Unavailable(Capability),
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricSample {
    pub descriptor: MetricDescriptor,
    pub observed_at: Timestamp,
    pub timestamp: Timestamp,
    pub interval: Option<Interval>,
    pub value: Option<f64>,
    pub status: SampleStatus,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CollectionBatch {
    pub observed_at: Timestamp,
    pub elapsed_ns: Option<u64>,
    /// One sample per discovered descriptor, in discovery order, including gaps.
    pub samples: Vec<MetricSample>,
}

pub trait Clock {
    fn now(&self) -> Timestamp;
}

/// Discovery and reads use the same capability vocabulary. Each reading fails
/// independently; missing readings become explicit temporary gaps.
pub trait Provider {
    fn discover(&mut self) -> Vec<MetricDescriptor>;
    fn read(&mut self) -> Vec<Reading>;
}

pub trait Collection {
    fn descriptors(&self) -> &[MetricDescriptor];
    fn collect(&mut self) -> CollectionBatch;
}

pub struct Collector<P, C> {
    provider: P,
    clock: C,
    descriptors: Vec<MetricDescriptor>,
    previous_observation: Option<Timestamp>,
    /// Scheduling policy only; never used to manufacture timestamps or windows.
    pub configured_period: Duration,
}

impl<P: Provider, C: Clock> Collector<P, C> {
    pub fn new(mut provider: P, clock: C, configured_period: Duration) -> Self {
        let descriptors = provider.discover();
        Self {
            provider,
            clock,
            descriptors,
            previous_observation: None,
            configured_period,
        }
    }
}

impl<P: Provider, C: Clock> Collection for Collector<P, C> {
    fn descriptors(&self) -> &[MetricDescriptor] {
        &self.descriptors
    }

    fn collect(&mut self) -> CollectionBatch {
        let observed_at = self.clock.now();
        let elapsed_ns = self.previous_observation.as_ref().and_then(|previous| {
            if previous.clock_domain == observed_at.clock_domain {
                observed_at.ns.checked_sub(previous.ns)
            } else {
                None
            }
        });
        self.previous_observation = Some(observed_at.clone());
        let mut readings: BTreeMap<_, _> = self
            .provider
            .read()
            .into_iter()
            .map(|reading| ((reading.metric_id, reading.entity_id), reading.result))
            .collect();
        let samples = self
            .descriptors
            .iter()
            .map(|descriptor| {
                let key = (descriptor.metric_id.clone(), descriptor.entity_id.clone());
                let result = match &descriptor.capability {
                    Capability::Unsupported(_) | Capability::PermissionDenied(_) => {
                        Err(descriptor.capability.clone())
                    }
                    _ => readings.remove(&key).unwrap_or_else(|| {
                        Err(Capability::TemporarilyUnavailable(
                            "Provider returned no reading".into(),
                        ))
                    }),
                };
                let result = result.and_then(|value| {
                    if value.value.is_finite() {
                        Ok(value)
                    } else {
                        Err(Capability::TemporarilyUnavailable(
                            "Provider returned a non-finite value".into(),
                        ))
                    }
                });
                let mut descriptor = descriptor.clone();
                match result {
                    Ok(value) => {
                        descriptor.capability = Capability::Available;
                        MetricSample {
                            descriptor,
                            observed_at: observed_at.clone(),
                            timestamp: value.timestamp.unwrap_or_else(|| observed_at.clone()),
                            interval: value.interval,
                            value: Some(value.value),
                            status: SampleStatus::Ok,
                        }
                    }
                    Err(capability) => {
                        let capability = match capability {
                            Capability::Available => Capability::TemporarilyUnavailable(
                                "Provider returned an error without a reason".into(),
                            ),
                            other => other,
                        };
                        descriptor.capability = capability.clone();
                        MetricSample {
                            descriptor,
                            observed_at: observed_at.clone(),
                            timestamp: observed_at.clone(),
                            interval: None,
                            value: None,
                            status: SampleStatus::Unavailable(capability),
                        }
                    }
                }
            })
            .collect();
        CollectionBatch {
            observed_at,
            elapsed_ns,
            samples,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::VecDeque;

    struct FakeClock(RefCell<VecDeque<u64>>);

    impl Clock for FakeClock {
        fn now(&self) -> Timestamp {
            Timestamp {
                ns: self.0.borrow_mut().pop_front().unwrap(),
                clock_domain: "test_observation_clock".into(),
            }
        }
    }

    struct FakeProvider {
        descriptors: Vec<MetricDescriptor>,
        batches: VecDeque<Vec<Reading>>,
    }

    impl Provider for FakeProvider {
        fn discover(&mut self) -> Vec<MetricDescriptor> {
            self.descriptors.clone()
        }

        fn read(&mut self) -> Vec<Reading> {
            self.batches.pop_front().unwrap()
        }
    }

    fn descriptor(id: &str, capability: Capability) -> MetricDescriptor {
        MetricDescriptor {
            metric_id: id.into(),
            entity_id: "fake-device".into(),
            entity_kind: EntityKind::Gpu,
            display_name: id.into(),
            unit: Unit::Percent,
            value_kind: ValueKind::Gauge,
            temporal_semantics: TemporalSemantics::VendorSampled,
            provider: "verifier".into(),
            source_semantics: "Deterministic vendor-window gauge".into(),
            semantics_version: 7,
            capability,
            role: MonitorRole::GpuUtilization,
        }
    }

    fn reading(id: &str, result: Result<SourceValue, Capability>) -> Reading {
        Reading {
            metric_id: id.into(),
            entity_id: "fake-device".into(),
            result,
        }
    }

    #[test]
    fn preserves_zero_identity_semantics_and_actual_time() {
        let descriptor = descriptor("busy", Capability::Available);
        let interval = Interval {
            start_ns: 300,
            end_ns: 900,
            clock_domain: "vendor_clock".into(),
        };
        let provider = FakeProvider {
            descriptors: vec![descriptor.clone()],
            batches: VecDeque::from([
                vec![reading("busy", Ok(SourceValue::gauge(37.0)))],
                vec![reading(
                    "busy",
                    Ok(SourceValue {
                        value: 0.0,
                        timestamp: None,
                        interval: Some(interval.clone()),
                    }),
                )],
            ]),
        };
        let clock = FakeClock(RefCell::new(VecDeque::from([1_000_000_000, 1_750_000_000])));
        let mut collector = Collector::new(provider, clock, Duration::from_secs(1));
        let first = collector.collect();
        let second = collector.collect();
        assert_eq!(first.samples[0].descriptor, descriptor);
        assert_eq!(second.samples[0].descriptor, descriptor);
        assert_eq!(first.samples[0].value, Some(37.0));
        assert_eq!(second.samples[0].value, Some(0.0));
        assert_eq!(first.samples[0].timestamp.ns, 1_000_000_000);
        assert_eq!(second.samples[0].timestamp.ns, 1_750_000_000);
        assert_eq!(second.elapsed_ns, Some(750_000_000));
        assert_eq!(second.samples[0].interval, Some(interval));
        assert_eq!(first.samples[0].status, SampleStatus::Ok);
        assert_eq!(second.samples[0].status, SampleStatus::Ok);
    }

    #[test]
    fn capabilities_and_temporary_gaps_do_not_erase_other_metrics() {
        let unsupported = Capability::Unsupported("not implemented".into());
        let denied = Capability::PermissionDenied("restricted".into());
        let temporary = Capability::TemporarilyUnavailable("retry later".into());
        let native_timestamp = Timestamp {
            ns: 123,
            clock_domain: "native_clock".into(),
        };
        let provider = FakeProvider {
            descriptors: vec![
                descriptor("good", Capability::Available),
                descriptor("unsupported", unsupported.clone()),
                descriptor("denied", denied.clone()),
                descriptor("flaky", Capability::Available),
            ],
            batches: VecDeque::from([vec![
                reading(
                    "good",
                    Ok(SourceValue {
                        value: 37.0,
                        timestamp: Some(native_timestamp.clone()),
                        interval: None,
                    }),
                ),
                reading("unsupported", Ok(SourceValue::gauge(0.0))),
                reading("denied", Ok(SourceValue::gauge(0.0))),
                reading("flaky", Err(temporary.clone())),
            ]]),
        };
        let clock = FakeClock(RefCell::new(VecDeque::from([1_000])));
        let mut collector = Collector::new(provider, clock, Duration::from_secs(1));
        let batch = collector.collect();
        assert_eq!(batch.samples[0].value, Some(37.0));
        assert_eq!(batch.samples[0].timestamp, native_timestamp);
        assert_eq!(batch.samples[0].observed_at.ns, 1_000);
        for (sample, expected) in batch.samples[1..]
            .iter()
            .zip([unsupported, denied, temporary])
        {
            assert_eq!(sample.value, None);
            assert_eq!(sample.descriptor.capability, expected);
            assert_eq!(sample.status, SampleStatus::Unavailable(expected));
        }
    }
}
