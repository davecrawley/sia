//! The sole sampling path. Providers and clocks can be replaced independently.

use crate::clock::Clock;
use crate::model::{
    Capability, CapabilityStatus, CollectionBatch, MetricDescriptor, MetricSample, Reading,
    SampleStatus, Unavailable,
};
use std::io;
use std::time::Duration;

pub trait Provider {
    fn discover(&mut self) -> Vec<Capability>;

    /// Refresh shared source state once per collection, before individual reads.
    fn prepare(&mut self, _observation_ns: u64, _previous_ns: Option<u64>, _clock_domain: &str) {}

    fn read(&mut self, metric: &MetricDescriptor) -> Result<Reading, Unavailable>;
}

/// Presentation depends on this interface and model values, not provider types.
pub trait Collection {
    fn capabilities(&self) -> Vec<Capability>;
    fn collect(&mut self) -> io::Result<CollectionBatch>;
}

struct Entry {
    provider: usize,
    capability: Capability,
}

pub struct Collector<C> {
    clock: C,
    providers: Vec<Box<dyn Provider>>,
    entries: Vec<Entry>,
    previous_ns: Option<u64>,
    period: Duration,
}

impl<C: Clock> Collector<C> {
    /// Construction discovers metrics but does not consume an observation time.
    pub fn new(mut providers: Vec<Box<dyn Provider>>, clock: C, period: Duration) -> Self {
        let mut entries = Vec::new();
        for (provider, source) in providers.iter_mut().enumerate() {
            entries.extend(source.discover().into_iter().map(|capability| Entry {
                provider,
                capability,
            }));
        }
        Self {
            clock,
            providers,
            entries,
            previous_ns: None,
            period,
        }
    }

    pub fn configured_period(&self) -> Duration {
        self.period
    }
}

impl<C: Clock> Collection for Collector<C> {
    fn capabilities(&self) -> Vec<Capability> {
        self.entries
            .iter()
            .map(|entry| entry.capability.clone())
            .collect()
    }

    fn collect(&mut self) -> io::Result<CollectionBatch> {
        let observation_ns = self.clock.now_ns()?;
        let clock = self.clock.identity();
        let elapsed_ns = self
            .previous_ns
            .and_then(|previous| observation_ns.checked_sub(previous));
        for provider in &mut self.providers {
            provider.prepare(observation_ns, self.previous_ns, &clock.domain);
        }
        self.previous_ns = Some(observation_ns);
        let mut samples = Vec::with_capacity(self.entries.len());
        for entry in &mut self.entries {
            let capability = &mut entry.capability;
            let result = match capability.status {
                CapabilityStatus::Unsupported | CapabilityStatus::PermissionDenied => {
                    Err(Unavailable {
                        status: capability.status,
                        reason: capability.reason.clone().unwrap_or_default(),
                    })
                }
                CapabilityStatus::Available | CapabilityStatus::TemporarilyUnavailable => {
                    self.providers[entry.provider].read(&capability.descriptor)
                }
            };
            let result = result.and_then(|reading| {
                if reading.value.is_finite() {
                    Ok(reading)
                } else {
                    Err(Unavailable::temporary("source returned a non-finite value"))
                }
            });
            let mut sample = MetricSample {
                descriptor: capability.descriptor.clone(),
                mono_ns: observation_ns,
                source_timestamp: None,
                window: None,
                value: None,
                status: SampleStatus::Ok,
            };
            match result {
                Ok(reading) => {
                    capability.status = CapabilityStatus::Available;
                    capability.reason = None;
                    sample.mono_ns = reading.mono_ns.unwrap_or(observation_ns);
                    sample.source_timestamp = reading.source_timestamp;
                    sample.window = reading.window;
                    sample.value = Some(reading.value);
                }
                Err(error) => {
                    capability.status = error.status;
                    capability.reason = Some(error.reason.clone());
                    sample.status = SampleStatus::Unavailable(error);
                }
            }
            samples.push(sample);
        }
        Ok(CollectionBatch {
            clock,
            observation_ns,
            elapsed_ns,
            samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        ClockIdentity, MetricKind, ObservationWindow, TemporalSemantics, Unit, ValueKind,
    };
    use std::collections::VecDeque;

    struct FakeClock(VecDeque<u64>);

    impl Clock for FakeClock {
        fn now_ns(&mut self) -> io::Result<u64> {
            Ok(self.0.pop_front().expect("unexpected clock read"))
        }

        fn identity(&self) -> ClockIdentity {
            ClockIdentity {
                domain: "fixture".into(),
                boot_id: None,
                time_namespace: None,
            }
        }
    }

    struct FakeProvider {
        descriptor: MetricDescriptor,
        readings: VecDeque<Result<Reading, Unavailable>>,
    }

    impl Provider for FakeProvider {
        fn discover(&mut self) -> Vec<Capability> {
            vec![Capability {
                descriptor: self.descriptor.clone(),
                status: CapabilityStatus::Available,
                reason: None,
            }]
        }

        fn read(&mut self, metric: &MetricDescriptor) -> Result<Reading, Unavailable> {
            assert_eq!(metric, &self.descriptor);
            self.readings.pop_front().expect("unexpected provider read")
        }
    }

    fn descriptor() -> MetricDescriptor {
        MetricDescriptor {
            metric_id: "fixture.gauge".into(),
            entity_id: "fixture:1".into(),
            display_name: "Fixture".into(),
            kind: MetricKind::CpuUtilization,
            unit: Unit::Percent,
            value_kind: ValueKind::Gauge,
            temporal_semantics: TemporalSemantics::PointSample,
            provider: "fixture".into(),
            source_semantics: "deterministic fixture gauge".into(),
            semantics_version: 1,
        }
    }

    #[test]
    fn preserves_zero_metadata_actual_time_and_source_window() {
        let descriptor = descriptor();
        let window = ObservationWindow {
            start_ns: 123,
            end_ns: 456,
            clock_domain: "source".into(),
        };
        let mut second = Reading::gauge(0.0);
        second.window = Some(window.clone());
        let provider = FakeProvider {
            descriptor: descriptor.clone(),
            readings: VecDeque::from([Ok(Reading::gauge(37.0)), Ok(second)]),
        };
        let clock = FakeClock(VecDeque::from([1_000_000_000, 1_750_000_000]));
        let mut collector = Collector::new(vec![Box::new(provider)], clock, Duration::from_secs(1));
        let first = collector.collect().unwrap();
        let second = collector.collect().unwrap();
        assert_eq!(first.samples[0].value, Some(37.0));
        assert_eq!(first.samples[0].mono_ns, 1_000_000_000);
        assert_eq!(second.samples[0].value, Some(0.0));
        assert_eq!(second.samples[0].mono_ns, 1_750_000_000);
        assert_eq!(second.elapsed_ns, Some(750_000_000));
        assert_eq!(second.samples[0].window, Some(window));
        assert_eq!(second.samples[0].descriptor, descriptor);
        assert_eq!(second.samples[0].status, SampleStatus::Ok);
    }

    #[test]
    fn unavailable_metric_does_not_erase_healthy_metric() {
        for status in [
            CapabilityStatus::Unsupported,
            CapabilityStatus::PermissionDenied,
            CapabilityStatus::TemporarilyUnavailable,
        ] {
            let failed = FakeProvider {
                descriptor: descriptor(),
                readings: VecDeque::from([Err(Unavailable {
                    status,
                    reason: "fixture failure".into(),
                })]),
            };
            let mut healthy_descriptor = descriptor();
            healthy_descriptor.entity_id = "fixture:2".into();
            let healthy = FakeProvider {
                descriptor: healthy_descriptor,
                readings: VecDeque::from([Ok(Reading::gauge(37.0))]),
            };
            let mut collector = Collector::new(
                vec![Box::new(failed), Box::new(healthy)],
                FakeClock(VecDeque::from([1])),
                Duration::from_secs(1),
            );
            let batch = collector.collect().unwrap();
            assert_eq!(batch.samples[0].value, None);
            assert!(matches!(
                &batch.samples[0].status,
                SampleStatus::Unavailable(error) if error.status == status
            ));
            assert_eq!(collector.capabilities()[0].status, status);
            assert_eq!(batch.samples[1].value, Some(37.0));
        }
    }
}
