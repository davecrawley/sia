use crate::model::{MetricDescriptor, MetricSample, MetricValue, SampleStatus, TelemetryModel};

pub trait Clock {
    fn now_ns(&mut self) -> u64;
}

#[derive(Debug)]
pub struct LinuxMonotonicClock;

impl Clock for LinuxMonotonicClock {
    fn now_ns(&mut self) -> u64 {
        let mut time = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: clock_gettime writes to a valid timespec and has no ownership requirements.
        let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) };
        if result != 0 {
            return 0;
        }
        (time.tv_sec as u64)
            .saturating_mul(1_000_000_000)
            .saturating_add(time.tv_nsec as u64)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Reading {
    pub metric_id: String,
    pub entity_id: String,
    pub value: Option<MetricValue>,
    pub status: SampleStatus,
    pub source_timestamp_ns: Option<u64>,
    pub window_start_ns: Option<u64>,
}

impl Reading {
    pub fn available(
        metric_id: impl Into<String>,
        entity_id: impl Into<String>,
        value: MetricValue,
    ) -> Self {
        Self {
            metric_id: metric_id.into(),
            entity_id: entity_id.into(),
            value: Some(value),
            status: SampleStatus::Ok,
            source_timestamp_ns: None,
            window_start_ns: None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProviderBatch {
    pub descriptors: Vec<MetricDescriptor>,
    pub readings: Vec<Reading>,
}

pub trait Provider {
    fn collect(&mut self) -> ProviderBatch;
}

pub struct Collector<C: Clock> {
    clock: C,
    providers: Vec<Box<dyn Provider>>,
    model: TelemetryModel,
}

impl<C: Clock> Collector<C> {
    pub fn new(clock: C) -> Self {
        Self {
            clock,
            providers: Vec::new(),
            model: TelemetryModel::default(),
        }
    }

    pub fn add_provider<P: Provider + 'static>(&mut self, provider: P) {
        self.providers.push(Box::new(provider));
    }

    pub fn collect_once(&mut self) {
        for provider in &mut self.providers {
            let batch = provider.collect();
            for descriptor in batch.descriptors {
                self.model.register(descriptor);
            }
            let needs_timestamp = batch
                .readings
                .iter()
                .any(|reading| reading.source_timestamp_ns.is_none());
            let observed = needs_timestamp.then(|| self.clock.now_ns());
            for reading in batch.readings {
                let observation_ns = reading
                    .source_timestamp_ns
                    .or(observed)
                    .expect("a collection timestamp is always available");
                self.model.ingest(MetricSample {
                    observation_ns,
                    window_start_ns: reading.window_start_ns,
                    metric_id: reading.metric_id,
                    entity_id: reading.entity_id,
                    value: reading.value,
                    status: reading.status,
                });
            }
        }
    }

    pub fn model(&self) -> &TelemetryModel {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut TelemetryModel {
        &mut self.model
    }

    pub fn into_model(self) -> TelemetryModel {
        self.model
    }
}
