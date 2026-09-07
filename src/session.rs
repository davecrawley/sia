use crate::collector::{CollectionError, MetricCollector};
use crate::model::{MetricDescriptor, MetricSample, SessionMetadata, SessionSnapshot};
use std::collections::BTreeSet;

pub trait SessionSource {
    fn metadata(&self) -> &SessionMetadata;
    fn descriptors(&self) -> &[MetricDescriptor];
    fn poll(&mut self) -> Result<Vec<MetricSample>, CollectionError>;
}

pub struct LocalLiveSource {
    metadata: SessionMetadata,
    descriptors: Vec<MetricDescriptor>,
    collectors: Vec<Box<dyn MetricCollector>>,
}
impl LocalLiveSource {
    pub fn new(collectors: Vec<Box<dyn MetricCollector>>) -> Self {
        let clock_domain = collectors
            .first()
            .map(|c| c.clock_domain())
            .unwrap_or("linux_clock_monotonic")
            .to_owned();
        let mut seen = BTreeSet::new();
        let mut descriptors = Vec::new();
        for collector in &collectors {
            for descriptor in collector.descriptors() {
                if seen.insert(descriptor.metric_id.clone()) {
                    descriptors.push(descriptor.clone());
                }
            }
        }
        Self {
            metadata: SessionMetadata {
                schema_version: 1,
                clock_domain,
            },
            descriptors,
            collectors,
        }
    }
    pub fn snapshot(&mut self) -> Result<SessionSnapshot, CollectionError> {
        Ok(SessionSnapshot {
            metadata: self.metadata.clone(),
            descriptors: self.descriptors.clone(),
            samples: self.poll()?,
        })
    }
}
impl SessionSource for LocalLiveSource {
    fn metadata(&self) -> &SessionMetadata {
        &self.metadata
    }
    fn descriptors(&self) -> &[MetricDescriptor] {
        &self.descriptors
    }
    fn poll(&mut self) -> Result<Vec<MetricSample>, CollectionError> {
        let mut samples = Vec::new();
        for collector in &mut self.collectors {
            samples.extend(collector.collect()?);
        }
        samples.sort_by_key(|sample| sample.observation_time);
        Ok(samples)
    }
}
