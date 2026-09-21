use std::io;

use crate::clock::{Clock, Timestamp};
use crate::model::{MetricDescriptor, ProviderBatch, Sample};

/// Previous is supplied only when the complete clock identity is compatible
/// and time has not gone backwards. Providers must reset interval baselines
/// when it is absent.
#[derive(Clone, Debug)]
pub struct CollectionContext {
    pub observed_at: Timestamp,
    pub previous: Option<Timestamp>,
}

impl CollectionContext {
    pub fn elapsed_ns(&self) -> Option<u64> {
        self.previous
            .as_ref()
            .and_then(|previous| self.observed_at.elapsed_since(previous))
    }
}

pub trait Provider {
    fn collect(&mut self, context: &CollectionContext) -> ProviderBatch;
}

#[derive(Clone, Debug, PartialEq)]
pub struct CollectionBatch {
    pub observed_at: Timestamp,
    pub elapsed_ns: Option<u64>,
    pub discontinuity: bool,
    pub descriptors: Vec<MetricDescriptor>,
    pub samples: Vec<Sample>,
}

pub struct Collector<C> {
    clock: C,
    providers: Vec<Box<dyn Provider>>,
    previous: Option<Timestamp>,
}

impl<C: Clock> Collector<C> {
    pub fn new(clock: C, providers: Vec<Box<dyn Provider>>) -> Self {
        Self {
            clock,
            providers,
            previous: None,
        }
    }

    pub fn collect(&mut self) -> io::Result<CollectionBatch> {
        let observed_at = match self.clock.now() {
            Ok(timestamp) => timestamp,
            Err(error) => {
                self.previous = None;
                return Err(error);
            }
        };
        let elapsed_ns = self
            .previous
            .as_ref()
            .and_then(|previous| observed_at.elapsed_since(previous));
        let discontinuity = self.previous.is_some() && elapsed_ns.is_none();
        let context = CollectionContext {
            observed_at: observed_at.clone(),
            previous: if elapsed_ns.is_some() {
                self.previous.clone()
            } else {
                None
            },
        };
        let mut batch = CollectionBatch {
            observed_at: observed_at.clone(),
            elapsed_ns,
            discontinuity,
            descriptors: Vec::new(),
            samples: Vec::new(),
        };
        for provider in &mut self.providers {
            let output = provider.collect(&context);
            // Provider-owned source timestamps and windows are passed through.
            batch.descriptors.extend(output.descriptors);
            batch.samples.extend(output.samples);
        }
        self.previous = Some(observed_at);
        Ok(batch)
    }
}
