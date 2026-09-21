use crate::clock::Clock;
use crate::model::{CollectionBatch, ProviderOutput, Timestamp};
use std::{fmt, io};

/// Providers receive only a compatible previous observation. A missing previous
/// observation also instructs stateful providers to reset interval baselines.
/// The timestamp is the actual clock reading at the start of this poll; vendor
/// timestamps and vendor windows, when known, belong in the sample separately.
pub trait Provider {
    fn collect(&mut self, now: &Timestamp, previous: Option<&Timestamp>) -> ProviderOutput;
}

#[derive(Debug)]
pub enum CollectionError {
    Clock(io::Error),
    InvalidTimestamp,
}

impl fmt::Display for CollectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clock(error) => write!(formatter, "collection clock: {error}"),
            Self::InvalidTimestamp => {
                write!(formatter, "incompatible or reversed source timestamp")
            }
        }
    }
}

impl std::error::Error for CollectionError {}

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

    pub fn collect(&mut self) -> Result<CollectionBatch, CollectionError> {
        let now = match self.clock.now() {
            Ok(now) => now,
            Err(error) => {
                self.previous = None;
                return Err(CollectionError::Clock(error));
            }
        };
        let elapsed = self
            .previous
            .as_ref()
            .and_then(|previous| now.elapsed_since(previous));
        let discontinuity = self.previous.is_some() && elapsed.is_none();
        if elapsed.is_none() {
            self.previous = None;
        }
        let mut batch = CollectionBatch {
            observed_at: now.clone(),
            elapsed,
            discontinuity,
            descriptors: Vec::new(),
            samples: Vec::new(),
        };
        for provider in &mut self.providers {
            let output = provider.collect(&now, self.previous.as_ref());
            let compatible = |timestamp: &Timestamp| timestamp.identity == now.identity;
            for sample in &output.samples {
                let source_valid = sample
                    .source_timestamp
                    .as_ref()
                    .map(&compatible)
                    .unwrap_or(true);
                let window_valid = sample
                    .observation_window
                    .as_ref()
                    .map(|window| {
                        compatible(&window.start)
                            && compatible(&window.end)
                            && window.elapsed().is_some()
                    })
                    .unwrap_or(true);
                if !compatible(&sample.observed_at) || !source_valid || !window_valid {
                    self.previous = None;
                    return Err(CollectionError::InvalidTimestamp);
                }
            }
            batch.descriptors.extend(output.descriptors);
            batch.samples.extend(output.samples);
        }
        self.previous = Some(now);
        Ok(batch)
    }
}
