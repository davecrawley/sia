use crate::clock::Clock;
use crate::model::{CollectionBatch, ProviderOutput, Timestamp};
use std::{fmt, io};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceState {
    Discovered,
    Available,
    TemporarilyUnavailable,
    Lost,
    Removed,
    Reappeared,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceMetadata {
    pub entity_id: String,
    pub display_name: String,
    pub uuid: Option<String>,
    pub pci_address: Option<String>,
    pub driver: String,
    pub driver_version: Option<String>,
    pub generation: u64,
    pub state: DeviceState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceEvent {
    pub observed_at: Timestamp,
    pub entity_id: String,
    pub generation: u64,
    pub state: DeviceState,
    pub reason: String,
    pub environment_changed: bool,
}

/// Events are retained for the session. A generation begins at its Discovered or
/// Reappeared event and applies to that entity's subsequent samples. NVIDIA also
/// emits device.generation at each observation for an explicit sample-time join.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionStatus {
    pub environment_changed: bool,
    pub devices: Vec<DeviceMetadata>,
    pub events: Vec<DeviceEvent>,
}

/// A missing previous observation instructs providers to reset interval baselines.
pub trait Provider {
    fn collect(&mut self, now: &Timestamp, previous: Option<&Timestamp>) -> ProviderOutput;

    fn take_device_events(&mut self) -> Vec<DeviceEvent> {
        Vec::new()
    }

    fn devices(&self) -> Vec<DeviceMetadata> {
        Vec::new()
    }
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
    status: SessionStatus,
}

impl<C: Clock> Collector<C> {
    pub fn new(clock: C, providers: Vec<Box<dyn Provider>>) -> Self {
        Self {
            clock,
            providers,
            previous: None,
            status: SessionStatus::default(),
        }
    }

    pub fn session_status(&self) -> &SessionStatus {
        &self.status
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
            for event in provider.take_device_events() {
                self.status.environment_changed |= event.environment_changed;
                self.status.events.push(event);
            }
            for device in provider.devices() {
                if let Some(existing) = self
                    .status
                    .devices
                    .iter_mut()
                    .find(|existing| existing.entity_id == device.entity_id)
                {
                    *existing = device;
                } else {
                    self.status.devices.push(device);
                }
            }
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
