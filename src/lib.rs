pub mod clock;
pub mod collector;
pub mod model;
pub mod provider;
pub mod session;

pub use clock::{Clock, ClockError, DeterministicClock, LinuxMonotonicClock};
pub use collector::{
    CollectionError, FakeMetricProvider, MetricCollector, MetricProvider, NativePoint,
    ProviderResult, TypedCollector,
};
pub use model::*;
pub use provider::{Discovery, NvidiaProvider, SystemProvider};
pub use session::{LocalLiveSource, SessionSource};

pub fn default_live_source() -> LocalLiveSource {
    let mut collectors: Vec<Box<dyn MetricCollector>> = Vec::new();
    collectors.push(Box::new(TypedCollector::new(
        SystemProvider::new(),
        LinuxMonotonicClock,
    )));

    #[cfg(feature = "nvidia")]
    if let Discovery::Available(provider) = NvidiaProvider::discover() {
        collectors.push(Box::new(TypedCollector::new(
            provider,
            LinuxMonotonicClock,
        )));
    }

    LocalLiveSource::new(collectors)
}