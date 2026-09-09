pub mod collect;
pub mod model;
#[cfg(feature = "nvidia")]
pub mod nvidia;
pub mod presentation;

pub use collect::{
    BackendOutcome, Clock, Coordinator, LinuxSystemBackend, MetricReading, MonotonicClock,
    NamedMeasurement, Provider, ProviderBatch, SourceReading, SystemBackend, SystemProvider,
};
pub use model::{
    CapabilityState, EntityKind, MetricDescriptor, MetricId, MetricValue, Observation,
    SampleStatus, Series, SeriesKey, TelemetryModel, TemporalSemantics, Unit, ValueKind,
    LIVE_RETENTION, LIVE_RETENTION_NS,
};
#[cfg(feature = "nvidia")]
pub use nvidia::{
    classify_nvml, nvidia_descriptor, NvidiaBackend, NvidiaDevice, NvidiaMetric, NvidiaProvider,
    NvmlBackend,
};
pub use presentation::{
    project_visible_traces, ModelQuery, PresentationSnapshot, SensorGroup, VisibleStatus,
    VisibleTrace,
};

pub type Collector<C> = Coordinator<C>;
pub type LinuxMonotonicClock = MonotonicClock;
pub type CapabilityStatus = CapabilityState;
pub type Reading = MetricReading;
pub const LIVE_WINDOW_NS: u64 = LIVE_RETENTION_NS;

pub fn coordinator_with_system<C, S>(clock: C, system: S) -> Coordinator<C>
where
    C: Clock,
    S: SystemBackend + 'static,
{
    Coordinator::from_providers(clock, vec![Box::new(SystemProvider::new(system))])
}

#[cfg(feature = "nvidia")]
pub fn coordinator_with_backends<C, S, N>(clock: C, system: S, nvidia: Option<N>) -> Coordinator<C>
where
    C: Clock,
    S: SystemBackend + 'static,
    N: NvidiaBackend + 'static,
{
    let mut providers: Vec<Box<dyn Provider>> = vec![Box::new(SystemProvider::new(system))];
    if let Some(backend) = nvidia {
        providers.push(Box::new(NvidiaProvider::new(backend)));
    }
    Coordinator::from_providers(clock, providers)
}

pub fn production_coordinator() -> Coordinator<MonotonicClock> {
    #[cfg(feature = "nvidia")]
    {
        coordinator_with_backends(
            MonotonicClock,
            LinuxSystemBackend::discover(),
            Some(NvmlBackend::default()),
        )
    }
    #[cfg(not(feature = "nvidia"))]
    {
        coordinator_with_system(MonotonicClock, LinuxSystemBackend::discover())
    }
}
