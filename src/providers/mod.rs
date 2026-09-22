mod native;
mod nvidia;

pub use native::NativeProvider;
pub use nvidia::{NvidiaInitializer, NvidiaProvider, NvidiaSource};

use crate::model::{
    Capability, EntityKind, MetricDescriptor, MetricSample, MetricValue, ObservationWindow,
    ProviderOutput, TemporalSemantics, Timestamp, Unavailable, Unit, ValueKind,
};

fn descriptor(
    metric_id: &str,
    entity_id: &str,
    entity_kind: EntityKind,
    display_name: &str,
    unit: Unit,
    provider: &str,
    source_semantics: &str,
) -> MetricDescriptor {
    MetricDescriptor {
        metric_id: metric_id.into(),
        entity_id: entity_id.into(),
        entity_kind,
        entity_display_name: display_name.into(),
        display_name: display_name.into(),
        unit,
        value_kind: ValueKind::Gauge,
        temporal_semantics: TemporalSemantics::PointSample,
        provider: provider.into(),
        source_semantics: source_semantics.into(),
        source_resolution_hint: None,
        comparability_group: None,
        semantics_version: 1,
        capability: Capability::Available,
    }
}

fn emit(
    output: &mut ProviderOutput,
    mut descriptor: MetricDescriptor,
    now: &Timestamp,
    window: Option<ObservationWindow>,
    value: Result<MetricValue, Unavailable>,
) {
    descriptor.capability = match &value {
        Ok(_) => Capability::Available,
        Err(reason) => Capability::from(reason),
    };
    output.samples.push(MetricSample {
        metric_id: descriptor.metric_id.clone(),
        entity_id: descriptor.entity_id.clone(),
        observed_at: now.clone(),
        source_timestamp: None,
        observation_window: window,
        value,
    });
    output.descriptors.push(descriptor);
}
