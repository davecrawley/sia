use crate::model::{CapabilityStatus, EntityId, MetricId, SampleStatus, SampleValue, SessionModel};

#[derive(Clone, Debug, PartialEq)]
pub enum CurrentValue {
    Value(SampleValue),
    Missing,
    Stale,
    Unsupported(String),
    PermissionDenied(String),
    TemporarilyUnavailable(String),
    Error(String),
}

pub fn current_value(model: &SessionModel, metric: &MetricId, entity: &EntityId) -> CurrentValue {
    if let Some(capability) = model.capability(metric, entity) {
        match capability.status {
            CapabilityStatus::Unsupported => {
                return CurrentValue::Unsupported(
                    capability
                        .detail
                        .clone()
                        .unwrap_or_else(|| "unsupported".into()),
                );
            }
            CapabilityStatus::PermissionDenied => {
                return CurrentValue::PermissionDenied(
                    capability
                        .detail
                        .clone()
                        .unwrap_or_else(|| "permission denied".into()),
                );
            }
            CapabilityStatus::TemporarilyUnavailable if model.latest(metric, entity).is_none() => {
                return CurrentValue::TemporarilyUnavailable(
                    capability
                        .detail
                        .clone()
                        .unwrap_or_else(|| "temporarily unavailable".into()),
                );
            }
            CapabilityStatus::Available | CapabilityStatus::TemporarilyUnavailable => {}
        }
    }
    let Some(sample) = model.latest(metric, entity) else {
        return CurrentValue::Missing;
    };
    match &sample.status {
        SampleStatus::Ok => sample
            .value
            .clone()
            .map(CurrentValue::Value)
            .unwrap_or(CurrentValue::Missing),
        SampleStatus::Stale => CurrentValue::Stale,
        SampleStatus::TemporarilyUnavailable(detail) => {
            CurrentValue::TemporarilyUnavailable(detail.clone())
        }
        SampleStatus::Error(detail) => CurrentValue::Error(detail.clone()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlotPoint {
    pub monotonic_ns: u64,
    pub interval_start_ns: Option<u64>,
    pub value: f64,
}

pub fn numeric_segments(
    model: &SessionModel,
    metric: &MetricId,
    entity: &EntityId,
) -> Vec<Vec<PlotPoint>> {
    let mut result = Vec::new();
    let mut segment = Vec::new();
    for sample in model.samples(metric, entity) {
        let value = if sample.status == SampleStatus::Ok {
            sample.value.as_ref().and_then(SampleValue::numeric)
        } else {
            None
        };
        if let Some(value) = value.filter(|value| value.is_finite()) {
            segment.push(PlotPoint {
                monotonic_ns: sample.observation_time.monotonic_ns,
                interval_start_ns: sample.interval_start.map(|time| time.monotonic_ns),
                value,
            });
        } else if !segment.is_empty() {
            result.push(std::mem::take(&mut segment));
        }
    }
    if !segment.is_empty() {
        result.push(segment);
    }
    result
}
