use crate::model::{CapabilityStatus, MetricDescriptor, MetricSample, TelemetryModel};
use std::collections::VecDeque;

pub trait ModelQuery {
    fn descriptors(&self) -> Vec<&MetricDescriptor>;
    fn series(&self) -> Vec<(&(String, String), &VecDeque<MetricSample>)>;
}

impl ModelQuery for TelemetryModel {
    fn descriptors(&self) -> Vec<&MetricDescriptor> {
        self.descriptors().collect()
    }

    fn series(&self) -> Vec<(&(String, String), &VecDeque<MetricSample>)> {
        self.all_series().collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisibleTrace {
    pub metric_id: String,
    pub entity_id: String,
    pub display_name: String,
    pub unit: String,
    pub points: Vec<(u64, f64)>,
}

pub fn project_visible_traces(model: &impl ModelQuery) -> Vec<VisibleTrace> {
    let descriptors = model.descriptors();
    model
        .series()
        .into_iter()
        .filter_map(|((metric_id, entity_id), samples)| {
            let descriptor = descriptors
                .iter()
                .copied()
                .find(|descriptor| descriptor.metric_id == *metric_id)?;
            if descriptor.capability_status != CapabilityStatus::Available {
                return None;
            }
            let points: Vec<_> = samples
                .iter()
                .filter(|sample| sample.is_plottable())
                .filter_map(|sample| {
                    Some((sample.observation_ns, sample.value.as_ref()?.as_f64()?))
                })
                .collect();
            if points.is_empty() {
                return None;
            }
            Some(VisibleTrace {
                metric_id: metric_id.clone(),
                entity_id: entity_id.clone(),
                display_name: descriptor.display_name.clone(),
                unit: descriptor.unit.clone(),
                points,
            })
        })
        .collect()
}
