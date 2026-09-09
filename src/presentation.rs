use crate::model::{
    CapabilityState, EntityKind, MetricDescriptor, Observation, SampleStatus, Series, SeriesKey,
    TelemetryModel, Unit,
};
use std::collections::BTreeMap;
use std::time::Duration;

pub trait ModelQuery {
    fn latest_observation_ns(&self) -> Option<u64>;
    fn series(&self) -> Vec<(&SeriesKey, &Series)>;
}

impl ModelQuery for TelemetryModel {
    fn latest_observation_ns(&self) -> Option<u64> {
        self.latest_observation_ns()
    }

    fn series(&self) -> Vec<(&SeriesKey, &Series)> {
        self.all_series().collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisibleTrace {
    pub key: SeriesKey,
    pub display_name: String,
    pub source_name: String,
    pub entity_kind: EntityKind,
    pub unit: Unit,
    pub points: Vec<(u64, f64)>,
    pub segments: Vec<Vec<(u64, f64)>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisibleStatus {
    pub key: SeriesKey,
    pub display_name: String,
    pub capability: CapabilityState,
    pub latest_sample_status: Option<SampleStatus>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SensorGroup {
    pub key: String,
    pub display_name: String,
    pub trace_keys: Vec<SeriesKey>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PresentationSnapshot {
    pub traces: Vec<VisibleTrace>,
    pub statuses: Vec<VisibleStatus>,
    pub groups: Vec<SensorGroup>,
    pub latest_observation_ns: Option<u64>,
}

impl PresentationSnapshot {
    pub fn from_model(model: &impl ModelQuery, window: Duration) -> Self {
        let latest = model.latest_observation_ns();
        let cutoff = latest
            .unwrap_or(0)
            .saturating_sub(window.as_nanos().min(u64::MAX as u128) as u64);
        let mut traces = Vec::new();
        let mut statuses = Vec::new();
        let mut groups: BTreeMap<String, SensorGroup> = BTreeMap::new();

        for (key, series) in model.series() {
            statuses.push(VisibleStatus {
                key: key.clone(),
                display_name: series.descriptor.display_name.clone(),
                capability: series.descriptor.capability.clone(),
                latest_sample_status: series
                    .observations
                    .last()
                    .map(|sample| sample.status.clone()),
            });

            if matches!(
                &series.descriptor.capability,
                CapabilityState::Unsupported(_)
            ) {
                continue;
            }
            let segments = plottable_segments(&series.observations, cutoff);
            let points: Vec<_> = segments.iter().flatten().copied().collect();
            if points.is_empty() {
                continue;
            }

            let group = group_for(&series.descriptor);
            groups
                .entry(group.0.clone())
                .or_insert_with(|| SensorGroup {
                    key: group.0,
                    display_name: group.1,
                    trace_keys: Vec::new(),
                })
                .trace_keys
                .push(key.clone());

            traces.push(VisibleTrace {
                key: key.clone(),
                display_name: series.descriptor.display_name.clone(),
                source_name: series.descriptor.provider.clone(),
                entity_kind: series.descriptor.entity_kind.clone(),
                unit: series.descriptor.unit.clone(),
                points,
                segments,
            });
        }

        for group in groups.values_mut() {
            group.trace_keys.sort();
        }
        traces.sort_by(|left, right| left.key.cmp(&right.key));
        statuses.sort_by(|left, right| left.key.cmp(&right.key));

        Self {
            traces,
            statuses,
            groups: groups.into_values().collect(),
            latest_observation_ns: latest,
        }
    }
}

pub fn project_visible_traces(model: &impl ModelQuery) -> Vec<VisibleTrace> {
    PresentationSnapshot::from_model(model, crate::model::LIVE_RETENTION).traces
}

fn plottable_segments(observations: &[Observation], cutoff: u64) -> Vec<Vec<(u64, f64)>> {
    let mut segments = Vec::new();
    let mut current = Vec::new();
    for observation in observations
        .iter()
        .filter(|sample| sample.observation_ns >= cutoff)
    {
        let point = if observation.is_plottable() {
            observation
                .value
                .as_ref()
                .and_then(|value| value.as_f64())
                .map(|value| (observation.observation_ns, value))
        } else {
            None
        };
        match point {
            Some(point) => current.push(point),
            None if !current.is_empty() => segments.push(std::mem::take(&mut current)),
            None => {}
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

fn group_for(descriptor: &MetricDescriptor) -> (String, String) {
    match &descriptor.entity_kind {
        EntityKind::Cpu => ("cpu".to_owned(), "CPU".to_owned()),
        EntityKind::Gpu => ("gpu".to_owned(), "GPU".to_owned()),
        EntityKind::System => ("system".to_owned(), "System".to_owned()),
        EntityKind::Thermal => {
            let text = format!(
                "{} {}",
                descriptor.display_name.to_lowercase(),
                descriptor.key.entity_id.to_lowercase()
            );
            if text.contains("nvme") {
                ("nvme".to_owned(), "NVMe SSD".to_owned())
            } else if text.contains("amdgpu") || text.contains("gpu") {
                ("gpu".to_owned(), "GPU".to_owned())
            } else if text.contains("coretemp")
                || text.contains("k10temp")
                || text.contains("cpu")
                || text.contains("package")
            {
                ("cpu-thermal".to_owned(), "CPU temperatures".to_owned())
            } else if text.contains("iwlwifi") {
                ("wifi".to_owned(), "Wi-Fi Controller".to_owned())
            } else if text.contains("r8169")
                || text.contains("r8125")
                || text.contains("e1000")
                || text.contains("igc")
            {
                ("ethernet".to_owned(), "Ethernet Controller".to_owned())
            } else {
                ("thermal".to_owned(), "Other temperatures".to_owned())
            }
        }
        EntityKind::Other(value) => (value.clone(), value.clone()),
    }
}
