use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

use manas_core::{HiddenNeuronMerge, ManasError, Network, ProtectionLevel};

use crate::backprop::cosine;
use crate::importance;

pub const DEFAULT_COMPRESSION_THRESHOLD: f32 = 0.10;
pub const DEFAULT_MIN_IDLE_DAYS: u64 = 30;
pub const DEFAULT_MIN_MERGE_SIMILARITY: f32 = 0.98;

const SECONDS_PER_DAY: u64 = 86_400;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompressionConfig {
    pub threshold: f32,
    pub min_idle_days: u64,
    pub min_merge_similarity: f32,
    pub now_secs: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompressionCandidate {
    pub source_index: usize,
    pub source_neuron_id: u64,
    pub target_index: usize,
    pub target_neuron_id: u64,
    pub importance_score: f32,
    pub idle_days: u64,
    pub merge_similarity: f32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompressionSkipCounts {
    pub protected: usize,
    pub high_importance: usize,
    pub recent: usize,
    pub no_merge_target: usize,
    pub unsupported_shape: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompressionPlan {
    pub threshold: f32,
    pub min_idle_days: u64,
    pub min_merge_similarity: f32,
    pub candidates: Vec<CompressionCandidate>,
    pub skipped: CompressionSkipCounts,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompressionReport {
    pub plan: CompressionPlan,
    pub neurons_before: u64,
    pub neurons_after: u64,
    pub neurons_removed: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_COMPRESSION_THRESHOLD,
            min_idle_days: DEFAULT_MIN_IDLE_DAYS,
            min_merge_similarity: DEFAULT_MIN_MERGE_SIMILARITY,
            now_secs: unix_now_secs(),
        }
    }
}

impl CompressionConfig {
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            threshold,
            ..Self::default()
        }
    }
}

impl CompressionPlan {
    pub fn projected_removed(&self) -> usize {
        self.candidates.len()
    }
}

pub fn plan_compression(
    network: &Network,
    config: &CompressionConfig,
) -> Result<CompressionPlan, ManasError> {
    validate_config(config)?;
    let mut plan = CompressionPlan {
        threshold: config.threshold,
        min_idle_days: config.min_idle_days,
        min_merge_similarity: config.min_merge_similarity,
        candidates: Vec::new(),
        skipped: CompressionSkipCounts::default(),
    };

    if network.neuron_count() == 0 {
        return Ok(plan);
    }
    if network.layers.len() != 2
        || network.layers[0].neurons.is_empty()
        || network.layers[1].neurons.is_empty()
    {
        plan.skipped.unsupported_shape = network.neuron_count() as usize;
        return Ok(plan);
    }

    let hidden_len = network.layers[0].neurons.len();
    let potential_sources = potential_sources(network, config, &mut plan.skipped)?;
    if potential_sources.is_empty() {
        return Ok(plan);
    }

    let source_set = potential_sources.iter().copied().collect::<HashSet<_>>();
    for source_index in potential_sources {
        if source_set.len() >= hidden_len {
            plan.skipped.no_merge_target += 1;
            continue;
        }

        let Some((target_index, merge_similarity)) =
            nearest_merge_target(network, source_index, &source_set, config)
        else {
            plan.skipped.no_merge_target += 1;
            continue;
        };

        let source = &network.layers[0].neurons[source_index];
        let target = &network.layers[0].neurons[target_index];
        plan.candidates.push(CompressionCandidate {
            source_index,
            source_neuron_id: source.id,
            target_index,
            target_neuron_id: target.id,
            importance_score: source.importance_score,
            idle_days: idle_days(source.last_activated, source.born_at, config.now_secs),
            merge_similarity,
        });
    }

    Ok(plan)
}

pub fn compress(
    network: &mut Network,
    config: &CompressionConfig,
) -> Result<CompressionReport, ManasError> {
    let plan = plan_compression(network, config)?;
    let neurons_before = network.neuron_count();
    let merges = plan
        .candidates
        .iter()
        .map(|candidate| HiddenNeuronMerge {
            source_index: candidate.source_index,
            target_index: candidate.target_index,
        })
        .collect::<Vec<_>>();

    let neurons_removed = network.merge_remove_hidden_neurons(&merges)?;
    recompute_importance_scores(network, config.now_secs);
    let neurons_after = network.neuron_count();

    Ok(CompressionReport {
        plan,
        neurons_before,
        neurons_after,
        neurons_removed,
    })
}

pub fn recompute_importance_scores(network: &mut Network, now_secs: u64) {
    for neuron in network
        .layers
        .iter_mut()
        .flat_map(|layer| layer.neurons.iter_mut())
    {
        neuron.importance_score = importance::compute_importance(neuron, now_secs);
    }
}

fn potential_sources(
    network: &Network,
    config: &CompressionConfig,
    skipped: &mut CompressionSkipCounts,
) -> Result<Vec<usize>, ManasError> {
    let mut sources = Vec::new();

    for (index, neuron) in network.layers[0].neurons.iter().enumerate() {
        if !matches!(neuron.protection_level, ProtectionLevel::Open) {
            skipped.protected += 1;
            continue;
        }
        if !source_output_edges_are_open(network, index)? {
            skipped.protected += 1;
            continue;
        }
        if neuron.importance_score >= config.threshold {
            skipped.high_importance += 1;
            continue;
        }

        let idle_days = idle_days(neuron.last_activated, neuron.born_at, config.now_secs);
        if idle_days < config.min_idle_days {
            skipped.recent += 1;
            continue;
        }

        sources.push(index);
    }

    Ok(sources)
}

fn nearest_merge_target(
    network: &Network,
    source_index: usize,
    removable_sources: &HashSet<usize>,
    config: &CompressionConfig,
) -> Option<(usize, f32)> {
    let source = &network.layers[0].neurons[source_index];
    let mut best: Option<(usize, f32)> = None;

    for (target_index, target) in network.layers[0].neurons.iter().enumerate() {
        if target_index == source_index
            || removable_sources.contains(&target_index)
            || matches!(target.protection_level, ProtectionLevel::Frozen)
            || !target_output_edges_are_mergeable(network, target_index)
        {
            continue;
        }

        let similarity = cosine(&source.weights, &target.weights);
        if similarity < config.min_merge_similarity {
            continue;
        }

        if best
            .as_ref()
            .map(|(_, best_similarity)| similarity > *best_similarity)
            .unwrap_or(true)
        {
            best = Some((target_index, similarity));
        }
    }

    best
}

fn source_output_edges_are_open(
    network: &Network,
    source_index: usize,
) -> Result<bool, ManasError> {
    for output_neuron in &network.layers[1].neurons {
        let Some(protection) = output_neuron.weight_protection.get(source_index) else {
            return Err(ManasError::InvalidNetwork(format!(
                "output neuron {} is missing source edge {}",
                output_neuron.id, source_index
            )));
        };
        if !matches!(protection, ProtectionLevel::Open) {
            return Ok(false);
        }
    }

    Ok(true)
}

fn target_output_edges_are_mergeable(network: &Network, target_index: usize) -> bool {
    network.layers[1].neurons.iter().all(|output_neuron| {
        output_neuron
            .weight_protection
            .get(target_index)
            .map(|protection| !matches!(protection, ProtectionLevel::Frozen))
            .unwrap_or(false)
    })
}

fn idle_days(last_activated: u64, born_at: u64, now_secs: u64) -> u64 {
    let last_seen = if last_activated > 0 {
        last_activated
    } else {
        born_at
    };
    if last_seen == 0 {
        0
    } else {
        now_secs.saturating_sub(last_seen) / SECONDS_PER_DAY
    }
}

fn validate_config(config: &CompressionConfig) -> Result<(), ManasError> {
    if !config.threshold.is_finite() || !(0.0..=1.0).contains(&config.threshold) {
        return Err(ManasError::InvalidNetwork(format!(
            "compression threshold must be between 0.0 and 1.0, found {}",
            config.threshold
        )));
    }
    if !config.min_merge_similarity.is_finite()
        || !(-1.0..=1.0).contains(&config.min_merge_similarity)
    {
        return Err(ManasError::InvalidNetwork(format!(
            "minimum merge similarity must be between -1.0 and 1.0, found {}",
            config.min_merge_similarity
        )));
    }
    Ok(())
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{
        ANCHOR_FACTS, ANCHOR_NEURONS_PER_FACT, ANCHOR_SURVIVAL_THRESHOLD, ANCHOR_TRAIN_EPOCHS,
        EMBED_DIM, HIDDEN_DIM, LEARNING_RATE, MAX_FORGETTING_DELTA, NEW_FACT_THRESHOLD,
        NOISE_FACTS, NOISE_TRAIN_EPOCHS, OUTPUT_DIM,
    };
    use crate::{EncodedFact, Trainer};
    use manas_core::Network;

    const NOW: u64 = 1_800_000_000;
    const DAY: u64 = 86_400;

    #[test]
    fn compression_reduces_neuron_count() {
        let mut network = mergeable_network();
        let before = network.neuron_count();

        let report = compress(&mut network, &test_config()).unwrap();

        assert_eq!(report.neurons_removed, 1);
        assert_eq!(network.neuron_count(), before - 1);
        assert_eq!(report.plan.candidates.len(), 1);
        assert_eq!(network.layers[0].neurons.len(), 2);
    }

    #[test]
    fn compression_never_touches_frozen_neurons() {
        let mut network = mergeable_network();
        let frozen_id = network.layers[0].neurons[1].id;
        network.layers[0].neurons[1].freeze_all();

        let report = compress(&mut network, &test_config()).unwrap();

        assert_eq!(report.neurons_removed, 0);
        assert!(
            network.layers[0]
                .neurons
                .iter()
                .any(|neuron| neuron.id == frozen_id
                    && matches!(neuron.protection_level, ProtectionLevel::Frozen))
        );
    }

    #[test]
    fn high_importance_neurons_survive() {
        let mut network = mergeable_network();
        let high_id = network.layers[0].neurons[1].id;
        network.layers[0].neurons[1].importance_score = 0.90;

        let report = compress(&mut network, &test_config()).unwrap();

        assert_eq!(report.neurons_removed, 0);
        assert!(
            network.layers[0]
                .neurons
                .iter()
                .any(|neuron| neuron.id == high_id)
        );
    }

    #[test]
    fn recent_low_importance_neurons_are_skipped() {
        let mut network = mergeable_network();
        network.layers[0].neurons[1].last_activated = NOW - DAY;

        let plan = plan_compression(&network, &test_config()).unwrap();

        assert_eq!(plan.projected_removed(), 0);
        assert_eq!(plan.skipped.recent, 1);
    }

    #[test]
    fn low_similarity_neurons_are_skipped() {
        let mut network = mergeable_network();
        network.layers[0].neurons[1].weights = vec![0.0, 0.0, 1.0, 0.0];

        let plan = plan_compression(&network, &test_config()).unwrap();

        assert_eq!(plan.projected_removed(), 0);
        assert_eq!(plan.skipped.no_merge_target, 1);
    }

    #[test]
    fn anti_forgetting_test_still_passes_after_compression() {
        let mut trainer = Trainer::with_seed(42 ^ 0x6a09_e667_f3bc_c909, EMBED_DIM, LEARNING_RATE);
        let anchors = trainer.encode_facts(&ANCHOR_FACTS);
        let noise = trainer.encode_facts(&NOISE_FACTS);
        let mut network = Network::with_seed(
            42 ^ 0xbb67_ae85_84ca_a73b,
            EMBED_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
        );

        trainer
            .train_facts(&mut network, &anchors, ANCHOR_TRAIN_EPOCHS)
            .unwrap();
        trainer
            .consolidate_anchors(&mut network, &anchors, ANCHOR_NEURONS_PER_FACT)
            .unwrap();
        let anchor_before = score_facts(&trainer, &network, &anchors);
        trainer
            .train_facts(&mut network, &noise, NOISE_TRAIN_EPOCHS)
            .unwrap();
        trainer
            .fit_new_facts(&mut network, &noise, &anchors)
            .unwrap();
        inject_redundant_open_neuron(&mut network);

        let report = compress(&mut network, &test_config()).unwrap();

        assert_eq!(report.neurons_removed, 1);
        let anchor_after = score_facts(&trainer, &network, &anchors);
        let new_scores = score_facts(&trainer, &network, &noise);
        for (index, (before, after)) in anchor_before.iter().zip(anchor_after.iter()).enumerate() {
            assert!(
                *after >= ANCHOR_SURVIVAL_THRESHOLD,
                "anchor {index} below survival threshold: before {before:.4}, after {after:.4}"
            );
            assert!(
                before - after <= MAX_FORGETTING_DELTA,
                "anchor {index} forgot too much: before {before:.4}, after {after:.4}"
            );
        }
        assert!(
            new_scores.iter().all(|score| *score >= NEW_FACT_THRESHOLD),
            "new fact scores below threshold after compression: {new_scores:?}"
        );
    }

    fn mergeable_network() -> Network {
        let mut network = Network::new_empty(4);
        for _ in 0..3 {
            network.grow_neuron(0, 4).unwrap();
        }

        network.layers[0].neurons[0].weights = vec![1.0, 0.0, 0.0, 0.0];
        network.layers[0].neurons[0].guard_all();
        network.layers[0].neurons[0].importance_score = 0.75;
        network.layers[0].neurons[0].last_activated = NOW;
        network.layers[0].neurons[0].born_at = NOW - 2 * DAY;

        network.layers[0].neurons[1].weights = vec![1.0, 0.0, 0.0, 0.0];
        network.layers[0].neurons[1].importance_score = 0.01;
        network.layers[0].neurons[1].last_activated = NOW - 31 * DAY;
        network.layers[0].neurons[1].born_at = NOW - 60 * DAY;

        network.layers[0].neurons[2].weights = vec![0.0, 1.0, 0.0, 0.0];
        network.layers[0].neurons[2].importance_score = 0.80;
        network.layers[0].neurons[2].last_activated = NOW;
        network.layers[0].neurons[2].born_at = NOW - 2 * DAY;

        for output_neuron in &mut network.layers[1].neurons {
            output_neuron.weights = vec![0.25, 0.05, 0.10];
            output_neuron.weight_protection = vec![ProtectionLevel::Open; 3];
        }

        network
    }

    fn inject_redundant_open_neuron(network: &mut Network) {
        let open_indices = network.open_hidden_indices();
        let target_index = open_indices[0];
        let source_index = open_indices[1];
        let target_weights = network.layers[0].neurons[target_index].weights.clone();
        network.layers[0].neurons[target_index].guard_all();
        network.layers[0].neurons[target_index].importance_score = 0.80;
        network.layers[0].neurons[target_index].last_activated = NOW;
        network.layers[0].neurons[target_index].born_at = NOW - 2 * DAY;

        network.layers[0].neurons[source_index].weights = target_weights;
        network.layers[0].neurons[source_index].importance_score = 0.01;
        network.layers[0].neurons[source_index].last_activated = NOW - 31 * DAY;
        network.layers[0].neurons[source_index].born_at = NOW - 60 * DAY;

        for output_neuron in &mut network.layers[1].neurons {
            output_neuron.weights[source_index] = 0.0;
            output_neuron.weight_protection[source_index] = ProtectionLevel::Open;
            if !matches!(
                output_neuron.weight_protection[target_index],
                ProtectionLevel::Frozen
            ) {
                output_neuron.weight_protection[target_index] = ProtectionLevel::Open;
            }
        }
    }

    fn test_config() -> CompressionConfig {
        CompressionConfig {
            threshold: 0.10,
            min_idle_days: 30,
            min_merge_similarity: 0.98,
            now_secs: NOW,
        }
    }

    fn score_facts(trainer: &Trainer, network: &Network, facts: &[EncodedFact]) -> Vec<f32> {
        facts
            .iter()
            .map(|fact| trainer.similarity_for_fact(network, fact))
            .collect()
    }
}
