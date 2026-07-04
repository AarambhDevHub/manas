use manas_core::{
    GROWTH_THRESHOLD, MAX_UPDATE_ATTEMPTS, ManasError, Network, ProtectionLevel, Source,
    TrainingExample,
};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::backprop::{compute_gradients, cosine, mse_loss};
use crate::decoder::{DecodedAnswer, decode_answer};
use crate::encoder::Encoder;
use crate::freshness::{FreshnessCategory, FreshnessWarning, detect_freshness, staleness_warning};
use crate::importance;

const DEFAULT_EMBED_TABLE_SIZE: usize = 8192;
const BOUND_REUSE_ACTIVATION: f32 = 0.55;
const EXACT_REUSE_ACTIVATION: f32 = 0.98;
const MIN_READOUT_ACTIVATION: f32 = 1.0e-6;
const FRESH_READOUT_TIE_EPSILON: f32 = 0.05;

/// Encoded input-target fact used by Stage 3 training.
#[derive(Clone, Debug)]
pub struct EncodedFact {
    pub input_text: String,
    pub target_text: String,
    pub input: Vec<f32>,
    pub target: Vec<f32>,
}

/// Protection transitions from a Stage 8 promotion pass.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProtectionReport {
    pub neurons_promoted: u32,
    pub neurons_frozen: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnswerSource {
    NeuralWeights,
    NotEnough,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryResult {
    pub answer: String,
    pub confidence: f32,
    pub answered_from: AnswerSource,
    pub freshness_warning: Option<FreshnessWarning>,
}

/// Growth-aware result from a single learn call.
#[derive(Clone, Debug)]
pub struct LearnReport {
    pub loss_before: f32,
    pub loss_after: f32,
    pub neurons_grown: u32,
    pub layers_grown: u32,
    pub neurons_promoted: u32,
    pub neurons_frozen: u32,
    pub total_neurons: u64,
    pub update_applied: bool,
}

/// Minimal trainer for the proven associative-memory engine.
pub struct Trainer {
    pub encoder: Encoder,
    pub learning_rate: f32,
    pub growth_threshold: f32,
    pub max_update_attempts: u32,
}

enum BoundHiddenSelection {
    BindExisting(HiddenSelection),
    ReadOnly(HiddenSelection),
    Grow,
}

#[derive(Clone, Copy, Debug)]
struct HiddenSelection {
    layer_index: usize,
    neuron_index: usize,
    neuron_id: u64,
}

struct BoundQueryCandidate {
    decoded: DecodedAnswer,
    score: f32,
    freshness_warning: Option<FreshnessWarning>,
    refreshed_at: u64,
}

impl Trainer {
    pub fn new(learning_rate: f32) -> Self {
        Self::with_seed(42, 32, learning_rate)
    }

    pub fn with_seed(seed: u64, embed_dim: usize, learning_rate: f32) -> Self {
        Self {
            encoder: Encoder::new(seed, embed_dim, DEFAULT_EMBED_TABLE_SIZE),
            learning_rate,
            growth_threshold: GROWTH_THRESHOLD,
            max_update_attempts: MAX_UPDATE_ATTEMPTS,
        }
    }

    pub fn encode_fact(&mut self, input: &str, target: &str) -> EncodedFact {
        EncodedFact {
            input_text: input.to_string(),
            target_text: target.to_string(),
            input: self.encoder.encode(input),
            target: self.encoder.encode_answer(target),
        }
    }

    pub fn encode_facts(&mut self, facts: &[(&str, &str)]) -> Vec<EncodedFact> {
        facts
            .iter()
            .map(|(input, target)| self.encode_fact(input, target))
            .collect()
    }

    pub fn learn_raw(
        &mut self,
        network: &mut Network,
        input: &str,
        target: &str,
    ) -> Result<f32, ManasError> {
        let fact = self.encode_fact(input, target);
        self.learn_fact(network, &fact)
    }

    pub fn learn(
        &mut self,
        network: &mut Network,
        input: &str,
        target: &str,
    ) -> Result<LearnReport, ManasError> {
        self.learn_with_source(network, input, target, Source::RawText)
    }

    pub fn learn_with_source(
        &mut self,
        network: &mut Network,
        input: &str,
        target: &str,
        source: Source,
    ) -> Result<LearnReport, ManasError> {
        self.learn_with_source_and_freshness(
            network,
            input,
            target,
            source,
            detected_freshness(input, target),
        )
    }

    pub fn learn_with_source_and_freshness(
        &mut self,
        network: &mut Network,
        input: &str,
        target: &str,
        source: Source,
        freshness: FreshnessCategory,
    ) -> Result<LearnReport, ManasError> {
        let fact = self.encode_fact(input, target);
        if network.keyed_hidden_memory() {
            return self.learn_bound_fact(network, &fact, source, freshness);
        }

        let now_secs = unix_now_secs();
        let activation_counts_before = activation_counts_by_id(network);
        let loss_before = loss_for_fact(network, &fact)?;
        let mut loss_after = loss_before;
        let mut neurons_grown = 0;
        let mut layers_grown = 0;
        let mut update_applied = false;

        if network.layers[0].neurons.is_empty()
            || network
                .layers
                .last()
                .map(|layer| layer.neurons.is_empty())
                .unwrap_or(true)
        {
            let growth = grow_for_fact(network, &fact)?;
            neurons_grown += 1;
            layers_grown += growth.layers_grown;
            update_applied = true;
            loss_after = loss_for_fact(network, &fact)?;
        }

        if loss_after > self.growth_threshold {
            for _ in 0..self.max_update_attempts {
                let (_, gradients) = compute_gradients(network, &fact.input, &fact.target)?;
                network.apply_gradients(&gradients, self.learning_rate)?;
                update_applied = true;
                loss_after = loss_for_fact(network, &fact)?;

                if loss_after <= self.growth_threshold {
                    break;
                }
            }

            if loss_after > self.growth_threshold {
                let growth = grow_for_fact(network, &fact)?;
                neurons_grown += 1;
                layers_grown += growth.layers_grown;
                update_applied = true;
                loss_after = loss_for_fact(network, &fact)?;
            }
        } else if !update_applied {
            let (_, gradients) = compute_gradients(network, &fact.input, &fact.target)?;
            network.apply_gradients(&gradients, self.learning_rate)?;
            update_applied = true;
            loss_after = loss_for_fact(network, &fact)?;
        }

        refresh_learning_metadata(network, &activation_counts_before, now_secs);
        let protection_report = self.update_protection_levels_at(network, now_secs);
        assign_metadata_to_best_hidden(network, &fact.input, &fact, source, freshness);

        Ok(LearnReport {
            loss_before,
            loss_after,
            neurons_grown,
            layers_grown,
            neurons_promoted: protection_report.neurons_promoted,
            neurons_frozen: protection_report.neurons_frozen,
            total_neurons: network.neuron_count(),
            update_applied,
        })
    }

    pub fn refresh_memory_at(
        &mut self,
        network: &mut Network,
        input: &str,
        target: &str,
        source: Source,
        freshness: FreshnessCategory,
        now_secs: u64,
    ) -> Result<LearnReport, ManasError> {
        let fact = self.encode_fact(input, target);
        self.refresh_bound_fact_at(network, &fact, source, freshness, now_secs)
    }

    fn refresh_bound_fact_at(
        &self,
        network: &mut Network,
        fact: &EncodedFact,
        source: Source,
        freshness: FreshnessCategory,
        now_secs: u64,
    ) -> Result<LearnReport, ManasError> {
        let activation_counts_before = activation_counts_by_id(network);
        let loss_before = loss_for_bound_fact(network, fact)?;
        let mut neurons_grown = 0;
        let mut layers_grown = 0;

        let selection = match select_bound_hidden(network, &fact.input) {
            BoundHiddenSelection::BindExisting(selection) => {
                network.bind_hidden_neuron_to_fact(
                    selection.neuron_id,
                    &fact.input,
                    &fact.target,
                )?;
                selection
            }
            BoundHiddenSelection::ReadOnly(_) | BoundHiddenSelection::Grow => {
                let growth = grow_bound_memory_for_fact(network, fact)?;
                layers_grown += growth.layers_grown;
                neurons_grown += 1;
                network.bind_hidden_neuron_to_fact(growth.neuron_id, &fact.input, &fact.target)?;
                hidden_selection_by_id(network, growth.neuron_id)?
            }
        };

        let loss_after = loss_for_bound_fact(network, fact)?;
        refresh_learning_metadata(network, &activation_counts_before, now_secs);
        let protection_report = self.update_protection_levels_at(network, now_secs);
        assign_metadata_to_hidden_selection(
            network,
            selection,
            fact,
            source,
            freshness,
            Some(now_secs),
        );

        Ok(LearnReport {
            loss_before,
            loss_after,
            neurons_grown,
            layers_grown,
            neurons_promoted: protection_report.neurons_promoted,
            neurons_frozen: protection_report.neurons_frozen,
            total_neurons: network.neuron_count(),
            update_applied: true,
        })
    }

    fn learn_bound_fact(
        &self,
        network: &mut Network,
        fact: &EncodedFact,
        source: Source,
        freshness: FreshnessCategory,
    ) -> Result<LearnReport, ManasError> {
        let now_secs = unix_now_secs();
        let activation_counts_before = activation_counts_by_id(network);
        let loss_before = loss_for_bound_fact(network, fact)?;
        let mut neurons_grown = 0;
        let mut layers_grown = 0;
        let mut update_applied = false;

        let selection = match select_bound_hidden(network, &fact.input) {
            BoundHiddenSelection::BindExisting(selection) => {
                update_applied = true;
                network.bind_hidden_neuron_to_fact(
                    selection.neuron_id,
                    &fact.input,
                    &fact.target,
                )?;
                selection
            }
            BoundHiddenSelection::ReadOnly(selection) => selection,
            BoundHiddenSelection::Grow => {
                let growth = grow_bound_memory_for_fact(network, fact)?;
                layers_grown += growth.layers_grown;
                neurons_grown += 1;
                update_applied = true;
                network.bind_hidden_neuron_to_fact(growth.neuron_id, &fact.input, &fact.target)?;
                hidden_selection_by_id(network, growth.neuron_id)?
            }
        };

        let loss_after = loss_for_bound_fact(network, fact)?;
        refresh_learning_metadata(network, &activation_counts_before, now_secs);
        let protection_report = self.update_protection_levels_at(network, now_secs);
        assign_metadata_to_hidden_selection(network, selection, fact, source, freshness, None);

        Ok(LearnReport {
            loss_before,
            loss_after,
            neurons_grown,
            layers_grown,
            neurons_promoted: protection_report.neurons_promoted,
            neurons_frozen: protection_report.neurons_frozen,
            total_neurons: network.neuron_count(),
            update_applied,
        })
    }

    pub fn update_protection_levels(&self, network: &mut Network) -> ProtectionReport {
        self.update_protection_levels_at(network, unix_now_secs())
    }

    pub fn update_protection_levels_at(
        &self,
        network: &mut Network,
        now_secs: u64,
    ) -> ProtectionReport {
        let mut report = ProtectionReport::default();

        for layer in &mut network.layers {
            for neuron in &mut layer.neurons {
                let before = neuron.protection_level;
                importance::promote_if_needed(neuron, now_secs);
                let after = neuron.protection_level;

                if after != before {
                    report.neurons_promoted = report.neurons_promoted.saturating_add(1);
                    if matches!(after, ProtectionLevel::Frozen) {
                        report.neurons_frozen = report.neurons_frozen.saturating_add(1);
                    }
                }
            }
        }

        report
    }

    pub fn learn_fact(&self, network: &mut Network, fact: &EncodedFact) -> Result<f32, ManasError> {
        let (loss, gradients) = compute_gradients(network, &fact.input, &fact.target)?;
        network.apply_gradients(&gradients, self.learning_rate)?;
        Ok(loss)
    }

    pub fn train_facts(
        &self,
        network: &mut Network,
        facts: &[EncodedFact],
        epochs: usize,
    ) -> Result<(), ManasError> {
        if facts.is_empty() {
            return Err(ManasError::EmptyInput);
        }

        for epoch in 0..epochs {
            for offset in 0..facts.len() {
                let index = (epoch + offset) % facts.len();
                self.learn_fact(network, &facts[index])?;
            }
        }

        Ok(())
    }

    pub fn consolidate_anchors(
        &self,
        network: &mut Network,
        anchors: &[EncodedFact],
        neurons_per_fact: usize,
    ) -> Result<(), ManasError> {
        let examples = training_examples(anchors);
        network.consolidate_anchor_facts(&examples, neurons_per_fact)?;
        Ok(())
    }

    pub fn fit_new_facts(
        &self,
        network: &mut Network,
        facts: &[EncodedFact],
        anchors: &[EncodedFact],
    ) -> Result<(), ManasError> {
        let fact_examples = training_examples(facts);
        let anchor_examples = training_examples(anchors);
        network.key_open_hidden_neurons_to_facts(&fact_examples)?;
        network.fit_open_output_weights_to_facts(&fact_examples, &anchor_examples)
    }

    pub fn query_vector(&self, network: &Network, input: &[f32]) -> Vec<f32> {
        network.forward(input)
    }

    pub fn query(&self, network: &Network, question: &str) -> Result<QueryResult, ManasError> {
        if network.neuron_count() == 0 {
            return Ok(not_enough());
        }

        if network.keyed_hidden_memory() {
            return Ok(self.query_bound_memory(network, question));
        }

        let input = self.encoder.encode_deterministic(question);
        if input.iter().all(|value| value.abs() <= f32::EPSILON) {
            return Ok(not_enough());
        }

        let output = network.forward(&input);
        Ok(match decode_answer(&output, &self.encoder, question) {
            Some(decoded) => {
                let freshness_warning = best_hidden_neuron(network, &input)
                    .and_then(|neuron| staleness_warning(neuron, unix_now_secs()));

                QueryResult {
                    answer: decoded.answer,
                    confidence: decoded.confidence,
                    answered_from: AnswerSource::NeuralWeights,
                    freshness_warning,
                }
            }
            None => not_enough(),
        })
    }

    fn query_bound_memory(&self, network: &Network, question: &str) -> QueryResult {
        let mut best: Option<BoundQueryCandidate> = None;

        for variant in query_variants(question) {
            let input = self.encoder.encode_deterministic(&variant);
            if input.iter().all(|value| value.abs() <= f32::EPSILON) {
                continue;
            }

            for readout in network.readouts_from_hidden(&input) {
                if readout.activation < MIN_READOUT_ACTIVATION {
                    continue;
                }

                let Some(decoded) = decode_answer(&readout.output, &self.encoder, question) else {
                    continue;
                };
                let score = decoded.confidence * readout.activation.clamp(0.0, 1.0);
                let selection = HiddenSelection {
                    layer_index: readout.layer_index,
                    neuron_index: readout.neuron_index,
                    neuron_id: readout.neuron_id,
                };
                let (freshness_warning, refreshed_at) = network
                    .layers
                    .get(selection.layer_index)
                    .and_then(|layer| layer.neurons.get(selection.neuron_index))
                    .map(|neuron| {
                        (
                            staleness_warning(neuron, unix_now_secs()),
                            neuron.refreshed_at,
                        )
                    })
                    .unwrap_or((None, 0));
                let candidate = BoundQueryCandidate {
                    decoded,
                    score,
                    freshness_warning,
                    refreshed_at,
                };

                if best
                    .as_ref()
                    .map(|current| better_bound_candidate(&candidate, current))
                    .unwrap_or(true)
                {
                    best = Some(candidate);
                }
            }
        }

        let Some(best) = best else {
            return not_enough();
        };

        QueryResult {
            answer: best.decoded.answer,
            confidence: best.score.clamp(0.0, 1.0),
            answered_from: AnswerSource::NeuralWeights,
            freshness_warning: best.freshness_warning,
        }
    }

    pub fn similarity_for_fact(&self, network: &Network, fact: &EncodedFact) -> f32 {
        cosine(&network.forward(&fact.input), &fact.target)
    }

    pub fn similarity_to_target(&mut self, network: &Network, input: &str, target: &str) -> f32 {
        let fact = self.encode_fact(input, target);
        self.similarity_for_fact(network, &fact)
    }
}

fn training_examples(facts: &[EncodedFact]) -> Vec<TrainingExample<'_>> {
    facts
        .iter()
        .map(|fact| TrainingExample {
            input: &fact.input,
            target: &fact.target,
        })
        .collect()
}

fn loss_for_fact(network: &Network, fact: &EncodedFact) -> Result<f32, ManasError> {
    mse_loss(&network.forward(&fact.input), &fact.target)
}

fn loss_for_bound_fact(network: &Network, fact: &EncodedFact) -> Result<f32, ManasError> {
    let output = network
        .readout_from_best_hidden(&fact.input)
        .map(|readout| readout.output)
        .unwrap_or_else(|| vec![0.0; network.output_dim]);
    mse_loss(&output, &fact.target)
}

fn better_bound_candidate(candidate: &BoundQueryCandidate, current: &BoundQueryCandidate) -> bool {
    if candidate.score > current.score + FRESH_READOUT_TIE_EPSILON {
        return true;
    }
    if current.score > candidate.score + FRESH_READOUT_TIE_EPSILON {
        return false;
    }

    match (
        candidate.freshness_warning.is_some(),
        current.freshness_warning.is_some(),
    ) {
        (false, true) => true,
        (true, false) => false,
        _ => candidate.refreshed_at > current.refreshed_at,
    }
}

fn select_bound_hidden(network: &Network, input: &[f32]) -> BoundHiddenSelection {
    let Some(readout) = network.readout_from_best_hidden(input) else {
        return BoundHiddenSelection::Grow;
    };
    let Some(neuron) = network
        .layers
        .get(readout.layer_index)
        .and_then(|layer| layer.neurons.get(readout.neuron_index))
    else {
        return BoundHiddenSelection::Grow;
    };
    let selection = HiddenSelection {
        layer_index: readout.layer_index,
        neuron_index: readout.neuron_index,
        neuron_id: readout.neuron_id,
    };

    if readout.activation >= EXACT_REUSE_ACTIVATION {
        return if matches!(neuron.protection_level, ProtectionLevel::Open) {
            BoundHiddenSelection::BindExisting(selection)
        } else {
            BoundHiddenSelection::ReadOnly(selection)
        };
    }

    if readout.activation >= BOUND_REUSE_ACTIVATION
        && matches!(neuron.protection_level, ProtectionLevel::Open)
    {
        BoundHiddenSelection::BindExisting(selection)
    } else {
        BoundHiddenSelection::Grow
    }
}

fn detected_freshness(input: &str, target: &str) -> FreshnessCategory {
    let mut text = String::with_capacity(input.len() + target.len() + 1);
    text.push_str(input);
    text.push(' ');
    text.push_str(target);
    detect_freshness(&text)
}

pub(crate) fn query_variants(question: &str) -> Vec<String> {
    let words = normalized_query_words(question);
    if words.is_empty() {
        return vec![question.trim().to_string()];
    }

    let mut variants = Vec::new();
    let entity_words = words
        .iter()
        .filter(|word| !is_query_stopword(word) && !is_relation_word(word))
        .cloned()
        .collect::<Vec<_>>();
    push_variant(&mut variants, entity_words.join(" "));

    let content_words = words
        .iter()
        .filter(|word| !is_query_stopword(word))
        .cloned()
        .collect::<Vec<_>>();
    push_variant(&mut variants, content_words.join(" "));

    for word in content_words {
        push_variant(&mut variants, word);
    }

    push_variant(&mut variants, question.trim().to_string());
    variants
}

fn push_variant(variants: &mut Vec<String>, variant: String) {
    let trimmed = variant.trim();
    if !trimmed.is_empty() && !variants.iter().any(|existing| existing == trimmed) {
        variants.push(trimmed.to_string());
    }
}

fn normalized_query_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|raw| {
            let cleaned = raw
                .chars()
                .filter(|ch| ch.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>();
            (!cleaned.is_empty()).then_some(cleaned)
        })
        .collect()
}

fn is_query_stopword(word: &str) -> bool {
    matches!(
        word,
        "a" | "an"
            | "and"
            | "are"
            | "at"
            | "by"
            | "did"
            | "do"
            | "does"
            | "in"
            | "is"
            | "of"
            | "on"
            | "the"
            | "to"
            | "was"
            | "were"
            | "what"
            | "when"
            | "where"
            | "which"
            | "who"
            | "why"
    )
}

fn is_relation_word(word: &str) -> bool {
    matches!(
        word,
        "boils"
            | "built"
            | "contains"
            | "converts"
            | "created"
            | "describes"
            | "develop"
            | "developed"
            | "fell"
            | "launched"
            | "located"
            | "painted"
            | "pulls"
            | "released"
            | "wrote"
    )
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn activation_counts_by_id(network: &Network) -> HashMap<u64, u64> {
    network
        .layers
        .iter()
        .flat_map(|layer| layer.neurons.iter())
        .map(|neuron| (neuron.id, neuron.activation_count))
        .collect()
}

fn refresh_learning_metadata(
    network: &mut Network,
    activation_counts_before: &HashMap<u64, u64>,
    now_secs: u64,
) {
    for neuron in network
        .layers
        .iter_mut()
        .flat_map(|layer| layer.neurons.iter_mut())
    {
        if neuron.born_at == 0 {
            neuron.born_at = now_secs;
        }

        let previous_count = activation_counts_before
            .get(&neuron.id)
            .copied()
            .unwrap_or(0);
        if neuron.activation_count > previous_count {
            neuron.last_activated = now_secs;
        }
    }
}

fn not_enough() -> QueryResult {
    QueryResult {
        answer: "Not enough knowledge yet.".to_string(),
        confidence: 0.0,
        answered_from: AnswerSource::NotEnough,
        freshness_warning: None,
    }
}

fn grow_for_fact(network: &mut Network, fact: &EncodedFact) -> Result<BoundGrowth, ManasError> {
    let growth = grow_bound_memory_for_fact(network, fact)?;
    network.key_hidden_neuron_to_input(growth.neuron_id, &fact.input)?;
    if let Some((layer_index, neuron_index)) = network.hidden_location_by_id(growth.neuron_id) {
        network.layers[layer_index].neurons[neuron_index].activation_count =
            network.layers[layer_index].neurons[neuron_index]
                .activation_count
                .saturating_add(1);
    }
    network.fit_open_output_weights_to_facts(
        &[TrainingExample {
            input: &fact.input,
            target: &fact.target,
        }],
        &[],
    )?;
    Ok(growth)
}

fn assign_metadata_to_best_hidden(
    network: &mut Network,
    input: &[f32],
    fact: &EncodedFact,
    source: Source,
    freshness: FreshnessCategory,
) {
    let Some(selection) = best_open_hidden_selection(network, input) else {
        return;
    };

    assign_metadata_to_hidden_selection(network, selection, fact, source, freshness, None);
}

fn assign_metadata_to_hidden_selection(
    network: &mut Network,
    selection: HiddenSelection,
    fact: &EncodedFact,
    source: Source,
    freshness: FreshnessCategory,
    refreshed_at: Option<u64>,
) {
    if let Some(neuron) = network
        .layers
        .get_mut(selection.layer_index)
        .and_then(|layer| layer.neurons.get_mut(selection.neuron_index))
    {
        neuron.source = source;
        neuron.freshness_category = freshness as u8;
        neuron.memory_input = fact.input_text.clone();
        neuron.memory_target = fact.target_text.clone();
        if let Some(timestamp) = refreshed_at {
            neuron.born_at = timestamp;
            neuron.last_activated = timestamp;
            neuron.refreshed_at = timestamp;
        }
    }
}

fn best_open_hidden_selection(network: &Network, input: &[f32]) -> Option<HiddenSelection> {
    let cache = network.forward_with_cache(input);
    let mut global_hidden_index = 0;
    let mut best: Option<(HiddenSelection, f32)> = None;

    for (layer_index, layer) in network
        .layers
        .iter()
        .take(network.layers.len().saturating_sub(1))
        .enumerate()
    {
        for (neuron_index, neuron) in layer.neurons.iter().enumerate() {
            let activation = cache
                .hidden
                .get(global_hidden_index)
                .copied()
                .unwrap_or_else(|| neuron.activate(input))
                .abs();
            if !matches!(neuron.protection_level, ProtectionLevel::Frozen)
                && best
                    .as_ref()
                    .map(|(_, best_activation)| activation > *best_activation)
                    .unwrap_or(true)
            {
                best = Some((
                    HiddenSelection {
                        layer_index,
                        neuron_index,
                        neuron_id: neuron.id,
                    },
                    activation,
                ));
            }
            global_hidden_index += 1;
        }
    }

    best.map(|(selection, _)| selection)
}

#[cfg(test)]
fn best_open_hidden_index(network: &Network, input: &[f32]) -> Option<usize> {
    let selection = best_open_hidden_selection(network, input)?;
    network.global_hidden_index(selection.layer_index, selection.neuron_index)
}

fn best_hidden_neuron<'a>(network: &'a Network, input: &[f32]) -> Option<&'a manas_core::Neuron> {
    let selection = best_open_hidden_selection(network, input)?;
    network
        .layers
        .get(selection.layer_index)
        .and_then(|layer| layer.neurons.get(selection.neuron_index))
}

fn hidden_selection_by_id(
    network: &Network,
    neuron_id: u64,
) -> Result<HiddenSelection, ManasError> {
    let (layer_index, neuron_index) = network
        .hidden_location_by_id(neuron_id)
        .ok_or(ManasError::NeuronNotFound(neuron_id))?;
    Ok(HiddenSelection {
        layer_index,
        neuron_index,
        neuron_id,
    })
}

struct BoundGrowth {
    neuron_id: u64,
    layers_grown: u32,
}

fn grow_bound_memory_for_fact(
    network: &mut Network,
    fact: &EncodedFact,
) -> Result<BoundGrowth, ManasError> {
    if network.should_grow_layer() {
        let input_size = network
            .layers
            .get(network.layers.len().saturating_sub(2))
            .map(|layer| layer.neurons.len())
            .unwrap_or(fact.input.len());
        let layer_id = network.grow_layer(input_size, 1)?;
        let neuron_id = network
            .layers
            .iter()
            .find(|layer| layer.id == layer_id)
            .and_then(|layer| layer.neurons.last())
            .map(|neuron| neuron.id)
            .ok_or_else(|| {
                ManasError::GrowthFailed("new layer did not create a neuron".to_string())
            })?;
        Ok(BoundGrowth {
            neuron_id,
            layers_grown: 1,
        })
    } else {
        let layer_index = deepest_growable_hidden_layer(network);
        let input_size = if layer_index == 0 {
            fact.input.len()
        } else {
            network.layers[layer_index - 1].neurons.len()
        };
        let layer_id = network.layers[layer_index].id;
        let neuron_id = network.grow_neuron(layer_id, input_size)?;
        Ok(BoundGrowth {
            neuron_id,
            layers_grown: 0,
        })
    }
}

fn deepest_growable_hidden_layer(network: &Network) -> usize {
    network
        .layers
        .iter()
        .take(network.layers.len().saturating_sub(1))
        .enumerate()
        .rev()
        .find(|(_, layer)| layer.neurons.len() < manas_core::MAX_NEURONS_PER_LAYER)
        .map(|(index, _)| index)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use manas_core::{GUARD_DELTA, ProtectionLevel};

    #[test]
    fn growth_trainer_learn_grows_empty_network() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        let report = trainer.learn(&mut network, "cat", "animal").unwrap();

        assert!(report.neurons_grown > 0);
        assert!(network.neuron_count() > 0);
        assert!(report.loss_after < report.loss_before);
        assert!(report.update_applied);
    }

    #[test]
    fn growth_repeated_teaching_does_not_explode_neurons() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        for _ in 0..100 {
            trainer.learn(&mut network, "cat", "animal").unwrap();
        }
        let count_after_100 = network.neuron_count();

        for _ in 0..100 {
            trainer.learn(&mut network, "cat", "animal").unwrap();
        }

        assert_eq!(network.neuron_count(), count_after_100);
    }

    #[test]
    fn growth_new_fact_grows_neuron_if_needed() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        for _ in 0..100 {
            trainer.learn(&mut network, "cat", "animal").unwrap();
        }
        let neurons_after_cat = network.neuron_count();

        for _ in 0..10 {
            trainer
                .learn(&mut network, "eiffel tower", "paris france")
                .unwrap();
        }

        assert!(network.neuron_count() >= neurons_after_cat);
    }

    #[test]
    fn layer_growth_trainer_adds_layer_when_hidden_memory_is_saturated() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        trainer
            .learn(&mut network, "cat", "small animal with fur")
            .unwrap();
        trainer
            .learn(
                &mut network,
                "Eiffel Tower",
                "located in Paris France and built in 1889",
            )
            .unwrap();
        for neuron in &mut network.layers[0].neurons {
            neuron.guard_all();
        }
        let layers_before = network.layer_count();

        let report = trainer
            .learn(
                &mut network,
                "Einstein",
                "theory of relativity in the early 20th century",
            )
            .unwrap();

        assert_eq!(report.layers_grown, 1);
        assert_eq!(network.layer_count(), layers_before + 1);
        assert_eq!(report.total_neurons, network.neuron_count());

        let cat = trainer.query(&network, "What is a cat?").unwrap();
        let einstein = trainer
            .query(&network, "What did Einstein develop?")
            .unwrap();
        assert_eq!(cat.answered_from, AnswerSource::NeuralWeights);
        assert_eq!(einstein.answered_from, AnswerSource::NeuralWeights);
        assert!(
            cat.answer.contains("animal") || cat.answer.contains("fur"),
            "{}",
            cat.answer
        );
        assert!(
            einstein.answer.contains("theory") || einstein.answer.contains("relativity"),
            "{}",
            einstein.answer
        );
    }

    #[test]
    fn growth_learn_report_records_growth_and_loss() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        let report = trainer.learn(&mut network, "rust", "language").unwrap();

        assert_eq!(report.layers_grown, 0);
        assert_eq!(report.neurons_promoted, 0);
        assert_eq!(report.neurons_frozen, 0);
        assert_eq!(report.total_neurons, network.neuron_count());
        assert!(report.loss_before >= report.loss_after);
    }

    #[test]
    fn protection_frozen_neuron_weight_never_changes() {
        let mut network = Network::new(32, 64, 32);
        network.layers[0].neurons[0].freeze_all();
        let weights_before = network.layers[0].neurons[0].weights.clone();
        let bias_before = network.layers[0].neurons[0].bias;
        let mut trainer = Trainer::new(0.1);
        trainer.growth_threshold = f32::MAX;

        for index in 0..1000 {
            trainer
                .learn(
                    &mut network,
                    &format!("protection fact {index}"),
                    &format!("protection value {index}"),
                )
                .unwrap();
        }

        assert_eq!(network.layers[0].neurons[0].weights, weights_before);
        assert_eq!(network.layers[0].neurons[0].bias, bias_before);
    }

    #[test]
    fn protection_guarded_neuron_updates_are_clamped() {
        let mut network = Network::new(32, 64, 32);
        network.layers[0].neurons[0].guard_all();
        let weights_before = network.layers[0].neurons[0].weights.clone();
        let bias_before = network.layers[0].neurons[0].bias;
        let mut trainer = Trainer::new(1.0);
        trainer.growth_threshold = f32::MAX;

        for index in 0..100 {
            trainer
                .learn(
                    &mut network,
                    &format!("guard stress {index}"),
                    &format!("guard target {index}"),
                )
                .unwrap();
        }

        for (before, after) in weights_before
            .iter()
            .zip(network.layers[0].neurons[0].weights.iter())
        {
            let delta = (after - before).abs();
            assert!(delta <= GUARD_DELTA * 100.0 + 1.0e-5);
        }
        let bias_delta = (network.layers[0].neurons[0].bias - bias_before).abs();
        assert!(bias_delta <= GUARD_DELTA * 100.0 + 1.0e-5);
    }

    #[test]
    fn protection_open_neuron_updates_freely() {
        let mut network = Network::new(32, 64, 32);
        network.layers[0].neurons[0].protection_level = ProtectionLevel::Open;
        let weights_before = network.layers[0].neurons[0].weights.clone();
        let mut trainer = Trainer::new(0.1);
        trainer.growth_threshold = f32::MAX;

        for _ in 0..100 {
            trainer.learn(&mut network, "hello", "world").unwrap();
        }

        let any_changed = weights_before
            .iter()
            .zip(network.layers[0].neurons[0].weights.iter())
            .any(|(before, after)| (after - before).abs() > 1.0e-6);
        assert!(any_changed);
    }

    #[test]
    fn protection_promotion_happens_automatically() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        for _ in 0..3000 {
            trainer.learn(&mut network, "cat", "animal").unwrap();
            trainer.update_protection_levels(&mut network);
        }

        let promoted = network
            .layers
            .iter()
            .flat_map(|layer| layer.neurons.iter())
            .filter(|neuron| !matches!(neuron.protection_level, ProtectionLevel::Open))
            .count();

        assert!(promoted > 0);
    }

    #[test]
    fn protection_update_uses_importance_thresholds_one_level_per_pass() {
        let mut network = Network::new(32, 64, 32);
        let now = unix_now_secs();
        make_neuron_high_importance(&mut network.layers[0].neurons[0], now);
        let trainer = Trainer::new(0.01);

        let first_report = trainer.update_protection_levels_at(&mut network, now);

        assert_eq!(first_report.neurons_promoted, 1);
        assert_eq!(first_report.neurons_frozen, 0);
        assert_eq!(
            network.layers[0].neurons[0].protection_level,
            ProtectionLevel::Guarded
        );

        let second_report = trainer.update_protection_levels_at(&mut network, now);

        assert_eq!(second_report.neurons_promoted, 1);
        assert_eq!(second_report.neurons_frozen, 1);
        assert_eq!(
            network.layers[0].neurons[0].protection_level,
            ProtectionLevel::Frozen
        );
    }

    #[test]
    fn protection_learn_report_records_transitions() {
        let mut network = Network::new(32, 64, 32);
        let now = unix_now_secs();
        make_neuron_high_importance(&mut network.layers[0].neurons[0], now);
        make_neuron_high_importance(&mut network.layers[0].neurons[1], now);
        network.layers[0].neurons[1].guard_all();
        let mut trainer = Trainer::new(0.01);
        trainer.growth_threshold = f32::MAX;

        let report = trainer.learn(&mut network, "rust", "language").unwrap();

        assert!(report.neurons_promoted >= 2);
        assert!(report.neurons_frozen >= 1);
        assert_eq!(
            network.layers[0].neurons[0].protection_level,
            ProtectionLevel::Guarded
        );
        assert_eq!(
            network.layers[0].neurons[1].protection_level,
            ProtectionLevel::Frozen
        );
    }

    #[test]
    fn importance_metadata_is_stamped_during_learning() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        trainer.learn(&mut network, "cat", "animal").unwrap();
        trainer.learn(&mut network, "cat", "animal").unwrap();

        assert!(
            network
                .layers
                .iter()
                .flat_map(|layer| layer.neurons.iter())
                .all(|neuron| neuron.born_at > 0)
        );
        assert!(
            network
                .layers
                .iter()
                .flat_map(|layer| layer.neurons.iter())
                .any(|neuron| neuron.last_activated > 0)
        );
        assert!(
            network
                .layers
                .iter()
                .flat_map(|layer| layer.neurons.iter())
                .any(|neuron| neuron.importance_score > 0.0)
        );
    }

    #[test]
    fn query_returns_neural_weights_for_learned_fact() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        trainer
            .learn(&mut network, "cat", "small animal with fur")
            .unwrap();

        let result = trainer.query(&network, "What is a cat?").unwrap();

        assert_eq!(result.answered_from, AnswerSource::NeuralWeights);
        assert!(result.confidence > 0.0);
        assert_eq!(result.freshness_warning, None);
        assert!(
            result.answer.contains("animal") || result.answer.contains("fur"),
            "answer was '{}'",
            result.answer
        );
    }

    #[test]
    fn query_returns_not_enough_for_unknown_question() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        trainer
            .learn(&mut network, "cat", "small animal with fur")
            .unwrap();

        let result = trainer.query(&network, "quasar").unwrap();

        assert_eq!(result.answered_from, AnswerSource::NotEnough);
        assert_eq!(result.confidence, 0.0);
        assert_eq!(result.freshness_warning, None);
    }

    #[test]
    fn query_bound_memory_returns_distinct_sequential_answers() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        trainer
            .learn(
                &mut network,
                "cat",
                "small domesticated animal with fur and whiskers",
            )
            .unwrap();
        trainer
            .learn(
                &mut network,
                "Eiffel Tower",
                "located in Paris France and built in 1889",
            )
            .unwrap();
        trainer
            .learn(
                &mut network,
                "Einstein",
                "theory of relativity in the early 20th century",
            )
            .unwrap();

        let cat = trainer.query(&network, "What is a cat?").unwrap();
        let eiffel = trainer
            .query(&network, "Where is the Eiffel Tower?")
            .unwrap();
        let einstein = trainer
            .query(&network, "What did Einstein develop?")
            .unwrap();

        assert_eq!(cat.answered_from, AnswerSource::NeuralWeights);
        assert_contains_any(&cat.answer, &["animal", "fur", "whiskers"]);
        assert_contains_any(&eiffel.answer, &["paris", "france", "1889"]);
        assert_contains_all(&einstein.answer, &["theory", "relativity"]);
        assert_ne!(cat.answer, eiffel.answer);
        assert_ne!(eiffel.answer, einstein.answer);
    }

    #[test]
    fn query_bound_memory_uses_question_variants() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        trainer
            .learn(&mut network, "Eiffel Tower", "located in Paris France")
            .unwrap();

        let result = trainer
            .query(&network, "Where is the Eiffel Tower?")
            .unwrap();

        assert_eq!(result.answered_from, AnswerSource::NeuralWeights);
        assert_contains_all(&result.answer, &["paris", "france"]);
    }

    #[test]
    fn learn_with_source_preserves_local_file_metadata() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);
        let source = Source::LocalFile {
            path: "notes/facts.txt".to_string(),
        };

        trainer
            .learn_with_source(&mut network, "cat", "small animal with fur", source.clone())
            .unwrap();

        assert!(
            network.layers[0]
                .neurons
                .iter()
                .any(|neuron| neuron.source == source)
        );
    }

    #[test]
    fn learn_stamps_refresh_memory_metadata() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        trainer
            .learn(&mut network, "stock price", "10 today")
            .unwrap();

        assert!(
            network
                .layers
                .iter()
                .take(network.layer_count().saturating_sub(1))
                .flat_map(|layer| layer.neurons.iter())
                .any(|neuron| {
                    neuron.memory_input == "stock price"
                        && neuron.memory_target == "10 today"
                        && neuron.refreshed_at == 0
                })
        );
    }

    #[test]
    fn learn_with_source_detects_freshness_metadata() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        trainer
            .learn_with_source(
                &mut network,
                "market",
                "Breaking news: the stock market fell today",
                Source::RawText,
            )
            .unwrap();

        assert!(
            network.layers[0]
                .neurons
                .iter()
                .any(|neuron| neuron.freshness_category == FreshnessCategory::Realtime as u8)
        );
    }

    #[test]
    fn learn_with_source_and_freshness_uses_explicit_category() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        trainer
            .learn_with_source_and_freshness(
                &mut network,
                "water",
                "Water is always composed of hydrogen and oxygen",
                Source::RawText,
                FreshnessCategory::Timeless,
            )
            .unwrap();

        assert!(
            network.layers[0]
                .neurons
                .iter()
                .any(|neuron| neuron.freshness_category == FreshnessCategory::Timeless as u8)
        );
    }

    #[test]
    fn query_returns_warning_for_stale_neural_answer() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        trainer
            .learn_with_source_and_freshness(
                &mut network,
                "cat",
                "small animal with fur",
                Source::RawText,
                FreshnessCategory::Fast,
            )
            .unwrap();

        let query_input = trainer.encoder.encode_deterministic("What is a cat?");
        let hidden_index = best_open_hidden_index(&network, &query_input).unwrap();
        network.layers[0].neurons[hidden_index].born_at = unix_now_secs() - 31 * 86_400;
        network.layers[0].neurons[hidden_index].freshness_category = FreshnessCategory::Fast as u8;

        let result = trainer.query(&network, "What is a cat?").unwrap();

        assert_eq!(result.answered_from, AnswerSource::NeuralWeights);
        assert_eq!(
            result.freshness_warning,
            Some(FreshnessWarning {
                category: FreshnessCategory::Fast,
                age_days: 31,
            })
        );
    }

    #[test]
    fn refresh_memory_grows_for_protected_stale_fact_and_query_prefers_fresh_answer() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);
        let now = unix_now_secs();

        trainer
            .learn_with_source_and_freshness(
                &mut network,
                "stock price",
                "10 today",
                Source::RawText,
                FreshnessCategory::Realtime,
            )
            .unwrap();
        let hidden_layers = network.layer_count().saturating_sub(1);
        let mut protected_before = None;
        for layer in network.layers.iter_mut().take(hidden_layers) {
            for neuron in &mut layer.neurons {
                if neuron.memory_input == "stock price" {
                    neuron.born_at = now.saturating_sub(2 * 86_400);
                    neuron.last_activated = neuron.born_at;
                    neuron.freeze_all();
                    protected_before = Some((neuron.id, neuron.weights.clone(), neuron.bias));
                }
            }
        }
        let neurons_before = network.neuron_count();

        let report = trainer
            .refresh_memory_at(
                &mut network,
                "stock price",
                "20 today",
                Source::Internet {
                    url: "https://example.test/stock".to_string(),
                },
                FreshnessCategory::Realtime,
                now,
            )
            .unwrap();

        assert_eq!(report.neurons_grown, 1);
        assert_eq!(network.neuron_count(), neurons_before + 1);
        let (protected_id, weights_before, bias_before) = protected_before.unwrap();
        let (layer_index, neuron_index) = network.hidden_location_by_id(protected_id).unwrap();
        let protected = &network.layers[layer_index].neurons[neuron_index];
        assert_eq!(protected.weights, weights_before);
        assert_eq!(protected.bias, bias_before);

        let result = trainer.query(&network, "What is stock price?").unwrap();
        assert_eq!(result.answered_from, AnswerSource::NeuralWeights);
        assert!(result.answer.contains("20"), "{}", result.answer);
        assert_eq!(result.freshness_warning, None);
    }

    fn make_neuron_high_importance(neuron: &mut manas_core::Neuron, now_secs: u64) {
        neuron.activation_count = 10_000;
        neuron.last_activated = now_secs;
        neuron.born_at = now_secs;
        for weight in &mut neuron.weights {
            *weight = 10.0;
        }
    }

    fn assert_contains_any(answer: &str, words: &[&str]) {
        let normalized = answer.to_lowercase();
        assert!(
            words.iter().any(|word| normalized.contains(word)),
            "answer '{answer}' did not contain any of {words:?}"
        );
    }

    fn assert_contains_all(answer: &str, words: &[&str]) {
        let normalized = answer.to_lowercase();
        for word in words {
            assert!(
                normalized.contains(word),
                "answer '{answer}' did not contain '{word}'"
            );
        }
    }
}
