use manas_core::{Activation, Network, ProtectionLevel, Source};

use crate::decoder::decode_answer;
use crate::freshness::{FreshnessCategory, freshness_age_days, is_stale};
use crate::trainer::{AnswerSource, Trainer, query_variants};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BrainDiagnostics {
    pub network: NetworkDiagnostics,
    pub learning: LearningDiagnostics,
    pub freshness: FreshnessDiagnostics,
    pub sources: SourceDiagnostics,
    pub layers: Vec<LayerDiagnostics>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NetworkDiagnostics {
    pub total_neurons: u64,
    pub total_layers: usize,
    pub open_neurons: u64,
    pub guarded_neurons: u64,
    pub frozen_neurons: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LearningDiagnostics {
    pub facts_taught: u64,
    pub total_learn_calls: u64,
    pub neurons_grown: u64,
    pub layers_grown: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FreshnessDiagnostics {
    pub timeless_neurons: u64,
    pub slow_neurons: u64,
    pub fast_neurons: u64,
    pub realtime_neurons: u64,
    pub stale_neurons: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SourceDiagnostics {
    pub raw_text_neurons: u64,
    pub local_file_neurons: u64,
    pub unknown_neurons: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayerDiagnostics {
    pub layer_index: usize,
    pub layer_id: u32,
    pub activation: Activation,
    pub neurons: usize,
    pub open_neurons: usize,
    pub guarded_neurons: usize,
    pub frozen_neurons: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NeuronDiagnostics {
    pub layer_index: usize,
    pub layer_id: u32,
    pub neuron_index: usize,
    pub neuron_id: u64,
    pub activation: Activation,
    pub protection: ProtectionLevel,
    pub importance_score: f32,
    pub activation_count: u64,
    pub freshness: FreshnessCategory,
    pub age_days: u64,
    pub stale: bool,
    pub source: Source,
    pub source_label: String,
    pub born_at: u64,
    pub last_activated: u64,
    pub weight_count: usize,
    pub protected_weight_count: usize,
    pub bias: f32,
    pub learned: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NeuronFilter {
    pub protection: Option<ProtectionLevel>,
    pub source_contains: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryTrace {
    pub question: String,
    pub answer: String,
    pub confidence: f32,
    pub answered_from: AnswerSource,
    pub selected_variant: Option<String>,
    pub variants: Vec<TraceVariant>,
    pub top_hidden_activations: Vec<NeuronActivationDiagnostic>,
    pub top_output_values: Vec<OutputValueDiagnostic>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TraceVariant {
    pub text: String,
    pub encoded: bool,
    pub selected: bool,
    pub hidden_index: Option<usize>,
    pub hidden_neuron_id: Option<u64>,
    pub hidden_activation: f32,
    pub decoded_answer: Option<String>,
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NeuronActivationDiagnostic {
    pub layer_index: usize,
    pub layer_id: u32,
    pub neuron_index: usize,
    pub neuron_id: u64,
    pub activation: f32,
    pub protection: ProtectionLevel,
    pub source_label: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutputValueDiagnostic {
    pub output_index: usize,
    pub neuron_id: Option<u64>,
    pub value: f32,
}

struct TraceCandidate {
    variant_index: usize,
    input: Vec<f32>,
    output: Vec<f32>,
    answer: String,
    score: f32,
}

impl BrainDiagnostics {
    pub fn from_network(network: &Network, now_secs: u64) -> Self {
        let neurons = neuron_diagnostics(network, now_secs);
        let learned_hidden = neurons
            .iter()
            .filter(|neuron| neuron.layer_index == 0 && neuron.learned)
            .collect::<Vec<_>>();

        let mut freshness = FreshnessDiagnostics::default();
        let mut sources = SourceDiagnostics::default();
        for neuron in &learned_hidden {
            match neuron.freshness {
                FreshnessCategory::Timeless => freshness.timeless_neurons += 1,
                FreshnessCategory::Slow => freshness.slow_neurons += 1,
                FreshnessCategory::Fast => freshness.fast_neurons += 1,
                FreshnessCategory::Realtime => freshness.realtime_neurons += 1,
            }
            if neuron.stale {
                freshness.stale_neurons += 1;
            }

            match &neuron.source {
                Source::RawText => sources.raw_text_neurons += 1,
                Source::LocalFile { .. } => sources.local_file_neurons += 1,
                Source::Unknown => sources.unknown_neurons += 1,
            }
        }

        let total_learn_calls = network
            .layers
            .first()
            .map(|layer| {
                layer
                    .neurons
                    .iter()
                    .map(|neuron| neuron.activation_count)
                    .sum()
            })
            .unwrap_or(0);

        Self {
            network: NetworkDiagnostics {
                total_neurons: network.neuron_count(),
                total_layers: network.layer_count(),
                open_neurons: network.open_neuron_count(),
                guarded_neurons: network.guarded_neuron_count(),
                frozen_neurons: network.frozen_neuron_count(),
            },
            learning: LearningDiagnostics {
                facts_taught: learned_hidden.len() as u64,
                total_learn_calls,
                neurons_grown: network
                    .layers
                    .first()
                    .map(|layer| layer.neurons.len())
                    .unwrap_or(0) as u64,
                layers_grown: network.layer_count().saturating_sub(2) as u64,
            },
            freshness,
            sources,
            layers: layer_diagnostics(network),
        }
    }
}

pub fn neuron_diagnostics(network: &Network, now_secs: u64) -> Vec<NeuronDiagnostics> {
    network
        .layers
        .iter()
        .enumerate()
        .flat_map(|(layer_index, layer)| {
            layer
                .neurons
                .iter()
                .enumerate()
                .map(move |(neuron_index, neuron)| {
                    let learned = layer_index == 0
                        && (neuron.activation_count > 0
                            || !matches!(neuron.source, Source::Unknown));
                    let freshness = FreshnessCategory::from(neuron.freshness_category);
                    let protected_weight_count = neuron
                        .weight_protection
                        .iter()
                        .filter(|protection| !matches!(protection, ProtectionLevel::Open))
                        .count();

                    NeuronDiagnostics {
                        layer_index,
                        layer_id: layer.id,
                        neuron_index,
                        neuron_id: neuron.id,
                        activation: neuron.activation,
                        protection: neuron.protection_level,
                        importance_score: neuron.importance_score,
                        activation_count: neuron.activation_count,
                        freshness,
                        age_days: freshness_age_days(neuron, now_secs),
                        stale: learned && is_stale(neuron, now_secs),
                        source: neuron.source.clone(),
                        source_label: source_label(&neuron.source),
                        born_at: neuron.born_at,
                        last_activated: neuron.last_activated,
                        weight_count: neuron.weights.len(),
                        protected_weight_count,
                        bias: neuron.bias,
                        learned,
                    }
                })
        })
        .collect()
}

pub fn filtered_neuron_diagnostics(
    network: &Network,
    now_secs: u64,
    filter: &NeuronFilter,
) -> Vec<NeuronDiagnostics> {
    neuron_diagnostics(network, now_secs)
        .into_iter()
        .filter(|neuron| filter.matches(neuron))
        .collect()
}

pub fn trace_query(
    trainer: &Trainer,
    network: &Network,
    question: &str,
    limit: usize,
) -> QueryTrace {
    let limit = limit.min(100);
    if network.neuron_count() == 0 {
        return QueryTrace {
            question: question.to_string(),
            answer: "Not enough knowledge yet.".to_string(),
            confidence: 0.0,
            answered_from: AnswerSource::NotEnough,
            selected_variant: None,
            variants: Vec::new(),
            top_hidden_activations: Vec::new(),
            top_output_values: Vec::new(),
        };
    }

    let variant_texts = if network.keyed_hidden_memory() {
        query_variants(question)
    } else {
        vec![question.trim().to_string()]
    };

    let mut variants = Vec::with_capacity(variant_texts.len());
    let mut best: Option<TraceCandidate> = None;

    for text in variant_texts {
        let input = trainer.encoder.encode_deterministic(&text);
        let encoded = input.iter().any(|value| value.abs() > f32::EPSILON);
        let mut variant = TraceVariant {
            text,
            encoded,
            selected: false,
            hidden_index: None,
            hidden_neuron_id: None,
            hidden_activation: 0.0,
            decoded_answer: None,
            score: 0.0,
        };

        if encoded {
            if network.keyed_hidden_memory() {
                if let Some(readout) = network.readout_from_best_hidden(&input) {
                    variant.hidden_index = Some(readout.hidden_index);
                    variant.hidden_neuron_id = hidden_neuron_id(network, readout.hidden_index);
                    variant.hidden_activation = readout.activation;

                    if let Some(decoded) =
                        decode_answer(&readout.output, &trainer.encoder, question)
                    {
                        let score = decoded.confidence * readout.activation.clamp(0.0, 1.0);
                        variant.decoded_answer = Some(decoded.answer.clone());
                        variant.score = score;
                        update_best(
                            &mut best,
                            TraceCandidate {
                                variant_index: variants.len(),
                                input,
                                output: readout.output,
                                answer: decoded.answer,
                                score,
                            },
                        );
                    }
                }
            } else {
                let cache = network.forward_with_cache(&input);
                let (hidden_index, hidden_activation) = best_hidden_activation(&cache.hidden);
                variant.hidden_index = hidden_index;
                variant.hidden_neuron_id =
                    hidden_index.and_then(|index| hidden_neuron_id(network, index));
                variant.hidden_activation = hidden_activation;

                if let Some(decoded) = decode_answer(&cache.output, &trainer.encoder, question) {
                    variant.decoded_answer = Some(decoded.answer.clone());
                    variant.score = decoded.confidence;
                    update_best(
                        &mut best,
                        TraceCandidate {
                            variant_index: variants.len(),
                            input,
                            output: cache.output,
                            answer: decoded.answer,
                            score: decoded.confidence,
                        },
                    );
                }
            }
        }

        variants.push(variant);
    }

    let Some(best) = best else {
        return QueryTrace {
            question: question.to_string(),
            answer: "Not enough knowledge yet.".to_string(),
            confidence: 0.0,
            answered_from: AnswerSource::NotEnough,
            selected_variant: None,
            variants,
            top_hidden_activations: Vec::new(),
            top_output_values: Vec::new(),
        };
    };

    if let Some(variant) = variants.get_mut(best.variant_index) {
        variant.selected = true;
    }

    let selected_variant = variants
        .get(best.variant_index)
        .map(|variant| variant.text.clone());
    let top_hidden_activations = top_hidden_activations(network, &best.input, limit);
    let top_output_values = top_output_values(network, &best.output, limit);

    QueryTrace {
        question: question.to_string(),
        answer: best.answer,
        confidence: best.score.clamp(0.0, 1.0),
        answered_from: AnswerSource::NeuralWeights,
        selected_variant,
        variants,
        top_hidden_activations,
        top_output_values,
    }
}

impl NeuronFilter {
    fn matches(&self, neuron: &NeuronDiagnostics) -> bool {
        if let Some(protection) = self.protection
            && neuron.protection != protection
        {
            return false;
        }

        if let Some(source_contains) = &self.source_contains {
            let needle = source_contains.to_lowercase();
            if !neuron.source_label.to_lowercase().contains(&needle) {
                return false;
            }
        }

        true
    }
}

fn layer_diagnostics(network: &Network) -> Vec<LayerDiagnostics> {
    network
        .layers
        .iter()
        .enumerate()
        .map(|(layer_index, layer)| {
            let open_neurons = layer
                .neurons
                .iter()
                .filter(|neuron| matches!(neuron.protection_level, ProtectionLevel::Open))
                .count();
            let guarded_neurons = layer
                .neurons
                .iter()
                .filter(|neuron| matches!(neuron.protection_level, ProtectionLevel::Guarded))
                .count();
            let frozen_neurons = layer
                .neurons
                .iter()
                .filter(|neuron| matches!(neuron.protection_level, ProtectionLevel::Frozen))
                .count();

            LayerDiagnostics {
                layer_index,
                layer_id: layer.id,
                activation: layer.activation,
                neurons: layer.neurons.len(),
                open_neurons,
                guarded_neurons,
                frozen_neurons,
            }
        })
        .collect()
}

fn source_label(source: &Source) -> String {
    match source {
        Source::RawText => "raw text".to_string(),
        Source::LocalFile { path } => path.clone(),
        Source::Unknown => "unknown".to_string(),
    }
}

fn update_best(best: &mut Option<TraceCandidate>, candidate: TraceCandidate) {
    if best
        .as_ref()
        .map(|current| candidate.score > current.score)
        .unwrap_or(true)
    {
        *best = Some(candidate);
    }
}

fn hidden_neuron_id(network: &Network, hidden_index: usize) -> Option<u64> {
    network
        .layers
        .first()
        .and_then(|layer| layer.neurons.get(hidden_index))
        .map(|neuron| neuron.id)
}

fn best_hidden_activation(hidden: &[f32]) -> (Option<usize>, f32) {
    hidden
        .iter()
        .enumerate()
        .max_by(|left, right| {
            left.1
                .abs()
                .partial_cmp(&right.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, activation)| (Some(index), *activation))
        .unwrap_or((None, 0.0))
}

fn top_hidden_activations(
    network: &Network,
    input: &[f32],
    limit: usize,
) -> Vec<NeuronActivationDiagnostic> {
    let Some(layer) = network.layers.first() else {
        return Vec::new();
    };
    let cache = network.forward_with_cache(input);
    let mut rows = layer
        .neurons
        .iter()
        .zip(cache.hidden.iter())
        .enumerate()
        .map(
            |(neuron_index, (neuron, activation))| NeuronActivationDiagnostic {
                layer_index: 0,
                layer_id: layer.id,
                neuron_index,
                neuron_id: neuron.id,
                activation: *activation,
                protection: neuron.protection_level,
                source_label: source_label(&neuron.source),
            },
        )
        .collect::<Vec<_>>();

    rows.sort_by(|left, right| {
        right
            .activation
            .abs()
            .partial_cmp(&left.activation.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows.truncate(limit);
    rows
}

fn top_output_values(
    network: &Network,
    output: &[f32],
    limit: usize,
) -> Vec<OutputValueDiagnostic> {
    let output_layer = network.layers.get(1);
    let mut rows = output
        .iter()
        .enumerate()
        .map(|(output_index, value)| OutputValueDiagnostic {
            output_index,
            neuron_id: output_layer
                .and_then(|layer| layer.neurons.get(output_index))
                .map(|neuron| neuron.id),
            value: *value,
        })
        .collect::<Vec<_>>();

    rows.sort_by(|left, right| {
        right
            .value
            .abs()
            .partial_cmp(&left.value.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows.truncate(limit);
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use manas_core::Source;

    const NOW: u64 = 1_800_000_000;

    #[test]
    fn diagnostics_counts_learned_hidden_neurons() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        trainer
            .learn_with_source(
                &mut network,
                "cat",
                "small animal with fur",
                Source::RawText,
            )
            .unwrap();

        let diagnostics = BrainDiagnostics::from_network(&network, NOW);

        assert_eq!(diagnostics.learning.facts_taught, 1);
        assert_eq!(diagnostics.sources.raw_text_neurons, 1);
        assert_eq!(diagnostics.network.total_layers, 2);
        assert!(diagnostics.learning.total_learn_calls > 0);
    }

    #[test]
    fn neuron_filter_matches_protection_and_source() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        trainer
            .learn_with_source(
                &mut network,
                "cat",
                "small animal with fur",
                Source::LocalFile {
                    path: "notes.txt".to_string(),
                },
            )
            .unwrap();
        network.layers[0].neurons[0].guard_all();

        let rows = filtered_neuron_diagnostics(
            &network,
            NOW,
            &NeuronFilter {
                protection: Some(ProtectionLevel::Guarded),
                source_contains: Some("notes".to_string()),
            },
        );

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].source_label, "notes.txt");
    }

    #[test]
    fn trace_query_matches_trainer_answer() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        trainer
            .learn(&mut network, "cat", "small animal with fur")
            .unwrap();

        let query = trainer.query(&network, "What is a cat?").unwrap();
        let trace = trace_query(&trainer, &network, "What is a cat?", 4);

        assert_eq!(trace.answered_from, AnswerSource::NeuralWeights);
        assert_eq!(trace.answer, query.answer);
        assert!(trace.selected_variant.is_some());
        assert!(!trace.top_hidden_activations.is_empty());
        assert!(!trace.top_output_values.is_empty());
    }
}
