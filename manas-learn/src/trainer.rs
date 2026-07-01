use manas_core::{
    GROWTH_THRESHOLD, MAX_UPDATE_ATTEMPTS, ManasError, Network, ProtectionLevel, TrainingExample,
};

use crate::backprop::{compute_gradients, cosine, mse_loss};
use crate::decoder::decode_answer;
use crate::encoder::Encoder;

const DEFAULT_EMBED_TABLE_SIZE: usize = 8192;
const OPEN_TO_GUARDED_ACTIVATIONS: u64 = 500;
const GUARDED_TO_FROZEN_ACTIVATIONS: u64 = 2_000;

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
            target: self.encoder.encode(target),
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
        let fact = self.encode_fact(input, target);
        let loss_before = loss_for_fact(network, &fact)?;
        let mut loss_after = loss_before;
        let mut neurons_grown = 0;
        let mut update_applied = false;

        if network.layers[0].neurons.is_empty() || network.layers[1].neurons.is_empty() {
            grow_for_fact(network, &fact)?;
            neurons_grown += 1;
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
                grow_for_fact(network, &fact)?;
                neurons_grown += 1;
                update_applied = true;
                loss_after = loss_for_fact(network, &fact)?;
            }
        } else if !update_applied {
            let (_, gradients) = compute_gradients(network, &fact.input, &fact.target)?;
            network.apply_gradients(&gradients, self.learning_rate)?;
            update_applied = true;
            loss_after = loss_for_fact(network, &fact)?;
        }

        let protection_report = self.update_protection_levels(network);

        Ok(LearnReport {
            loss_before,
            loss_after,
            neurons_grown,
            layers_grown: 0,
            neurons_promoted: protection_report.neurons_promoted,
            neurons_frozen: protection_report.neurons_frozen,
            total_neurons: network.neuron_count(),
            update_applied,
        })
    }

    pub fn update_protection_levels(&self, network: &mut Network) -> ProtectionReport {
        let mut report = ProtectionReport::default();

        for layer in &mut network.layers {
            for neuron in &mut layer.neurons {
                neuron.importance_score = (neuron.activation_count as f32
                    / GUARDED_TO_FROZEN_ACTIVATIONS as f32)
                    .clamp(0.0, 1.0);

                let before = neuron.protection_level;
                if neuron.activation_count >= GUARDED_TO_FROZEN_ACTIVATIONS {
                    if !matches!(before, ProtectionLevel::Frozen) {
                        neuron.freeze_all();
                    }
                } else if neuron.activation_count >= OPEN_TO_GUARDED_ACTIVATIONS
                    && matches!(before, ProtectionLevel::Open)
                {
                    neuron.guard_all();
                }

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
        let input = self.encoder.encode_deterministic(question);
        if input.iter().all(|value| value.abs() <= f32::EPSILON) || network.neuron_count() == 0 {
            return Ok(not_enough());
        }

        let output = network.forward(&input);
        Ok(match decode_answer(&output, &self.encoder, question) {
            Some(decoded) => QueryResult {
                answer: decoded.answer,
                confidence: decoded.confidence,
                answered_from: AnswerSource::NeuralWeights,
            },
            None => not_enough(),
        })
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

fn not_enough() -> QueryResult {
    QueryResult {
        answer: "Not enough knowledge yet.".to_string(),
        confidence: 0.0,
        answered_from: AnswerSource::NotEnough,
    }
}

fn grow_for_fact(network: &mut Network, fact: &EncodedFact) -> Result<(), ManasError> {
    let neuron_id = network.grow_neuron(0, fact.input.len())?;
    network.key_hidden_neuron_to_input(neuron_id, &fact.input)?;
    network.fit_open_output_weights_to_facts(
        &[TrainingExample {
            input: &fact.input,
            target: &fact.target,
        }],
        &[],
    )
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
    fn protection_learn_report_records_transitions() {
        let mut network = Network::new(32, 64, 32);
        network.layers[0].neurons[0].activation_count = OPEN_TO_GUARDED_ACTIVATIONS - 1;
        network.layers[0].neurons[1].guard_all();
        network.layers[0].neurons[1].activation_count = GUARDED_TO_FROZEN_ACTIVATIONS - 1;
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
    fn query_returns_neural_weights_for_learned_fact() {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);

        trainer
            .learn(&mut network, "cat", "small animal with fur")
            .unwrap();

        let result = trainer.query(&network, "What is a cat?").unwrap();

        assert_eq!(result.answered_from, AnswerSource::NeuralWeights);
        assert!(result.confidence > 0.0);
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
    }
}
