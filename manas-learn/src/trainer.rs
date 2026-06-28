use manas_core::{ManasError, Network, TrainingExample};

use crate::backprop::{compute_gradients, cosine};
use crate::encoder::Encoder;

const DEFAULT_EMBED_TABLE_SIZE: usize = 8192;

/// Encoded input-target fact used by Stage 3 training.
#[derive(Clone, Debug)]
pub struct EncodedFact {
    pub input_text: String,
    pub target_text: String,
    pub input: Vec<f32>,
    pub target: Vec<f32>,
}

/// Minimal trainer for the proven associative-memory engine.
pub struct Trainer {
    pub encoder: Encoder,
    pub learning_rate: f32,
}

impl Trainer {
    pub fn new(learning_rate: f32) -> Self {
        Self::with_seed(42, 32, learning_rate)
    }

    pub fn with_seed(seed: u64, embed_dim: usize, learning_rate: f32) -> Self {
        Self {
            encoder: Encoder::new(seed, embed_dim, DEFAULT_EMBED_TABLE_SIZE),
            learning_rate,
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
