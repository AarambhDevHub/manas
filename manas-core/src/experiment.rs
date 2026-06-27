//! Stage 1 standalone associative-memory proof.
//!
//! Compile directly:
//!   rustc --edition=2024 -O -D warnings manas-core/src/experiment.rs -o /tmp/manas-stage1
//!   /tmp/manas-stage1
//!
//! Run tests directly:
//!   rustc --edition=2024 --test -D warnings manas-core/src/experiment.rs -o /tmp/manas-stage1-tests
//!   /tmp/manas-stage1-tests

use std::process;

const EMBED_DIM: usize = 32;
const HIDDEN_DIM: usize = 64;
const OUTPUT_DIM: usize = 32;
const EMBED_TABLE_SIZE: usize = 8192;
const MAX_EPOCHS: usize = 2_000;
const LEARNING_RATE: f32 = 0.02;
const GRAD_CLIP: f32 = 1.0;
const CHECK_EVERY: usize = 25;
const CORRECT_THRESHOLD: f32 = 0.70;
const WRONG_THRESHOLD: f32 = 0.35;

const PROOF_FACTS: [(&str, &str); 3] = [
    ("cat", "small animal with fur"),
    ("paris", "city in france"),
    ("rust", "systems programming language"),
];

const SEEDS: [u64; 5] = [1, 7, 42, 2026, 99_991];

fn main() {
    let mut all_passed = true;

    for seed in SEEDS {
        let report = run_proof(seed);
        print_report(&report);
        println!();

        if !report.passed {
            all_passed = false;
        }
    }

    if all_passed {
        println!("PASS: associative memory proof succeeded for all seeds");
    } else {
        eprintln!("FAIL: associative memory proof did not meet thresholds");
        process::exit(1);
    }
}

#[derive(Clone)]
struct Neuron {
    weights: Vec<f32>,
    bias: f32,
}

struct Layer {
    neurons: Vec<Neuron>,
}

struct Network {
    layers: Vec<Layer>,
}

struct ForwardCache {
    input: Vec<f32>,
    hidden: Vec<f32>,
    output: Vec<f32>,
}

impl Network {
    fn new(seed: u64, input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = SplitMix64::new(seed);
        let hidden_limit = xavier_limit(input_dim, hidden_dim);
        let output_limit = xavier_limit(hidden_dim, output_dim);

        let hidden = Layer {
            neurons: (0..hidden_dim)
                .map(|_| Neuron::random(&mut rng, input_dim, hidden_limit))
                .collect(),
        };

        let output = Layer {
            neurons: (0..output_dim)
                .map(|_| Neuron::random(&mut rng, hidden_dim, output_limit))
                .collect(),
        };

        Self {
            layers: vec![hidden, output],
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.forward_cached(input).output
    }

    fn forward_cached(&self, input: &[f32]) -> ForwardCache {
        debug_assert_eq!(self.layers.len(), 2);

        let hidden = self.layers[0]
            .neurons
            .iter()
            .map(|neuron| tanh(dot(&neuron.weights, input) + neuron.bias))
            .collect::<Vec<_>>();

        let output = self.layers[1]
            .neurons
            .iter()
            .map(|neuron| dot(&neuron.weights, &hidden) + neuron.bias)
            .collect::<Vec<_>>();

        ForwardCache {
            input: input.to_vec(),
            hidden,
            output,
        }
    }

    fn backprop(&mut self, input: &[f32], target: &[f32], lr: f32) -> f32 {
        assert_eq!(target.len(), OUTPUT_DIM);

        let cache = self.forward_cached(input);
        let loss = mse_loss(&cache.output, target);

        let output_deltas = cache
            .output
            .iter()
            .zip(target.iter())
            .map(|(actual, expected)| 2.0 * (actual - expected) / OUTPUT_DIM as f32)
            .collect::<Vec<_>>();

        let mut hidden_deltas = vec![0.0; HIDDEN_DIM];
        for (output_delta, output_neuron) in
            output_deltas.iter().zip(self.layers[1].neurons.iter())
        {
            for (hidden_delta, weight) in hidden_deltas.iter_mut().zip(output_neuron.weights.iter())
            {
                *hidden_delta += output_delta * weight;
            }
        }

        for (hidden_delta, hidden_activation) in hidden_deltas.iter_mut().zip(cache.hidden.iter()) {
            *hidden_delta *= tanh_derivative_from_activation(*hidden_activation);
        }

        for (output_neuron, output_delta) in self.layers[1]
            .neurons
            .iter_mut()
            .zip(output_deltas.iter())
        {
            apply_weight_updates(output_neuron, &cache.hidden, *output_delta, lr);
        }

        for (hidden_neuron, hidden_delta) in self.layers[0]
            .neurons
            .iter_mut()
            .zip(hidden_deltas.iter())
        {
            apply_weight_updates(hidden_neuron, &cache.input, *hidden_delta, lr);
        }

        loss
    }
}

impl Neuron {
    fn random(rng: &mut SplitMix64, input_dim: usize, limit: f32) -> Self {
        let weights = (0..input_dim)
            .map(|_| rng.uniform_range(-limit, limit))
            .collect();

        Self { weights, bias: 0.0 }
    }
}

#[derive(Clone)]
enum EmbeddingSlot {
    Empty,
    Occupied { hash: u64, vector: Vec<f32> },
}

struct Encoder {
    slots: Vec<EmbeddingSlot>,
    seed: u64,
    dim: usize,
}

impl Encoder {
    fn new(seed: u64, dim: usize, table_size: usize) -> Self {
        Self {
            slots: vec![EmbeddingSlot::Empty; table_size],
            seed,
            dim,
        }
    }

    fn encode(&mut self, text: &str) -> Vec<f32> {
        let mut encoded = vec![0.0; self.dim];

        for word in normalized_words(text) {
            let word_vec = self.word_vector(&word);
            for (dst, src) in encoded.iter_mut().zip(word_vec.iter()) {
                *dst += src;
            }
        }

        encoded
    }

    fn word_vector(&mut self, word: &str) -> Vec<f32> {
        let hash = stable_hash(word);
        let mut index = hash as usize % self.slots.len();

        loop {
            match &self.slots[index] {
                EmbeddingSlot::Occupied {
                    hash: existing_hash,
                    vector,
                } if *existing_hash == hash => return vector.clone(),
                EmbeddingSlot::Occupied { .. } => {
                    index = (index + 1) % self.slots.len();
                }
                EmbeddingSlot::Empty => {
                    let vector = make_embedding(hash ^ self.seed, self.dim);
                    self.slots[index] = EmbeddingSlot::Occupied {
                        hash,
                        vector: vector.clone(),
                    };
                    return vector;
                }
            }
        }
    }
}

#[derive(Clone)]
struct FactVector {
    input_text: &'static str,
    target_text: &'static str,
    input: Vec<f32>,
    target: Vec<f32>,
}

struct FactScore {
    input_text: &'static str,
    target_text: &'static str,
    correct_similarity: f32,
    max_wrong_similarity: f32,
}

struct ProofReport {
    seed: u64,
    epochs_run: usize,
    first_loss: f32,
    final_loss: f32,
    scores: Vec<FactScore>,
    passed: bool,
}

fn run_proof(seed: u64) -> ProofReport {
    let mut encoder = Encoder::new(seed ^ 0x9e37_79b9_7f4a_7c15, EMBED_DIM, EMBED_TABLE_SIZE);
    let facts = encode_facts(&mut encoder);
    let mut network = Network::new(seed ^ 0xd1b5_4a32_d192_ed03, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM);

    let mut first_loss = 0.0;
    let mut final_loss = 0.0;
    let mut epochs_run = MAX_EPOCHS;

    for epoch in 1..=MAX_EPOCHS {
        let mut epoch_loss = 0.0;

        for offset in 0..facts.len() {
            let index = (epoch + offset) % facts.len();
            epoch_loss += network.backprop(&facts[index].input, &facts[index].target, LEARNING_RATE);
        }

        epoch_loss /= facts.len() as f32;

        if epoch == 1 {
            first_loss = epoch_loss;
        }
        final_loss = epoch_loss;

        if epoch % CHECK_EVERY == 0 {
            let scores = evaluate(&network, &facts);
            if proof_passed(&scores) {
                epochs_run = epoch;
                return ProofReport {
                    seed,
                    epochs_run,
                    first_loss,
                    final_loss,
                    scores,
                    passed: true,
                };
            }
        }
    }

    let scores = evaluate(&network, &facts);
    ProofReport {
        seed,
        epochs_run,
        first_loss,
        final_loss,
        passed: proof_passed(&scores),
        scores,
    }
}

fn encode_facts(encoder: &mut Encoder) -> Vec<FactVector> {
    PROOF_FACTS
        .iter()
        .map(|(input_text, target_text)| FactVector {
            input_text,
            target_text,
            input: encoder.encode(input_text),
            target: encoder.encode(target_text),
        })
        .collect()
}

fn evaluate(network: &Network, facts: &[FactVector]) -> Vec<FactScore> {
    facts
        .iter()
        .map(|fact| {
            let output = network.forward(&fact.input);
            let correct_similarity = cosine(&output, &fact.target);
            let max_wrong_similarity = facts
                .iter()
                .filter(|candidate| candidate.target_text != fact.target_text)
                .map(|candidate| cosine(&output, &candidate.target))
                .fold(f32::NEG_INFINITY, f32::max);

            FactScore {
                input_text: fact.input_text,
                target_text: fact.target_text,
                correct_similarity,
                max_wrong_similarity,
            }
        })
        .collect()
}

fn proof_passed(scores: &[FactScore]) -> bool {
    scores.iter().all(|score| {
        score.correct_similarity > CORRECT_THRESHOLD
            && score.max_wrong_similarity < WRONG_THRESHOLD
    })
}

fn print_report(report: &ProofReport) {
    println!("Seed: {}", report.seed);
    println!("Epochs run: {}", report.epochs_run);
    println!("Loss: {:.6} -> {:.6}", report.first_loss, report.final_loss);

    for score in &report.scores {
        println!(
            "{:<6} similarity to target ({:<28}) : {:.4}",
            score.input_text, score.target_text, score.correct_similarity
        );
        println!(
            "{:<6} max similarity to wrong target          : {:.4}",
            score.input_text, score.max_wrong_similarity
        );
    }

    println!("Result: {}", if report.passed { "PASS" } else { "FAIL" });
}

fn normalized_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|raw| {
            let cleaned = raw
                .chars()
                .filter(|ch| ch.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>();

            if cleaned.is_empty() {
                None
            } else {
                Some(cleaned)
            }
        })
        .collect()
}

fn make_embedding(seed: u64, dim: usize) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut vector = (0..dim)
        .map(|_| rng.uniform_range(-1.0, 1.0))
        .collect::<Vec<_>>();

    normalize_in_place(&mut vector);
    vector
}

fn normalize_in_place(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();

    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

fn stable_hash(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;

    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }

    splitmix64(hash)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        splitmix64(self.state)
    }

    fn next_f32(&mut self) -> f32 {
        let bits = self.next_u64() >> 40;
        bits as f32 / (1_u32 << 24) as f32
    }

    fn uniform_range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }
}

fn xavier_limit(fan_in: usize, fan_out: usize) -> f32 {
    (6.0 / (fan_in + fan_out) as f32).sqrt()
}

fn apply_weight_updates(neuron: &mut Neuron, input: &[f32], delta: f32, lr: f32) {
    for (weight, input_value) in neuron.weights.iter_mut().zip(input.iter()) {
        let gradient = (delta * input_value).clamp(-GRAD_CLIP, GRAD_CLIP);
        *weight -= lr * gradient;
    }

    neuron.bias -= lr * delta.clamp(-GRAD_CLIP, GRAD_CLIP);
}

fn mse_loss(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(left, right)| {
            let error = left - right;
            error * error
        })
        .sum::<f32>()
        / actual.len() as f32
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| left_value * right_value)
        .sum()
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let numerator = dot(left, right);
    let left_norm = dot(left, left).sqrt();
    let right_norm = dot(right, right).sqrt();

    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        numerator / (left_norm * right_norm)
    }
}

fn tanh(value: f32) -> f32 {
    value.tanh()
}

fn tanh_derivative_from_activation(activation: f32) -> f32 {
    1.0 - activation * activation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_are_deterministic_for_same_seed() {
        let mut first = Encoder::new(42, EMBED_DIM, EMBED_TABLE_SIZE);
        let mut second = Encoder::new(42, EMBED_DIM, EMBED_TABLE_SIZE);

        assert_eq!(first.encode("cat"), second.encode("cat"));
        assert_eq!(
            first.encode("small animal with fur"),
            second.encode("small animal with fur")
        );
    }

    #[test]
    fn unrelated_phrases_are_not_highly_similar() {
        let mut encoder = Encoder::new(42, EMBED_DIM, EMBED_TABLE_SIZE);
        let cat = encoder.encode("small animal with fur");
        let rust = encoder.encode("systems programming language");

        assert!(
            cosine(&cat, &rust) < 0.50,
            "unrelated phrase vectors were too similar"
        );
    }

    #[test]
    fn single_fact_training_reduces_loss() {
        let mut encoder = Encoder::new(42, EMBED_DIM, EMBED_TABLE_SIZE);
        let input = encoder.encode("cat");
        let target = encoder.encode("small animal with fur");
        let mut network = Network::new(42, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM);

        let before = mse_loss(&network.forward(&input), &target);
        for _ in 0..500 {
            network.backprop(&input, &target, LEARNING_RATE);
        }
        let after = mse_loss(&network.forward(&input), &target);

        assert!(after < before * 0.25, "loss did not fall enough");
    }

    #[test]
    fn three_fact_proof_passes_for_primary_seed() {
        let report = run_proof(42);
        assert!(report.passed, "primary proof seed failed");
    }

    #[test]
    fn three_fact_proof_passes_for_five_fixed_seeds() {
        for seed in SEEDS {
            let report = run_proof(seed);
            assert!(report.passed, "proof failed for seed {}", seed);
        }
    }
}
