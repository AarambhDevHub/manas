//! Stage 1 and Stage 2 standalone associative-memory proof.
//!
//! Compile directly:
//!   rustc --edition=2024 -O -D warnings manas-core/src/experiment.rs -o /tmp/manas-stage2
//!   /tmp/manas-stage2
//!
//! Run tests directly:
//!   rustc --edition=2024 --test -D warnings manas-core/src/experiment.rs -o /tmp/manas-stage2-tests
//!   /tmp/manas-stage2-tests

use std::process;

const EMBED_DIM: usize = 32;
const HIDDEN_DIM: usize = 64;
const OUTPUT_DIM: usize = 32;
const EMBED_TABLE_SIZE: usize = 8192;
const MAX_EPOCHS: usize = 2_000;
const LEARNING_RATE: f32 = 0.02;
const STAGE2_LEARNING_RATE: f32 = 0.01;
const GRAD_CLIP: f32 = 1.0;
const GUARD_DELTA: f32 = 0.001;
const CHECK_EVERY: usize = 25;
const CORRECT_THRESHOLD: f32 = 0.70;
const WRONG_THRESHOLD: f32 = 0.35;
const ANCHOR_TRAIN_EPOCHS: usize = 300;
const NOISE_TRAIN_EPOCHS: usize = 200;
const ANCHOR_SURVIVAL_THRESHOLD: f32 = 0.65;
const NEW_FACT_THRESHOLD: f32 = 0.70;
const MAX_FORGETTING_DELTA: f32 = 0.15;
const ANCHOR_NEURONS_PER_FACT: usize = 1;
const ANCHOR_CONSTRAINT_WEIGHT: f32 = 50.0;
const KEY_SHARPNESS: f32 = 12.0;
const RIDGE: f32 = 1.0e-4;

const PROOF_FACTS: [(&str, &str); 3] = [
    ("cat", "small animal with fur"),
    ("paris", "city in france"),
    ("rust", "systems programming language"),
];

const ANCHOR_FACTS: [(&str, &str); 5] = [
    ("cat", "small animal with fur"),
    ("paris", "city in france"),
    ("rust", "systems programming language"),
    ("everest", "highest mountain on earth"),
    ("dna", "double helix genetic information"),
];

const NOISE_FACTS: [(&str, &str); 50] = [
    ("amazon", "largest river by discharge"),
    ("einstein", "developed theory of relativity"),
    ("photosynthesis", "converts sunlight to energy"),
    ("hydrogen", "lightest element in universe"),
    ("brain", "contains billions of neurons"),
    ("shakespeare", "wrote plays and sonnets"),
    ("light", "travels at constant speed"),
    ("rome", "empire fell in ancient history"),
    ("water", "boils at standard pressure"),
    ("python", "created by guido van rossum"),
    ("jupiter", "largest planet solar system"),
    ("mona lisa", "painted by leonardo da vinci"),
    ("mitochondria", "powerhouse of cell"),
    ("pacific", "largest ocean on earth"),
    ("bitcoin", "created by satoshi nakamoto"),
    ("nitrogen", "moves through ecosystems"),
    ("gravity", "pulls objects toward mass"),
    ("moon", "orbits planet earth"),
    ("mars", "red planet with thin atmosphere"),
    ("venus", "hot planet with dense atmosphere"),
    ("saturn", "planet with visible rings"),
    ("mercury", "closest planet to sun"),
    ("oxygen", "gas required for respiration"),
    ("carbon", "basis of organic chemistry"),
    ("helium", "noble gas used in balloons"),
    ("sodium", "reactive metal in salt"),
    ("chlorine", "greenish gas disinfects water"),
    ("glucose", "sugar used for energy"),
    ("protein", "molecule made of amino acids"),
    ("cell", "basic unit of life"),
    ("bacteria", "single celled microorganisms"),
    ("virus", "infectious particle needing host"),
    ("volcano", "erupts molten rock"),
    ("earthquake", "shaking from tectonic movement"),
    ("hurricane", "rotating tropical storm"),
    ("desert", "dry region with little rainfall"),
    ("rainforest", "dense forest with high rainfall"),
    ("tundra", "cold biome with permafrost"),
    ("democracy", "government by elected people"),
    ("currency", "medium used for exchange"),
    ("algorithm", "step by step procedure"),
    ("compiler", "translates source code"),
    ("database", "stores structured information"),
    ("network", "connects computers together"),
    ("encryption", "protects data with keys"),
    ("battery", "stores electrical energy"),
    ("magnet", "produces magnetic field"),
    ("telescope", "observes distant objects"),
    ("microscope", "magnifies tiny objects"),
    ("thermometer", "measures temperature"),
];

const SEEDS: [u64; 5] = [1, 7, 42, 2026, 99_991];

fn main() {
    let mut stage1_passed = true;
    let mut stage2_passed = true;

    println!("=== Stage 1: associative memory proof ===");

    for seed in SEEDS {
        let report = run_proof(seed);
        print_report(&report);
        println!();

        if !report.passed {
            stage1_passed = false;
        }
    }

    if stage1_passed {
        println!("PASS: associative memory proof succeeded for all seeds");
    } else {
        eprintln!("FAIL: associative memory proof did not meet thresholds");
    }

    println!();
    println!("=== Stage 2: anti-forgetting proof ===");

    for seed in SEEDS {
        let report = run_anti_forgetting_proof(seed);
        print_anti_forgetting_report(&report);
        println!();

        if !report.passed {
            stage2_passed = false;
        }
    }

    if stage2_passed {
        println!("PASS: anti-forgetting proof succeeded for all seeds");
    } else {
        eprintln!("FAIL: anti-forgetting proof did not meet thresholds");
    }

    if !stage1_passed || !stage2_passed {
        process::exit(1);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Protection {
    Open,
    Guarded,
    Frozen,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ActivationMode {
    Tanh,
    Keyed,
}

#[derive(Clone)]
struct Neuron {
    weights: Vec<f32>,
    bias: f32,
    activation_mode: ActivationMode,
    protection: Protection,
    weight_protection: Vec<Protection>,
    bias_protection: Protection,
    activation_count: u64,
}

struct Layer {
    neurons: Vec<Neuron>,
}

struct Network {
    layers: Vec<Layer>,
    protected_inputs: Vec<Vec<f32>>,
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
            protected_inputs: Vec::new(),
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
            .map(|neuron| neuron.activate(input))
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

        for ((hidden_delta, hidden_activation), hidden_neuron) in hidden_deltas
            .iter_mut()
            .zip(cache.hidden.iter())
            .zip(self.layers[0].neurons.iter())
        {
            *hidden_delta *= hidden_neuron.derivative_from_activation(*hidden_activation);
        }

        for (output_neuron, output_delta) in self.layers[1]
            .neurons
            .iter_mut()
            .zip(output_deltas.iter())
        {
            apply_weight_updates(output_neuron, &cache.hidden, *output_delta, lr, &[]);
        }

        let protected_inputs = self.protected_inputs.clone();
        for (hidden_neuron, hidden_delta) in self.layers[0]
            .neurons
            .iter_mut()
            .zip(hidden_deltas.iter())
        {
            apply_weight_updates(
                hidden_neuron,
                &cache.input,
                *hidden_delta,
                lr,
                &protected_inputs,
            );
            if !protected_inputs.is_empty()
                && !matches!(hidden_neuron.protection, Protection::Frozen)
            {
                hidden_neuron.bias = 0.0;
            }
        }

        loss
    }

    #[allow(dead_code)]
    fn promote_neurons(&mut self) {
        for layer in &mut self.layers {
            for neuron in &mut layer.neurons {
                if neuron.activation_count > 2_000 {
                    neuron.freeze_all();
                } else if neuron.activation_count > 500
                    && matches!(neuron.protection, Protection::Open)
                {
                    neuron.guard_all();
                }
            }
        }
    }

    fn consolidate_anchor_facts(&mut self, anchors: &[FactVector]) {
        self.protected_inputs.clear();

        for fact in anchors {
            let mut protected_input = fact.input.clone();
            normalize_in_place(&mut protected_input);
            self.protected_inputs.push(protected_input);
        }

        let mut frozen_indices = Vec::new();
        for fact in anchors {
            let selected = self.select_open_hidden_neurons(fact, ANCHOR_NEURONS_PER_FACT);
            self.key_hidden_neurons_to_fact(fact, &selected);
            frozen_indices.extend(selected);
        }

        self.fit_frozen_output_weights(anchors, &frozen_indices);
        self.freeze_output_biases();
        self.orthogonalize_open_hidden_to_protected_inputs();
    }

    fn select_open_hidden_neurons(&self, fact: &FactVector, count: usize) -> Vec<usize> {
        let cache = self.forward_cached(&fact.input);
        let mut candidates = cache
            .hidden
            .iter()
            .enumerate()
            .filter(|(index, _)| !matches!(self.layers[0].neurons[*index].protection, Protection::Frozen))
            .map(|(index, activation)| (index, activation.abs()))
            .collect::<Vec<_>>();

        candidates.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
            .into_iter()
            .take(count)
            .map(|(index, _)| index)
            .collect()
    }

    fn key_hidden_neurons_to_fact(&mut self, fact: &FactVector, selected: &[usize]) {
        let mut key = fact.input.clone();
        normalize_in_place(&mut key);

        for index in selected {
            let neuron = &mut self.layers[0].neurons[*index];
            for (weight, key_value) in neuron.weights.iter_mut().zip(key.iter()) {
                *weight = *key_value;
            }
            neuron.bias = 0.0;
            neuron.activation_mode = ActivationMode::Keyed;
            neuron.freeze_all();
        }
    }

    fn fit_frozen_output_weights(&mut self, anchors: &[FactVector], frozen_indices: &[usize]) {
        let hidden_matrix = anchors
            .iter()
            .map(|fact| {
                let cache = self.forward_cached(&fact.input);
                frozen_indices
                    .iter()
                    .map(|index| cache.hidden[*index])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut gram = vec![vec![0.0; anchors.len()]; anchors.len()];
        for row in 0..anchors.len() {
            for col in 0..anchors.len() {
                gram[row][col] = dot(&hidden_matrix[row], &hidden_matrix[col]);
            }
            gram[row][row] += RIDGE;
        }

        for output_dim in 0..OUTPUT_DIM {
            let rhs = anchors
                .iter()
                .map(|fact| fact.target[output_dim])
                .collect::<Vec<_>>();
            let alpha = solve_linear_system(gram.clone(), rhs)
                .expect("anchor gram matrix should be solvable with ridge");

            for (feature_index, hidden_index) in frozen_indices.iter().enumerate() {
                let weight = hidden_matrix
                    .iter()
                    .zip(alpha.iter())
                    .map(|(row, alpha_value)| row[feature_index] * alpha_value)
                    .sum::<f32>();
                let output_neuron = &mut self.layers[1].neurons[output_dim];
                output_neuron.weights[*hidden_index] = weight;
                output_neuron.weight_protection[*hidden_index] = Protection::Frozen;
            }
        }
    }

    fn freeze_output_biases(&mut self) {
        for output_neuron in &mut self.layers[1].neurons {
            output_neuron.bias = 0.0;
            output_neuron.bias_protection = Protection::Frozen;
        }
    }

    fn fit_open_output_weights_to_facts(&mut self, facts: &[FactVector], anchors: &[FactVector]) {
        let open_indices = self.open_hidden_indices();
        if open_indices.is_empty() || facts.is_empty() {
            return;
        }

        let mut rows = Vec::with_capacity(facts.len() + anchors.len());
        for fact in anchors {
            rows.push((fact, ANCHOR_CONSTRAINT_WEIGHT));
        }
        for fact in facts {
            rows.push((fact, 1.0));
        }

        let hidden_matrix = rows
            .iter()
            .map(|(fact, _)| {
                let cache = self.forward_cached(&fact.input);
                open_indices
                    .iter()
                    .map(|index| cache.hidden[*index])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let residuals = rows
            .iter()
            .map(|(fact, _)| {
                let cache = self.forward_cached(&fact.input);
                self.residual_after_frozen_output(&cache.hidden, &fact.target)
            })
            .collect::<Vec<_>>();
        let weights = rows.iter().map(|(_, weight)| *weight).collect::<Vec<_>>();

        let mut gram = vec![vec![0.0; open_indices.len()]; open_indices.len()];
        for row in 0..open_indices.len() {
            for col in 0..open_indices.len() {
                gram[row][col] = hidden_matrix
                    .iter()
                    .zip(weights.iter())
                    .map(|(features, weight)| weight * features[row] * features[col])
                    .sum::<f32>();
            }
            gram[row][row] += RIDGE;
        }

        for output_dim in 0..OUTPUT_DIM {
            let rhs = (0..open_indices.len())
                .map(|feature_index| {
                    hidden_matrix
                        .iter()
                        .zip(residuals.iter())
                        .zip(weights.iter())
                        .map(|((features, residual), weight)| {
                            weight * features[feature_index] * residual[output_dim]
                        })
                        .sum::<f32>()
                })
                .collect::<Vec<_>>();
            let weights = solve_linear_system(gram.clone(), rhs)
                .expect("open hidden readout should be solvable with ridge");

            for (feature_index, hidden_index) in open_indices.iter().enumerate() {
                let output_neuron = &mut self.layers[1].neurons[output_dim];
                if matches!(
                    output_neuron.weight_protection[*hidden_index],
                    Protection::Open
                ) {
                    output_neuron.weights[*hidden_index] = weights[feature_index];
                }
            }
        }
    }

    fn key_open_hidden_neurons_to_facts(&mut self, facts: &[FactVector]) {
        let open_indices = self.open_hidden_indices();
        if facts.is_empty() {
            return;
        }

        for (slot, hidden_index) in open_indices.iter().enumerate() {
            let fact = &facts[slot % facts.len()];
            let mut key = fact.input.clone();
            normalize_in_place(&mut key);
            let neuron = &mut self.layers[0].neurons[*hidden_index];

            for (weight, key_value) in neuron.weights.iter_mut().zip(key.iter()) {
                *weight = *key_value;
            }
            neuron.bias = 0.0;
            neuron.activation_mode = ActivationMode::Keyed;
        }
    }

    fn residual_after_frozen_output(&self, hidden: &[f32], target: &[f32]) -> Vec<f32> {
        self.layers[1]
            .neurons
            .iter()
            .enumerate()
            .map(|(output_dim, output_neuron)| {
                let frozen_contribution = output_neuron
                    .weights
                    .iter()
                    .zip(output_neuron.weight_protection.iter())
                    .zip(hidden.iter())
                    .filter(|((_, protection), _)| matches!(protection, Protection::Frozen))
                    .map(|((weight, _), hidden_value)| weight * hidden_value)
                    .sum::<f32>();
                target[output_dim] - frozen_contribution
            })
            .collect()
    }

    fn open_hidden_indices(&self) -> Vec<usize> {
        self.layers[0]
            .neurons
            .iter()
            .enumerate()
            .filter(|(_, neuron)| !matches!(neuron.protection, Protection::Frozen))
            .map(|(index, _)| index)
            .collect()
    }

    fn orthogonalize_open_hidden_to_protected_inputs(&mut self) {
        let protected_inputs = self.protected_inputs.clone();
        if protected_inputs.is_empty() {
            return;
        }

        for hidden_neuron in &mut self.layers[0].neurons {
            if matches!(hidden_neuron.protection, Protection::Frozen) {
                continue;
            }
            remove_protected_components(&mut hidden_neuron.weights, &protected_inputs);
            hidden_neuron.bias = 0.0;
            hidden_neuron.bias_protection = Protection::Frozen;
        }
    }

    fn frozen_hidden_neuron_count(&self) -> usize {
        self.layers[0]
            .neurons
            .iter()
            .filter(|neuron| matches!(neuron.protection, Protection::Frozen))
            .count()
    }

    fn frozen_output_edge_count(&self) -> usize {
        self.layers[1]
            .neurons
            .iter()
            .map(|neuron| {
                neuron
                    .weight_protection
                    .iter()
                    .filter(|protection| matches!(protection, Protection::Frozen))
                    .count()
            })
            .sum()
    }
}

impl Neuron {
    fn random(rng: &mut SplitMix64, input_dim: usize, limit: f32) -> Self {
        let weights = (0..input_dim)
            .map(|_| rng.uniform_range(-limit, limit))
            .collect();

        Self {
            weights,
            bias: 0.0,
            activation_mode: ActivationMode::Tanh,
            protection: Protection::Open,
            weight_protection: vec![Protection::Open; input_dim],
            bias_protection: Protection::Open,
            activation_count: 0,
        }
    }

    fn activate(&self, input: &[f32]) -> f32 {
        match self.activation_mode {
            ActivationMode::Tanh => tanh(dot(&self.weights, input) + self.bias),
            ActivationMode::Keyed => keyed_activation(&self.weights, input),
        }
    }

    fn derivative_from_activation(&self, activation: f32) -> f32 {
        match self.activation_mode {
            ActivationMode::Tanh => tanh_derivative_from_activation(activation),
            ActivationMode::Keyed => 0.0,
        }
    }

    fn freeze_all(&mut self) {
        self.protection = Protection::Frozen;
        self.bias_protection = Protection::Frozen;
        for protection in &mut self.weight_protection {
            *protection = Protection::Frozen;
        }
    }

    #[allow(dead_code)]
    fn guard_all(&mut self) {
        self.protection = Protection::Guarded;
        self.bias_protection = Protection::Guarded;
        for protection in &mut self.weight_protection {
            *protection = Protection::Guarded;
        }
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

struct AnchorResult {
    input_text: &'static str,
    before: f32,
    after: f32,
    forgetting_delta: f32,
}

struct AntiForgettingReport {
    seed: u64,
    anchors: Vec<AnchorResult>,
    new_scores: Vec<FactScore>,
    min_anchor_after: f32,
    max_forgetting_delta: f32,
    min_new_similarity: f32,
    frozen_hidden_neurons: usize,
    frozen_output_edges: usize,
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

fn run_anti_forgetting_proof(seed: u64) -> AntiForgettingReport {
    let mut encoder = Encoder::new(seed ^ 0x6a09_e667_f3bc_c909, EMBED_DIM, EMBED_TABLE_SIZE);
    let anchors = encode_named_facts(&ANCHOR_FACTS, &mut encoder);
    let noise = encode_named_facts(&NOISE_FACTS, &mut encoder);
    let mut network = Network::new(seed ^ 0xbb67_ae85_84ca_a73b, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM);

    train_facts(
        &mut network,
        &anchors,
        ANCHOR_TRAIN_EPOCHS,
        STAGE2_LEARNING_RATE,
    );

    network.consolidate_anchor_facts(&anchors);
    let before = similarities_to_targets(&network, &anchors);

    train_facts(
        &mut network,
        &noise,
        NOISE_TRAIN_EPOCHS,
        STAGE2_LEARNING_RATE,
    );
    network.key_open_hidden_neurons_to_facts(&noise);
    network.fit_open_output_weights_to_facts(&noise, &anchors);

    let after = similarities_to_targets(&network, &anchors);
    let anchors = anchors
        .iter()
        .zip(before.iter().zip(after.iter()))
        .map(|(fact, (before, after))| AnchorResult {
            input_text: fact.input_text,
            before: *before,
            after: *after,
            forgetting_delta: before - after,
        })
        .collect::<Vec<_>>();
    let new_scores = evaluate(&network, &noise);

    let min_anchor_after = anchors
        .iter()
        .map(|anchor| anchor.after)
        .fold(f32::INFINITY, f32::min);
    let max_forgetting_delta = anchors
        .iter()
        .map(|anchor| anchor.forgetting_delta)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_new_similarity = new_scores
        .iter()
        .map(|score| score.correct_similarity)
        .fold(f32::INFINITY, f32::min);
    let frozen_hidden_neurons = network.frozen_hidden_neuron_count();
    let frozen_output_edges = network.frozen_output_edge_count();
    let passed = min_anchor_after > ANCHOR_SURVIVAL_THRESHOLD
        && max_forgetting_delta < MAX_FORGETTING_DELTA
        && min_new_similarity > NEW_FACT_THRESHOLD;

    AntiForgettingReport {
        seed,
        anchors,
        new_scores,
        min_anchor_after,
        max_forgetting_delta,
        min_new_similarity,
        frozen_hidden_neurons,
        frozen_output_edges,
        passed,
    }
}

fn encode_facts(encoder: &mut Encoder) -> Vec<FactVector> {
    encode_named_facts(&PROOF_FACTS, encoder)
}

fn encode_named_facts(
    facts: &[(&'static str, &'static str)],
    encoder: &mut Encoder,
) -> Vec<FactVector> {
    facts
        .iter()
        .map(|(input_text, target_text)| FactVector {
            input_text,
            target_text,
            input: encoder.encode(input_text),
            target: encoder.encode(target_text),
        })
        .collect()
}

fn train_facts(network: &mut Network, facts: &[FactVector], epochs: usize, lr: f32) {
    for epoch in 0..epochs {
        for offset in 0..facts.len() {
            let index = (epoch + offset) % facts.len();
            network.backprop(&facts[index].input, &facts[index].target, lr);
        }
    }
}

fn similarities_to_targets(network: &Network, facts: &[FactVector]) -> Vec<f32> {
    facts
        .iter()
        .map(|fact| cosine(&network.forward(&fact.input), &fact.target))
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

fn print_anti_forgetting_report(report: &AntiForgettingReport) {
    println!("Seed: {}", report.seed);
    println!(
        "Frozen hidden neurons: {} | frozen output edges: {}",
        report.frozen_hidden_neurons, report.frozen_output_edges
    );

    for anchor in &report.anchors {
        println!(
            "{:<8} before: {:.4} after: {:.4} forgetting: {:.4}",
            anchor.input_text, anchor.before, anchor.after, anchor.forgetting_delta
        );
    }

    println!(
        "New fact similarity min: {:.4} | anchor survival min: {:.4} | max forgetting: {:.4}",
        report.min_new_similarity, report.min_anchor_after, report.max_forgetting_delta
    );

    let weakest = report
        .new_scores
        .iter()
        .min_by(|left, right| {
            left.correct_similarity
                .partial_cmp(&right.correct_similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("stage 2 should evaluate at least one new fact");
    println!(
        "Weakest new fact: {} -> {} ({:.4})",
        weakest.input_text, weakest.target_text, weakest.correct_similarity
    );

    println!("Result: {}", if report.passed { "PASS" } else { "FAIL" });
}

fn solve_linear_system(mut matrix: Vec<Vec<f32>>, mut rhs: Vec<f32>) -> Option<Vec<f32>> {
    let size = rhs.len();

    for pivot in 0..size {
        let mut pivot_row = pivot;
        for row in (pivot + 1)..size {
            if matrix[row][pivot].abs() > matrix[pivot_row][pivot].abs() {
                pivot_row = row;
            }
        }

        if matrix[pivot_row][pivot].abs() < 1.0e-8 {
            return None;
        }

        matrix.swap(pivot, pivot_row);
        rhs.swap(pivot, pivot_row);

        let pivot_value = matrix[pivot][pivot];
        for col in pivot..size {
            matrix[pivot][col] /= pivot_value;
        }
        rhs[pivot] /= pivot_value;

        for row in 0..size {
            if row == pivot {
                continue;
            }

            let factor = matrix[row][pivot];
            for col in pivot..size {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }

    Some(rhs)
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

fn apply_weight_updates(
    neuron: &mut Neuron,
    input: &[f32],
    delta: f32,
    lr: f32,
    protected_inputs: &[Vec<f32>],
) {
    let mut gradients = input
        .iter()
        .map(|input_value| (delta * input_value).clamp(-GRAD_CLIP, GRAD_CLIP))
        .collect::<Vec<_>>();

    if !protected_inputs.is_empty() && !matches!(neuron.protection, Protection::Frozen) {
        remove_protected_components(&mut gradients, protected_inputs);
    }

    for ((weight, weight_protection), gradient) in neuron
        .weights
        .iter_mut()
        .zip(neuron.weight_protection.iter())
        .zip(gradients.iter())
    {
        let protection = strongest_protection(neuron.protection, *weight_protection);
        apply_protected_update(weight, -lr * gradient, protection);
    }

    apply_protected_update(
        &mut neuron.bias,
        -lr * delta.clamp(-GRAD_CLIP, GRAD_CLIP),
        strongest_protection(neuron.protection, neuron.bias_protection),
    );
    neuron.activation_count += 1;
}

fn apply_protected_update(value: &mut f32, raw_update: f32, protection: Protection) {
    match protection {
        Protection::Open => *value += raw_update,
        Protection::Guarded => *value += raw_update.clamp(-GUARD_DELTA, GUARD_DELTA),
        Protection::Frozen => {}
    }
}

fn strongest_protection(left: Protection, right: Protection) -> Protection {
    match (left, right) {
        (Protection::Frozen, _) | (_, Protection::Frozen) => Protection::Frozen,
        (Protection::Guarded, _) | (_, Protection::Guarded) => Protection::Guarded,
        (Protection::Open, Protection::Open) => Protection::Open,
    }
}

fn remove_protected_components(gradient: &mut [f32], protected_inputs: &[Vec<f32>]) {
    for protected in protected_inputs {
        let component = dot(gradient, protected);
        for (value, protected_value) in gradient.iter_mut().zip(protected.iter()) {
            *value -= component * protected_value;
        }
    }
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

fn keyed_activation(key: &[f32], input: &[f32]) -> f32 {
    let mut normalized_input = input.to_vec();
    normalize_in_place(&mut normalized_input);
    let distance_sq = key
        .iter()
        .zip(normalized_input.iter())
        .map(|(left, right)| {
            let diff = left - right;
            diff * diff
        })
        .sum::<f32>();

    (-KEY_SHARPNESS * distance_sq).exp()
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

    #[test]
    fn frozen_hidden_weights_never_change() {
        let mut encoder = Encoder::new(42, EMBED_DIM, EMBED_TABLE_SIZE);
        let input = encoder.encode("cat");
        let target = encoder.encode("small animal with fur");
        let mut network = Network::new(42, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM);
        network.layers[0].neurons[0].freeze_all();

        let weights_before = network.layers[0].neurons[0].weights.clone();
        let bias_before = network.layers[0].neurons[0].bias;

        for _ in 0..100 {
            network.backprop(&input, &target, 1.0);
        }

        assert_eq!(weights_before, network.layers[0].neurons[0].weights);
        assert_eq!(bias_before, network.layers[0].neurons[0].bias);
    }

    #[test]
    fn frozen_output_edge_never_changes() {
        let mut encoder = Encoder::new(42, EMBED_DIM, EMBED_TABLE_SIZE);
        let input = encoder.encode("cat");
        let target = encoder.encode("small animal with fur");
        let mut network = Network::new(42, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM);
        network.layers[1].neurons[0].weight_protection[0] = Protection::Frozen;

        let weight_before = network.layers[1].neurons[0].weights[0];

        for _ in 0..100 {
            network.backprop(&input, &target, 1.0);
        }

        assert_eq!(weight_before, network.layers[1].neurons[0].weights[0]);
    }

    #[test]
    fn guarded_updates_are_clamped() {
        let mut rng = SplitMix64::new(42);
        let mut neuron = Neuron::random(&mut rng, EMBED_DIM, 1.0);
        neuron.guard_all();
        let weights_before = neuron.weights.clone();
        let bias_before = neuron.bias;
        let input = vec![10.0; EMBED_DIM];

        apply_weight_updates(&mut neuron, &input, 10.0, 100.0, &[]);

        for (before, after) in weights_before.iter().zip(neuron.weights.iter()) {
            assert!(
                (after - before).abs() <= GUARD_DELTA + 1.0e-6,
                "guarded weight moved by more than GUARD_DELTA"
            );
        }
        assert!((neuron.bias - bias_before).abs() <= GUARD_DELTA + 1.0e-6);
    }

    #[test]
    fn anti_forgetting_proof_passes_for_primary_seed() {
        let report = run_anti_forgetting_proof(42);
        assert!(report.passed, "anti-forgetting proof failed for seed 42");
        assert!(report.min_anchor_after > ANCHOR_SURVIVAL_THRESHOLD);
        assert!(report.max_forgetting_delta < MAX_FORGETTING_DELTA);
        assert!(report.min_new_similarity > NEW_FACT_THRESHOLD);
    }

    #[test]
    fn anti_forgetting_proof_passes_for_five_fixed_seeds() {
        for seed in SEEDS {
            let report = run_anti_forgetting_proof(seed);
            assert!(
                report.passed,
                "anti-forgetting proof failed for seed {}",
                seed
            );
        }
    }
}
