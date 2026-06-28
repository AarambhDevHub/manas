use crate::activation::Activation;
use crate::error::ManasError;
use crate::layer::Layer;
use crate::neuron::{Neuron, ProtectionLevel, dot, normalize_in_place};

pub const GUARD_DELTA: f32 = 0.001;
const GRAD_CLIP: f32 = 1.0;
const DEFAULT_SEED: u64 = 42;
const RIDGE: f32 = 1.0e-4;
const ANCHOR_CONSTRAINT_WEIGHT: f32 = 50.0;

/// Borrowed vector pair used by consolidation and readout fitting.
#[derive(Clone, Copy)]
pub struct TrainingExample<'a> {
    pub input: &'a [f32],
    pub target: &'a [f32],
}

/// Gradients for one neuron.
#[derive(Clone, Debug)]
pub struct NeuronGradients {
    pub weight_gradients: Vec<f32>,
    pub bias_gradient: f32,
}

/// Cached forward pass values needed by backpropagation.
#[derive(Clone, Debug)]
pub struct ForwardCache {
    pub input: Vec<f32>,
    pub hidden: Vec<f32>,
    pub output: Vec<f32>,
}

/// Summary of anchor consolidation.
#[derive(Clone, Debug)]
pub struct ConsolidationReport {
    pub frozen_hidden_neurons: usize,
    pub frozen_output_edges: usize,
}

/// The minimal associative network proven in Stage 1 and Stage 2.
#[derive(Clone, Debug)]
pub struct Network {
    pub layers: Vec<Layer>,
    pub total_neurons: u64,
    pub created_at: u64,
    pub version: u8,
    pub next_id: u64,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    protected_inputs: Vec<Vec<f32>>,
}

impl Network {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self::with_seed(DEFAULT_SEED, input_dim, hidden_dim, output_dim)
    }

    pub fn with_seed(seed: u64, input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = SplitMix64::new(seed);
        let hidden_limit = xavier_limit(input_dim, hidden_dim);
        let output_limit = xavier_limit(hidden_dim, output_dim);
        let mut next_id = 0;

        let hidden = Layer {
            id: 0,
            neurons: (0..hidden_dim)
                .map(|_| {
                    let id = next_id;
                    next_id += 1;
                    Neuron::random(id, &mut rng, input_dim, hidden_limit, Activation::Tanh)
                })
                .collect(),
            activation: Activation::Tanh,
        };

        let output = Layer {
            id: 1,
            neurons: (0..output_dim)
                .map(|_| {
                    let id = next_id;
                    next_id += 1;
                    Neuron::random(id, &mut rng, hidden_dim, output_limit, Activation::Linear)
                })
                .collect(),
            activation: Activation::Linear,
        };

        Self {
            layers: vec![hidden, output],
            total_neurons: next_id,
            created_at: 0,
            version: 2,
            next_id,
            input_dim,
            hidden_dim,
            output_dim,
            protected_inputs: Vec::new(),
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.forward_with_cache(input).output
    }

    pub fn forward_with_cache(&self, input: &[f32]) -> ForwardCache {
        let hidden = self.layers[0].forward(input);
        let output = self.layers[1].forward(&hidden);

        ForwardCache {
            input: input.to_vec(),
            hidden,
            output,
        }
    }

    pub fn apply_gradients(
        &mut self,
        gradients: &[(u64, NeuronGradients)],
        lr: f32,
    ) -> Result<(), ManasError> {
        let protected_inputs = self.protected_inputs.clone();

        for (neuron_id, gradient) in gradients {
            let neuron = self.find_neuron_mut(*neuron_id)?;
            if neuron.weights.len() != gradient.weight_gradients.len() {
                return Err(ManasError::GradientShapeMismatch {
                    neuron_id: *neuron_id,
                    expected: neuron.weights.len(),
                    found: gradient.weight_gradients.len(),
                });
            }

            apply_weight_updates(neuron, gradient, lr, &protected_inputs);
        }

        Ok(())
    }

    pub fn consolidate_anchor_facts(
        &mut self,
        anchors: &[TrainingExample<'_>],
        neurons_per_fact: usize,
    ) -> Result<ConsolidationReport, ManasError> {
        if anchors.is_empty() {
            return Err(ManasError::EmptyInput);
        }
        self.validate_examples(anchors)?;
        self.protected_inputs.clear();

        for anchor in anchors {
            let mut protected_input = anchor.input.to_vec();
            normalize_in_place(&mut protected_input);
            self.protected_inputs.push(protected_input);
        }

        let mut frozen_indices = Vec::new();
        for anchor in anchors {
            let selected = self.select_open_hidden_neurons(anchor.input, neurons_per_fact);
            self.key_hidden_neurons_to_input(anchor.input, &selected, true);
            frozen_indices.extend(selected);
        }

        self.fit_frozen_output_weights(anchors, &frozen_indices)?;
        self.freeze_output_biases();
        self.orthogonalize_open_hidden_to_protected_inputs();

        Ok(ConsolidationReport {
            frozen_hidden_neurons: self.frozen_hidden_neuron_count(),
            frozen_output_edges: self.frozen_output_edge_count(),
        })
    }

    pub fn key_open_hidden_neurons_to_facts(
        &mut self,
        facts: &[TrainingExample<'_>],
    ) -> Result<(), ManasError> {
        if facts.is_empty() {
            return Ok(());
        }
        self.validate_examples(facts)?;
        let open_indices = self.open_hidden_indices();

        for (slot, hidden_index) in open_indices.iter().enumerate() {
            let fact = facts[slot % facts.len()];
            self.key_hidden_neurons_to_input(fact.input, &[*hidden_index], false);
        }

        Ok(())
    }

    pub fn fit_open_output_weights_to_facts(
        &mut self,
        facts: &[TrainingExample<'_>],
        anchors: &[TrainingExample<'_>],
    ) -> Result<(), ManasError> {
        if facts.is_empty() {
            return Ok(());
        }
        self.validate_examples(facts)?;
        self.validate_examples(anchors)?;
        let open_indices = self.open_hidden_indices();
        if open_indices.is_empty() {
            return Ok(());
        }

        let mut rows = Vec::with_capacity(facts.len() + anchors.len());
        for anchor in anchors {
            rows.push((*anchor, ANCHOR_CONSTRAINT_WEIGHT));
        }
        for fact in facts {
            rows.push((*fact, 1.0));
        }

        let hidden_matrix = rows
            .iter()
            .map(|(fact, _)| {
                let cache = self.forward_with_cache(fact.input);
                open_indices
                    .iter()
                    .map(|index| cache.hidden[*index])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let residuals = rows
            .iter()
            .map(|(fact, _)| {
                let cache = self.forward_with_cache(fact.input);
                self.residual_after_frozen_output(&cache.hidden, fact.target)
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

        for output_dim in 0..self.output_dim {
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
            let weights = solve_linear_system(gram.clone(), rhs)?;

            for (feature_index, hidden_index) in open_indices.iter().enumerate() {
                let output_neuron = &mut self.layers[1].neurons[output_dim];
                if matches!(
                    output_neuron.weight_protection[*hidden_index],
                    ProtectionLevel::Open
                ) {
                    output_neuron.weights[*hidden_index] = weights[feature_index];
                }
            }
        }

        Ok(())
    }

    pub fn frozen_hidden_neuron_count(&self) -> usize {
        self.layers[0]
            .neurons
            .iter()
            .filter(|neuron| matches!(neuron.protection_level, ProtectionLevel::Frozen))
            .count()
    }

    pub fn frozen_output_edge_count(&self) -> usize {
        self.layers[1]
            .neurons
            .iter()
            .map(|neuron| {
                neuron
                    .weight_protection
                    .iter()
                    .filter(|protection| matches!(protection, ProtectionLevel::Frozen))
                    .count()
            })
            .sum()
    }

    pub fn open_hidden_indices(&self) -> Vec<usize> {
        self.layers[0]
            .neurons
            .iter()
            .enumerate()
            .filter(|(_, neuron)| !matches!(neuron.protection_level, ProtectionLevel::Frozen))
            .map(|(index, _)| index)
            .collect()
    }

    fn validate_examples(&self, examples: &[TrainingExample<'_>]) -> Result<(), ManasError> {
        for example in examples {
            if example.input.len() != self.input_dim {
                return Err(ManasError::InvalidNetwork(format!(
                    "input dimension mismatch: expected {}, found {}",
                    self.input_dim,
                    example.input.len()
                )));
            }
            if example.target.len() != self.output_dim {
                return Err(ManasError::InvalidNetwork(format!(
                    "target dimension mismatch: expected {}, found {}",
                    self.output_dim,
                    example.target.len()
                )));
            }
        }
        Ok(())
    }

    fn find_neuron_mut(&mut self, neuron_id: u64) -> Result<&mut Neuron, ManasError> {
        self.layers
            .iter_mut()
            .flat_map(|layer| layer.neurons.iter_mut())
            .find(|neuron| neuron.id == neuron_id)
            .ok_or(ManasError::NeuronNotFound(neuron_id))
    }

    fn select_open_hidden_neurons(&self, input: &[f32], count: usize) -> Vec<usize> {
        let cache = self.forward_with_cache(input);
        let mut candidates = cache
            .hidden
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                !matches!(
                    self.layers[0].neurons[*index].protection_level,
                    ProtectionLevel::Frozen
                )
            })
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

    fn key_hidden_neurons_to_input(&mut self, input: &[f32], selected: &[usize], freeze: bool) {
        let mut key = input.to_vec();
        normalize_in_place(&mut key);

        for index in selected {
            let neuron = &mut self.layers[0].neurons[*index];
            for (weight, key_value) in neuron.weights.iter_mut().zip(key.iter()) {
                *weight = *key_value;
            }
            neuron.bias = 0.0;
            neuron.activation = Activation::Keyed;
            if freeze {
                neuron.freeze_all();
            }
        }
    }

    fn fit_frozen_output_weights(
        &mut self,
        anchors: &[TrainingExample<'_>],
        frozen_indices: &[usize],
    ) -> Result<(), ManasError> {
        let hidden_matrix = anchors
            .iter()
            .map(|anchor| {
                let cache = self.forward_with_cache(anchor.input);
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

        for output_dim in 0..self.output_dim {
            let rhs = anchors
                .iter()
                .map(|anchor| anchor.target[output_dim])
                .collect::<Vec<_>>();
            let alpha = solve_linear_system(gram.clone(), rhs)?;

            for (feature_index, hidden_index) in frozen_indices.iter().enumerate() {
                let weight = hidden_matrix
                    .iter()
                    .zip(alpha.iter())
                    .map(|(row, alpha_value)| row[feature_index] * alpha_value)
                    .sum::<f32>();
                let output_neuron = &mut self.layers[1].neurons[output_dim];
                output_neuron.weights[*hidden_index] = weight;
                output_neuron.weight_protection[*hidden_index] = ProtectionLevel::Frozen;
            }
        }

        Ok(())
    }

    fn freeze_output_biases(&mut self) {
        for output_neuron in &mut self.layers[1].neurons {
            output_neuron.bias = 0.0;
            output_neuron.bias_protection = ProtectionLevel::Frozen;
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
                    .filter(|((_, protection), _)| matches!(protection, ProtectionLevel::Frozen))
                    .map(|((weight, _), hidden_value)| weight * hidden_value)
                    .sum::<f32>();
                target[output_dim] - frozen_contribution
            })
            .collect()
    }

    fn orthogonalize_open_hidden_to_protected_inputs(&mut self) {
        let protected_inputs = self.protected_inputs.clone();
        if protected_inputs.is_empty() {
            return;
        }

        for hidden_neuron in &mut self.layers[0].neurons {
            if matches!(hidden_neuron.protection_level, ProtectionLevel::Frozen) {
                continue;
            }
            remove_protected_components(&mut hidden_neuron.weights, &protected_inputs);
            hidden_neuron.bias = 0.0;
            hidden_neuron.bias_protection = ProtectionLevel::Frozen;
        }
    }
}

fn apply_weight_updates(
    neuron: &mut Neuron,
    gradient: &NeuronGradients,
    lr: f32,
    protected_inputs: &[Vec<f32>],
) {
    let mut gradients = gradient
        .weight_gradients
        .iter()
        .map(|value| value.clamp(-GRAD_CLIP, GRAD_CLIP))
        .collect::<Vec<_>>();

    if !protected_inputs.is_empty()
        && !matches!(neuron.protection_level, ProtectionLevel::Frozen)
        && !matches!(neuron.activation, Activation::Keyed)
    {
        remove_protected_components(&mut gradients, protected_inputs);
    }

    for ((weight, weight_protection), gradient) in neuron
        .weights
        .iter_mut()
        .zip(neuron.weight_protection.iter())
        .zip(gradients.iter())
    {
        let protection = strongest_protection(neuron.protection_level, *weight_protection);
        apply_protected_update(weight, -lr * gradient, protection);
    }

    apply_protected_update(
        &mut neuron.bias,
        -lr * gradient.bias_gradient.clamp(-GRAD_CLIP, GRAD_CLIP),
        strongest_protection(neuron.protection_level, neuron.bias_protection),
    );
    neuron.activation_count += 1;
}

fn apply_protected_update(value: &mut f32, raw_update: f32, protection: ProtectionLevel) {
    match protection {
        ProtectionLevel::Open => *value += raw_update,
        ProtectionLevel::Guarded => *value += raw_update.clamp(-GUARD_DELTA, GUARD_DELTA),
        ProtectionLevel::Frozen => {}
    }
}

fn strongest_protection(left: ProtectionLevel, right: ProtectionLevel) -> ProtectionLevel {
    match (left, right) {
        (ProtectionLevel::Frozen, _) | (_, ProtectionLevel::Frozen) => ProtectionLevel::Frozen,
        (ProtectionLevel::Guarded, _) | (_, ProtectionLevel::Guarded) => ProtectionLevel::Guarded,
        (ProtectionLevel::Open, ProtectionLevel::Open) => ProtectionLevel::Open,
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

fn solve_linear_system(
    mut matrix: Vec<Vec<f32>>,
    mut rhs: Vec<f32>,
) -> Result<Vec<f32>, ManasError> {
    let size = rhs.len();

    for pivot in 0..size {
        let mut pivot_row = pivot;
        for row in (pivot + 1)..size {
            if matrix[row][pivot].abs() > matrix[pivot_row][pivot].abs() {
                pivot_row = row;
            }
        }

        if matrix[pivot_row][pivot].abs() < 1.0e-8 {
            return Err(ManasError::InvalidNetwork(
                "linear solve failed: singular matrix".to_string(),
            ));
        }

        matrix.swap(pivot, pivot_row);
        rhs.swap(pivot, pivot_row);

        let pivot_value = matrix[pivot][pivot];
        for value in matrix[pivot].iter_mut().skip(pivot) {
            *value /= pivot_value;
        }
        rhs[pivot] /= pivot_value;

        let pivot_tail = matrix[pivot][pivot..].to_vec();
        for row in 0..size {
            if row == pivot {
                continue;
            }

            let factor = matrix[row][pivot];
            for (value, pivot_value) in matrix[row].iter_mut().skip(pivot).zip(pivot_tail.iter()) {
                *value -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }

    Ok(rhs)
}

fn xavier_limit(fan_in: usize, fan_out: usize) -> f32 {
    (6.0 / (fan_in + fan_out) as f32).sqrt()
}

pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
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

    pub(crate) fn uniform_range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_output_dimension_matches_network_output() {
        let network = Network::new(32, 64, 32);
        let output = network.forward(&[0.1; 32]);
        assert_eq!(output.len(), 32);
    }

    #[test]
    fn frozen_neuron_weights_never_change() {
        let mut network = Network::new(32, 64, 32);
        network.layers[0].neurons[0].freeze_all();
        let weights_before = network.layers[0].neurons[0].weights.clone();
        let bias_before = network.layers[0].neurons[0].bias;
        let gradients = vec![(
            network.layers[0].neurons[0].id,
            NeuronGradients {
                weight_gradients: vec![10.0; 32],
                bias_gradient: 10.0,
            },
        )];

        network.apply_gradients(&gradients, 1.0).unwrap();

        assert_eq!(network.layers[0].neurons[0].weights, weights_before);
        assert_eq!(network.layers[0].neurons[0].bias, bias_before);
    }

    #[test]
    fn frozen_output_edge_never_changes() {
        let mut network = Network::new(32, 64, 32);
        network.layers[1].neurons[0].weight_protection[0] = ProtectionLevel::Frozen;
        let edge_before = network.layers[1].neurons[0].weights[0];
        let gradients = vec![(
            network.layers[1].neurons[0].id,
            NeuronGradients {
                weight_gradients: vec![10.0; 64],
                bias_gradient: 0.0,
            },
        )];

        network.apply_gradients(&gradients, 1.0).unwrap();

        assert_eq!(network.layers[1].neurons[0].weights[0], edge_before);
    }

    #[test]
    fn guarded_updates_are_clamped() {
        let mut network = Network::new(32, 64, 32);
        network.layers[0].neurons[0].guard_all();
        let weights_before = network.layers[0].neurons[0].weights.clone();
        let bias_before = network.layers[0].neurons[0].bias;
        let gradients = vec![(
            network.layers[0].neurons[0].id,
            NeuronGradients {
                weight_gradients: vec![10.0; 32],
                bias_gradient: 10.0,
            },
        )];

        network.apply_gradients(&gradients, 1.0).unwrap();

        for (before, after) in weights_before
            .iter()
            .zip(network.layers[0].neurons[0].weights.iter())
        {
            assert!((after - before).abs() <= GUARD_DELTA + 1.0e-6);
        }
        assert!((network.layers[0].neurons[0].bias - bias_before).abs() <= GUARD_DELTA + 1.0e-6);
    }
}
