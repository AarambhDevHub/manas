use crate::activation::Activation;
use crate::network::SplitMix64;

const KEY_SHARPNESS: f32 = 12.0;

/// Per-neuron update protection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProtectionLevel {
    Open,
    Guarded,
    Frozen,
}

/// Metadata about where a neuron came from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Source {
    RawText,
    LocalFile { path: String },
    Unknown,
}

/// A single neural unit and its protection metadata.
#[derive(Clone, Debug)]
pub struct Neuron {
    pub id: u64,
    pub weights: Vec<f32>,
    pub bias: f32,
    pub activation: Activation,
    pub importance_score: f32,
    pub protection_level: ProtectionLevel,
    pub weight_protection: Vec<ProtectionLevel>,
    pub bias_protection: ProtectionLevel,
    pub born_at: u64,
    pub last_activated: u64,
    pub activation_count: u64,
    pub source: Source,
    pub freshness_category: u8,
}

impl Neuron {
    pub(crate) fn random(
        id: u64,
        rng: &mut SplitMix64,
        input_dim: usize,
        limit: f32,
        activation: Activation,
    ) -> Self {
        let weights = (0..input_dim)
            .map(|_| rng.uniform_range(-limit, limit))
            .collect();

        Self {
            id,
            weights,
            bias: 0.0,
            activation,
            importance_score: 0.0,
            protection_level: ProtectionLevel::Open,
            weight_protection: vec![ProtectionLevel::Open; input_dim],
            bias_protection: ProtectionLevel::Open,
            born_at: 0,
            last_activated: 0,
            activation_count: 0,
            source: Source::Unknown,
            freshness_category: 1,
        }
    }

    /// Activate this neuron for a vector input.
    pub fn activate(&self, input: &[f32]) -> f32 {
        match self.activation {
            Activation::Keyed => keyed_activation(&self.weights, input),
            activation => activation.apply(dot(&self.weights, input) + self.bias),
        }
    }

    pub fn derivative_from_output(&self, output: f32) -> f32 {
        self.activation.derivative_from_output(output)
    }

    pub fn freeze_all(&mut self) {
        self.protection_level = ProtectionLevel::Frozen;
        self.bias_protection = ProtectionLevel::Frozen;
        for protection in &mut self.weight_protection {
            *protection = ProtectionLevel::Frozen;
        }
    }

    pub fn guard_all(&mut self) {
        self.protection_level = ProtectionLevel::Guarded;
        self.bias_protection = ProtectionLevel::Guarded;
        for protection in &mut self.weight_protection {
            *protection = ProtectionLevel::Guarded;
        }
    }
}

pub(crate) fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| left_value * right_value)
        .sum()
}

pub(crate) fn normalize_in_place(vector: &mut [f32]) {
    let norm = dot(vector, vector).sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

fn keyed_activation(key: &[f32], input: &[f32]) -> f32 {
    let input_norm = dot(input, input).sqrt();
    if input_norm == 0.0 {
        return 0.0;
    }

    let distance_sq = key
        .iter()
        .zip(input.iter())
        .map(|(key_value, input_value)| {
            let normalized_input = input_value / input_norm;
            let diff = key_value - normalized_input;
            diff * diff
        })
        .sum::<f32>();

    (-KEY_SHARPNESS * distance_sq).exp()
}
