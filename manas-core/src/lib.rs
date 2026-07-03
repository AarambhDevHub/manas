//! Core neural network engine for Manas.

pub mod activation;
pub mod error;
pub mod layer;
pub mod network;
pub mod neuron;

pub use activation::Activation;
pub use error::ManasError;
pub use layer::Layer;
pub use network::{
    ConsolidationReport, ForwardCache, GROWTH_THRESHOLD, GUARD_DELTA, HiddenReadout, MAX_LAYERS,
    MAX_NEURONS_PER_LAYER, MAX_UPDATE_ATTEMPTS, Network, NeuronGradients, TrainingExample,
};
pub use neuron::{Neuron, ProtectionLevel, Source};
