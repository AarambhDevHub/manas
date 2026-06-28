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
    ConsolidationReport, ForwardCache, GUARD_DELTA, Network, NeuronGradients, TrainingExample,
};
pub use neuron::{Neuron, ProtectionLevel, Source};
