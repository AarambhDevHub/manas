//! Learning utilities for the Manas associative-memory engine.

pub mod backprop;
pub mod embedder;
pub mod encoder;
pub mod fixtures;
pub mod tokenizer;
pub mod trainer;

pub use embedder::Embedder;
pub use encoder::Encoder;
pub use tokenizer::Tokenizer;
pub use trainer::{EncodedFact, LearnReport, ProtectionReport, Trainer};
