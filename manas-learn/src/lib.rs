//! Learning utilities for the Stage 3 Manas engine.

pub mod backprop;
pub mod encoder;
pub mod fixtures;
pub mod trainer;

pub use encoder::Encoder;
pub use trainer::{EncodedFact, Trainer};
