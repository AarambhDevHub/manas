//! Learning utilities for the Manas associative-memory engine.

pub mod backprop;
pub mod decoder;
pub mod embedder;
pub mod encoder;
pub mod fixtures;
pub mod tokenizer;
pub mod trainer;

pub use embedder::Embedder;
pub use encoder::{Encoder, EncoderVocabEntry};
pub use tokenizer::Tokenizer;
pub use trainer::{AnswerSource, EncodedFact, LearnReport, ProtectionReport, QueryResult, Trainer};
