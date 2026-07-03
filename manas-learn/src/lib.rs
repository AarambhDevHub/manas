//! Learning utilities for the Manas associative-memory engine.

pub mod backprop;
pub mod decoder;
pub mod embedder;
pub mod encoder;
pub mod fixtures;
pub mod freshness;
pub mod importance;
pub mod tokenizer;
pub mod trainer;

pub use embedder::Embedder;
pub use encoder::{Encoder, EncoderVocabEntry};
pub use freshness::{
    FreshnessCategory, FreshnessWarning, detect_freshness, freshness_age_days, is_stale,
    staleness_warning,
};
pub use importance::{GUARDED_TO_FROZEN_IMPORTANCE, OPEN_TO_GUARDED_IMPORTANCE};
pub use tokenizer::Tokenizer;
pub use trainer::{AnswerSource, EncodedFact, LearnReport, ProtectionReport, QueryResult, Trainer};
