//! Learning utilities for the Manas associative-memory engine.

pub mod backprop;
pub mod compression;
pub mod decoder;
pub mod diagnostics;
pub mod embedder;
pub mod encoder;
pub mod fixtures;
pub mod freshness;
pub mod importance;
pub mod tokenizer;
pub mod trainer;

pub use compression::{
    CompressionCandidate, CompressionConfig, CompressionPlan, CompressionReport,
    CompressionSkipCounts, DEFAULT_COMPRESSION_THRESHOLD, DEFAULT_MIN_IDLE_DAYS,
    DEFAULT_MIN_MERGE_SIMILARITY, compress, plan_compression,
};
pub use diagnostics::{
    BrainDiagnostics, FreshnessDiagnostics, LayerDiagnostics, LearningDiagnostics,
    NetworkDiagnostics, NeuronActivationDiagnostic, NeuronDiagnostics, NeuronFilter,
    OutputValueDiagnostic, QueryTrace, SourceDiagnostics, TraceVariant,
    filtered_neuron_diagnostics, neuron_diagnostics, trace_query,
};
pub use embedder::Embedder;
pub use encoder::{Encoder, EncoderVocabEntry};
pub use freshness::{
    FreshnessCategory, FreshnessWarning, detect_freshness, freshness_age_days, is_stale,
    staleness_warning,
};
pub use importance::{GUARDED_TO_FROZEN_IMPORTANCE, OPEN_TO_GUARDED_IMPORTANCE};
pub use tokenizer::Tokenizer;
pub use trainer::{
    AnswerSource, EncodedFact, LearnReport, ProtectionReport, QueryResult, QueryStyle, Trainer,
};
