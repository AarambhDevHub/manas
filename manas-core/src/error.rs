use std::path::PathBuf;

/// Error type shared by Manas core and higher-level crates.
#[derive(Debug)]
pub enum ManasError {
    NeuronNotFound(u64),
    LayerNotFound(u32),
    GrowthFailed(String),
    FileReadError {
        path: PathBuf,
        source: std::io::Error,
    },
    FileWriteError {
        path: PathBuf,
        source: std::io::Error,
    },
    CorruptBrain {
        reason: String,
    },
    ChecksumMismatch {
        expected: u32,
        found: u32,
    },
    EncodingError(String),
    EmptyInput,
    InvalidNetwork(String),
    GradientShapeMismatch {
        neuron_id: u64,
        expected: usize,
        found: usize,
    },
}

impl std::fmt::Display for ManasError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NeuronNotFound(id) => write!(formatter, "neuron not found: {id}"),
            Self::LayerNotFound(id) => write!(formatter, "layer not found: {id}"),
            Self::GrowthFailed(reason) => write!(formatter, "growth failed: {reason}"),
            Self::FileReadError { path, source } => {
                write!(formatter, "failed to read {}: {source}", path.display())
            }
            Self::FileWriteError { path, source } => {
                write!(formatter, "failed to write {}: {source}", path.display())
            }
            Self::CorruptBrain { reason } => write!(formatter, "corrupt brain: {reason}"),
            Self::ChecksumMismatch { expected, found } => {
                write!(
                    formatter,
                    "checksum mismatch: expected {expected}, found {found}"
                )
            }
            Self::EncodingError(reason) => write!(formatter, "encoding error: {reason}"),
            Self::EmptyInput => write!(formatter, "input is empty"),
            Self::InvalidNetwork(reason) => write!(formatter, "invalid network: {reason}"),
            Self::GradientShapeMismatch {
                neuron_id,
                expected,
                found,
            } => write!(
                formatter,
                "gradient shape mismatch for neuron {neuron_id}: expected {expected}, found {found}"
            ),
        }
    }
}

impl std::error::Error for ManasError {}
