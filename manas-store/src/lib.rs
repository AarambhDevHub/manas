use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use manas_core::{Activation, Layer, ManasError, Network, Neuron, ProtectionLevel, Source};

const MAGIC: &[u8; 4] = b"MANS";
const FORMAT_VERSION: u8 = 2;
const CRC_SIZE: usize = 4;
const CRC32_POLYNOMIAL: u32 = 0xEDB8_8320;

/// A single-file persisted Manas brain.
#[derive(Clone, Debug)]
pub struct ManasBrain {
    pub path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct BrainState {
    pub network: Network,
    pub vocab_entries: Vec<VocabEntry>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VocabEntry {
    pub token: String,
    pub id: u32,
    pub embedding: Vec<f32>,
}

impl ManasBrain {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn save(&self, network: &Network) -> Result<(), ManasError> {
        let bytes = encode_state(network, &[])?;
        fs::write(&self.path, bytes).map_err(|source| ManasError::FileWriteError {
            path: self.path.clone(),
            source,
        })
    }

    pub fn save_state(&self, state: &BrainState) -> Result<(), ManasError> {
        let bytes = encode_state(&state.network, &state.vocab_entries)?;
        fs::write(&self.path, bytes).map_err(|source| ManasError::FileWriteError {
            path: self.path.clone(),
            source,
        })
    }

    pub fn load(&self) -> Result<Network, ManasError> {
        Ok(self.load_state()?.network)
    }

    pub fn load_state(&self) -> Result<BrainState, ManasError> {
        let bytes = fs::read(&self.path).map_err(|source| ManasError::FileReadError {
            path: self.path.clone(),
            source,
        })?;

        decode_state(&bytes)
    }

    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    pub fn size_bytes(&self) -> u64 {
        fs::metadata(&self.path)
            .map(|metadata| metadata.len())
            .unwrap_or(0)
    }
}

fn encode_state(network: &Network, vocab_entries: &[VocabEntry]) -> Result<Vec<u8>, ManasError> {
    validate_network_for_save(network)?;
    let vocab_size = validate_vocab_entries(vocab_entries, network.input_dim)?;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    write_u8(&mut bytes, FORMAT_VERSION);
    write_u64(&mut bytes, network.created_at);
    write_u64(&mut bytes, unix_now());
    write_u64(&mut bytes, network.total_neurons);
    write_u32(
        &mut bytes,
        usize_to_u32(network.layers.len(), "layer count")?,
    );
    write_u32(
        &mut bytes,
        usize_to_u32(network.input_dim, "input dimension")?,
    );
    write_u32(&mut bytes, vocab_size);

    write_vocab(&mut bytes, vocab_entries)?;
    write_layers(&mut bytes, &network.layers)?;

    let checksum = crc32(&bytes);
    write_u32(&mut bytes, checksum);

    Ok(bytes)
}

fn decode_state(bytes: &[u8]) -> Result<BrainState, ManasError> {
    if bytes.len() < MAGIC.len() + 1 + CRC_SIZE {
        return corrupt("file is too short");
    }

    let checksum_offset = bytes.len() - CRC_SIZE;
    let stored_checksum = read_u32_at(bytes, checksum_offset)?;
    let computed_checksum = crc32(&bytes[..checksum_offset]);
    if stored_checksum != computed_checksum {
        return Err(ManasError::ChecksumMismatch {
            expected: stored_checksum,
            found: computed_checksum,
        });
    }

    let content = &bytes[..checksum_offset];
    let mut reader = Reader::new(content);
    reader.expect_bytes(MAGIC)?;
    let version = reader.read_u8()?;
    if version != FORMAT_VERSION {
        return corrupt(format!("unsupported format version {version}"));
    }

    let created_at = reader.read_u64()?;
    let _modified_at = reader.read_u64()?;
    let expected_total_neurons = reader.read_u64()?;
    let expected_layer_count = reader.read_u32()?;
    let input_dim = u32_to_usize(reader.read_u32()?, "input dimension")?;
    let vocab_size = reader.read_u32()?;
    let vocab_entries = read_vocab(&mut reader, vocab_size, input_dim)?;

    let layers = read_layers(&mut reader, expected_layer_count)?;
    reader.finish()?;

    let network = Network::from_persisted_layers(created_at, version, input_dim, layers)
        .map_err(network_error_to_corrupt)?;
    if network.total_neurons != expected_total_neurons {
        return corrupt(format!(
            "header neuron count {} does not match decoded count {}",
            expected_total_neurons, network.total_neurons
        ));
    }

    Ok(BrainState {
        network,
        vocab_entries,
    })
}

fn write_vocab(bytes: &mut Vec<u8>, entries: &[VocabEntry]) -> Result<(), ManasError> {
    let mut sorted_entries = entries.iter().collect::<Vec<_>>();
    sorted_entries.sort_by_key(|entry| entry.id);

    write_u32(
        bytes,
        usize_to_u32(sorted_entries.len(), "vocab entry count")?,
    );
    for entry in sorted_entries {
        let token_bytes = entry.token.as_bytes();
        write_u16(
            bytes,
            usize_to_u16(token_bytes.len(), "vocab token length")?,
        );
        bytes.extend_from_slice(token_bytes);
        write_u32(bytes, entry.id);
        for value in &entry.embedding {
            write_f32(bytes, *value);
        }
    }

    Ok(())
}

fn read_vocab(
    reader: &mut Reader<'_>,
    vocab_size: u32,
    embed_dim: usize,
) -> Result<Vec<VocabEntry>, ManasError> {
    let entry_count = reader.read_u32()?;
    if vocab_size == 0 && entry_count != 0 {
        return corrupt("vocab entries present with zero vocab size");
    }
    if entry_count > vocab_size {
        return corrupt(format!(
            "vocab entry count {entry_count} exceeds vocab size {vocab_size}"
        ));
    }

    let mut seen_tokens = HashSet::new();
    let mut seen_ids = HashSet::new();
    let mut entries = Vec::with_capacity(u32_to_usize(entry_count, "vocab entry count")?);

    for _ in 0..entry_count {
        let token_len = reader.read_u16()? as usize;
        let token_bytes = reader.read_bytes(token_len)?;
        let token = std::str::from_utf8(token_bytes)
            .map_err(|_| corrupt_error("vocab token is not valid UTF-8"))?
            .to_string();
        let id = reader.read_u32()?;

        if token.is_empty() {
            return corrupt("vocab token cannot be empty");
        }
        if id >= vocab_size {
            return corrupt(format!("vocab id {id} exceeds vocab size {vocab_size}"));
        }
        if !seen_tokens.insert(token.clone()) {
            return corrupt(format!("duplicate vocab token '{token}'"));
        }
        if !seen_ids.insert(id) {
            return corrupt(format!("duplicate vocab id {id}"));
        }

        let mut embedding = Vec::with_capacity(embed_dim);
        for _ in 0..embed_dim {
            embedding.push(reader.read_f32()?);
        }

        entries.push(VocabEntry {
            token,
            id,
            embedding,
        });
    }

    Ok(entries)
}

fn write_layers(bytes: &mut Vec<u8>, layers: &[Layer]) -> Result<(), ManasError> {
    write_u32(bytes, usize_to_u32(layers.len(), "layer section count")?);

    for layer in layers {
        write_u32(bytes, layer.id);
        write_u8(bytes, encode_activation(layer.activation));
        write_u32(bytes, usize_to_u32(layer.neurons.len(), "neuron count")?);

        for neuron in &layer.neurons {
            write_neuron(bytes, neuron)?;
        }
    }

    Ok(())
}

fn write_neuron(bytes: &mut Vec<u8>, neuron: &Neuron) -> Result<(), ManasError> {
    write_u64(bytes, neuron.id);
    write_u32(bytes, usize_to_u32(neuron.weights.len(), "weight count")?);
    for weight in &neuron.weights {
        write_f32(bytes, *weight);
    }
    write_f32(bytes, neuron.bias);
    write_u8(bytes, encode_activation(neuron.activation));
    write_f32(bytes, neuron.importance_score);
    write_u8(bytes, encode_protection(neuron.protection_level));
    write_u32(
        bytes,
        usize_to_u32(neuron.weight_protection.len(), "weight protection count")?,
    );
    for protection in &neuron.weight_protection {
        write_u8(bytes, encode_protection(*protection));
    }
    write_u8(bytes, encode_protection(neuron.bias_protection));
    write_u64(bytes, neuron.born_at);
    write_u64(bytes, neuron.last_activated);
    write_u64(bytes, neuron.activation_count);
    write_source(bytes, &neuron.source)?;
    write_u8(bytes, neuron.freshness_category);
    Ok(())
}

fn write_source(bytes: &mut Vec<u8>, source: &Source) -> Result<(), ManasError> {
    match source {
        Source::RawText => {
            write_u8(bytes, 0);
            write_u16(bytes, 0);
        }
        Source::LocalFile { path } => {
            write_u8(bytes, 1);
            let path_bytes = path.as_bytes();
            write_u16(bytes, usize_to_u16(path_bytes.len(), "source path length")?);
            bytes.extend_from_slice(path_bytes);
        }
        Source::Unknown => {
            write_u8(bytes, 2);
            write_u16(bytes, 0);
        }
    }
    Ok(())
}

fn read_layers(reader: &mut Reader<'_>, expected_count: u32) -> Result<Vec<Layer>, ManasError> {
    let layer_count = reader.read_u32()?;
    if layer_count != expected_count {
        return corrupt(format!(
            "header layer count {expected_count} does not match section count {layer_count}"
        ));
    }

    let mut layers = Vec::with_capacity(u32_to_usize(layer_count, "layer count")?);
    for _ in 0..layer_count {
        let id = reader.read_u32()?;
        let activation = decode_activation(reader.read_u8()?)?;
        let neuron_count = reader.read_u32()?;
        let mut neurons = Vec::with_capacity(u32_to_usize(neuron_count, "neuron count")?);
        for _ in 0..neuron_count {
            neurons.push(read_neuron(reader)?);
        }
        layers.push(Layer {
            id,
            neurons,
            activation,
        });
    }

    Ok(layers)
}

fn read_neuron(reader: &mut Reader<'_>) -> Result<Neuron, ManasError> {
    let id = reader.read_u64()?;
    let weight_count = u32_to_usize(reader.read_u32()?, "weight count")?;
    let mut weights = Vec::with_capacity(weight_count);
    for _ in 0..weight_count {
        weights.push(reader.read_f32()?);
    }

    let bias = reader.read_f32()?;
    let activation = decode_activation(reader.read_u8()?)?;
    let importance_score = reader.read_f32()?;
    let protection_level = decode_protection(reader.read_u8()?)?;
    let weight_protection_count = u32_to_usize(reader.read_u32()?, "weight protection count")?;
    let mut weight_protection = Vec::with_capacity(weight_protection_count);
    for _ in 0..weight_protection_count {
        weight_protection.push(decode_protection(reader.read_u8()?)?);
    }
    let bias_protection = decode_protection(reader.read_u8()?)?;
    let born_at = reader.read_u64()?;
    let last_activated = reader.read_u64()?;
    let activation_count = reader.read_u64()?;
    let source = read_source(reader)?;
    let freshness_category = reader.read_u8()?;

    Ok(Neuron {
        id,
        weights,
        bias,
        activation,
        importance_score,
        protection_level,
        weight_protection,
        bias_protection,
        born_at,
        last_activated,
        activation_count,
        source,
        freshness_category,
    })
}

fn read_source(reader: &mut Reader<'_>) -> Result<Source, ManasError> {
    let tag = reader.read_u8()?;
    let len = reader.read_u16()? as usize;
    let bytes = reader.read_bytes(len)?;
    let text = std::str::from_utf8(bytes)
        .map_err(|_| corrupt_error("source path is not valid UTF-8"))?
        .to_string();

    match tag {
        0 if text.is_empty() => Ok(Source::RawText),
        1 => Ok(Source::LocalFile { path: text }),
        2 if text.is_empty() => Ok(Source::Unknown),
        0 => corrupt("raw text source cannot contain source bytes"),
        2 => corrupt("unknown source cannot contain source bytes"),
        _ => corrupt(format!("invalid source tag {tag}")),
    }
}

fn validate_vocab_entries(entries: &[VocabEntry], embed_dim: usize) -> Result<u32, ManasError> {
    let mut seen_tokens = HashSet::new();
    let mut seen_ids = HashSet::new();
    let mut max_id = None;

    for entry in entries {
        if entry.token.is_empty() {
            return Err(ManasError::InvalidNetwork(
                "vocab token cannot be empty".to_string(),
            ));
        }
        usize_to_u16(entry.token.len(), "vocab token length")?;
        if entry.embedding.len() != embed_dim {
            return Err(ManasError::InvalidNetwork(format!(
                "embedding for token '{}' has dimension {}, expected {}",
                entry.token,
                entry.embedding.len(),
                embed_dim
            )));
        }
        if !seen_tokens.insert(entry.token.clone()) {
            return Err(ManasError::InvalidNetwork(format!(
                "duplicate vocab token '{}'",
                entry.token
            )));
        }
        if !seen_ids.insert(entry.id) {
            return Err(ManasError::InvalidNetwork(format!(
                "duplicate vocab id {}",
                entry.id
            )));
        }
        max_id = Some(max_id.map_or(entry.id, |current: u32| current.max(entry.id)));
    }

    usize_to_u32(entries.len(), "vocab entry count")?;
    Ok(max_id.map_or(0, |id| id.saturating_add(1)))
}

fn validate_network_for_save(network: &Network) -> Result<(), ManasError> {
    if network.layers.len() != 2 {
        return Err(ManasError::InvalidNetwork(format!(
            "Stage 4 persistence expects exactly 2 layers, found {}",
            network.layers.len()
        )));
    }
    if network.layers[0].neurons.len() != network.hidden_dim {
        return Err(ManasError::InvalidNetwork(
            "hidden dimension does not match hidden layer size".to_string(),
        ));
    }
    if network.layers[1].neurons.is_empty() {
        if !network.layers[0].neurons.is_empty() {
            return Err(ManasError::InvalidNetwork(
                "non-empty hidden layer requires output neurons".to_string(),
            ));
        }
        if network.output_dim != network.input_dim {
            return Err(ManasError::InvalidNetwork(
                "empty network output dimension must match input dimension".to_string(),
            ));
        }
    } else if network.layers[1].neurons.len() != network.output_dim {
        return Err(ManasError::InvalidNetwork(
            "output dimension does not match output layer size".to_string(),
        ));
    }
    let rebuilt = Network::from_persisted_layers(
        network.created_at,
        network.version,
        network.input_dim,
        network.layers.clone(),
    )?;
    if rebuilt.total_neurons != network.total_neurons {
        return Err(ManasError::InvalidNetwork(format!(
            "total_neurons is {}, but layers contain {}",
            network.total_neurons, rebuilt.total_neurons
        )));
    }
    Ok(())
}

fn encode_activation(activation: Activation) -> u8 {
    match activation {
        Activation::ReLU => 0,
        Activation::Sigmoid => 1,
        Activation::Tanh => 2,
        Activation::Linear => 3,
        Activation::Keyed => 4,
    }
}

fn decode_activation(value: u8) -> Result<Activation, ManasError> {
    match value {
        0 => Ok(Activation::ReLU),
        1 => Ok(Activation::Sigmoid),
        2 => Ok(Activation::Tanh),
        3 => Ok(Activation::Linear),
        4 => Ok(Activation::Keyed),
        _ => corrupt(format!("invalid activation tag {value}")),
    }
}

fn encode_protection(protection: ProtectionLevel) -> u8 {
    match protection {
        ProtectionLevel::Open => 0,
        ProtectionLevel::Guarded => 1,
        ProtectionLevel::Frozen => 2,
    }
}

fn decode_protection(value: u8) -> Result<ProtectionLevel, ManasError> {
    match value {
        0 => Ok(ProtectionLevel::Open),
        1 => Ok(ProtectionLevel::Guarded),
        2 => Ok(ProtectionLevel::Frozen),
        _ => corrupt(format!("invalid protection tag {value}")),
    }
}

fn write_u8(bytes: &mut Vec<u8>, value: u8) {
    bytes.push(value);
}

fn write_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_f32(bytes: &mut Vec<u8>, value: f32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn read_u32_at(bytes: &[u8], offset: usize) -> Result<u32, ManasError> {
    let chunk = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| corrupt_error("missing checksum"))?;
    Ok(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
}

fn usize_to_u16(value: usize, field: &str) -> Result<u16, ManasError> {
    u16::try_from(value)
        .map_err(|_| ManasError::InvalidNetwork(format!("{field} exceeds u16::MAX")))
}

fn usize_to_u32(value: usize, field: &str) -> Result<u32, ManasError> {
    u32::try_from(value)
        .map_err(|_| ManasError::InvalidNetwork(format!("{field} exceeds u32::MAX")))
}

fn u32_to_usize(value: u32, field: &str) -> Result<usize, ManasError> {
    usize::try_from(value).map_err(|_| corrupt_error(format!("{field} does not fit usize")))
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF;

    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            if crc & 1 == 1 {
                crc = (crc >> 1) ^ CRC32_POLYNOMIAL;
            } else {
                crc >>= 1;
            }
        }
    }

    !crc
}

fn network_error_to_corrupt(error: ManasError) -> ManasError {
    match error {
        ManasError::InvalidNetwork(reason) => ManasError::CorruptBrain { reason },
        other => other,
    }
}

fn corrupt<T>(reason: impl Into<String>) -> Result<T, ManasError> {
    Err(corrupt_error(reason))
}

fn corrupt_error(reason: impl Into<String>) -> ManasError {
    ManasError::CorruptBrain {
        reason: reason.into(),
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn expect_bytes(&mut self, expected: &[u8]) -> Result<(), ManasError> {
        let actual = self.read_bytes(expected.len())?;
        if actual == expected {
            Ok(())
        } else {
            corrupt("invalid magic bytes")
        }
    }

    fn read_u8(&mut self) -> Result<u8, ManasError> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, ManasError> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, ManasError> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, ManasError> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_f32(&mut self) -> Result<f32, ManasError> {
        let bytes = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], ManasError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| corrupt_error("read offset overflow"))?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| corrupt_error("unexpected end of file"))?;
        self.offset = end;
        Ok(bytes)
    }

    fn finish(&self) -> Result<(), ManasError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            corrupt(format!(
                "trailing bytes after network section: {}",
                self.bytes.len() - self.offset
            ))
        }
    }
}
