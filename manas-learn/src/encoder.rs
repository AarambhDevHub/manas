use std::collections::{HashMap, HashSet};

use manas_core::ManasError;

use crate::embedder::Embedder;
use crate::tokenizer::Tokenizer;

const DEFAULT_TABLE_SIZE: usize = 8192;
pub(crate) const ANSWER_CODEC_MARKER: f32 = -1.0;
pub(crate) const ANSWER_COUNT_SCALE: f32 = 32.0;
pub(crate) const ANSWER_ID_SCALE: f32 = 4096.0;

/// Deterministic tokenizer-backed encoder for Stage 6.
pub struct Encoder {
    tokenizer: Tokenizer,
    embedder: Embedder,
}

/// Persistable tokenizer/embedding entry used by the Stage 9 CLI brain state.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderVocabEntry {
    pub token: String,
    pub id: u32,
    pub embedding: Vec<f32>,
}

impl Encoder {
    pub fn new(seed: u64, dim: usize, table_size: usize) -> Self {
        Self {
            tokenizer: Tokenizer::default(),
            embedder: Embedder::with_seed_and_capacity(dim, seed, table_size.max(1)),
        }
    }

    pub fn with_dim(dim: usize) -> Self {
        Self::new(42, dim, DEFAULT_TABLE_SIZE)
    }

    pub fn encode(&mut self, text: &str) -> Vec<f32> {
        let token_ids = self.tokenizer.encode(text);
        self.embedder.encode_sequence(&token_ids)
    }

    pub fn encode_answer(&mut self, text: &str) -> Vec<f32> {
        let words = answer_words(text);
        if words.is_empty() {
            return self.encode(text);
        }

        let mut encoded = vec![0.0; self.dim()];
        if encoded.len() < 3 {
            return self.encode(text);
        }

        let max_words = encoded.len().saturating_sub(2);
        let word_ids = words
            .into_iter()
            .take(max_words)
            .filter_map(|word| {
                let _ = self.encode(&word);
                self.tokenizer.vocab.get(&format!("#{word}")).copied()
            })
            .collect::<Vec<_>>();

        if word_ids.is_empty() {
            return encoded;
        }

        encoded[0] = ANSWER_CODEC_MARKER;
        encoded[1] = word_ids.len() as f32 / ANSWER_COUNT_SCALE;
        for (slot, word_id) in word_ids.into_iter().enumerate() {
            encoded[slot + 2] = word_id.saturating_add(1) as f32 / ANSWER_ID_SCALE;
        }

        encoded
    }

    pub fn encode_deterministic(&self, text: &str) -> Vec<f32> {
        let token_ids = self.tokenizer.encode_deterministic(text);
        self.embedder.encode_existing_sequence(&token_ids)
    }

    pub fn dim(&self) -> usize {
        self.embedder.embed_dim
    }

    pub fn vocab_size(&self) -> u32 {
        self.tokenizer.vocab_size()
    }

    pub fn export_vocab(&self) -> Vec<EncoderVocabEntry> {
        let mut entries = self
            .tokenizer
            .id_to_token
            .iter()
            .map(|(id, token)| EncoderVocabEntry {
                token: token.clone(),
                id: *id,
                embedding: self.embedder.base_embedding(*id),
            })
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| entry.id);
        entries
    }

    pub fn import_vocab(&mut self, entries: &[EncoderVocabEntry]) -> Result<(), ManasError> {
        let mut seen_tokens = HashSet::new();
        let mut seen_ids = HashSet::new();
        let mut vocab = HashMap::with_capacity(entries.len());
        let mut id_to_token = HashMap::with_capacity(entries.len());
        let mut embed_table = HashMap::with_capacity(entries.len());
        let mut next_id = 0_u32;

        for entry in entries {
            if entry.token.is_empty() {
                return Err(ManasError::EncodingError(
                    "vocab entry token cannot be empty".to_string(),
                ));
            }
            if entry.embedding.len() != self.dim() {
                return Err(ManasError::EncodingError(format!(
                    "embedding for token '{}' has dimension {}, expected {}",
                    entry.token,
                    entry.embedding.len(),
                    self.dim()
                )));
            }
            if !seen_tokens.insert(entry.token.clone()) {
                return Err(ManasError::EncodingError(format!(
                    "duplicate vocab token '{}'",
                    entry.token
                )));
            }
            if !seen_ids.insert(entry.id) {
                return Err(ManasError::EncodingError(format!(
                    "duplicate vocab id {}",
                    entry.id
                )));
            }

            vocab.insert(entry.token.clone(), entry.id);
            id_to_token.insert(entry.id, entry.token.clone());
            embed_table.insert(entry.id, entry.embedding.clone());
            next_id = next_id.max(entry.id.saturating_add(1));
        }

        self.tokenizer.vocab = vocab;
        self.tokenizer.id_to_token = id_to_token;
        self.tokenizer.next_id = next_id;
        self.embedder.embed_table = embed_table;
        Ok(())
    }

    pub fn known_words(&self) -> Vec<String> {
        let mut words = self
            .tokenizer
            .id_to_token
            .values()
            .filter_map(|token| token.strip_prefix('#'))
            .map(str::to_string)
            .collect::<Vec<_>>();
        words.sort();
        words.dedup();
        words
    }

    pub fn known_word_ids(&self) -> Vec<(u32, String)> {
        let mut words = self
            .tokenizer
            .id_to_token
            .iter()
            .filter_map(|(id, token)| token.strip_prefix('#').map(|word| (*id, word.to_string())))
            .collect::<Vec<_>>();
        words.sort_by_key(|(id, _)| *id);
        words
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn tokenizer_mut(&mut self) -> &mut Tokenizer {
        &mut self.tokenizer
    }

    pub fn embedder(&self) -> &Embedder {
        &self.embedder
    }

    pub fn embedder_mut(&mut self) -> &mut Embedder {
        &mut self.embedder
    }
}

fn answer_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|raw| {
            let cleaned = raw
                .chars()
                .filter(|ch| ch.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>();
            if cleaned.is_empty() || is_answer_stopword(&cleaned) {
                None
            } else {
                Some(cleaned)
            }
        })
        .collect()
}

fn is_answer_stopword(word: &str) -> bool {
    matches!(
        word,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "by"
            | "for"
            | "from"
            | "in"
            | "is"
            | "it"
            | "of"
            | "on"
            | "or"
            | "the"
            | "to"
            | "was"
            | "were"
            | "with"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_are_deterministic_for_same_seed() {
        let mut first = Encoder::new(42, 32, DEFAULT_TABLE_SIZE);
        let mut second = Encoder::new(42, 32, DEFAULT_TABLE_SIZE);

        assert_eq!(first.encode("cat"), second.encode("cat"));
    }

    #[test]
    fn encoder_vocab_grows_through_encode() {
        let mut encoder = Encoder::new(42, 32, DEFAULT_TABLE_SIZE);
        let before = encoder.vocab_size();

        encoder.encode("cat");

        assert!(encoder.vocab_size() > before);
    }

    #[test]
    fn deterministic_encode_does_not_grow_vocab() {
        let mut encoder = Encoder::new(42, 32, DEFAULT_TABLE_SIZE);
        encoder.encode("cat");
        let before = encoder.vocab_size();
        let before_embeddings = encoder.embedder().embed_table.len();

        let known = encoder.encode_deterministic("cat");
        let unknown = encoder.encode_deterministic("dog");

        assert!(known.iter().any(|value| *value != 0.0));
        assert!(unknown.iter().all(|value| *value == 0.0));
        assert_eq!(encoder.vocab_size(), before);
        assert_eq!(encoder.embedder().embed_table.len(), before_embeddings);
    }

    #[test]
    fn encoder_preserves_token_order() {
        let mut encoder = Encoder::new(42, 32, DEFAULT_TABLE_SIZE);

        let cat_dog = encoder.encode("cat dog");
        let dog_cat = encoder.encode("dog cat");

        assert_ne!(cat_dog, dog_cat);
    }

    #[test]
    fn encoder_vocab_export_import_preserves_vectors() {
        let mut original = Encoder::new(42, 32, DEFAULT_TABLE_SIZE);
        let original_cat = original.encode("cat");
        original.encode("small animal with fur");
        let entries = original.export_vocab();

        let mut loaded = Encoder::new(42, 32, DEFAULT_TABLE_SIZE);
        loaded.import_vocab(&entries).unwrap();

        assert_eq!(loaded.vocab_size(), original.vocab_size());
        assert_eq!(loaded.encode_deterministic("cat"), original_cat);
        assert_eq!(
            loaded.encode_deterministic("small animal"),
            original.encode_deterministic("small animal")
        );
    }
}
