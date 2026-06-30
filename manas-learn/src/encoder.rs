use crate::embedder::Embedder;
use crate::tokenizer::Tokenizer;

const DEFAULT_TABLE_SIZE: usize = 8192;

/// Deterministic tokenizer-backed encoder for Stage 6.
pub struct Encoder {
    tokenizer: Tokenizer,
    embedder: Embedder,
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
}
