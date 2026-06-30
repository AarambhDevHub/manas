use crate::tokenizer::Tokenizer;

const DEFAULT_TABLE_SIZE: usize = 8192;

/// Deterministic tokenizer-backed encoder for Stage 5.
pub struct Encoder {
    embeddings: Vec<Option<Vec<f32>>>,
    tokenizer: Tokenizer,
    seed: u64,
    dim: usize,
}

impl Encoder {
    pub fn new(seed: u64, dim: usize, table_size: usize) -> Self {
        Self {
            embeddings: vec![None; table_size.max(1)],
            tokenizer: Tokenizer::default(),
            seed,
            dim,
        }
    }

    pub fn with_dim(dim: usize) -> Self {
        Self::new(42, dim, DEFAULT_TABLE_SIZE)
    }

    pub fn encode(&mut self, text: &str) -> Vec<f32> {
        let token_ids = self.tokenizer.encode(text);
        self.encode_token_ids(&token_ids)
    }

    pub fn encode_deterministic(&self, text: &str) -> Vec<f32> {
        let token_ids = self.tokenizer.encode_deterministic(text);
        self.encode_token_ids_deterministic(&token_ids)
    }

    pub fn dim(&self) -> usize {
        self.dim
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

    fn encode_token_ids(&mut self, token_ids: &[u32]) -> Vec<f32> {
        let mut encoded = vec![0.0; self.dim];

        for token_id in token_ids {
            let token_vec = self.token_vector(*token_id);
            for (dst, src) in encoded.iter_mut().zip(token_vec.iter()) {
                *dst += src;
            }
        }

        encoded
    }

    fn encode_token_ids_deterministic(&self, token_ids: &[u32]) -> Vec<f32> {
        let mut encoded = vec![0.0; self.dim];

        for token_id in token_ids {
            let token_vec = self.existing_or_generated_token_vector(*token_id);
            for (dst, src) in encoded.iter_mut().zip(token_vec.iter()) {
                *dst += src;
            }
        }

        encoded
    }

    fn token_vector(&mut self, token_id: u32) -> Vec<f32> {
        let index = token_id as usize;
        if index >= self.embeddings.len() {
            self.embeddings.resize(index + 1, None);
        }

        if let Some(vector) = &self.embeddings[index] {
            return vector.clone();
        }

        let vector = make_embedding(token_seed(self.seed, token_id), self.dim);
        self.embeddings[index] = Some(vector.clone());
        vector
    }

    fn existing_or_generated_token_vector(&self, token_id: u32) -> Vec<f32> {
        self.embeddings
            .get(token_id as usize)
            .and_then(Option::as_ref)
            .cloned()
            .unwrap_or_else(|| make_embedding(token_seed(self.seed, token_id), self.dim))
    }
}

fn token_seed(seed: u64, token_id: u32) -> u64 {
    seed ^ u64::from(token_id)
}

fn make_embedding(seed: u64, dim: usize) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut vector = (0..dim)
        .map(|_| rng.uniform_range(-1.0, 1.0))
        .collect::<Vec<_>>();

    normalize_in_place(&mut vector);
    vector
}

fn normalize_in_place(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        splitmix64(self.state)
    }

    fn next_f32(&mut self) -> f32 {
        let bits = self.next_u64() >> 40;
        bits as f32 / (1_u32 << 24) as f32
    }

    fn uniform_range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
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

        let known = encoder.encode_deterministic("cat");
        let unknown = encoder.encode_deterministic("dog");

        assert!(known.iter().any(|value| *value != 0.0));
        assert!(unknown.iter().all(|value| *value == 0.0));
        assert_eq!(encoder.vocab_size(), before);
    }
}
