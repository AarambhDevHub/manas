use std::collections::HashMap;

const DEFAULT_SEED: u64 = 42;
const DEFAULT_POSITIONAL_SCALE: f32 = 0.1;

/// Deterministic token embedder with position-sensitive sequence encoding.
#[derive(Clone, Debug)]
pub struct Embedder {
    pub embed_table: HashMap<u32, Vec<f32>>,
    pub embed_dim: usize,
    pub positional_scale: f32,
    seed: u64,
}

impl Embedder {
    pub fn new(embed_dim: usize) -> Self {
        Self::with_seed(embed_dim, DEFAULT_SEED)
    }

    pub fn with_seed(embed_dim: usize, seed: u64) -> Self {
        Self::with_seed_and_capacity(embed_dim, seed, 0)
    }

    pub(crate) fn with_seed_and_capacity(embed_dim: usize, seed: u64, capacity: usize) -> Self {
        Self {
            embed_table: HashMap::with_capacity(capacity),
            embed_dim: embed_dim.max(1),
            positional_scale: DEFAULT_POSITIONAL_SCALE,
            seed,
        }
    }

    pub fn get_or_create(&mut self, token_id: u32) -> &Vec<f32> {
        let seed = self.seed;
        let embed_dim = self.embed_dim;

        self.embed_table
            .entry(token_id)
            .or_insert_with(|| make_embedding(token_seed(seed, token_id), embed_dim))
    }

    pub fn base_embedding(&self, token_id: u32) -> Vec<f32> {
        self.embed_table
            .get(&token_id)
            .cloned()
            .unwrap_or_else(|| make_embedding(token_seed(self.seed, token_id), self.embed_dim))
    }

    pub fn embed_with_position(&self, token_id: u32, position: usize) -> Vec<f32> {
        match self.embed_table.get(&token_id) {
            Some(base) => positioned_embedding(base, position, self.positional_scale),
            None => {
                let base = make_embedding(token_seed(self.seed, token_id), self.embed_dim);
                positioned_embedding(&base, position, self.positional_scale)
            }
        }
    }

    pub fn encode_sequence(&mut self, token_ids: &[u32]) -> Vec<f32> {
        let mut encoded = vec![0.0; self.embed_dim];
        let positional_scale = self.positional_scale;

        for (position, token_id) in token_ids.iter().enumerate() {
            let base = self.get_or_create(*token_id);
            add_positioned_embedding(&mut encoded, base, position, positional_scale);
        }

        encoded
    }

    pub fn encode_existing_sequence(&self, token_ids: &[u32]) -> Vec<f32> {
        let mut encoded = vec![0.0; self.embed_dim];

        for (position, token_id) in token_ids.iter().enumerate() {
            match self.embed_table.get(token_id) {
                Some(base) => {
                    add_positioned_embedding(&mut encoded, base, position, self.positional_scale)
                }
                None => {
                    let base = make_embedding(token_seed(self.seed, *token_id), self.embed_dim);
                    add_positioned_embedding(&mut encoded, &base, position, self.positional_scale);
                }
            }
        }

        encoded
    }
}

fn positioned_embedding(base: &[f32], position: usize, positional_scale: f32) -> Vec<f32> {
    let mut positioned = vec![0.0; base.len()];
    add_positioned_embedding(&mut positioned, base, position, positional_scale);
    positioned
}

fn add_positioned_embedding(
    encoded: &mut [f32],
    base: &[f32],
    position: usize,
    positional_scale: f32,
) {
    for dimension in (0..base.len()).step_by(2) {
        let even = base[dimension];
        let odd = base.get(dimension + 1).copied().unwrap_or(0.0);
        let angle = positional_angle(position, dimension, base.len());
        let (sin, cos) = angle.sin_cos();

        let rotated_even = even * cos - odd * sin;
        encoded[dimension] += rotated_even * (1.0 + positional_scale * sin);

        if let Some(dst) = encoded.get_mut(dimension + 1) {
            let rotated_odd = even * sin + odd * cos;
            *dst += rotated_odd * (1.0 + positional_scale * cos);
        }
    }
}

fn positional_angle(position: usize, dimension: usize, embed_dim: usize) -> f32 {
    let pair_index = (dimension / 2) as f32;
    let exponent = 2.0 * pair_index / embed_dim.max(1) as f32;
    position as f32 / 10_000.0_f32.powf(exponent)
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
    use crate::backprop::cosine;
    use crate::tokenizer::Tokenizer;

    use super::*;

    #[test]
    fn same_sequence_same_vector() {
        let mut embedder = Embedder::new(32);
        let mut tokenizer = Tokenizer::new(4);

        let ids = tokenizer.encode("rust programming language");
        let first = embedder.encode_sequence(&ids);
        let second = embedder.encode_sequence(&ids);

        let similarity = cosine(&first, &second);
        assert!(
            (similarity - 1.0).abs() < 1e-5,
            "same sequence must give same vector, got {similarity:.6}"
        );
    }

    #[test]
    fn order_matters() {
        let mut embedder = Embedder::new(32);
        let mut tokenizer = Tokenizer::new(4);

        let cat_dog = embedder.encode_sequence(&tokenizer.encode("cat dog"));
        let dog_cat = embedder.encode_sequence(&tokenizer.encode("dog cat"));

        let similarity = cosine(&cat_dog, &dog_cat);
        assert!(
            similarity < 0.99,
            "order should matter: cosine was {similarity:.4}"
        );
        assert!(
            similarity > 0.50,
            "same words should remain related: cosine was {similarity:.4}"
        );
    }

    #[test]
    fn similar_words_similar_vectors() {
        let mut embedder = Embedder::new(32);
        let mut tokenizer = Tokenizer::new(4);

        let cat = embedder.encode_sequence(&tokenizer.encode("cat"));
        let cats = embedder.encode_sequence(&tokenizer.encode("cats"));
        let dog = embedder.encode_sequence(&tokenizer.encode("dog"));

        let cat_cats = cosine(&cat, &cats);
        let cat_dog = cosine(&cat, &dog);

        assert!(
            cat_cats > cat_dog,
            "cat-cats ({cat_cats:.4}) should be more similar than cat-dog ({cat_dog:.4})"
        );
    }

    #[test]
    fn deterministic_existing_sequence_does_not_grow_table() {
        let mut embedder = Embedder::new(32);
        let mut tokenizer = Tokenizer::new(4);
        let ids = tokenizer.encode("cat");

        let first = embedder.encode_sequence(&ids);
        let before = embedder.embed_table.len();
        let second = embedder.encode_existing_sequence(&ids);

        assert_eq!(embedder.embed_table.len(), before);
        assert_eq!(first, second);
    }

    #[test]
    fn embed_with_position_is_deterministic_without_inserting() {
        let embedder = Embedder::new(32);

        let first = embedder.embed_with_position(7, 3);
        let second = embedder.embed_with_position(7, 3);

        assert_eq!(first, second);
        assert!(embedder.embed_table.is_empty());
    }
}
