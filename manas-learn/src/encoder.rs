const DEFAULT_TABLE_SIZE: usize = 8192;

#[derive(Clone)]
enum EmbeddingSlot {
    Empty,
    Occupied { hash: u64, vector: Vec<f32> },
}

/// Deterministic hash-based encoder from the Stage 1-2 experiment.
pub struct Encoder {
    slots: Vec<EmbeddingSlot>,
    seed: u64,
    dim: usize,
}

impl Encoder {
    pub fn new(seed: u64, dim: usize, table_size: usize) -> Self {
        Self {
            slots: vec![EmbeddingSlot::Empty; table_size],
            seed,
            dim,
        }
    }

    pub fn with_dim(dim: usize) -> Self {
        Self::new(42, dim, DEFAULT_TABLE_SIZE)
    }

    pub fn encode(&mut self, text: &str) -> Vec<f32> {
        let mut encoded = vec![0.0; self.dim];

        for word in normalized_words(text) {
            let word_vec = self.word_vector(&word);
            for (dst, src) in encoded.iter_mut().zip(word_vec.iter()) {
                *dst += src;
            }
        }

        encoded
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    fn word_vector(&mut self, word: &str) -> Vec<f32> {
        let hash = stable_hash(word);
        let mut index = hash as usize % self.slots.len();

        loop {
            match &self.slots[index] {
                EmbeddingSlot::Occupied {
                    hash: existing_hash,
                    vector,
                } if *existing_hash == hash => return vector.clone(),
                EmbeddingSlot::Occupied { .. } => {
                    index = (index + 1) % self.slots.len();
                }
                EmbeddingSlot::Empty => {
                    let vector = make_embedding(hash ^ self.seed, self.dim);
                    self.slots[index] = EmbeddingSlot::Occupied {
                        hash,
                        vector: vector.clone(),
                    };
                    return vector;
                }
            }
        }
    }
}

fn normalized_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|raw| {
            let cleaned = raw
                .chars()
                .filter(|ch| ch.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>();

            if cleaned.is_empty() {
                None
            } else {
                Some(cleaned)
            }
        })
        .collect()
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

fn stable_hash(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;

    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }

    splitmix64(hash)
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
}
