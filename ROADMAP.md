# ROADMAP.md — Manas v2

> **"A brain that starts empty, learns from experience, and remembers what it learned — forever."**
>
> This roadmap is built around one core principle: **prove the engine works first,
> then add everything else.** Every milestone has a mandatory test that must pass
> before the next milestone begins. No exceptions.

---

## The Single Most Important Test

Before any milestone is considered done, this test must pass:

```bash
# teach 22 facts
./manas teach "A cat is a small domesticated animal with fur and whiskers."
./manas teach "The Eiffel Tower is located in Paris France."
./manas teach "The Amazon River is the largest river by discharge in the world."
./manas teach "Photosynthesis converts sunlight into energy in plants."
./manas teach "Hydrogen is the lightest element in the universe."
./manas teach "The human brain contains approximately 86 billion neurons."
./manas teach "Mount Everest is the highest mountain on Earth at 8849 meters."
./manas teach "Shakespeare wrote 37 plays and 154 sonnets."
./manas teach "The speed of light is approximately 299792458 meters per second."
./manas teach "DNA is a double helix structure that carries genetic information."
./manas teach "The Roman Empire fell in 476 AD."
./manas teach "Water boils at 100 degrees Celsius at standard pressure."
./manas teach "Python was created by Guido van Rossum in 1991."
./manas teach "Jupiter is the largest planet in our solar system."
./manas teach "The Mona Lisa was painted by Leonardo da Vinci."
./manas teach "Rust was first released by Mozilla Research in 2010."
./manas teach "The mitochondria is the powerhouse of the cell."
./manas teach "Einstein developed the theory of relativity."
./manas teach "The Pacific Ocean is the largest ocean on Earth."
./manas teach "Bitcoin was created by Satoshi Nakamoto in 2009."
./manas teach "The nitrogen cycle describes how nitrogen moves through ecosystems."
./manas teach "Gravity pulls objects toward each other with a force proportional to mass."

# delete ALL sidecars — neural weights only
rm -f brain.manas.sources brain.manas.sourceindex brain.manas.seq
rm -f brain.manas.transformer brain.manas.langmeta

# ask — must answer from weights alone
./manas ask "What is a cat?"
./manas ask "Where is the Eiffel Tower?"
./manas ask "What did Einstein develop?"
```

**Expected:**
```
Answer: A cat is a small domesticated animal with fur and whiskers.
Answered from: neural weights

Answer: The Eiffel Tower is located in Paris France.
Answered from: neural weights

Answer: Einstein developed the theory of relativity.
Answered from: neural weights
```

**This test does not pass in v1. It must pass at the end of every milestone
from Stage 2 onward.**

---

## Status Table

| Milestone | Name | Status |
|---|---|---|
| Stage 0 | Workspace and foundation | Planned |
| Stage 1 | The engine — associative memory proof | Planned |
| Stage 2 | Anti-forgetting proof | Planned |
| Stage 3 | Crate structure | Planned |
| Stage 4 | Persistence — `.manas` binary format | Planned |
| Stage 5 | Character n-gram tokenizer | Planned |
| Stage 6 | Positional embeddings | Planned |
| Stage 7 | Growth system | Planned |
| Stage 8 | Protection system hardened | Planned |
| Stage 9 | `manas-cli` v1 — teach and ask | Planned |
| Stage 10 | File and folder ingestion | Planned |
| Stage 11 | Importance scoring and promotion | Planned |
| Stage 12 | Freshness system | Planned |
| Stage 13 | The real demo | Planned |
| Stage 14 | Inspect, neurons, and debug commands | Planned |
| Stage 15 | Compression and forget command | Planned |
| Stage 16 | Benchmarks and test suite | Planned |
| Stage 17 | Layer growth | Planned |
| Stage 18 | Internet agent (future) | Planned |
| Stage 19 | Language generation (future) | Planned |

---

## Stage 0 — Workspace and Foundation

**Goal:** Create the new Manas workspace. Nothing from v1 is carried over.

### What to Build

Create the Rust workspace with the correct crate structure:

```
manas/
├── Cargo.toml          ← workspace
├── ARCHITECTURE.md
├── ROADMAP.md
├── README.md
├── manas-core/
├── manas-store/
├── manas-learn/
├── manas-ingest/
└── manas-cli/
```

`Cargo.toml` (workspace root):
```toml
[workspace]
resolver = "2"
members = [
    "manas-core",
    "manas-store",
    "manas-learn",
    "manas-ingest",
    "manas-cli",
]
```

Create empty `lib.rs` in each crate. Verify `cargo build` succeeds with zero errors.

### What NOT to Do

- Do not copy any code from v1
- Do not create `manas-language`, `manas-agent`, or `manas-memory` crates
- Do not add any dependencies yet except `rand` in `manas-core`

### Test

```bash
cargo build
# must succeed with zero errors and zero warnings
```

### Done When

- [ ] `cargo build` passes clean
- [ ] All 5 crates exist with empty `lib.rs`
- [ ] Workspace `Cargo.toml` compiles correctly

---

## Stage 1 — The Engine: Associative Memory Proof

**Goal:** Prove that a neural network can store and retrieve a fact from its weights.
This is one file, no crates, no CLI — just proof.

This is the most important stage in the entire roadmap.
If this does not work, nothing else matters.

### What to Build

Create a **single standalone file**: `manas-core/src/experiment.rs`

It does not need to be clean. It does not need to be fast.
It needs to prove one thing:

```
teach("cat", "small animal with fur")
query("cat") → output resembles "small animal with fur"
```

### The Minimal Network

Build the simplest possible associative network:

```rust
struct Neuron {
    weights: Vec<f32>,
    bias: f32,
}

struct Layer {
    neurons: Vec<Neuron>,
}

struct Network {
    layers: Vec<Layer>,
}

impl Network {
    fn forward(&self, input: &[f32]) -> Vec<f32>
    fn backprop(&mut self, input: &[f32], target: &[f32], lr: f32) -> f32
    // returns loss
}
```

### The Minimal Encoder

No tokenizer yet. Just convert a word to a fixed-size float vector
using a simple hash → index → lookup table:

```rust
fn word_to_vec(word: &str, dim: usize) -> Vec<f32> {
    // hash the word → index into a fixed embedding table
    // return the embedding at that index
    // embeddings are randomly initialized once, then fixed
}

fn encode(text: &str, dim: usize) -> Vec<f32> {
    // split on whitespace
    // embed each word
    // sum (not average) to preserve some positional bias
}
```

### The Proof

```rust
fn main() {
    let mut network = Network::new(embed_dim: 32, hidden: 64, output: 32);
    let lr = 0.01;

    // teach 3 facts
    for _ in 0..1000 {
        let input  = encode("cat", 32);
        let target = encode("small animal with fur", 32);
        network.backprop(&input, &target, lr);

        let input  = encode("paris", 32);
        let target = encode("city in france", 32);
        network.backprop(&input, &target, lr);

        let input  = encode("rust", 32);
        let target = encode("systems programming language", 32);
        network.backprop(&input, &target, lr);
    }

    // query all 3 — must work
    let cat_out   = network.forward(&encode("cat", 32));
    let paris_out = network.forward(&encode("paris", 32));
    let rust_out  = network.forward(&encode("rust", 32));

    // measure: are output vectors closer to their targets than to each other?
    println!("cat   similarity to target : {:.4}", cosine(cat_out, encode("small animal with fur", 32)));
    println!("paris similarity to target : {:.4}", cosine(paris_out, encode("city in france", 32)));
    println!("rust  similarity to target : {:.4}", cosine(rust_out, encode("systems programming language", 32)));

    // cross-check: cat should NOT match paris target
    println!("cat   similarity to paris  : {:.4}", cosine(cat_out, encode("city in france", 32)));
}
```

### Expected Output

```
cat   similarity to target : 0.8+
paris similarity to target : 0.8+
rust  similarity to target : 0.8+
cat   similarity to paris  : 0.2 or below
```

The exact numbers do not matter. What matters is:
- Each query output is much closer to its own target than to other targets
- The similarity difference is clear and consistent

### Test

```bash
cargo run --example experiment
# or
rustc experiment.rs && ./experiment
```

Check:
- [ ] Each fact's similarity to its own target > 0.70
- [ ] Each fact's similarity to wrong targets < 0.35
- [ ] The difference is consistent across 5 independent runs

### Done When

- [ ] Similarity to correct target consistently > 0.70
- [ ] Similarity to wrong targets consistently < 0.35
- [ ] Works reliably across multiple random seeds
- [ ] You understand exactly why it works — not just that it works

**Do not move to Stage 2 until this is solid.**

---

## Stage 2 — Anti-Forgetting Proof

**Goal:** Prove that fact #1 survives after learning 50 more facts.
This is the hardest problem in the project.

### What to Build

Extend `experiment.rs` to test for catastrophic forgetting:

```rust
fn main() {
    let mut network = Network::new(embed_dim: 32, hidden: 64, output: 32);
    let lr = 0.01;

    // --- FIRST: teach fact #1 ---
    let fact1_input  = encode("cat", 32);
    let fact1_target = encode("small animal with fur", 32);
    for _ in 0..200 {
        network.backprop(&fact1_input, &fact1_target, lr);
    }

    // --- measure fact #1 BEFORE new learning ---
    let before = cosine(
        network.forward(&fact1_input),
        fact1_target.clone()
    );
    println!("fact1 before: {:.4}", before);

    // --- NOW: teach 50 completely different facts ---
    let other_facts = vec![
        ("amazon",      "largest river by discharge"),
        ("everest",     "highest mountain on earth"),
        ("einstein",    "developed theory of relativity"),
        ("photosynthesis", "converts sunlight to energy"),
        // ... 46 more ...
    ];

    for _ in 0..200 {
        for (input, target) in &other_facts {
            network.backprop(
                &encode(input, 32),
                &encode(target, 32),
                lr
            );
        }
    }

    // --- measure fact #1 AFTER new learning ---
    let after = cosine(
        network.forward(&fact1_input),
        fact1_target.clone()
    );
    println!("fact1 after : {:.4}", after);
    println!("forgetting  : {:.4}", before - after);

    // --- also check a few of the new facts ---
    let amazon_sim = cosine(
        network.forward(&encode("amazon", 32)),
        encode("largest river by discharge", 32)
    );
    println!("amazon after : {:.4}", amazon_sim);
}
```

### What You Will Likely See First

```
fact1 before: 0.8432
fact1 after : 0.1204   ← catastrophic forgetting
forgetting  : 0.7228
```

This is expected. This is the problem. Now fix it.

### The Fix — Protection Levels in the Experiment

Add protection to `Neuron`:

```rust
#[derive(Clone)]
enum Protection {
    Open,
    Guarded,
    Frozen,
}

struct Neuron {
    weights: Vec<f32>,
    bias: f32,
    protection: Protection,
    activation_count: u64,
}
```

Add promotion logic to `backprop`:

```rust
fn apply_gradient(&mut self, neuron: &mut Neuron, weight_grad: Vec<f32>, bias_grad: f32, lr: f32) {
    match neuron.protection {
        Protection::Frozen  => return,                          // zero update
        Protection::Guarded => {
            // clamp each weight update to [-0.001, +0.001]
            for (w, g) in neuron.weights.iter_mut().zip(weight_grad) {
                *w += (g * lr).clamp(-0.001, 0.001);
            }
            neuron.bias += (bias_grad * lr).clamp(-0.001, 0.001);
        }
        Protection::Open => {
            for (w, g) in neuron.weights.iter_mut().zip(weight_grad) {
                *w += g * lr;
            }
            neuron.bias += bias_grad * lr;
        }
    }
    neuron.activation_count += 1;
}
```

Add automatic promotion:

```rust
fn promote_neurons(&mut self) {
    for layer in &mut self.layers {
        for neuron in &mut layer.neurons {
            if neuron.activation_count > 500 {
                neuron.protection = Protection::Guarded;
            }
            if neuron.activation_count > 2000 {
                neuron.protection = Protection::Frozen;
            }
        }
    }
}
```

### Required Test Result

```
fact1 before: 0.84+
fact1 after : 0.75+    ← must stay high after 50 new facts
forgetting  : 0.09 or less
amazon after : 0.80+   ← new facts also learned well
```

Both old AND new knowledge must be strong simultaneously.

### The Checklist Test

Run this full sequence and verify every line:

```rust
// teach 5 facts, then check all 5 survive after 50 more facts
let anchors = ["cat", "paris", "rust", "everest", "dna"];
let targets = [
    "small animal with fur",
    "city in france",
    "systems programming language",
    "highest mountain on earth",
    "double helix genetic information",
];

// learn anchors
for (a, t) in anchors.iter().zip(targets.iter()) {
    for _ in 0..300 { network.backprop(&encode(a, 32), &encode(t, 32), lr); }
}

// learn 50 new facts
for _ in 0..200 { /* 50 unrelated facts */ }

// verify all 5 anchors survived
for (a, t) in anchors.iter().zip(targets.iter()) {
    let sim = cosine(network.forward(&encode(a, 32)), encode(t, 32));
    assert!(sim > 0.65, "FORGOT: {} similarity dropped to {:.4}", a, sim);
    println!("{}: {:.4} ✅", a, sim);
}
```

### Done When

- [ ] All 5 anchor facts survive at > 0.65 similarity after 50 new facts
- [ ] New facts also learn to > 0.70 similarity
- [ ] Forgetting delta < 0.15 for each anchor
- [ ] Test passes reliably across 5 independent runs with different random seeds

**This stage is the hardest. Take as long as needed. Do not move on until it is solid.**

---

## Stage 3 — Crate Structure

**Goal:** Move the proven engine from `experiment.rs` into proper crates.
No new features. Just clean structure.

### What to Build

Move the working code from Stage 1 and Stage 2 into the right crates:

**`manas-core`** gets:
- `activation.rs` — ReLU, Sigmoid, Tanh, Linear
- `neuron.rs` — Neuron, ProtectionLevel, Source
- `layer.rs` — Layer, forward()
- `network.rs` — Network, forward(), apply_gradients()
- `error.rs` — ManasError

**`manas-learn`** gets:
- `encoder.rs` — simple hash-based encoder from the experiment (not the full tokenizer yet)
- `backprop.rs` — gradient computation (moved from experiment)
- `trainer.rs` — learn(), query() — the two core operations

### Rules

- `manas-core` must have **zero** dependencies outside std and rand
- `manas-learn` depends only on `manas-core`
- No CLI yet
- No file I/O yet

### Test

```rust
// tests/anti_forgetting.rs in manas-learn crate

#[test]
fn five_facts_survive_fifty_new_facts() {
    use manas_core::Network;
    use manas_learn::Trainer;

    let mut network = Network::new(32, 64, 32);
    let mut trainer = Trainer::new(0.01);

    let anchors = ["cat", "paris", "rust", "everest", "dna"];
    let anchor_targets = [
        "small animal with fur",
        "city in france",
        "systems programming language",
        "highest mountain on earth",
        "double helix genetic",
    ];

    // learn anchors
    for (a, t) in anchors.iter().zip(anchor_targets.iter()) {
        for _ in 0..300 {
            trainer.learn_raw(&mut network, a, t);
        }
    }

    // learn 50 unrelated facts
    let noise = vec![
        ("amazon", "river in south america"),
        ("bitcoin", "digital currency"),
        ("jupiter", "largest planet"),
        // ... 47 more
    ];
    for _ in 0..200 {
        for (k, v) in &noise {
            trainer.learn_raw(&mut network, k, v);
        }
    }

    // verify all 5 anchors survived
    for (a, t) in anchors.iter().zip(anchor_targets.iter()) {
        let sim = trainer.similarity_to_target(&network, a, t);
        assert!(
            sim > 0.65,
            "Catastrophic forgetting: '{}' similarity dropped to {:.4}",
            a, sim
        );
    }
}
```

```bash
cargo test -p manas-learn anti_forgetting
# must pass
```

### Done When

- [ ] `manas-core` compiles clean with only `rand` dependency
- [ ] `manas-learn` compiles clean depending only on `manas-core`
- [ ] `anti_forgetting` test passes
- [ ] `cargo test` runs with zero failures

---

## Stage 4 — Persistence: The `.manas` Binary Format

**Goal:** The brain survives a restart. Teach facts, quit, restart, ask — still works.

### What to Build

Build `manas-store` completely:

```rust
// manas-store/src/lib.rs
pub struct ManasBrain {
    pub path: PathBuf,
}

impl ManasBrain {
    pub fn new(path: impl Into<PathBuf>) -> Self
    pub fn save(&self, network: &Network) -> Result<(), ManasError>
    pub fn load(&self) -> Result<Network, ManasError>
    pub fn exists(&self) -> bool
    pub fn size_bytes(&self) -> u64
}
```

**Binary format** (see ARCHITECTURE.md Section 11 for full spec):
```
[MAGIC: MANS] [VERSION: 2] [HEADER] [VOCAB] [LAYERS+NEURONS] [CRC32]
```

### Tests

```rust
// tests/persistence.rs in manas-store

#[test]
fn brain_survives_save_and_load() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("brain.manas");

    // build a trained network
    let mut network = Network::new(32, 64, 32);
    let mut trainer = Trainer::new(0.01);
    for _ in 0..300 {
        trainer.learn_raw(&mut network, "cat", "small animal with fur");
    }

    // save
    let brain = ManasBrain::new(&path);
    brain.save(&network).unwrap();
    assert!(path.exists());

    // load into fresh network
    let loaded = brain.load().unwrap();

    // verify same answer
    let original_sim = trainer.similarity_to_target(&network, "cat", "small animal with fur");
    let loaded_sim   = trainer.similarity_to_target(&loaded, "cat", "small animal with fur");

    assert!(
        (original_sim - loaded_sim).abs() < 0.01,
        "Loaded network gave different answer: {:.4} vs {:.4}",
        original_sim, loaded_sim
    );
}

#[test]
fn checksum_catches_corruption() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("brain.manas");

    let network = Network::new(32, 64, 32);
    ManasBrain::new(&path).save(&network).unwrap();

    // corrupt one byte in the middle
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[40] ^= 0xFF;
    std::fs::write(&path, bytes).unwrap();

    // load must fail with ChecksumMismatch
    let result = ManasBrain::new(&path).load();
    assert!(matches!(result, Err(ManasError::ChecksumMismatch { .. })));
}

#[test]
fn protection_levels_survive_save_load() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("brain.manas");

    let mut network = Network::new(32, 64, 32);
    // force a neuron to Frozen
    network.layers[0].neurons[0].protection_level = ProtectionLevel::Frozen;

    ManasBrain::new(&path).save(&network).unwrap();
    let loaded = ManasBrain::new(&path).load().unwrap();

    assert_eq!(
        loaded.layers[0].neurons[0].protection_level,
        ProtectionLevel::Frozen,
        "Protection level not preserved through save/load"
    );
}
```

### Done When

- [ ] `brain_survives_save_and_load` passes
- [ ] `checksum_catches_corruption` passes
- [ ] `protection_levels_survive_save_load` passes
- [ ] Magic bytes and version validated on load
- [ ] File size grows predictably as neurons grow
- [ ] `cargo test -p manas-store` passes clean

---

## Stage 5 — Character N-Gram Tokenizer

**Goal:** Replace the hash-based encoder with a real character n-gram tokenizer.
Knowledge representation improves significantly.

### Why Character N-Grams

```
word splitter (v1):  "cats" and "cat" → completely different tokens, no shared structure
char n-grams  (v2):  "cats" and "cat" share ["c", "ca", "cat"] → model sees they're related
```

### What to Build

```rust
// manas-learn/src/tokenizer.rs

pub struct Tokenizer {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub next_id: u32,
    pub max_ngram: usize,   // default: 4
}

impl Tokenizer {
    pub fn new(max_ngram: usize) -> Self

    pub fn encode(&mut self, text: &str) -> Vec<u32>
    // 1. lowercase
    // 2. for each word, extract n-grams of length 1..=max_ngram
    // 3. add word boundary marker: "#word"
    // 4. assign IDs (add to vocab if new)
    // returns: Vec<u32> of token IDs

    pub fn decode(&self, ids: &[u32]) -> String
    // return the longest n-gram for each ID, joined by spaces

    pub fn vocab_size(&self) -> u32

    pub fn encode_deterministic(&self, text: &str) -> Vec<u32>
    // same as encode() but never adds new vocab entries
    // used during query (don't grow vocab from questions)
}
```

**Example:**

```
tokenize("cat sat on the mat")
→ ngrams for "cat": ["c", "ca", "cat", "#cat"]
→ ngrams for "sat": ["s", "sa", "sat", "#sat"]
→ ngrams for "on":  ["o", "on", "#on"]
→ ngrams for "the": ["t", "th", "the", "#the"]
→ ngrams for "mat": ["m", "ma", "mat", "#mat"]
→ IDs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]
```

### Tests

```rust
#[test]
fn cat_and_cats_share_tokens() {
    let mut tok = Tokenizer::new(4);
    let cat_ids  = tok.encode("cat");
    let cats_ids = tok.encode("cats");

    // they must share at least one token ID (the "cat" trigram)
    let cat_set: HashSet<u32>  = cat_ids.into_iter().collect();
    let cats_set: HashSet<u32> = cats_ids.into_iter().collect();
    let shared = cat_set.intersection(&cats_set).count();
    assert!(shared >= 3, "cat and cats should share at least 3 tokens, got {}", shared);
}

#[test]
fn encode_is_deterministic() {
    let mut tok = Tokenizer::new(4);
    let first  = tok.encode("the quick brown fox");
    let second = tok.encode("the quick brown fox");
    assert_eq!(first, second);
}

#[test]
fn empty_string_returns_empty() {
    let mut tok = Tokenizer::new(4);
    assert!(tok.encode("").is_empty());
}

#[test]
fn vocab_grows_on_new_words() {
    let mut tok = Tokenizer::new(4);
    let before = tok.vocab_size();
    tok.encode("xyzqwerty");
    let after = tok.vocab_size();
    assert!(after > before);
}

#[test]
fn decode_encode_roundtrip_reasonable() {
    let mut tok = Tokenizer::new(4);
    let ids = tok.encode("rust programming");
    let decoded = tok.decode(&ids);
    // decoded may not be identical but must contain recognizable substrings
    assert!(decoded.contains("rust") || decoded.contains("programming"),
        "Decoded '{}' does not resemble original", decoded);
}
```

### Done When

- [ ] All tokenizer tests pass
- [ ] `cat_and_cats_share_tokens` confirms structural relationship
- [ ] Anti-forgetting test from Stage 3 still passes with new tokenizer
- [ ] Vocab grows correctly and deterministically
- [ ] `cargo test -p manas-learn tokenizer` passes clean

---

## Stage 6 — Positional Embeddings

**Goal:** Word order now matters. "cat eats dog" ≠ "dog eats cat".

### What to Build

```rust
// manas-learn/src/embedder.rs

pub struct Embedder {
    pub embed_table: HashMap<u32, Vec<f32>>,
    pub embed_dim: usize,
    pub positional_scale: f32,   // default: 0.1
}

impl Embedder {
    pub fn new(embed_dim: usize) -> Self

    pub fn get_or_create(&mut self, token_id: u32) -> &Vec<f32>
    // if token_id not in table: initialize with small random values

    pub fn embed_with_position(&self, token_id: u32, position: usize) -> Vec<f32>
    // embed[i] += positional_scale × sin(position / 10000^(2i/embed_dim))
    // this encodes position into the embedding itself

    pub fn encode_sequence(&mut self, token_ids: &[u32]) -> Vec<f32>
    // embed each token with its position
    // sum all positioned embeddings → single Vec<f32> of length embed_dim
    // this is the input vector to the network
}
```

### Tests

```rust
#[test]
fn order_matters() {
    let mut emb = Embedder::new(32);
    let mut tok = Tokenizer::new(4);

    let ab = emb.encode_sequence(&tok.encode("cat dog"));
    let ba = emb.encode_sequence(&tok.encode("dog cat"));

    let sim = cosine(&ab, &ba);
    // same words, different order → similar but not identical
    assert!(sim < 0.99, "Order should matter: cosine was {:.4}", sim);
    assert!(sim > 0.50, "Too dissimilar for same words: cosine was {:.4}", sim);
}

#[test]
fn same_sequence_same_vector() {
    let mut emb = Embedder::new(32);
    let mut tok = Tokenizer::new(4);

    let ids = tok.encode("rust programming language");
    let v1 = emb.encode_sequence(&ids);
    let v2 = emb.encode_sequence(&ids);

    let sim = cosine(&v1, &v2);
    assert!((sim - 1.0).abs() < 1e-5, "Same sequence must give same vector");
}

#[test]
fn similar_words_similar_vectors() {
    let mut emb = Embedder::new(32);
    let mut tok = Tokenizer::new(4);

    let cat  = emb.encode_sequence(&tok.encode("cat"));
    let cats = emb.encode_sequence(&tok.encode("cats"));
    let dog  = emb.encode_sequence(&tok.encode("dog"));

    let cat_cats = cosine(&cat, &cats);
    let cat_dog  = cosine(&cat, &dog);

    // cat and cats should be more similar to each other than to dog
    assert!(
        cat_cats > cat_dog,
        "cat-cats ({:.4}) should be more similar than cat-dog ({:.4})",
        cat_cats, cat_dog
    );
}
```

### Done When

- [ ] All embedding tests pass
- [ ] Anti-forgetting test still passes end-to-end with tokenizer + embedder
- [ ] Order sensitivity confirmed (same words, different order → different vectors)
- [ ] Structural similarity confirmed (cat closer to cats than to dog)
- [ ] `cargo test -p manas-learn embedder` passes clean

---

## Stage 7 — Growth System

**Goal:** The network grows new neurons when it cannot represent something well enough.
Starts with zero neurons. Grows exactly when needed.

### What to Build

Add to `manas-core/src/network.rs`:

```rust
pub const GROWTH_THRESHOLD: f32 = 0.35;
pub const MAX_UPDATE_ATTEMPTS: u32 = 3;
pub const MAX_NEURONS_PER_LAYER: usize = 512;
pub const MAX_LAYERS: usize = 16;
pub const GUARD_DELTA: f32 = 0.001;

impl Network {
    pub fn grow_neuron(&mut self, layer_id: u32, input_size: usize)
        -> Result<u64, ManasError>
    // add one new Open neuron to layer_id
    // return new neuron's ID

    pub fn grow_layer(&mut self, input_size: usize, neuron_count: usize) -> u32
    // add a new layer with neuron_count Open neurons
    // return new layer's ID

    pub fn neuron_count(&self) -> u64
    pub fn layer_count(&self) -> usize
    pub fn open_neuron_count(&self) -> u64
    pub fn frozen_neuron_count(&self) -> u64
}
```

Add to `manas-learn/src/trainer.rs`:

```rust
impl Trainer {
    pub fn learn(&mut self, network: &mut Network, input: &str, target: &str)
        -> Result<LearnReport, ManasError>
    {
        let input_vec  = self.encode(input);
        let target_vec = self.encode(target);

        let output_vec = network.forward(&input_vec);
        let loss = mse_loss(&output_vec, &target_vec);

        if loss > GROWTH_THRESHOLD {
            // try updating open neurons first
            for attempt in 0..MAX_UPDATE_ATTEMPTS {
                let grads = compute_gradients(network, &input_vec, &target_vec);
                network.apply_gradients(&grads, self.learning_rate);
                let new_loss = mse_loss(&network.forward(&input_vec), &target_vec);
                if new_loss <= GROWTH_THRESHOLD { break; }
                if attempt == MAX_UPDATE_ATTEMPTS - 1 {
                    // still too high → grow
                    network.grow_neuron(0, input_vec.len())?;
                    grew_neurons += 1;
                }
            }
        } else {
            // loss is fine — normal update
            let grads = compute_gradients(network, &input_vec, &target_vec);
            network.apply_gradients(&grads, self.learning_rate);
        }

        Ok(LearnReport { loss, neurons_grown: grew_neurons, ... })
    }
}
```

### Tests

```rust
#[test]
fn network_starts_empty_and_grows() {
    let mut network = Network::new_empty(32);
    assert_eq!(network.neuron_count(), 0);

    let mut trainer = Trainer::new(0.01);
    trainer.learn(&mut network, "cat", "animal").unwrap();

    assert!(network.neuron_count() > 0, "Network should have grown at least one neuron");
}

#[test]
fn repeated_teaching_does_not_explode_neurons() {
    let mut network = Network::new_empty(32);
    let mut trainer = Trainer::new(0.01);

    // teach same fact 100 times
    for _ in 0..100 {
        trainer.learn(&mut network, "cat", "animal").unwrap();
    }

    // neuron count should stabilize, not grow every iteration
    let count_after_100 = network.neuron_count();

    for _ in 0..100 {
        trainer.learn(&mut network, "cat", "animal").unwrap();
    }

    let count_after_200 = network.neuron_count();
    assert_eq!(
        count_after_100, count_after_200,
        "Repeated teaching of same fact should not grow neurons indefinitely"
    );
}

#[test]
fn new_fact_grows_neuron_if_needed() {
    let mut network = Network::new_empty(32);
    let mut trainer = Trainer::new(0.01);

    // train until stable on cat
    for _ in 0..500 {
        trainer.learn(&mut network, "cat", "animal").unwrap();
    }
    let neurons_after_cat = network.neuron_count();

    // teach completely different fact — may need new neuron
    for _ in 0..10 {
        trainer.learn(&mut network, "eiffel tower", "paris france").unwrap();
    }
    let neurons_after_eiffel = network.neuron_count();

    // neurons may have grown for the new fact
    // this is acceptable and expected
    println!("After cat:   {} neurons", neurons_after_cat);
    println!("After eiffel: {} neurons", neurons_after_eiffel);
}

#[test]
fn growth_respects_max_neurons_per_layer() {
    let mut network = Network::new_empty(8);
    let mut trainer = Trainer::new(0.01);

    // teach many different facts aggressively
    for i in 0..1000 {
        let input  = format!("fact{}", i);
        let target = format!("value{}", i);
        let _ = trainer.learn(&mut network, &input, &target);
    }

    for layer in &network.layers {
        assert!(
            layer.neurons.len() <= MAX_NEURONS_PER_LAYER,
            "Layer exceeded max: {} neurons", layer.neurons.len()
        );
    }
}
```

### Done When

- [ ] `network_starts_empty_and_grows` passes
- [ ] `repeated_teaching_does_not_explode_neurons` passes
- [ ] `new_fact_grows_neuron_if_needed` passes
- [ ] `growth_respects_max_neurons_per_layer` passes
- [ ] Anti-forgetting test from Stage 3 still passes with growth system active
- [ ] `cargo test -p manas-core growth` passes clean

---

## Stage 8 — Protection System Hardened

**Goal:** Prove that protection levels are enforced structurally and cannot be bypassed.

### What to Verify

The protection system was introduced in Stage 2 as an experiment.
This stage formalizes it, hardens it, and proves it works under stress.

### Tests

```rust
#[test]
fn frozen_neuron_weight_never_changes() {
    let mut network = Network::new(32, 64, 32);
    // freeze first neuron
    network.layers[0].neurons[0].protection_level = ProtectionLevel::Frozen;
    let weights_before = network.layers[0].neurons[0].weights.clone();

    // run 1000 learning steps — anything could happen
    let mut trainer = Trainer::new(0.1); // high LR to stress test
    for i in 0..1000 {
        let _ = trainer.learn(&mut network, &format!("fact{}", i), &format!("value{}", i));
    }

    let weights_after = &network.layers[0].neurons[0].weights;
    assert_eq!(
        &weights_before, weights_after,
        "Frozen neuron weights changed — protection system broken"
    );
}

#[test]
fn guarded_neuron_updates_are_clamped() {
    let mut network = Network::new(32, 64, 32);
    network.layers[0].neurons[0].protection_level = ProtectionLevel::Guarded;
    let weights_before = network.layers[0].neurons[0].weights.clone();

    let mut trainer = Trainer::new(1.0); // extreme LR to stress test
    for i in 0..100 {
        let _ = trainer.learn(&mut network, &format!("stress{}", i), &format!("test{}", i));
    }

    let weights_after = &network.layers[0].neurons[0].weights;
    for (before, after) in weights_before.iter().zip(weights_after.iter()) {
        let delta = (after - before).abs();
        assert!(
            delta <= 0.001 * 100.0 + 1e-5,
            "Guarded weight changed by {:.6} — clamping failed", delta
        );
    }
}

#[test]
fn open_neuron_updates_freely() {
    let mut network = Network::new(32, 64, 32);
    network.layers[0].neurons[0].protection_level = ProtectionLevel::Open;
    let weights_before = network.layers[0].neurons[0].weights.clone();

    let mut trainer = Trainer::new(0.1);
    for _ in 0..100 {
        let _ = trainer.learn(&mut network, "hello", "world");
    }

    let weights_after = &network.layers[0].neurons[0].weights;
    let any_changed = weights_before.iter().zip(weights_after.iter()).any(|(b, a)| (a - b).abs() > 1e-6);
    assert!(any_changed, "Open neuron should have its weights updated");
}

#[test]
fn protection_survives_save_and_load() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("brain.manas");

    let mut network = Network::new(32, 64, 32);
    network.layers[0].neurons[0].protection_level = ProtectionLevel::Frozen;
    network.layers[0].neurons[1].protection_level = ProtectionLevel::Guarded;

    ManasBrain::new(&path).save(&network).unwrap();
    let loaded = ManasBrain::new(&path).load().unwrap();

    assert_eq!(loaded.layers[0].neurons[0].protection_level, ProtectionLevel::Frozen);
    assert_eq!(loaded.layers[0].neurons[1].protection_level, ProtectionLevel::Guarded);
}

#[test]
fn promotion_happens_automatically() {
    let mut network = Network::new_empty(32);
    let mut trainer = Trainer::new(0.01);

    // teach same fact thousands of times to trigger promotion
    for _ in 0..3000 {
        trainer.learn(&mut network, "cat", "animal").unwrap();
        trainer.update_protection_levels(&mut network);
    }

    // at least one neuron should have been promoted above Open
    let promoted = network.layers.iter()
        .flat_map(|l| l.neurons.iter())
        .filter(|n| !matches!(n.protection_level, ProtectionLevel::Open))
        .count();

    assert!(promoted > 0, "No neurons were promoted after 3000 learning steps");
}
```

### Done When

- [ ] All 5 protection tests pass
- [ ] `frozen_neuron_weight_never_changes` passes under extreme LR
- [ ] `guarded_neuron_updates_are_clamped` confirms delta bounds
- [ ] `protection_survives_save_and_load` confirms persistence
- [ ] `promotion_happens_automatically` confirms auto-promotion
- [ ] Full anti-forgetting test still passes
- [ ] `cargo test -p manas-core protection` passes clean

---

## Stage 9 — `manas-cli` v1: Teach and Ask

**Goal:** The full loop works from the command line for the first time.
Teach a fact. Quit. Restart. Ask. Get the answer from neural weights.

### What to Build

```rust
// manas-cli/src/main.rs

// manas teach "A cat is a small animal."
// manas ask "What is a cat?"
// manas inspect
// manas reset
```

This is the first time everything connects:
`manas-cli` → `manas-learn` → `manas-core` → `manas-store`

### The Critical End-to-End Test

```bash
# fresh brain
rm -f brain.manas
./manas reset

# teach facts
./manas teach "A cat is a small domesticated animal with fur and whiskers."
./manas teach "The Eiffel Tower is located in Paris France."
./manas teach "Rust is a systems programming language focused on safety."

# quit — all state is in brain.manas

# now ask — must answer from neural weights, not memory
./manas ask "What is a cat?"
./manas ask "Where is the Eiffel Tower?"
./manas ask "What is Rust?"
```

Expected:
```
Answer: small domesticated animal fur whiskers
Confidence: 0.82
Answered from: neural weights

Answer: paris france
Confidence: 0.78
Answered from: neural weights

Answer: systems programming language safety
Confidence: 0.80
Answered from: neural weights
```

Answers do not need to be verbatim. They need to capture the key concepts.

### The Sidecar Test

```bash
# teach facts
./manas teach "A cat is a small domesticated animal with fur and whiskers."
./manas teach "The Eiffel Tower is located in Paris France."

# delete everything except the brain file
rm -f brain.manas.sources brain.manas.sourceindex brain.manas.seq
rm -f brain.manas.transformer brain.manas.langmeta

# ask — MUST still work — weights only
./manas ask "What is a cat?"
```

Expected:
```
Answer: small domesticated animal fur whiskers
Answered from: neural weights
```

**This is the test v1 failed. v2 must pass it.**

### Done When

- [ ] `manas teach` works for raw text
- [ ] `manas ask` answers from neural weights
- [ ] Sidecar test passes — no text files needed
- [ ] Brain persists across restart
- [ ] `manas inspect` shows neuron count, layer count, protection stats
- [ ] `manas reset` deletes brain and starts fresh

---

## Stage 10 — File and Folder Ingestion

**Goal:** Teach Manas from a file or a folder of files.

### What to Build

Complete `manas-ingest` crate and connect it to the CLI:

```bash
./manas teach notes.md
./manas teach ./docs/
./manas teach ./docs/ --recursive
```

### Supported Formats

| Format | Parsed As |
|---|---|
| `.txt` | Plain paragraphs |
| `.md` | Markdown stripped, sections as chunks |
| `.rs` | Doc comments + function signatures |
| `.toml` | Key-value pairs as sentences |
| `.json` | Flattened key-value sentences |
| `.csv` | Each row as a sentence |

### Tests

```rust
#[test]
fn txt_file_ingests_correctly() {
    let content = "The cat sat on the mat.\nRust is a programming language.";
    let tmp = write_tmp_file("test.txt", content);
    let chunks = ingest(IngestSource::File(tmp.path().to_path_buf())).unwrap();
    assert!(chunks.len() >= 1);
    assert!(chunks[0].text.contains("cat") || chunks[0].text.contains("Rust"));
}

#[test]
fn folder_walk_finds_all_supported_files() {
    let dir = tempdir().unwrap();
    write_file(dir.path().join("a.txt"), "fact a");
    write_file(dir.path().join("b.md"),  "fact b");
    write_file(dir.path().join("c.rs"),  "/// fact c\nfn main() {}");
    write_file(dir.path().join("skip.exe"), "ignored");

    let chunks = ingest(IngestSource::Folder(dir.path().to_path_buf())).unwrap();
    let sources: Vec<_> = chunks.iter().map(|c| c.source.clone()).collect();

    assert!(sources.iter().any(|s| s.to_string().contains("a.txt")));
    assert!(sources.iter().any(|s| s.to_string().contains("b.md")));
    assert!(sources.iter().any(|s| s.to_string().contains("c.rs")));
    assert!(!sources.iter().any(|s| s.to_string().contains("skip.exe")));
}

#[test]
fn markdown_strips_syntax() {
    let md = "# Title\n\n**bold** text and `code` here.";
    let chunks = ingest(IngestSource::RawText(md.to_string())).unwrap();
    let text = &chunks[0].text;
    assert!(!text.contains('#'), "Markdown headers should be stripped");
    assert!(!text.contains("**"), "Markdown bold should be stripped");
    assert!(!text.contains('`'), "Markdown code should be stripped");
}
```

### Done When

- [ ] All format parsers work and are tested
- [ ] Folder walking is recursive and respects supported extensions
- [ ] Source metadata (`Source::LocalFile`) is preserved per chunk
- [ ] Anti-forgetting test still passes after real file ingestion
- [ ] `cargo test -p manas-ingest` passes clean

---

## Stage 11 — Importance Scoring and Promotion

**Goal:** Importance scores are computed correctly and drive promotion automatically.

### What to Build

Formalize and test the importance scoring system in `manas-learn/src/importance.rs`:

```rust
pub fn compute_importance(neuron: &Neuron, now_secs: u64) -> f32 {
    let freq      = (neuron.activation_count as f32 / 10_000.0).clamp(0.0, 1.0);
    let days_idle = (now_secs - neuron.last_activated) as f32 / 86_400.0;
    let recency   = (-0.1 * days_idle).exp();
    let magnitude = (l2_norm(&neuron.weights) / 10.0).clamp(0.0, 1.0);
    let age_grace = if neuron.born_at + 7 * 86400 > now_secs { 1.0 } else { 0.0 };

    0.40 * freq + 0.30 * recency + 0.20 * magnitude + 0.10 * age_grace
}
```

### Tests

```rust
#[test]
fn frequently_used_neuron_gets_high_importance() {
    let mut neuron = Neuron::default();
    neuron.activation_count = 8000;
    neuron.last_activated   = unix_now();
    let score = compute_importance(&neuron, unix_now());
    assert!(score > 0.40, "Frequently used neuron should score > 0.40, got {:.4}", score);
}

#[test]
fn idle_neuron_gets_low_recency() {
    let mut neuron = Neuron::default();
    neuron.activation_count = 100;
    let thirty_days_ago = unix_now() - 30 * 86400;
    neuron.last_activated = thirty_days_ago;
    let score = compute_importance(&neuron, unix_now());
    let recency_component = (-0.1_f32 * 30.0).exp();
    assert!(recency_component < 0.05, "30-day idle recency should be near zero");
}

#[test]
fn age_grace_only_applies_in_first_week() {
    let now   = unix_now();
    let young = { let mut n = Neuron::default(); n.born_at = now - 3 * 86400; n };
    let old   = { let mut n = Neuron::default(); n.born_at = now - 8 * 86400; n };

    let young_score = compute_importance(&young, now);
    let old_score   = compute_importance(&old, now);

    assert!(
        young_score > old_score,
        "Young neuron ({:.4}) should score higher than old ({:.4}) due to age grace",
        young_score, old_score
    );
}

#[test]
fn importance_drives_promotion() {
    let mut network = Network::new(32, 64, 32);
    // simulate high usage on first neuron
    network.layers[0].neurons[0].activation_count = 6000;
    network.layers[0].neurons[0].last_activated   = unix_now();

    let mut trainer = Trainer::new(0.01);
    trainer.update_protection_levels(&mut network);

    assert!(
        !matches!(network.layers[0].neurons[0].protection_level, ProtectionLevel::Open),
        "Highly-used neuron should have been promoted above Open"
    );
}
```

### Done When

- [ ] All importance scoring tests pass
- [ ] Promotion threshold logic verified (Open → Guarded at 0.50, Guarded → Frozen at 0.85)
- [ ] `age_grace` uses smooth exponential decay not a cliff
- [ ] `cargo test -p manas-learn importance` passes clean

---

## Stage 12 — Freshness System

**Goal:** Every fact knows how time-sensitive it is. Stale facts are flagged.

### What to Build

```rust
// manas-learn/src/freshness.rs

pub enum FreshnessCategory {
    Timeless  = 0,   // definitions, proofs, laws of physics
    Slow      = 1,   // historical facts, biographies
    Fast      = 2,   // software versions, news
    Realtime  = 3,   // stock prices, live scores
}

pub fn detect_freshness(text: &str) -> FreshnessCategory

pub fn is_stale(neuron: &Neuron, now_secs: u64) -> bool {
    let age_days = (now_secs - neuron.born_at) / 86400;
    match FreshnessCategory::from(neuron.freshness_category) {
        FreshnessCategory::Timeless  => false,
        FreshnessCategory::Slow      => age_days > 365,
        FreshnessCategory::Fast      => age_days > 30,
        FreshnessCategory::Realtime  => age_days > 1,
    }
}
```

### Tests

```rust
#[test]
fn timeless_keywords_detected() {
    assert_eq!(detect_freshness("The Pythagorean theorem states that a²+b²=c²"),
               FreshnessCategory::Timeless);
    assert_eq!(detect_freshness("Water is always composed of hydrogen and oxygen"),
               FreshnessCategory::Timeless);
}

#[test]
fn realtime_keywords_detected() {
    assert_eq!(detect_freshness("Breaking news: the stock market fell today"),
               FreshnessCategory::Realtime);
    assert_eq!(detect_freshness("Live scores updated every minute"),
               FreshnessCategory::Realtime);
}

#[test]
fn fast_keywords_detected() {
    assert_eq!(detect_freshness("Rust 2.0 was released last month"),
               FreshnessCategory::Fast);
}

#[test]
fn timeless_fact_never_stale() {
    let mut neuron = Neuron::default();
    neuron.freshness_category = FreshnessCategory::Timeless as u8;
    neuron.born_at = 0; // born at unix epoch — ancient
    assert!(!is_stale(&neuron, unix_now()));
}

#[test]
fn fast_fact_stale_after_30_days() {
    let mut neuron = Neuron::default();
    neuron.freshness_category = FreshnessCategory::Fast as u8;
    neuron.born_at = unix_now() - 31 * 86400; // 31 days ago
    assert!(is_stale(&neuron, unix_now()));
}
```

### Done When

- [ ] Keyword detection tests pass for all 4 categories
- [ ] Staleness detection tests pass for all 4 categories
- [ ] `manas ask` appends a "Note: may be outdated" line when answering from a stale neuron
- [ ] `cargo test -p manas-learn freshness` passes clean

---

## Stage 13 — The Real Demo

**Goal:** Run the definitive test that v1 failed. This is the milestone that proves
the whole project works.

### The Demo Script

```bash
#!/bin/bash
# demo.sh — the test that matters

set -e

echo "=== Starting fresh ==="
rm -f brain.manas
./manas reset

echo ""
echo "=== Teaching 22 facts ==="
./manas teach "A cat is a small domesticated animal with fur and whiskers."
./manas teach "The Eiffel Tower is located in Paris France and was built in 1889."
./manas teach "The Amazon River is the largest river by discharge in the world."
./manas teach "Photosynthesis is the process by which plants convert sunlight into energy."
./manas teach "Hydrogen is the lightest and most abundant element in the universe."
./manas teach "The human brain contains approximately 86 billion neurons."
./manas teach "Mount Everest is the highest mountain on Earth at 8849 meters."
./manas teach "Shakespeare wrote 37 plays and 154 sonnets during his lifetime."
./manas teach "The speed of light in vacuum is approximately 299792458 meters per second."
./manas teach "DNA is a double helix structure that carries genetic information."
./manas teach "The Roman Empire fell in 476 AD when Romulus Augustulus was deposed."
./manas teach "Water boils at 100 degrees Celsius at standard atmospheric pressure."
./manas teach "The Python programming language was created by Guido van Rossum in 1991."
./manas teach "Jupiter is the largest planet in our solar system with 95 known moons."
./manas teach "The Mona Lisa was painted by Leonardo da Vinci in the early 16th century."
./manas teach "Rust programming language was first released by Mozilla Research in 2010."
./manas teach "The mitochondria is the powerhouse of the cell in biology."
./manas teach "Albert Einstein developed the theory of relativity in the early 20th century."
./manas teach "The Pacific Ocean is the largest and deepest ocean on Earth."
./manas teach "Bitcoin was created by Satoshi Nakamoto and launched in January 2009."
./manas teach "The nitrogen cycle describes how nitrogen moves through ecosystems."
./manas teach "Gravity pulls objects toward each other with a force proportional to mass."

echo ""
echo "=== Deleting ALL sidecars — neural weights only ==="
rm -f brain.manas.sources brain.manas.sourceindex brain.manas.seq
rm -f brain.manas.transformer brain.manas.langmeta

echo ""
echo "=== Asking — must answer from neural weights ==="
./manas ask "What is a cat?"
echo "---"
./manas ask "Where is the Eiffel Tower?"
echo "---"
./manas ask "What did Einstein develop?"
echo "---"
./manas ask "What is the mitochondria?"
echo "---"
./manas ask "When was Bitcoin created?"

echo ""
echo "=== Brain state ==="
./manas inspect
```

### Required Output

Every `ask` must show:
```
Answered from: neural weights
```

Not:
```
Not enough local memory to answer this yet.
```

Not:
```
Search results from DuckDuckGo...
```

### Done When

- [ ] All 5 `ask` calls return answers
- [ ] All 5 show `Answered from: neural weights`
- [ ] No sidecar files exist when the test runs
- [ ] Brain file is under 500KB for 22 facts
- [ ] `manas inspect` shows correct neuron and protection stats

**This is v0.1.0. The first real version of Manas.**

---

## Stage 14 — Inspect, Neurons, and Debug Commands

**Goal:** Make the brain state visible and understandable.

### Commands to Build

```bash
# full brain overview
./manas inspect

# list all neurons with importance scores
./manas neurons

# show neurons above a protection level
./manas neurons --protection frozen
./manas neurons --protection guarded

# show neurons for a specific source
./manas neurons --source "notes.md"

# trace how a query flows through the network
./manas trace "What is a cat?"
```

### `manas inspect` Output

```
Brain
  file              : brain.manas
  size              : 48 KB
  created           : 2026-01-15
  last modified     : 2026-01-15

Network
  total neurons     : 47
  total layers      : 3
  open neurons      : 21
  guarded neurons   : 19
  frozen neurons    : 7

Learning
  facts taught      : 22
  total learn calls : 220
  neurons grown     : 12
  layers grown      : 1

Freshness
  timeless neurons  : 31
  slow neurons      : 14
  fast neurons      : 2
  realtime neurons  : 0
  stale neurons     : 0
```

### Done When

- [ ] `manas inspect` shows all sections above
- [ ] `manas neurons` lists each neuron with ID, importance, protection, source
- [ ] `manas neurons --protection frozen` filters correctly
- [ ] `manas trace` shows forward pass activation per layer
- [ ] `cargo test -p manas-cli inspect` passes

---

## Stage 15 — Compression and Forget Command

**Goal:** Low-importance neurons can be merged or removed to keep the brain small.

### What to Build

```bash
# show which neurons would be compressed
./manas forget --dry-run

# compress neurons with importance < 0.10
./manas forget

# compress with custom threshold
./manas forget --threshold 0.20
```

### Compression Logic

```rust
pub fn compress(network: &mut Network, threshold: f32) -> CompressionReport {
    // find all Open neurons with importance_score < threshold
    // that have not been activated in > 30 days
    // merge their weights into the nearest Guarded neighbor
    // remove the low-importance neuron
    // recompute importance scores
}
```

### Tests

```rust
#[test]
fn compression_reduces_neuron_count() {
    let mut network = Network::new(32, 64, 32);
    // inject low-importance neurons
    // run compression
    // verify count dropped
}

#[test]
fn compression_never_touches_frozen_neurons() {
    // all Frozen neurons must survive compression
}

#[test]
fn high_importance_neurons_survive() {
    // neurons above threshold must survive
}

#[test]
fn anti_forgetting_test_still_passes_after_compression() {
    // the 5-fact survival test must still pass after compression
}
```

### Done When

- [ ] All compression tests pass
- [ ] Frozen neurons are never touched
- [ ] Anti-forgetting test still passes after compression
- [ ] Brain file size decreases after compression
- [ ] `cargo test -p manas-learn compression` passes

---

## Stage 16 — Benchmarks and Test Suite

**Goal:** Every claim is measurable. Every regression is catchable.

### Benchmarks to Build

```rust
// manas-benches/benches/bench.rs (new crate)

// B1: How fast is a single teach call?
// B2: How fast is a single ask call?
// B3: How fast is brain.manas save?
// B4: How fast is brain.manas load?
// B5: How fast is the tokenizer on 1000 words?
// B6: How fast is the full anti-forgetting test?
// B7: How much memory does a 1000-neuron brain use?
// B8: How much does the brain file grow per new fact?
```

### Integration Tests

```rust
// tests/integration/

// IT-1: 22-fact demo test (Stage 13) runs in CI
// IT-2: Anti-forgetting: 5 facts survive 50 new facts
// IT-3: Persistence: brain survives save/load/save/load cycle
// IT-4: Growth: network grows correctly under novel input
// IT-5: Protection: frozen neurons never change under 1000 steps
// IT-6: Compression: brain shrinks after forget command
// IT-7: Freshness: stale facts are flagged correctly
// IT-8: File ingestion: teach from .txt .md .rs works
```

### CI Requirements

```yaml
# .github/workflows/ci.yml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - cargo test --workspace
      - cargo clippy --workspace -- -D warnings
      - cargo fmt --check
```

### Done When

- [ ] All 8 benchmarks produce stable numbers
- [ ] All 8 integration tests pass in CI
- [ ] `cargo test --workspace` passes with zero failures
- [ ] `cargo clippy` passes with zero warnings
- [ ] Benchmark results documented in `BENCHMARKS.md`

---

## Stage 17 — Layer Growth

**Goal:** Not just neurons grow — entire new layers grow when needed.

This is a significant step. The network can now deepen itself, not just widen.

### What to Build

```rust
impl Network {
    pub fn should_grow_layer(&self) -> bool {
        // true when:
        // - ALL neurons in layer 0 are Frozen or Guarded
        // - loss is still above GROWTH_THRESHOLD
        // - total layers < MAX_LAYERS
    }

    pub fn grow_layer(&mut self, input_size: usize, neuron_count: usize) -> u32 {
        // add a new layer with fresh Open neurons
        // connect it to the previous final layer
        // return new layer ID
    }
}
```

### Tests

```rust
#[test]
fn layer_grows_when_all_neurons_saturated() {
    // freeze all neurons in a single-layer network
    // verify should_grow_layer() returns true
    // verify grow_layer() adds a new layer
}

#[test]
fn layer_count_respects_max_layers() {
    // push the network to grow many layers
    // verify it never exceeds MAX_LAYERS
}

#[test]
fn anti_forgetting_still_passes_with_two_layers() {
    // 5 facts survive 50 new facts with a 2-layer network
}

#[test]
fn deeper_network_can_represent_more_facts() {
    // a 2-layer network should answer more diverse facts
    // than a 1-layer network of equal total neuron count
}
```

### Done When

- [ ] All layer growth tests pass
- [ ] Anti-forgetting test passes with 2-layer networks
- [ ] Layer count never exceeds `MAX_LAYERS`
- [ ] `manas inspect` shows correct layer count and per-layer stats
- [ ] `cargo test -p manas-core layer_growth` passes clean

---

## Stage 18 — Internet Agent (Future)

**Goal:** Manas can fetch fresh facts from the internet when its knowledge is stale.

**Not started until Stage 17 is complete and the demo from Stage 13 is stable.**

### Planned Behavior

```bash
./manas refresh          # re-fetch all Realtime neurons
./manas refresh --fast   # re-fetch all Fast neurons older than 30 days
```

### Planned Architecture

```
detect stale neuron (freshness system)
  → build search query from neuron's source context
  → fetch from DuckDuckGo (no API key required)
  → parse result
  → re-teach the updated fact
  → update neuron's born_at timestamp
  → re-run importance + protection scoring
```

---

## Stage 19 — Language Generation (Future)

**Goal:** Manas can generate sentences from what it has learned, not just retrieve facts.

**Not started until Stage 17 is complete.**

This is the only stage where a small transformer-style language path may be added.
But unlike v1, it will be built ON TOP of the working associative memory engine,
not instead of it. The associative memory answers questions.
Language generation is a separate capability for producing fluent text.

---

## Principles

These principles are the law. No milestone may violate them.

1. **Knowledge lives in weights** — `ask` never reads a text file to answer a taught fact.
2. **Never forget** — once a neuron is Frozen, its weights never change.
3. **Grow when needed** — new neurons are added only when loss stays above threshold.
4. **Prove before building** — every stage has a test that must pass before the next begins.
5. **From scratch** — no Candle, no HuggingFace, no burn, no external ML framework.
6. **One file** — the entire brain lives in `brain.manas`. No sidecars for answering.
7. **Honest claims** — Manas is a research project, not a ChatGPT replacement.
8. **Local first** — runs on any laptop CPU with no internet required for taught facts.

---

## Version History

| Version | Stage | What it proves |
|---|---|---|
| v0.0.1 | Stage 1-2 | Associative memory works. Anti-forgetting works. |
| v0.1.0 | Stage 13 | The full demo. 22 facts. Neural weights only. No sidecars. |
| v0.2.0 | Stage 14-15 | Brain is inspectable and compressable. |
| v0.3.0 | Stage 16 | Fully tested and benchmarked. |
| v0.4.0 | Stage 17 | Network grows new layers automatically. |
| v1.0.0 | All stages | Stable, tested, documented, released. |
