# ARCHITECTURE.md — Manas v2

> **"A brain that starts empty, learns from experience, and remembers what it learned — forever."**
>
> Manas (Sanskrit: *मनस्* — mind, intellect, the seat of thought) is a self-growing
> local AI brain written entirely from scratch in Rust. It starts with zero knowledge,
> learns one fact at a time, stores that knowledge directly inside neural network weights,
> and never forgets what it learned — even as it keeps learning new things.
>
> No cloud. No GPU. No external ML framework. Runs on your laptop CPU.
> Everything lives in one `.manas` file.

---

## Table of Contents

1. [Why v2 Exists](#1-why-v2-exists)
2. [The Core Goal](#2-the-core-goal)
3. [Core Principles](#3-core-principles)
4. [What Makes Manas Different](#4-what-makes-manas-different)
5. [System Overview](#5-system-overview)
6. [The Engine — Associative Memory](#6-the-engine--associative-memory)
7. [Anti-Forgetting System](#7-anti-forgetting-system)
8. [The Growth System](#8-the-growth-system)
9. [Crate Structure](#9-crate-structure)
10. [Crate Details](#10-crate-details)
    - [manas-core](#101-manas-core)
    - [manas-store](#102-manas-store)
    - [manas-learn](#103-manas-learn)
    - [manas-ingest](#104-manas-ingest)
    - [manas-cli](#105-manas-cli)
11. [The .manas Binary Format](#11-the-manas-binary-format)
12. [Data Flow — Full Pipeline](#12-data-flow--full-pipeline)
13. [Neuron Lifecycle](#13-neuron-lifecycle)
14. [The Importance Scoring System](#14-the-importance-scoring-system)
15. [The Freshness System](#15-the-freshness-system)
16. [Error Handling Strategy](#16-error-handling-strategy)
17. [Benchmarks and Integration Gates](#17-benchmarks-and-integration-gates)
18. [What Manas Is Not](#18-what-manas-is-not)

---

## 1. Why v2 Exists

Manas v1 was built with the right vision but the wrong engine.

The v1 `ask` command answered questions by searching a text file (`brain.manas.sources`)
using token overlap. The neural network — all its neurons, backprop, importance scoring,
and protection levels — played no role in answering questions. Knowledge did not live in
the weights. It lived in a text file on disk.

This was proven by a simple test:

```bash
# teach Manas a fact
./manas teach "A cat is a small domesticated animal."
./manas ask "What is a cat?"
# → correct answer ✅

# delete the text sidecar
rm brain.manas.sources brain.manas.sourceindex

# ask again — neural weights only
./manas ask "What is a cat?"
# → "Not enough local memory to answer this yet." ✅ confirmed broken
# → falls back to DuckDuckGo web search ✅ confirmed broken
```

The neural network knew nothing. It was decoration.

**Manas v2 fixes this at the foundation.** Knowledge lives in weights. The neural network
IS the memory. No text sidecar. No keyword search. The network answers directly from
what it learned.

---

## 2. The Core Goal

> Build a neural network that works like a human brain learning from experience —
> starting empty, growing as it learns, never forgetting what it knew, and running
> entirely on a local CPU.

In concrete terms:

```
Step 1:  Brain starts empty — zero neurons, zero knowledge
Step 2:  Teach it "A cat is a small animal with fur"
Step 3:  The neural weights now encode the concept "cat → animal, fur, small"
Step 4:  Teach it 100 more completely unrelated facts
Step 5:  Ask "What is a cat?"
Step 6:  The neural weights answer correctly — no text file involved
Step 7:  The brain grew neurons during steps 2-4 as needed
Step 8:  Old knowledge was never destroyed by new learning
```

This is the demo that matters. Everything in this architecture exists to make
that demo work reliably.

---

## 3. Core Principles

### Principle 1 — Knowledge Lives in Weights
The neural network weights ARE the memory. `ask` queries the network directly.
No text sidecars. No keyword search. No fallback to the internet for taught facts.

### Principle 2 — Never Forget
Once a fact is learned and its neurons are protected, no future learning can destroy it.
Anti-forgetting is built into the learning objective itself, not bolted on afterward.

### Principle 3 — Grow When Needed
The network starts with zero neurons and grows a new neuron only when it cannot
represent something well enough. Growth is measured, not unbounded.

### Principle 4 — Full Local Ownership
One `.manas` file. No account. No API key. No internet required after facts are taught.
The user owns their brain completely.

### Principle 5 — Built From Scratch
No Candle. No HuggingFace. No burn. No tch. No external ML framework.
Every neuron, every gradient, every matrix multiplication is Rust written by hand.
This gives full control over the growth system, the anti-forgetting system,
and the storage format — none of which exist in any external framework.

### Principle 6 — Honest Claims
Manas is not a ChatGPT replacement. It is not a general-purpose LLM.
It is a local associative memory system that learns and remembers facts.
Every milestone is honest about what works and what does not.

### Principle 7 — Small Safe Steps
Every version is testable before the next begins.
No feature is added until the foundation beneath it is proven to work.

---

## 4. What Makes Manas Different

| Problem | How Everyone Else Solves It | How Manas Solves It |
|---|---|---|
| Storing knowledge locally | Save text to a file, search it | Knowledge lives in neural weights |
| Learning new facts | Retrain the whole model | Online learning — one fact at a time |
| Forgetting old facts | Accepted as normal (catastrophic forgetting) | Protection system prevents it |
| Growing with new knowledge | Fixed parameter count forever | Network grows new neurons as needed |
| Running locally | Requires GPU or cloud | Runs on any laptop CPU |
| Knowledge cutoff | Hard cutoff date, stale forever | Teach it new facts anytime |
| Water-cooled datacenters | Required for training | Not required — ever |

---

## 5. System Overview

```
Input
─────
  [Raw Text]   [Local Files]   [Folders]
       │              │             │
       └──────────────┼─────────────┘
                      │
                      ▼
             ┌─────────────────┐
             │  manas-ingest   │  normalize, tokenize, chunk
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │  manas-learn    │  associative learning engine
             │                 │  anti-forgetting system
             │                 │  growth signal
             └────────┬────────┘
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
    ┌──────────────┐   ┌──────────────────┐
    │  manas-core  │   │  importance +    │
    │  neurons     │   │  protection      │
    │  layers      │   │  system          │
    │  network     │   │  (inside core)   │
    │  growth      │   └──────────────────┘
    └──────┬───────┘
           │
           ▼
  ┌────────────────┐
  │  manas-store   │  .manas binary file
  │  read / write  │  append-only growth
  │  integrity     │  CRC32 checksum
  └────────────────┘

Query
──────
  manas ask "What is a cat?"
       │
       ▼
  manas-learn: encode question → query vector
       │
       ▼
  manas-core: forward pass → retrieve answer activation
       │
       ▼
  manas-learn: decode activation → human-readable answer
```

---

## 6. The Engine — Associative Memory

This is the heart of Manas v2. Everything else exists to support this.

### What v1 Did Wrong

v1 used **MSE loss on next-token prediction** as its learning objective.
The network learned: given token A, predict token B.
This is a language model objective — good for generating text, useless for
storing and retrieving facts.

### What v2 Does

v2 uses **associative memory learning** as its core objective.

The network learns to associate an input pattern with an output pattern:

```
Input pattern:  [tokens of "cat"]
Output pattern: [tokens of "animal", "fur", "small", "domesticated"]

Input pattern:  [tokens of "paris"]
Output pattern: [tokens of "city", "france", "eiffel"]
```

When you later ask "What is a cat?", the network encodes the question into
an input vector, runs a forward pass, and the output activations reconstruct
the associated concepts.

### How It Works — Step by Step

**Encoding:**
```
"cat is a small animal" 
  → tokenize → [cat, is, a, small, animal]
  → embed each token → Vec<f32> per token
  → average over tokens → single Vec<f32> (the input vector)
```

**Learning:**
```
input_vec  = encode("cat")
target_vec = pack_answer_words("small domesticated animal with fur and whiskers")
loss       = mse(network.forward(input_vec), target_vec)
gradients  = backprop(loss)
update weights (respecting protection levels)
```

**Querying:**
```
question_vec = encode("What is a cat")
output_vec   = read best hidden output column
answer       = decode(output_vec)  → recover packed known answer words
```

### Why This Stores Knowledge in Weights

When the network learns `cat → animal fur whiskers small domesticated`,
the weights between neurons that activated for "cat" and neurons that activate for
"animal fur whiskers" are strengthened. That connection IS the knowledge.
No text file. No index. The weights are the memory.

---

## 7. Anti-Forgetting System

Catastrophic forgetting is the #1 enemy of continual learning.
When a standard network learns fact B, it overwrites the weights that stored fact A.

Manas v2 solves this with **three layers of protection**:

### Layer 1 — Protection Levels on Every Neuron

Every neuron has one of three protection states:

```rust
pub enum ProtectionLevel {
    Open,     // newly created — full weight updates allowed
    Guarded,  // recently used — small weight updates only
    Frozen,   // important/core — zero weight updates, ever
}
```

Rules:
- A neuron starts as `Open` when created
- Stage 11 promotion uses weighted importance scores:
  `Open → Guarded` at 0.50, then `Guarded → Frozen` at 0.85
- `Frozen` neurons are **never** updated by backprop — zero gradient applied
- `Guarded` neurons receive updates clamped to `[-GUARD_DELTA, +GUARD_DELTA]`
- Promotion is monotonic: protection can be strengthened but never weakened

This is enforced inside `apply_gradients()` in `manas-core` — not in the
trainer, not in the CLI. The protection is structural, not optional.

### Layer 2 — Importance Scoring

Every neuron has an `importance_score: f32` computed from:

```
importance = 0.40 × activation_frequency
           + 0.30 × recency_score
           + 0.20 × weight_magnitude
           + 0.10 × age_grace
```

Neurons with high importance are promoted to `Guarded` or `Frozen` automatically
after each learning step. Low-importance neurons stay `Open` and are candidates
for reuse or compression.

Stage 11 formalizes this weighted score and recomputes it after learning.

### Layer 3 — Growth Instead of Overwrite

When a new fact cannot be represented well by existing neurons (loss stays above
`GROWTH_THRESHOLD` after `MAX_UPDATE_ATTEMPTS`), Manas grows a new neuron
rather than forcing existing neurons to compromise their stored knowledge.

This means old knowledge is never diluted. New knowledge gets fresh capacity.

### The Critical Guarantee

```
After teaching 1000 facts:
  fact #1 (learned first) → still answerable ✅
  fact #500 (learned middle) → still answerable ✅
  fact #1000 (learned last) → still answerable ✅
```

This is the guarantee Manas v2 is designed to deliver and test against.

---

## 8. The Growth System

The network starts empty and grows exactly when needed.

### When a Neuron Grows

```
teach("paris is the capital of france")
  → encode → input_vec
  → forward pass → output_vec
  → compute loss against target_vec
  → loss > GROWTH_THRESHOLD?
      → try updating existing Open neurons (up to MAX_ATTEMPTS)
      → loss still > GROWTH_THRESHOLD?
          → grow new neuron in most appropriate layer
          → initialize with small random weights
          → set ProtectionLevel::Open
          → importance_score = 0.0
```

### When a Layer Grows

Real new-depth layer growth is a later milestone. Stage 7 widens the current
hidden layer and keeps the engine in its proven two-layer shape.

A new layer is added when:
- All neurons in every layer are `Frozen` or `Guarded`
- Loss is still above threshold after MAX_LAYER_ATTEMPTS
- Network depth is below MAX_LAYERS

### Growth is Bounded

```rust
pub const MAX_NEURONS_PER_LAYER: usize = 512;
pub const MAX_LAYERS: usize = 16;
pub const GROWTH_THRESHOLD: f32 = 0.35;
pub const MAX_UPDATE_ATTEMPTS: u32 = 3;
```

The brain cannot grow infinitely. When it hits the bounds, it compresses
low-importance neurons before growing further.

### Growth is Visible

Every `teach` operation reports exactly what grew:

```
Teaching complete
  neurons grown     : 2
  layers grown      : 0
  neurons protected : 14
  neurons frozen    : 7
  total neurons     : 23
```

---

## 9. Crate Structure

```
manas/
├── Cargo.toml              ← workspace root, edition 2024
├── ARCHITECTURE.md         ← this file
├── ROADMAP.md
├── README.md
│
├── manas-core/             ← THE ENGINE
│   ├── Cargo.toml          ← deps: rand only
│   └── src/
│       ├── lib.rs
│       ├── activation.rs   ← ReLU, Sigmoid, Tanh, Linear
│       ├── neuron.rs       ← Neuron, ProtectionLevel, Source
│       ├── layer.rs        ← Layer struct
│       ├── network.rs      ← Network, forward, grow, apply_gradients
│       └── error.rs        ← ManasError enum
│
├── manas-store/            ← PERSISTENCE
│   ├── Cargo.toml          ← no external deps
│   └── src/
│       ├── lib.rs
│       ├── format.rs       ← binary layout constants, magic bytes
│       ├── writer.rs       ← write full brain to .manas
│       ├── reader.rs       ← read full brain from .manas
│       ├── patcher.rs      ← append single neuron without full rewrite
│       └── integrity.rs    ← CRC32 checksum, header validation
│
├── manas-learn/            ← LEARNING ENGINE
│   ├── Cargo.toml          ← deps: manas-core, manas-store
│   └── src/
│       ├── lib.rs
│       ├── tokenizer.rs    ← character n-gram tokenizer
│       ├── embedder.rs     ← token → Vec<f32>, positional encoding
│       ├── encoder.rs      ← text → single input vector
│       ├── decoder.rs      ← output vector → human-readable text
│       ├── backprop.rs     ← MSE loss, gradient computation
│       ├── trainer.rs      ← learn(), query(), grow decision logic
│       ├── importance.rs   ← importance scoring, promotion logic
│       ├── diagnostics.rs  ← inspect, neuron list, trace data models
│       └── compression.rs  ← forget plans and safe compaction reports
│
├── manas-ingest/           ← INPUT PIPELINE
│   ├── Cargo.toml          ← deps: manas-core
│   └── src/
│       ├── lib.rs
│       ├── chunker.rs      ← split text into overlapping chunks
│       ├── normalizer.rs   ← clean text before learning
│       ├── file_reader.rs  ← read a single file
│       ├── folder_walker.rs ← walk folder recursively
│       └── format/
│           ├── mod.rs
│           ├── plaintext.rs
│           ├── markdown.rs
│           ├── rust_source.rs
│           ├── json.rs
│           ├── toml.rs
│           └── csv.rs
│
├── manas-cli/              ← USER INTERFACE
    ├── Cargo.toml          ← deps: all crates above
    └── src/
        └── main.rs         ← std-only arg parsing, command routing, formatting
│
└── manas-benches/          ← TOOLING ONLY
    └── benches/
        └── bench.rs        ← B1-B8 benchmark harness, BENCHMARKS.md generator
```

**Runtime crates: 5** (v1 had 9 — simpler is better)

**Tooling crates: 1** (`manas-benches`, not part of the runtime path)

**No `manas-language` crate** — the transformer path from v1 is removed.
Language generation is a future milestone, not the foundation.

**No `manas-agent` crate** — internet search is a future milestone.
The brain must prove it can store knowledge in weights before fetching more.

**No `manas-memory` crate** — importance scoring and protection now live
directly inside `manas-core` and `manas-learn` where they belong.
In v1, they were separate but didn't actually influence the learning path.
In v2, they are structural — built into `apply_gradients()` and `trainer.rs`.

---

## 10. Crate Details

### 10.1 `manas-core`

The neural network runtime. No ML frameworks. No external math libraries.
Every operation is plain Rust.

**Dependencies:** `rand` only (for weight initialization)

#### Key Types

```rust
// activation.rs
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
}

impl Activation {
    pub fn apply(&self, x: f32) -> f32 { ... }
    pub fn derivative(&self, x: f32) -> f32 { ... }
}
```

```rust
// neuron.rs
pub enum ProtectionLevel {
    Open,       // full updates allowed
    Guarded,    // clamped updates only: [-GUARD_DELTA, +GUARD_DELTA]
    Frozen,     // zero updates, always
}

pub enum Source {
    RawText,
    LocalFile { path: String },
    Unknown,
}

pub struct Neuron {
    pub id: u64,
    pub weights: Vec<f32>,
    pub bias: f32,
    pub activation: Activation,
    pub importance_score: f32,
    pub protection_level: ProtectionLevel,
    pub born_at: u64,            // unix timestamp
    pub last_activated: u64,     // unix timestamp
    pub activation_count: u64,
    pub source: Source,
    pub freshness_category: u8,  // 0=timeless 1=slow 2=fast 3=realtime
}
```

```rust
// layer.rs
pub struct Layer {
    pub id: u32,
    pub neurons: Vec<Neuron>,
    pub activation: Activation,
}

impl Layer {
    pub fn forward(&self, input: &[f32]) -> Vec<f32> { ... }
}
```

```rust
// network.rs
pub struct Network {
    pub layers: Vec<Layer>,
    pub total_neurons: u64,
    pub created_at: u64,
    pub version: u8,
    pub next_id: u64,            // monotonic neuron ID counter
}

impl Network {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self { ... }
    pub fn new_empty(embed_dim: usize) -> Self { ... }
    pub fn forward(&self, input: &[f32]) -> Vec<f32> { ... }
    pub fn forward_with_cache(&self, input: &[f32]) -> ForwardCache { ... }
    pub fn grow_neuron(&mut self, layer_id: u32, input_size: usize) -> Result<u64, ManasError> { ... }
    pub fn grow_layer(&mut self, input_size: usize, neuron_count: usize) -> Result<u32, ManasError> { ... }
    pub fn bind_hidden_neuron_to_fact(&mut self, neuron_id: u64, input: &[f32], target: &[f32])
        -> Result<usize, ManasError> { ... }
    pub fn readout_from_best_hidden(&self, input: &[f32]) -> Option<HiddenReadout> { ... }
    pub fn neuron_count(&self) -> u64 { ... }
    pub fn layer_count(&self) -> usize { ... }
    pub fn open_neuron_count(&self) -> u64 { ... }
    pub fn frozen_neuron_count(&self) -> u64 { ... }
    pub fn apply_gradients(&mut self, gradients: &[(u64, NeuronGradients)], lr: f32) { ... }
    pub fn recompute_next_id(&mut self) { ... }
}
```

#### The Critical Method — `apply_gradients`

This is where anti-forgetting is enforced structurally:

```rust
pub fn apply_gradients(&mut self, gradients: &[(u64, NeuronGradients)], lr: f32) {
    for (neuron_id, grad) in gradients {
        let neuron = self.find_neuron_mut(*neuron_id);
        match neuron.protection_level {
            ProtectionLevel::Frozen  => continue,             // zero update, always
            ProtectionLevel::Guarded => apply_clamped(neuron, grad, lr, GUARD_DELTA),
            ProtectionLevel::Open    => apply_full(neuron, grad, lr),
        }
    }
}
```

Protection is not a suggestion. It is enforced here and cannot be bypassed
by any layer above `manas-core`.

---

### 10.2 `manas-store`

Custom binary persistence. Everything about the brain lives in one `.manas` file.

**Dependencies:** none (pure Rust std only)

#### File Format Overview

```
[MAGIC: 4 bytes] [VERSION: 1 byte] [HEADER: variable] [NEURONS: variable] [CRC32: 4 bytes]
```

See Section 11 for the full binary format specification.

#### Key API

```rust
pub struct ManasBrain {
    pub path: PathBuf,
}

impl ManasBrain {
    pub fn new(path: impl Into<PathBuf>) -> Self { ... }
    pub fn save(&self, network: &Network) -> Result<(), ManasError> { ... }
    pub fn save_state(&self, state: &BrainState) -> Result<(), ManasError> { ... }
    pub fn load(&self) -> Result<Network, ManasError> { ... }
    pub fn load_state(&self) -> Result<BrainState, ManasError> { ... }
    pub fn metadata(&self) -> Result<BrainMetadata, ManasError> { ... }
    pub fn exists(&self) -> bool { ... }
    pub fn size_bytes(&self) -> u64 { ... }
}
```

Stage 14 exposes `BrainMetadata` from the existing header fields:
format version, created time, modified time, total neurons, layer count,
input dimension, and vocab size. This does not change the version 2 binary
format.

#### Why Append-Only Growth Matters

Future storage patching can append newly grown neurons without rewriting the
whole brain. The current implementation saves validated full `BrainState`
snapshots with CRC32 integrity.

---

### 10.3 `manas-learn`

The learning engine. This is where the associative memory objective lives.

**Dependencies:** `manas-core`, `manas-store`, `rand`

#### Tokenizer — Character N-Gram

v1 used a whitespace word splitter. v2 uses character n-grams.

```
"cat" → ["c", "ca", "cat", "#cat"]
"cats" → ["c", "ca", "cat", "cats", "#cats"]
"category" → ["c", "ca", "cat", "cate", "categ", ...]
```

Why:
- "cat" and "cats" share substructure → the model learns they are related
- Order information is preserved inside n-grams
- No external tokenizer library needed
- Works for any language, including Sanskrit

```rust
pub struct Tokenizer {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub vocab_size: u32,
    pub max_ngram: usize,   // default: 4
}

impl Tokenizer {
    pub fn encode(&mut self, text: &str) -> Vec<u32> { ... }
    pub fn decode(&self, ids: &[u32]) -> String { ... }
    pub fn vocab_size(&self) -> u32 { ... }
}
```

#### Embedder — Positional Embeddings

v1 averaged all token embeddings (order-blind).
v2 uses positional encoding so order is preserved.

```
embed("cat sat") ≠ embed("sat cat")
```

```rust
pub struct Embedder {
    pub embed_table: HashMap<u32, Vec<f32>>,
    pub embed_dim: usize,
    pub positional_scale: f32,
}

impl Embedder {
    pub fn new(embed_dim: usize) -> Self { ... }
    pub fn with_seed(embed_dim: usize, seed: u64) -> Self { ... }
    pub fn get_or_create(&mut self, id: u32) -> &Vec<f32> { ... }
    pub fn embed_with_position(&self, id: u32, position: usize) -> Vec<f32> { ... }
    pub fn encode_sequence(&mut self, ids: &[u32]) -> Vec<f32> { ... }
    pub fn encode_existing_sequence(&self, ids: &[u32]) -> Vec<f32> { ... }
    // positional encoding: sinusoidal pair rotation + bounded modulation
}
```

#### Encoder — Text to Vector

```rust
pub fn encode(text: &str, tokenizer: &mut Tokenizer, embedder: &mut Embedder) -> Vec<f32> {
    let ids = tokenizer.encode(text);
    embedder.encode_sequence(&ids)
    // returns single Vec<f32> of length embed_dim
    // this is the input vector to the network
}
```

#### Decoder — Vector to Text

```rust
pub fn decode(output_vec: &[f32], tokenizer: &Tokenizer, embedder: &Embedder) -> String {
    // find the token whose embedding is closest (cosine similarity)
    // to each dimension cluster in output_vec
    // reconstruct the most likely answer tokens
    // join into human-readable string
}
```

#### Trainer — The Core Learning Loop

```rust
pub struct Trainer {
    pub tokenizer: Tokenizer,
    pub embedder: Embedder,
    pub learning_rate: f32,
    pub growth_threshold: f32,
    pub max_update_attempts: u32,
    pub freshness_category: u8,
    pub source: Source,
}

impl Trainer {
    // Teach one fact. Returns what happened.
    pub fn learn(&mut self, network: &mut Network, input: &str, target: &str)
        -> Result<LearnReport, ManasError> { ... }

    // Teach one fact and preserve source metadata on the best matching neuron.
    pub fn learn_with_source(
        &mut self,
        network: &mut Network,
        input: &str,
        target: &str,
        source: Source,
    ) -> Result<LearnReport, ManasError> { ... }

    // Teach one fact with explicit freshness metadata.
    pub fn learn_with_source_and_freshness(
        &mut self,
        network: &mut Network,
        input: &str,
        target: &str,
        source: Source,
        freshness: FreshnessCategory,
    ) -> Result<LearnReport, ManasError> { ... }

    // Ask the network a question. Returns best answer from weights.
    pub fn query(&mut self, network: &Network, question: &str)
        -> Result<QueryResult, ManasError> { ... }

    // Promote neurons based on current importance scores
    pub fn update_protection_levels(&self, network: &mut Network) { ... }
}

pub struct LearnReport {
    pub loss_before: f32,
    pub loss_after: f32,
    pub neurons_grown: u32,
    pub layers_grown: u32,
    pub neurons_promoted: u32,    // Open → Guarded or Guarded → Frozen
    pub neurons_frozen: u32,
    pub total_neurons: u64,
    pub update_applied: bool,
}

pub struct QueryResult {
    pub answer: String,
    pub confidence: f32,          // 0.0 → 1.0
    pub answered_from: AnswerSource,
    pub freshness_warning: Option<FreshnessWarning>,
}

pub struct HiddenReadout {
    pub hidden_index: usize,
    pub activation: f32,
    pub output: Vec<f32>,
}

pub struct FreshnessWarning {
    pub category: FreshnessCategory,
    pub age_days: u64,
}

pub enum AnswerSource {
    NeuralWeights,    // answered from network weights directly
    NotEnough,        // network does not have enough activation
}
```

#### Importance Scoring

```rust
// importance.rs
pub fn compute_importance(neuron: &Neuron, now: u64) -> f32 {
    let freq     = (neuron.activation_count as f32 / 10_000.0).clamp(0.0, 1.0);
    let recency  = recency_score(neuron.last_activated, now);
    let magnitude = weight_magnitude(neuron);
    let age_grace = age_grace_score(neuron.born_at, now);

    0.40 * freq + 0.30 * recency + 0.20 * magnitude + 0.10 * age_grace
}

pub fn promote_if_needed(neuron: &mut Neuron, now: u64) {
    let score = compute_importance(neuron, now);
    neuron.importance_score = score;
    match neuron.protection_level {
        ProtectionLevel::Open    if score >= 0.50 => neuron.guard_all(),
        ProtectionLevel::Guarded if score >= 0.85 => neuron.freeze_all(),
        _ => {}
    }
}
```

---

### 10.4 `manas-ingest`

Unified input pipeline. Converts anything into clean text chunks for learning.
Stage 10 implements this with std-only parsing and deterministic folder order.

**Dependencies:** `manas-core` (for `Source` type)

```rust
pub const CHUNK_SIZE: usize = 512;    // characters
pub const CHUNK_OVERLAP: usize = 64;  // characters of overlap between chunks

pub enum IngestSource {
    RawText(String),
    File(PathBuf),
    Folder(PathBuf),
}

pub struct TextChunk {
    pub text: String,
    pub source: Source,
    pub chunk_id: u64,
}

pub fn ingest(source: IngestSource) -> Result<Vec<TextChunk>, ManasError> { ... }
pub fn chunk_text(text: &str) -> Vec<String> { ... }
pub fn normalize(text: &str) -> String { ... }
```

#### Supported File Formats

| Format | How Parsed |
|---|---|
| `.txt` | Read as-is, chunk by paragraph |
| `.md` | Strip markdown syntax, chunk by section |
| `.rs` | Extract doc comments + function signatures |
| `.toml` | Extract key-value pairs as sentences |
| `.json` | Flatten nested keys to readable sentences |
| `.csv` | Each row becomes a sentence |

---

### 10.5 `manas-cli`

User-facing commands. Thin layer over the learning engine.
**All business logic lives in the crates above. The CLI only routes and formats.**

**Dependencies:** all crates above. Stage 9 uses std-only argument parsing;
external CLI parsing can be added later if needed.

#### Commands

```
manas teach <text|file|folder> [--recursive]
                          Teach raw text, a supported file, or a folder
manas ask "<QUESTION>"    Ask a question — answered from neural weights
manas inspect             Show brain, network, learning, freshness, source, layer stats
manas neurons             List/filter neurons with importance, protection, source
manas trace "<QUESTION>"  Trace query variants, activations, and output values
manas forget [--dry-run]  Compress stale low-importance mergeable neurons
manas reset               Delete the brain and start fresh
```

`forget` is conservative: it never removes `Frozen` neurons, never rewrites
frozen output edges, and only removes low-importance `Open` hidden neurons when
their output column can be merged into a highly similar retained neighbor.

#### `manas teach` Output Format

```
Teaching complete

Input
  mode                  : text
  chunks processed      : 1
  facts learned         : 1

Network
  neurons grown         : 2
  layers grown          : 0
  neurons promoted      : 3
  neurons frozen        : 1
  total neurons         : 18
  total layers          : 2

Learning
  loss (before)         : 0.4821
  loss (after)          : 0.1203
  update applied        : yes

Protection
  open neurons          : 11
  guarded neurons       : 5
  frozen neurons        : 2
```

#### `manas ask` Output Format

```
Answer
  A cat is a small domesticated animal with fur and whiskers.

Confidence
  0.87

Answered from
  neural weights
```

---

## 11. The .manas Binary Format

Everything about the brain lives in one file. No sidecars. No `.sources`. No `.sourceindex`.
In v1, the answering came from sidecar files. In v2, the network answers directly.

### File Layout

```
Offset  Size    Field
──────  ──────  ─────────────────────────────
0       4       Magic bytes: 0x4D 0x41 0x4E 0x53  ("MANS")
4       1       Format version: 2
5       8       Created at (unix timestamp, u64 LE)
13      8       Last modified (unix timestamp, u64 LE)
21      8       Total neurons (u64 LE)
29      4       Total layers (u32 LE)
33      4       Embed dim (u32 LE)
37      4       Vocab size (u32 LE)
41      N       Vocab section (variable length)
41+N    M       Layer + neuron section (variable length)
41+N+M  4       CRC32 checksum of entire file content before this field
```

### Vocab Section

```
[vocab_entry_count: u32 LE]
for each entry:
  [token_len: u16 LE]
  [token_bytes: token_len bytes UTF-8]
  [token_id: u32 LE]
  [embed_vec: embed_dim × 4 bytes (f32 LE)]
```

### Layer + Neuron Section

```
[layer_count: u32 LE]
for each layer:
  [layer_id: u32 LE]
  [activation: u8]   0=ReLU 1=Sigmoid 2=Tanh 3=Linear
  [neuron_count: u32 LE]
  for each neuron:
    [neuron_id: u64 LE]
    [weight_count: u32 LE]
    [weights: weight_count × 4 bytes (f32 LE)]
    [bias: f32 LE]
    [activation: u8]
    [importance_score: f32 LE]
    [protection_level: u8]  0=Open 1=Guarded 2=Frozen
    [born_at: u64 LE]
    [last_activated: u64 LE]
    [activation_count: u64 LE]
    [source_type: u8]   0=RawText 1=LocalFile 2=Unknown
    [source_len: u16 LE]
    [source_bytes: source_len bytes UTF-8]
    [freshness_category: u8]
```

### Why No Sidecars

v1 had: `.manas` + `.manas.sources` + `.manas.sourceindex` + `.manas.seq`
+ `.manas.transformer` + `.manas.langmeta`

v2 has: `.manas` — one file, everything inside.

The answering logic uses the network weights directly. There is no text to store
in a sidecar. The vocab section stores embeddings. The neuron section stores
the learned associations. That is everything needed.

---

## 12. Data Flow — Full Pipeline

### Teaching a Fact

```
manas teach "A cat is a small domesticated animal with fur and whiskers."

  1. manas-cli: parse command, detect raw text input
  2. manas-ingest: normalize → "a cat is a small domesticated animal with fur and whiskers"
  3. manas-ingest: chunk (single chunk, text is short)
  4. manas-learn: tokenize chunk → token IDs
  5. manas-learn: embed with positional encoding → input_vec (Vec<f32>)
  6. manas-learn: pack meaningful answer word IDs into target_vec
  7. manas-core: grow or reuse an Open keyed hidden neuron
  8. manas-core: bind the hidden neuron to input_vec and write target_vec into its output column
  9. manas-learn: update importance, source, freshness, and protection metadata
  10. manas-store: persist network weights, vocab, and neuron metadata in .manas
  11. manas-cli: print LearnReport
```

### Asking a Question

```
manas ask "What is a cat?"

  1. manas-cli: parse command
  2. manas-learn: build query variants such as "cat" from "What is a cat?"
  3. manas-learn: encode each query variant → question_vec (Vec<f32>)
  4. manas-core: select best activated hidden neuron and read only its output column
  5. manas-learn: decode packed answer word IDs and score by hidden activation
  6. confidence > MIN_CONFIDENCE?
       → yes: decode(output_vec) → "small domesticated animal with fur and whiskers"
              answered_from = AnswerSource::NeuralWeights
       → no:  "Not enough knowledge yet."
              answered_from = AnswerSource::NotEnough
  7. manas-cli: print QueryResult
```

No text file. No sidecar. No internet. The network answers from its own weights.

---

## 13. Neuron Lifecycle

```
                   ┌─────────────┐
                   │   Created   │
                   │  (Open)     │ ← importance = 0.0
                   └──────┬──────┘
                          │ used in forward passes
                          │ activation_count grows
                          │ importance_score rises above 0.50
                          ▼
                   ┌─────────────┐
                   │  Guarded    │ ← clamped updates only
                   └──────┬──────┘
                          │ continues being activated
                          │ importance_score rises above 0.85
                          ▼
                   ┌─────────────┐
                   │   Frozen    │ ← zero updates, ever
                   └──────┬──────┘
                          │
               ┌──────────┴───────────┐
               │                      │
               ▼                      ▼
      stays frozen forever     (future) compression
      — knowledge preserved    if neuron cluster is
                               redundant
```

---

## 14. The Importance Scoring System

Importance determines whether a neuron gets promoted to a higher protection level.

```
importance = 0.40 × freq + 0.30 × recency + 0.20 × magnitude + 0.10 × age_grace
```

| Component | Formula | Meaning |
|---|---|---|
| `freq` | `activation_count / 10_000` clamped to [0, 1] | How often this neuron fires |
| `recency` | `exp(-0.1 × days_since_last_activation)` | How recently it was used |
| `magnitude` | `L2_norm(weights) / 10.0` clamped to [0, 1] | How strong its connections are |
| `age_grace` | `exp(-age_days / 7.0)` | Smooth grace period for new neurons |

Promotion thresholds:
- `Open → Guarded`: importance >= 0.50
- `Guarded → Frozen`: importance >= 0.85

Scores are recomputed after every `teach` call.

---

## 15. The Freshness System

Some knowledge goes stale. Facts about current events, software versions, and prices
change over time. The freshness system tracks this.

Every neuron has a `freshness_category: u8`:

| Category | Value | Examples | Re-learn trigger |
|---|---|---|---|
| Timeless | 0 | Mathematical proofs, definitions, laws of physics | Never |
| Slow | 1 | Historical facts, biographies | 1 year |
| Fast | 2 | Software versions, news, prices | 1 month |
| Realtime | 3 | Stock prices, live scores | 1 day |

The freshness category is detected automatically from the text content during `teach`:

```rust
pub fn detect_freshness(text: &str) -> FreshnessCategory {
    // keywords like "theorem", "law", "always" → 0 (Timeless)
    // keywords like "today", "breaking", "live" → 3 (Realtime)
    // keywords like "released", "version" → 2 (Fast)
    // default → 1 (Slow)
}
```

When `ask` retrieves an answer from a neuron with a stale freshness category,
it notes the staleness in the output:

```
Answer
  Rust 1.70 was released in June 2023.

Confidence
  0.81

Answered from
  neural weights

Note
  This knowledge may be outdated (Fast freshness, learned 47 days ago).
```

The internet agent (future milestone) will use freshness to decide when to re-fetch.

---

## 16. Error Handling Strategy

All errors go through `ManasError`. No `.unwrap()` in library code.
The CLI is the only place that handles errors with user-visible messages.

```rust
// manas-core/src/error.rs
#[derive(Debug)]
pub enum ManasError {
    NeuronNotFound(u64),
    LayerNotFound(u32),
    GrowthFailed(String),
    FileReadError { path: PathBuf, source: std::io::Error },
    FileWriteError { path: PathBuf, source: std::io::Error },
    CorruptBrain { reason: String },
    ChecksumMismatch { expected: u32, found: u32 },
    EncodingError(String),
    EmptyInput,
}

impl std::fmt::Display for ManasError { ... }
impl std::error::Error for ManasError { ... }
```

---

## 17. Benchmarks and Integration Gates

Stage 16 keeps performance and regression claims measurable.

`manas-benches` is a dedicated non-runtime crate with a custom no-dependency
benchmark harness:

| ID | What it measures |
|---|---|
| B1 | Single `teach` call |
| B2 | Single `ask` call |
| B3 | `.manas` save |
| B4 | `.manas` load |
| B5 | Tokenizer throughput on 1000 words |
| B6 | Full anti-forgetting proof |
| B7 | Estimated 1000-neuron memory footprint |
| B8 | Brain file growth per new fact |

The committed benchmark report is generated with:

```bash
cargo bench -p manas-benches -- --write-markdown BENCHMARKS.md
```

CI runs the quick benchmark smoke:

```bash
cargo bench -p manas-benches -- --quick
```

The Stage 16 integration gate lives in `manas-cli/tests/stage16_integration.rs`
because the workspace root is virtual and `cargo test --workspace` only runs
tests attached to workspace packages. It covers the real demo, anti-forgetting,
persistence, growth, protection, compression, freshness, and ingestion.

---

## 18. What Manas Is Not

Manas v2 is honest about its scope:

| Manas IS | Manas IS NOT |
|---|---|
| A local associative memory system | A ChatGPT replacement |
| A self-growing neural network | A general-purpose LLM |
| A continual learning research project | A production AI system |
| A from-scratch Rust implementation | A wrapper around Candle or HuggingFace |
| Knowledge stored in neural weights | A text search engine |
| Runs on your laptop CPU | Requires a GPU or cloud |
| Answers from what it was taught | Answers from the internet by default |

The goal is to prove that a neural network can store and retrieve knowledge from
weights alone, grow as it learns, and never forget — running on anyone's laptop.
That is enough. That is the project.
