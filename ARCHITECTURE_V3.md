# ARCHITECTURE.md — Manas v3

> **"v2 proved a brain can remember. v3 teaches it to speak — conditioned on what it remembers, not on the internet."**
>
> Manas v3 adds a small, from-scratch transformer decoder that generates fluent
> language, conditioned on concept vectors retrieved from v2's associative
> memory. It adds multi-fact reasoning, conversational context, and closes the
> four open problems from v2's SELF_LEARNING.md. Nothing in v2 is replaced.
> The associative memory still decides *what* the answer is. v3 only decides
> *how to phrase it*.
>
> Still no cloud. Still no GPU. Still no external ML framework.
> Everything lives in one `.manas` file — now format version 4.

---

## Table of Contents

1. [Why v3 Exists](#1-why-v3-exists)
2. [The Core Goal](#2-the-core-goal)
3. [Core Principles](#3-core-principles)
4. [What Makes v3 Different](#4-what-makes-v3-different)
5. [System Overview](#5-system-overview)
6. [The Generation Engine — Conditioned Transformer](#6-the-generation-engine--conditioned-transformer)
7. [Anti-Forgetting Extended to the Transformer](#7-anti-forgetting-extended-to-the-transformer)
8. [The Reasoning System](#8-the-reasoning-system)
9. [Adaptive Growth and Validated Compression](#9-adaptive-growth-and-validated-compression)
10. [Crate Structure](#10-crate-structure)
11. [Crate Details](#11-crate-details)
    - [manas-tokenizer](#111-manas-tokenizer)
    - [manas-transformer](#112-manas-transformer)
    - [manas-reason](#113-manas-reason)
    - [manas-eval](#114-manas-eval)
    - [manas-core (extensions)](#115-manas-core-extensions)
    - [manas-cli (extensions)](#116-manas-cli-extensions)
12. [The .manas Binary Format v4](#12-the-manas-binary-format-v4)
13. [Data Flow — Full Pipeline](#13-data-flow--full-pipeline)
14. [Transformer Weight Lifecycle](#14-transformer-weight-lifecycle)
15. [The Evaluation System](#15-the-evaluation-system)
16. [Error Handling Strategy](#16-error-handling-strategy)
17. [Benchmarks and Integration Gates](#17-benchmarks-and-integration-gates)
18. [What Manas v3 Is Not](#18-what-manas-v3-is-not)

---

## 1. Why v3 Exists

v2 proved knowledge can live in weights and survive forgetting. But v2's
`manas generate` produced text through **fixed intent templates** — definition,
location, time, action, fallback. The neural network decided *what* to say;
a hand-written template decided *how* to say it. This was proven adequate for
the v2 demo but was never claimed to be more than that.

This was documented honestly in v2's own SELF_LEARNING.md, Problem 3:

> "Reading connections back as human language is non-trivial... This works
> for facts the network was explicitly taught. It is less clear how it
> handles implicit reasoning or combining multiple facts."

v3 exists to replace template realization with **learned, conditioned
generation** — a small transformer decoder that produces text token-by-token,
the same generation mechanism real LLMs use, but conditioned on the
associative memory's concept vector so it can learn phrasing from a handful
of taught sentences instead of a web-scale corpus.

This is proven by an extended version of v2's core test:

```bash
# same 22 facts as v2, sidecars deleted, weights-only

./manas generate "What is a cat?"
# v2: fixed template realization
# v3: token-by-token transformer decode, conditioned on the retrieved concept

./manas ask "What did Einstein develop and when did the Roman Empire fall?"
# v2: could only answer one bound fact per query
# v3: composes two separately-bound facts into one answer

./manas ask "What is a cat?"
./manas ask "What does it eat?"
# v3: second question resolves "it" using conversational context
```

**Manas v3 does not touch how facts are stored.** It only changes how they
are spoken.

---

## 2. The Core Goal

> Build a small transformer that generates fluent language from a handful of
> taught examples — by conditioning it on concept vectors the associative
> memory already knows, instead of asking it to learn language from scratch
> off a massive corpus.

In concrete terms:

```
Step 1:  v2 brain already knows 22 facts, stored in associative memory weights
Step 2:  Teach the same 22 facts again through the v3 pipeline
Step 3:  Each teach call trains the transformer with teacher-forced
         next-token prediction on that one sentence, conditioned on the
         concept vector the associative memory already retrieves for it
Step 4:  Ask "What is a cat?"
Step 5:  The transformer generates token-by-token, cross-attending to the
         concept vector — not filling in a template
Step 6:  Ask a question combining two facts
Step 7:  The answer references both, composed from two retrieved concepts
Step 8:  Old phrasing quality is preserved as new facts are taught —
         anti-forgetting now applies to transformer weights too
```

This is the demo that matters for v3. Everything in this architecture exists
to make that demo work reliably, with the same "prove before building"
discipline as v2.

---

## 3. Core Principles

All seven of v2's principles carry forward unchanged. v3 adds one:

### Principle 1 — Knowledge Lives in Weights *(v2, unchanged)*

The associative memory still decides what the answer is. No text sidecars.

### Principle 2 — Never Forget *(v2, unchanged, now extended)*

Protection now applies to transformer weights as well as associative memory
neurons. See Section 7.

### Principle 3 — Grow When Needed *(v2, unchanged, now diagnosed)*

Growth decisions are now diagnosed (Plateau vs Ceiling) rather than triggered
by a single fixed threshold. See Section 9.

### Principle 4 — Full Local Ownership *(v2, unchanged)*

One `.manas` file, now format version 4. No account, no API key, no internet
required.

### Principle 5 — Built From Scratch *(v2, unchanged)*

The transformer's attention, backprop, and training loop are hand-rolled
Rust, same discipline as v2's `Network`. No Candle, no burn, no tch.

### Principle 6 — Honest Claims *(v2, unchanged, now sharper)*

v3 generates like an LLM mechanically (token-by-token, sampled), but it is
not a general-purpose LLM — it can only fluently discuss what it has been
taught. This distinction is reported explicitly in generation output
(Section 11.6).

### Principle 7 — Small Safe Steps *(v2, unchanged)*

Every v3 stage has a test that must pass before the next begins, same as v2's
18-stage discipline.

### Principle 8 — Conditioning, Not Replacement *(new in v3)*

The transformer never learns world knowledge independently. It is always
conditioned on a concept vector supplied by the associative memory. This is
the single design rule that makes small-data generation viable at all, and
no future stage may weaken it (e.g. by training the transformer directly on
raw uncontrolled text without a concept vector).

---

## 4. What Makes v3 Different

| Problem                          | How v2 Solved It                        | How v3 Solves It                                  |
| --------------------------------- | ---------------------------------------- | -------------------------------------------------- |
| Turning a concept into a sentence | Fixed intent templates                   | Small transformer, conditioned on the concept vector |
| Answering multi-part questions    | One bound fact per query, no composition | `query_multi` + concept merge (Section 8)          |
| Follow-up questions               | Each query stateless                     | Conversational context window (Section 8)          |
| Forgetting during generation      | Not applicable — no learned generation   | Frozen/Guarded parity extended to transformer weights |
| Knowing when to grow              | Fixed loss threshold                     | Plateau vs Ceiling diagnosis (Section 9)            |
| Knowing what to protect           | Fixed importance formula                 | Formula weights learned from compression outcomes   |
| Compression safety                | Fixed threshold, no validation           | Compress → verify recall → commit or roll back      |
| Measuring generation quality      | Not measured — no learned generation     | `manas eval --generation`, fluency/repetition/accuracy |

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
             │  manas-ingest   │  (v2, unchanged)
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │  manas-learn    │  associative learning (v2, unchanged)
             │                 │  anti-forgetting, growth signal
             └────────┬────────┘
                      │
             ┌────────┼────────────────────────┐
             │        │                        │
             ▼        ▼                        ▼
    ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐
    │  manas-core  │  │  manas-tokenizer │  │  manas-transformer  │
    │  neurons     │  │  GrowingVocab    │  │  TinyTransformer    │
    │  layers      │  │  (Stage 20)      │  │  ConditionedGenerator│
    │  network     │  └──────────────────┘  │  (Stages 21-24)     │
    │  growth      │                        └──────────┬──────────┘
    └──────┬───────┘                                   │
           │                                            │
           │              ┌─────────────────────────────┘
           │              │
           ▼              ▼
    ┌──────────────────────────┐
    │      manas-reason        │  multi-fact composition (Stage 26)
    │  ConversationContext     │  conversational context (Stage 27)
    └────────────┬──────────────┘
                 │
                 ▼
        ┌────────────────┐
        │  manas-store   │  .manas v4 binary file (v2 format, extended)
        │  read / write  │  vocab + transformer sections added
        │  integrity     │  CRC32 checksum, unchanged mechanism
        └────────────────┘

Query with generation
──────────────────────
  manas generate "What is a cat?"
       │
       ▼
  manas-learn: encode question → concept vector       (v2, unchanged)
       │
       ▼
  manas-transformer: cross-attend to concept vector,
                      decode token-by-token             (v3, new)
       │
       ▼
  fluent generated sentence
```

---

## 6. The Generation Engine — Conditioned Transformer

This is the heart of v3, the way associative memory was the heart of v2.

### What v2 Did

v2's `manas-language` crate took a decoded concept (a set of retrieved
words) and slotted it into one of five fixed sentence templates
(definition/location/time/action/fallback). This produced grammatical output,
but the phrasing was always one of five shapes, never learned, never varied.

### What v3 Does

v3 replaces template realization with a **small causal transformer decoder,
conditioned on the concept vector via cross-attention**:

```
input:  concept_vector (from associative memory, v2 unchanged)
        + partial generated sequence so far
output: probability distribution over the growing vocab for the next token

for each generation step:
    logits = transformer.forward(generated_so_far, cross_attend = concept_vector)
    next_token = sample(logits, temperature, top_k)
    generated_so_far.push(next_token)
    stop when next_token is end-of-sequence or max_words reached
```

### Why Conditioning Makes Small Data Viable

A standard transformer trained from scratch on 22 sentences would learn
almost nothing generalizable — 22 examples is far too little to learn both
grammar and facts simultaneously. By conditioning on a concept vector that
already encodes *what* the answer is, the transformer's only job is learning
*how English sentences are typically shaped* — subject-verb-object order,
"is a," "is located in," "was created by" — patterns that repeat across
facts. This is why the same architectural trick used for image captioning
(condition a decoder on a fixed feature vector) works here for text.

### Architecture Detail

```
pub struct DecoderLayer {
    self_attn: MultiHeadAttention,   // causal masked, over generated tokens so far
    cross_attn: MultiHeadAttention,  // attends to the concept vector(s)
    ffn: FeedForward,
    ln1: LayerNorm,
    ln2: LayerNorm,
    ln3: LayerNorm,
}

pub struct TinyTransformer {
    layers: Vec<DecoderLayer>,       // 2-4 layers
    vocab: GrowingVocab,             // Section 11.1
    embed_dim: usize,                // 64-128
    num_heads: usize,                // 2-4
    max_seq_len: usize,
}
```

Weight tying between the input embedding table and the output projection
matrix halves the parameters that must be learned — important given the
target is dozens of taught sentences, not billions of tokens.

### Why This Is Still "From Neural Weights"

The generation pipeline draws on two weight sets, and both are neural
weights — no text file is read at any point:

```
Answer: Einstein developed the theory of relativity.
Answered from: neural weights
  - concept retrieval: associative memory (manas-core, v2)
  - phrasing:          transformer (manas-transformer, v3)
```

This provenance line is a required part of every `generate` output
(Section 11.6) — it keeps Principle 1 verifiable even as the pipeline gains
a second weight set.

---

## 7. Anti-Forgetting Extended to the Transformer

v2's three-layer protection system (protection levels, importance scoring,
growth-instead-of-overwrite) was designed for associative memory neurons. v3
discovered the identical failure mode in the transformer: teaching fact #23
measurably degraded the fluency of fact #1's generated answer, because the
transformer's weights were being updated by every subsequent `teach` call
with no protection at all.

### The Fix — Same Mechanism, New Substrate

```
pub enum ProtectionLevel { Open, Guarded, Frozen }   // reused from manas-core

impl TinyTransformer {
    pub fn importance_of_weight(&self, layer: usize, idx: usize) -> f32 {
        // same formula family as manas-core's neuron importance:
        // 0.40 * freq + 0.30 * recency + 0.20 * magnitude + 0.10 * age_grace
        // applied per transformer weight instead of per neuron
    }

    pub fn apply_protection(&mut self, grad: &mut Gradient) {
        match self.protection_level_of(grad.target) {
            ProtectionLevel::Frozen  => grad.zero(),
            ProtectionLevel::Guarded => grad.clamp(GUARD_DELTA),
            ProtectionLevel::Open    => {}, // full update
        }
    }
}
```

### The Critical Guarantee (extended from v2)

```
After teaching 50 facts through the transformer:
  fact #1's generated phrasing  → fluency unchanged within tolerance ✅
  fact #25's generated phrasing → fluency unchanged within tolerance ✅
  fact #50's generated phrasing → learned normally ✅
```

This is tested the same way v2 tested the associative memory guarantee:
generate fact #1's answer before and after teaching 49 more facts, and
compare fluency scores.

---

## 8. The Reasoning System

v2 could retrieve exactly one bound fact per query and answer with no
memory of prior turns. v3 adds two capabilities on top of retrieval, both
living in the new `manas-reason` crate.

### Multi-Fact Composition

```
pub fn query_multi(trainer: &Trainer, prompt: &str, k: usize) -> Vec<ConceptVector> {
    // returns the top-k activated concept vectors, not just the single best match
}

pub fn compose(concepts: Vec<ConceptVector>) -> ConceptVector {
    // v3.0: a weighted merge of concept vectors — deliberately simple.
    // Not a learned composition network. See SELF_LEARNING.md Problem 5:
    // this merges *retrieved* facts, it does not perform novel inference
    // across facts that were never activated together during training.
}
```

### Conversational Context

```
pub struct ConversationContext {
    history: VecDeque<(String, String)>,   // (prompt, answer), capped at N=5
}

pub fn query_with_context(
    trainer: &Trainer,
    prompt: &str,
    ctx: &ConversationContext,
) -> ConceptVector {
    // folds recent history into the query encoding before matching memory,
    // so "What does it eat?" resolves "it" to the prior turn's subject
}
```

**Conversational context is deliberately not persisted into `.manas`.** It is
session-scoped, in-memory only. Persisting it would blur the project's
central claim that knowledge lives in weights, not in transient chat state.

---

## 9. Adaptive Growth and Validated Compression

v2 grew the network whenever loss stayed above a single fixed threshold, and
compressed at a single fixed threshold with no feedback loop. Both were
documented as open problems in v2's SELF_LEARNING.md. v3 addresses both —
partially, and the roadmap says so explicitly rather than overclaiming.

### Adaptive Growth — Plateau vs Ceiling

```
pub enum StallType { Plateau, Ceiling }

pub fn diagnose_stall(loss_history: &[f32]) -> StallType {
    // Plateau: loss is not decreasing right now, but trending down recently
    //          -> more training iterations, no growth
    // Ceiling:  loss stable at a high value regardless of iteration count
    //          -> genuinely lacks capacity, grow
}
```

This distinguishes two situations v2's single threshold could not tell
apart: a network that just needs more iterations (growing would waste
capacity) versus a network that genuinely lacks capacity (not growing wastes
time). It remains a heuristic on loss trajectory, not a principled capacity
estimate — this is stated plainly in SELF_LEARNING.md v3 rather than
presented as fully solved.

### Validated Compression

```
pub fn compress_validated(trainer: &mut Trainer, threshold: f32) -> CompressionReport {
    let snapshot = trainer.snapshot();
    trainer.compress(threshold);                      // v2 mechanism, unchanged
    let regressions = trainer.check_all_taught_facts_recall();
    if !regressions.is_empty() {
        trainer.restore(snapshot);                      // roll back
    }
    CompressionReport { regressions, committed: regressions.is_empty() }
}
```

This guarantees compression never silently breaks recall of a previously
correct fact. It does not find the *optimal* compression threshold — only a
*safe* outcome at whatever threshold is attempted.

### Depth-vs-Width Growth Decision

```
pub fn choose_growth_strategy(saturation_pattern: &SaturationPattern) -> GrowthStrategy {
    // consults a small history of past growth outcomes for similar patterns
    // instead of v2's fixed rule (always width first, depth only after
    // width is exhausted)
}
```

---

## 10. Crate Structure

```
manas/
├── Cargo.toml
├── ARCHITECTURE.md          ← this file
├── ROADMAP.md
├── SELF_LEARNING.md
├── BENCHMARKS_V3.md
│
├── manas-core/              ← v2, unchanged: associative memory, growth, protection
├── manas-store/             ← v2, extended: .manas format bumped to v4
├── manas-learn/             ← v2, extended: query_multi, query_with_context hooks
├── manas-ingest/             ← v2, unchanged
├── manas-agent/               ← v2, unchanged
├── manas-language/           ← v2's template realizer — deprecated after Stage 25,
│                                 kept only as a fallback for degenerate cases
│
├── manas-tokenizer/          ← NEW — GrowingVocab (Section 11.1)
│   └── src/
│       ├── lib.rs
│       └── vocab.rs
│
├── manas-transformer/        ← NEW — TinyTransformer, ConditionedGenerator (Section 11.2)
│   └── src/
│       ├── lib.rs
│       ├── attention.rs
│       ├── decoder_layer.rs
│       ├── transformer.rs
│       ├── training.rs
│       └── protection.rs
│
├── manas-reason/             ← NEW — composition, conversational context (Section 11.3)
│   └── src/
│       ├── lib.rs
│       ├── compose.rs
│       └── context.rs
│
├── manas-eval/               ← NEW — evaluation harness (Section 11.4)
│   └── src/
│       ├── lib.rs
│       ├── fluency.rs
│       ├── reasoning.rs
│       └── growth.rs
│
├── manas-cli/                ← v2, extended: new commands (Section 11.6)
│
└── manas-benches/             ← v2, unchanged tooling crate, extended with v3 benchmarks
```

**Runtime crates: 11** (7 from v2, 4 new)
**Tooling crates: 1** (`manas-benches`, unchanged role)

**`manas-tokenizer` is separated from `manas-core`** because vocabulary
growth has a different lifecycle than neuron growth — it can be tested and
benchmarked independently of the full engine.

**`manas-transformer` is isolated from `manas-learn`** for the same reason
v2 isolated `manas-agent` from the core engine: it has a training-loop
dependency footprint that should never leak into the associative memory's
own code paths.

**`manas-reason` sits above both `manas-core` and `manas-transformer`**
because composition and context resolution need to talk to both.

**`manas-language` is not deleted** — it remains available as a deterministic
fallback if the transformer produces degenerate output (e.g. during the
first few `teach` calls, before enough phrasing patterns exist to generalize
from).

---

## 11. Crate Details

### 11.1 `manas-tokenizer`

Extends v2's character n-gram tokenizer (`manas-learn::tokenizer`) into a
**growing** vocabulary — new n-grams get new embedding slots the moment they
are seen, with no pretrained BPE and no fixed vocab size.

**Dependencies:** `manas-core` (for `ProtectionLevel`)

```
pub struct GrowingVocab {
    pub ngram_to_id: HashMap<String, u32>,
    pub embeddings: Vec<Vec<f32>>,
    pub protection: Vec<ProtectionLevel>,   // reused from manas-core
}

impl GrowingVocab {
    pub fn encode(&mut self, word: &str) -> Vec<u32> {
        // same n-gram splitting rules as v2: "cat" -> ["c","ca","cat","#cat"]
        // any n-gram not yet seen gets a new slot + freshly initialized embedding
    }
    pub fn embedding_of(&self, id: u32) -> &[f32]
    pub fn vocab_size(&self) -> usize
}
```

Growth is driven purely by "has this n-gram been seen before" — no
threshold, no heuristic, matching v2's growth philosophy but applied to
vocabulary instead of neurons.

---

### 11.2 `manas-transformer`

The generation engine. See Section 6 for the architecture rationale.

**Dependencies:** `manas-core`, `manas-tokenizer`, `rand`

```
pub struct TinyTransformer {
    pub layers: Vec<DecoderLayer>,
    pub vocab: GrowingVocab,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub max_seq_len: usize,
}

pub struct ConditionedGenerator {
    pub memory: Trainer,           // v2 associative memory, unchanged
    pub transformer: TinyTransformer,
}

impl ConditionedGenerator {
    pub fn generate(&self, prompt: &str, max_words: usize) -> GenerationResult { ... }
    pub fn teach(&mut self, sentence: &str) -> TeachReport { ... }
}

pub struct GenerationResult {
    pub text: String,
    pub confidence: f32,
    pub answered_from: AnswerSource,        // reused from manas-learn
    pub generation_provenance: GenerationProvenance,
}

pub struct GenerationProvenance {
    pub concept_source: &'static str,   // "associative memory"
    pub phrasing_source: &'static str,  // "transformer"
}
```

#### Training — `train_step`

```
pub fn train_step(&mut self, tokens: &[u32], concept: &[f32]) {
    // teacher-forced next-token cross-entropy loss against the taught sentence
    // gradients computed via hand-rolled backprop through attention + FFN
    // apply_protection() called before weight update (Section 7)
}
```

---

### 11.3 `manas-reason`

Multi-fact composition and conversational context. See Section 8.

**Dependencies:** `manas-core`, `manas-learn`, `manas-transformer`

```
pub struct ConversationContext {
    pub history: VecDeque<(String, String)>,
}

pub fn query_with_context(trainer: &Trainer, prompt: &str, ctx: &ConversationContext) -> ConceptVector
pub fn query_multi(trainer: &Trainer, prompt: &str, k: usize) -> Vec<ConceptVector>
pub fn compose(concepts: Vec<ConceptVector>) -> ConceptVector
```

---

### 11.4 `manas-eval`

Evaluation harness for everything v3 adds. Produces `BENCHMARKS_V3.md`.

**Dependencies:** `manas-core`, `manas-learn`, `manas-transformer`, `manas-reason`

```
pub struct EvalReport {
    pub fluency: f32,
    pub repetition_rate: f32,
    pub next_token_accuracy: f32,
    pub multi_fact_accuracy: f32,
    pub context_resolution_accuracy: f32,
    pub spurious_growth_rate: f32,
    pub compression_regression_rate: f32,
}

pub fn eval_generation(gen: &ConditionedGenerator, held_out: &[String]) -> EvalReport { ... }
pub fn eval_reasoning(gen: &ConditionedGenerator, multi_fact_prompts: &[String]) -> EvalReport { ... }
pub fn eval_growth(trainer: &Trainer, history: &[LearnReport]) -> EvalReport { ... }
```

Fluency is a perplexity-style proxy computed from the transformer's own
next-token predictions on held-out word positions — there is no external
reference LM, consistent with "local first, no internet required." All
metrics are deterministic given a seeded RNG so CI can gate on regression.

---

### 11.5 `manas-core` (extensions)

v3 extends `manas-core` with two additions used by both associative memory
neurons and transformer weights:

```
pub struct ImportanceScorer {
    pub weights: [f32; 4],   // learned (Stage 29), replaces v2's fixed
                              // 0.40/0.30/0.20/0.10 constants
}

impl ImportanceScorer {
    pub fn update_from_compression_outcome(&mut self, target_id: u64, recall_survived: bool) { ... }
}

pub enum StallType { Plateau, Ceiling }   // Section 9
pub enum GrowthStrategy { Width, Depth }  // Section 9
```

These live in `manas-core` (not `manas-transformer`) specifically so both
neurons and transformer weights share one importance-scoring and
growth-diagnosis implementation — avoiding the drift risk of two parallel
formulas.

---

### 11.6 `manas-cli` (extensions)

New and changed commands in v3:

```
manas generate "<PROMPT>" [--max-words N]
                          Now routes through ConditionedGenerator (transformer)
                          instead of Stage 19's template realizer

manas ask [--fluent] [--no-context] "<QUESTION>"
                          --fluent now uses the transformer; --no-context
                          disables conversational context for this query

manas eval --generation   Fluency, repetition rate, next-token accuracy
manas eval --reasoning    Multi-fact composition accuracy
manas eval --context      Context-resolution accuracy
manas eval --growth       Spurious growth rate, strategy accuracy
manas eval --compression  Regression rate under validated compression

manas forget --validate   Compress, verify recall, commit or roll back (Section 9)

manas inspect --vocab        Show GrowingVocab size and growth history
manas inspect --transformer  Show per-layer protection level distribution
manas inspect --growth-history  Show diagnosed stall types and chosen strategies
```

#### `manas generate` Output Format (v3)

```
Generated
  A cat is a small domesticated animal with fur and whiskers.

Confidence
  0.87

Generated from
  neural weights
    - concept retrieval : associative memory
    - phrasing          : transformer
```

---

## 12. The .manas Binary Format v4

Extends the v2/v3 refresh-metadata format with two new sections. All v2
sections (magic bytes, version, neurons, layers, refresh metadata, CRC32
checksum) are unchanged.

```
Offset            Size    Field
────────────────  ──────  ─────────────────────────────
0                 4       Magic bytes: 0x4D 0x41 0x4E 0x53  ("MANS")
4                 1       Format version: 4
...               ...     [v2/v3 header + vocab + layer/neuron sections, unchanged]
...               V       Growing vocab section (NEW — see below)
...+V             T       Transformer section (NEW — see below)
...+V+T           4       CRC32 checksum of entire file content before this field
```

### Growing Vocab Section (new)

```
[ngram_entry_count: u32 LE]
for each entry:
  [ngram_len: u16 LE]
  [ngram_bytes: ngram_len bytes UTF-8]
  [ngram_id: u32 LE]
  [embed_vec: embed_dim × 4 bytes (f32 LE)]
  [protection_level: u8]   0=Open 1=Guarded 2=Frozen
```

### Transformer Section (new)

```
[layer_count: u8]              // 2-4
[embed_dim: u32 LE]
[num_heads: u8]
for each decoder layer:
  [self_attn_weights: variable, f32 LE]
  [cross_attn_weights: variable, f32 LE]
  [ffn_weights: variable, f32 LE]
  [layer_norm_params: variable, f32 LE]
  [protection_levels: one u8 per weight group]
```

### Backward and Forward Compatibility

A v4 loader reading a v2/v3 file (no vocab/transformer sections) initializes
a fresh empty `GrowingVocab` and `TinyTransformer`, so old brains still load
and can be incrementally upgraded by continuing to `teach` them under v3.

Forward compatibility is not guaranteed — v2/v3 binaries reading a v4 file
fail the version check and refuse to load, same policy as every prior format
bump.

---

## 13. Data Flow — Full Pipeline

### Teaching a Fact (v3)

```
sentence
  → manas-core::Trainer::teach()                    (v2, unchanged: associative binding)
  → manas-tokenizer::GrowingVocab::encode_sentence()  (Stage 20: grow vocab if needed)
  → manas-core::Trainer::query()                     (get concept vector for this sentence)
  → manas-transformer::train_step(tokens, concept)    (Stage 23: teacher-forced training)
  → manas-transformer::apply_protection(gradients)    (Stage 24: anti-forgetting parity)
  → manas-core::diagnose_stall(loss_history)           (Stage 28: Plateau vs Ceiling)
  → if Ceiling: choose_growth_strategy() → grow        (Stage 31: Width vs Depth)
```

### Asking a Question, With Context and Composition (v3)

```
prompt + ConversationContext
  → manas-reason::query_with_context()      (Stage 27: fold history into query)
  → manas-core::Trainer::query_multi()      (Stage 26: top-k concept vectors)
  → manas-reason::compose()                  (Stage 26: merge concepts if multiple)
  → manas-transformer::ConditionedGenerator::generate()   (Stage 22: cross-attend, decode)
  → generated sentence, with provenance line
```

### Compressing (v3)

```
manas forget --validate
  → manas-core::compress_validated(threshold)
      → snapshot current state
      → compress()                          (v2 mechanism, unchanged)
      → check_all_taught_facts_recall()      (Stage 30: validation)
      → commit or restore(snapshot)
```

No text file. No sidecar. No internet. Both weight sets answer from
themselves.

---

## 14. Transformer Weight Lifecycle

Mirrors v2's neuron lifecycle (Section 13 of v2's ARCHITECTURE.md), applied
to transformer weights:

```
             ┌─────────────┐
             │   Created   │
             │  (Open)     │ ← importance = 0.0
             └──────┬──────┘
                    │ used in training steps
                    │ activation frequency grows
                    │ importance_score rises above 0.50
                    ▼
             ┌─────────────┐
             │  Guarded    │ ← clamped updates only
             └──────┬──────┘
                    │ continues being used
                    │ importance_score rises above 0.85
                    ▼
             ┌─────────────┐
             │   Frozen    │ ← zero updates, ever
             └──────┬──────┘
                    │
                    ▼
         phrasing pattern preserved forever
         (e.g. the "X is located in Y" pattern
         learned from an early fact stays intact
         no matter how many later facts are taught)
```

---

## 15. The Evaluation System

v2 had no generation to evaluate — Stage 16's benchmarks (B1-B9) measured
speed and the anti-forgetting proof, not output quality. v3 adds quality
metrics because, for the first time, Manas produces learned (not templated)
output whose quality can meaningfully vary.

| Metric                       | What it measures                                             |
| ----------------------------- | -------------------------------------------------------------- |
| Fluency                       | Perplexity-style proxy from the transformer's own predictions |
| Repetition rate               | N-gram repetition within a single generated output             |
| Next-token accuracy           | Accuracy on held-out word positions from taught sentences      |
| Multi-fact accuracy           | Whether composed answers reference all activated facts         |
| Context-resolution accuracy   | Whether follow-up pronouns/references resolve correctly         |
| Spurious growth rate          | How often growth triggers when it did not need to (Stage 28)   |
| Compression regression rate   | How often a compression pass would have broken recall (Stage 30) |

All metrics are recorded in `BENCHMARKS_V3.md`, generated the same way v2's
`BENCHMARKS.md` was — via a dedicated bench command, gated in CI against
recorded baselines.

---

## 16. Error Handling Strategy

Extends v2's `ManasError` enum. No `.unwrap()` in library code, same
discipline as v2.

```
// manas-transformer/src/error.rs (new variants added to the shared ManasError)
pub enum ManasError {
    // ... all v2 variants unchanged ...
    TransformerForwardFailed(String),
    VocabGrowthFailed { ngram: String, reason: String },
    CompressionRegressionDetected { fact: String },
    ContextWindowOverflow { max: usize, attempted: usize },
}
```

---

## 17. Benchmarks and Integration Gates

Extends v2's Stage 16 benchmark harness (`manas-benches`) with v3-specific
measurements:

| ID  | What it measures                          |
| --- | ------------------------------------------ |
| B1-B9 | Unchanged from v2                         |
| B10 | Single transformer forward pass            |
| B11 | Single transformer train_step              |
| B12 | Full transformer anti-forgetting proof      |
| B13 | Multi-fact composition query latency        |
| B14 | Conversational context resolution latency   |
| B15 | Validated compression pass (with rollback path) |

```
cargo bench -p manas-benches -- --write-markdown BENCHMARKS_V3.md
```

CI runs the same quick smoke as v2, extended to cover the v3 integration
test suite (`manas-cli/tests/stage33_integration.rs`), which exercises the
full 22-fact demo through the transformer generation path, multi-fact
composition, conversational context, and validated compression together.

---

## 18. What Manas v3 Is Not

| Manas v3 IS                                          | Manas v3 IS NOT                                  |
| ----------------------------------------------------- | -------------------------------------------------- |
| A conditioned transformer generating from few examples | A general-purpose LLM that can discuss any topic  |
| Token-by-token generation, the real LLM mechanism      | Fluent about anything it wasn't taught             |
| Multi-fact composition over retrieved memories         | Novel inference across facts never taught together |
| Conversational context within a session                | Long-term memory of conversations (that's `teach`) |
| Growth diagnosis (Plateau vs Ceiling)                  | A solved theory of optimal network capacity        |
| Validated, rollback-safe compression                   | A compression threshold chosen optimally, not just safely |
| Still fully from-scratch Rust, no ML framework          | A wrapper around any external transformer library  |

The goal is unchanged from v2, extended by one clause: prove that a neural
network can store knowledge in weights, grow, never forget, **and now speak
fluently about what it knows — conditioned on it, not despite it.** That is
the v3 project.