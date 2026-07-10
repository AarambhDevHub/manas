# ARCHITECTURE.md — Manas v3

> This document describes the system design of Manas v3. It assumes familiarity
> with Manas v2's architecture (associative memory, protection levels, growth,
> `.manas` binary format). v3 adds a conditioned transformer generation path,
> reasoning over multiple memories, and conversational context — without
> altering how the associative memory itself stores or protects knowledge.

---

## Design Philosophy

v2 proved that facts can live in weights and survive forgetting. v3's job is
narrower and harder: **make the network speak fluently and reason across facts,
using the same few-shot, from-scratch, no-framework discipline.**

The central architectural decision of v3 is **conditioning, not replacement**:

```
v2:  prompt -> associative memory -> concept vector -> template realization -> text
v3:  prompt -> associative memory -> concept vector -> transformer (conditioned) -> text
```

The associative memory is unchanged. It still decides *what* the answer is.
The transformer only decides *how to phrase it*, using the concept vector as
cross-attention context. This is why a 2-4 layer transformer can work from
22 taught facts instead of needing a web-scale corpus — it is not learning
world knowledge, it is learning phrasing patterns conditioned on knowledge
it is handed.

---

## Crate Structure (v3 additions)

```
manas/
├── Cargo.toml
├── ARCHITECTURE.md
├── ROADMAP.md
├── SELF_LEARNING.md
├── BENCHMARKS_V3.md
├── manas-core/          # v2, unchanged — associative memory, growth, protection
├── manas-store/         # v2, extended — .manas format bumped to v4 for transformer + vocab
├── manas-learn/         # v2, extended — query_multi, query_with_context
├── manas-ingest/        # v2, unchanged
├── manas-agent/         # v2, unchanged — internet refresh
├── manas-language/      # v2's template realizer — deprecated after Stage 25, kept for fallback
├── manas-tokenizer/      # NEW — GrowingVocab (Stage 20)
├── manas-transformer/    # NEW — TinyTransformer, ConditionedGenerator (Stages 21-24)
├── manas-reason/         # NEW — multi-fact composition, conversational context (Stages 26-27)
├── manas-eval/           # NEW — evaluation harness (Stage 33)
└── manas-cli/            # v2, extended — new commands below
```

### Why new crates, not extensions of existing ones

- `manas-tokenizer` is separated from `manas-core` because vocabulary growth
  has a different lifecycle than neuron growth — it can be swapped, tested,
  and benchmarked independently (Stage 20 tests do not require the full engine).
- `manas-transformer` is isolated from `manas-learn` for the same reason v2
  isolated `manas-agent` from the core engine: it has an external-shaped
  dependency footprint (attention math, training loop) that should never leak
  into the associative memory's own code paths.
- `manas-reason` sits above both `manas-core` and `manas-transformer` because
  multi-fact composition and context resolution need to talk to both.

---

## Data Flow (End to End)

### Teaching (`manas teach`)

```
sentence
  -> manas-core::Trainer::teach()          [v2, unchanged: associative binding]
  -> manas-tokenizer::GrowingVocab::encode_sentence()   [Stage 20: grow vocab if needed]
  -> manas-core::Trainer::query()          [get concept vector for this sentence]
  -> manas-transformer::train_step(tokens, concept)      [Stage 23: teacher-forced training]
  -> manas-transformer::apply_protection(gradients)      [Stage 24: anti-forgetting parity]
```

### Asking (`manas ask`, with context)

```
prompt + ConversationContext
  -> manas-reason::query_with_context()     [Stage 27: fold history into query]
  -> manas-core::Trainer::query_multi()     [Stage 26: top-k concept vectors]
  -> manas-reason::compose()                [Stage 26: merge concepts if multiple]
  -> manas-transformer::ConditionedGenerator::generate()   [Stage 22: cross-attend, decode]
  -> generated sentence
```

### Growing (during `teach`, automatic)

```
loss after update
  -> manas-core::Trainer::diagnose_stall()   [Stage 28: Plateau vs Ceiling]
  -> if Ceiling: choose_growth_strategy()    [Stage 31: Width vs Depth]
  -> grow_neurons() or grow_layer()          [v2 mechanisms, reused]
  -> manas-tokenizer grows in parallel if new n-grams were seen
```

### Compressing (`manas forget --validate`)

```
compress_validated(threshold)
  -> snapshot current state
  -> compress()                              [v2 mechanism, unchanged]
  -> check_all_taught_facts_recall()         [Stage 30: validation]
  -> commit or restore(snapshot)
```

---

## Module Design

### `manas-tokenizer` (Stage 20)

```rust
pub struct GrowingVocab {
    ngram_to_id: HashMap<String, u32>,
    embeddings: Vec<Vec<f32>>,
    protection: Vec<ProtectionLevel>,   // reuses manas-core's protection enum
}
```

Design notes:
- Encoding reuses v2's exact n-gram splitting rules for continuity
  ("c", "ca", "cat", "#cat") — this is why "cat"/"cats" generalize.
- Growth is driven purely by "has this n-gram been seen before," no threshold
  or heuristic — every new n-gram gets a slot, same as v1 principle "grow when
  needed" but for vocabulary instead of neurons.
- Persisted inside `.manas` v4 as a new section; v2/v3 brains without this
  section fall back to a default single-slot-per-character vocab on load.

### `manas-transformer` (Stages 21-24)

```rust
pub struct TinyTransformer {
    layers: Vec<DecoderLayer>,      // 2-4 layers
    vocab: GrowingVocab,
    max_seq_len: usize,
}

pub struct DecoderLayer {
    self_attn: MultiHeadAttention,   // causal
    cross_attn: MultiHeadAttention,  // attends to concept vector
    ffn: FeedForward,
    ln1: LayerNorm, ln2: LayerNorm, ln3: LayerNorm,
}
```

Design notes:
- All matrix ops and backprop are hand-rolled in `manas-core`-style plain
  Rust, no external tensor library — same discipline as v2's `Network`.
- Weight tying between input embedding and output projection halves the
  parameter count that needs training, important on the i3/8GB target.
- Cross-attention context is a single concept vector per generation call
  (Stage 22), or a merged set of concept vectors for multi-fact composition
  (Stage 26) — the mechanism is identical, only the input differs.
- Protection levels (Frozen/Guarded/Open) apply per-weight, same importance
  formula family as v2's neurons, computed via `manas-core::ImportanceScorer`
  (Stage 29) shared between neurons and transformer weights.

### `manas-reason` (Stages 26-27)

```rust
pub struct ConversationContext {
    history: VecDeque<(String, String)>,
}

pub fn query_with_context(trainer: &Trainer, prompt: &str, ctx: &ConversationContext) -> ConceptVector
pub fn compose(concepts: Vec<ConceptVector>) -> ConceptVector
```

Design notes:
- `compose()` is deliberately simple in v3.0: a weighted merge of concept
  vectors, not a learned composition network — keeping with "prove before
  building," a learned composer is a candidate for v4 if this proves
  insufficient.
- Context window is capped (default N=5 exchanges) and stored in memory only
  — it is not persisted into `.manas`, since conversational context is
  session-scoped, not knowledge.

### `manas-eval` (Stage 33)

```rust
pub struct EvalReport {
    fluency: f32,
    repetition_rate: f32,
    next_token_accuracy: f32,
    multi_fact_accuracy: f32,
    context_resolution_accuracy: f32,
}
```

Design notes:
- Fluency is a perplexity-style proxy computed against the transformer's own
  next-token predictions on held-out word positions — there is no external
  reference LM, consistent with "local first, no internet required."
- All eval metrics are deterministic given a seeded RNG, so CI can gate on
  regression against `BENCHMARKS_V3.md`.

---

## `.manas` Binary Format v4

Extends v3's refresh-metadata format (v3 in the v2 roadmap's numbering,
i.e. the format bump introduced in Stage 18) with two new sections:

```
[v2/v3 sections, unchanged: magic bytes, version, neurons, layers,
 refresh metadata, CRC32 checksum]
+
[vocab section]      — GrowingVocab n-grams, embeddings, protection levels
[transformer section] — TinyTransformer layer weights, protection levels
```

Backward compatibility: a v4 loader reading a v2/v3 file (no vocab/transformer
sections) initializes a fresh empty vocab and transformer, so old brains still
load and can be incrementally upgraded by continuing to `teach` them under v3.

Forward compatibility is not guaranteed — v2/v3 binaries reading a v4 file
will fail the version check and refuse to load, same policy as v2's format
bumps.

---

## What Does Not Change From v2

- `manas-core`'s associative memory, protection levels, and growth mechanics
- The `.manas` file being the single source of truth (no sidecars for answering)
- `manas-agent`'s internet refresh behavior
- `manas-ingest`'s file/folder ingestion
- The core claim: `ask` never reads a text file to answer a taught fact

v3 is additive. If every new crate were deleted, v2's behavior would be
fully intact.
