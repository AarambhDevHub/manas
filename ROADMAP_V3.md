# ROADMAP.md — Manas v3

> **"The brain remembers. Now it learns to speak — and to reason across what it remembers."**
>
> v3 builds directly on top of the working v2 engine. Nothing in v2 is replaced.
> The associative memory still answers questions from weights alone. v3 adds a
> small conditioned transformer for fluent generation, multi-fact reasoning,
> conversational context, and closes the four open problems documented in
> SELF_LEARNING.md v2.

---

## The Single Most Important Test (v3)

Everything from the v2 test still must pass. v3 adds four more checks on top:

```bash
# same 22 facts as v2, same sidecar deletion, weights-only

./manas generate "What is a cat?"
# must produce a fluent, non-templated sentence via the transformer decoder,
# not the Stage 19 intent-realization templates

./manas ask "What did Einstein develop and when did the Roman Empire fall?"
# must combine two separately-bound facts into one answer (Stage 26)

./manas ask "What is a cat?"
./manas ask "What does it eat?"
# second question must resolve "it" -> "cat" using conversational context (Stage 27)

./manas eval --generation
# must report fluency, repetition rate, next-token accuracy, and multi-fact
# accuracy above the baselines recorded in BENCHMARKS_V3.md (Stage 33)
```

**This test does not pass at the start of v3. It must pass at the end of
every stage from Stage 25 onward (generation baseline), and fully by Stage 33.**

---

## Status Table

| Milestone | Name | Status |
|---|---|---|
| Stage 20 | Growing subword tokenizer | Planned |
| Stage 21 | Tiny transformer decoder (skeleton) | Planned |
| Stage 22 | Conditioning — memory to transformer | Planned |
| Stage 23 | Online next-token training | Planned |
| Stage 24 | Anti-forgetting parity for the transformer | Planned |
| Stage 25 | Generation test and fluency benchmark | Planned |
| Stage 26 | Multi-fact reasoning and composition | Planned |
| Stage 27 | Conversational context window | Planned |
| Stage 28 | Adaptive growth policy | Planned |
| Stage 29 | Learned importance scoring | Planned |
| Stage 30 | Validated compression | Planned |
| Stage 31 | Depth-vs-width growth decision | Planned |
| Stage 32 | Attention-lite retrieval ranking | Planned |
| Stage 33 | Full evaluation harness | Planned |

---

## Stage 20 — Growing Subword Tokenizer

**Goal:** Extend the v2 character n-gram tokenizer into a vocabulary that grows
the same way the network grows — new n-grams get new embedding slots the
moment they are seen, with no pretrained BPE and no fixed vocab size.

### What to Build

```rust
pub struct GrowingVocab {
    ngram_to_id: HashMap<String, u32>,
    embeddings: Vec<Vec<f32>>,       // parallel to ngram_to_id
    frozen: Vec<bool>,               // reuses protection levels from v2
}

impl GrowingVocab {
    pub fn encode(&mut self, word: &str) -> Vec<u32> {
        // same n-gram splitting as v2 ("cat" -> ["c","ca","cat","#cat"])
        // any n-gram not yet in ngram_to_id gets a new slot + fresh embedding
    }

    pub fn embedding_of(&self, id: u32) -> &[f32]
    pub fn vocab_size(&self) -> usize
}
```

### What NOT to Do

- Do not import a BPE library or pretrained tokenizer
- Do not fix a maximum vocab size — growth is unbounded like neuron growth,
  subject to the same compression logic from Stage 30

### Tests

```rust
#[test]
fn new_ngram_gets_new_slot() { /* teach a new word, vocab_size increases */ }

#[test]
fn shared_ngrams_reuse_slots() { /* "cat" and "cats" share 3 of 4 n-gram ids */ }

#[test]
fn vocab_persists_across_save_load() { /* .manas v4 round-trip */ }
```

### Done When

- [ ] Vocab grows automatically during `teach`, no manual vocab step
- [ ] `.manas` format bumped to v4, still loads v2/v3 brains
- [ ] `manas inspect --vocab` shows vocab size and growth history

---

## Stage 21 — Tiny Transformer Decoder (Skeleton)

**Goal:** Build a small causal transformer decoder — forward pass only, no
training. Deliberately small: 2-4 layers, dim 64-128, 2-4 heads. This is not
a GPT clone; it exists to phrase concepts the associative memory already knows.

### What to Build

```rust
pub struct DecoderLayer {
    self_attn: MultiHeadAttention,   // causal masked
    cross_attn: MultiHeadAttention,  // attends to the concept vector (Stage 22)
    ffn: FeedForward,
    ln1: LayerNorm,
    ln2: LayerNorm,
    ln3: LayerNorm,
}

pub struct TinyTransformer {
    layers: Vec<DecoderLayer>,
    vocab: GrowingVocab,             // Stage 20
    max_seq_len: usize,
}

impl TinyTransformer {
    pub fn forward(&self, tokens: &[u32], concept: &[f32]) -> Vec<Vec<f32>>
    // returns per-position logits over the growing vocab
}
```

### What NOT to Do

- Do not use Candle, burn, or any external tensor/autograd library — matrix
  ops and backprop are hand-rolled, same as v2's `Network`
- Do not hardcode a fixed vocab size in the output projection — it must track
  `GrowingVocab::vocab_size()` dynamically

### Tests

```rust
#[test]
fn forward_pass_produces_correct_shape() { /* [seq_len, vocab_size] logits */ }

#[test]
fn causal_mask_prevents_future_leakage() { /* position i logits unaffected by token i+1 */ }
```

### Done When

- [ ] Forward pass runs on CPU within acceptable latency on the i3 target hardware
- [ ] Causal masking verified by test, not just by construction
- [ ] No training wired yet — this stage is structure only

---

## Stage 22 — Conditioning: Memory to Transformer

**Goal:** Wire the associative memory's retrieved concept vector into the
transformer as cross-attention context. This is the core design move of v3:
the transformer does not learn language from scratch, it learns phrasing
conditioned on a concept it already knows the answer to.

### What to Build

```rust
pub struct ConditionedGenerator {
    memory: Trainer,              // v2 associative memory, unchanged
    transformer: TinyTransformer, // Stage 21
}

impl ConditionedGenerator {
    pub fn generate(&self, prompt: &str, max_words: usize) -> String {
        let concept = self.memory.query_with_style(prompt, QueryStyle::Expanded);
        // concept vector becomes the cross-attention context for every
        // decoder layer, for every generated token
        self.transformer.autoregressive_decode(concept, max_words)
    }
}
```

### Tests

```rust
#[test]
fn concept_vector_changes_generation_output() {
    // same transformer weights, two different concept vectors
    // -> generated text differs, proving conditioning actually influences output
}

#[test]
fn generation_without_concept_is_incoherent() {
    // sanity check: unconditioned decode should be visibly worse than conditioned
}
```

### Done When

- [ ] `manas generate` and `manas ask --fluent` route through
      `ConditionedGenerator` instead of Stage 19's template realizer
- [ ] Stage 19's intent-template code path is deprecated but not deleted until
      Stage 25 benchmark confirms the new path is strictly better

---

## Stage 23 — Online Next-Token Training

**Goal:** Every `teach` call trains the transformer with teacher-forced
next-token prediction on that sentence, immediately — true online learning,
matching the associative memory's update model from v2.

### What to Build

```rust
impl ConditionedGenerator {
    pub fn teach(&mut self, sentence: &str) {
        self.memory.teach(sentence);                    // v2 path, unchanged
        let concept = self.memory.query(sentence);
        let tokens = self.transformer.vocab.encode_sentence(sentence);
        self.transformer.train_step(tokens, concept);    // Stage 23, new
    }
}
```

- Weight-tie input embedding and output projection matrices — halves the
  parameters that need to be learned, important given the i3/8GB target
- Loss: standard causal cross-entropy per position, teacher-forced against
  the actual taught sentence

### Tests

```rust
#[test]
fn training_reduces_loss_on_repeated_teach() { /* loss decreases over calls */ }

#[test]
fn weight_tying_holds_after_vocab_growth() { /* new vocab slots stay tied */ }
```

### Done When

- [ ] `teach` updates both the associative memory and the transformer in one call
- [ ] Loss curve logged and inspectable via `manas inspect --transformer`

---

## Stage 24 — Anti-Forgetting Parity for the Transformer

**Goal:** Extend v2's Frozen/Guarded protection levels to transformer weights.
Without this, teaching fact #23 degrades the phrasing quality learned from
fact #1 — the same catastrophic forgetting problem v2 solved for the
associative memory, now showing up in the generation path.

### What to Build

```rust
pub enum ProtectionLevel { Open, Guarded, Frozen } // reused from v2

impl TinyTransformer {
    pub fn importance_of_weight(&self, layer: usize, idx: usize) -> f32 {
        // same formula family as v2 Stage 11, applied to transformer weights:
        // 0.40*freq + 0.30*recency + 0.20*magnitude + 0.10*age_grace
    }

    pub fn apply_protection(&mut self, grad: &mut Gradient) {
        // zero out Frozen weight gradients, clamp Guarded weight gradients
    }
}
```

### Tests

```rust
#[test]
fn frozen_transformer_weights_never_change() { /* teach 50 new facts, check */ }

#[test]
fn phrasing_quality_for_fact_1_survives_fact_50() {
    // generate fact #1's answer before and after teaching 49 more facts
    // fluency score must not regress beyond a small tolerance
}
```

### Done When

- [ ] Anti-forgetting test passes for the transformer, mirroring v2's Stage 2 test
- [ ] `manas inspect --transformer` shows per-layer protection level distribution

---

## Stage 25 — Generation Test and Fluency Benchmark

**Goal:** Replace Stage 19's template realization entirely. Re-run the full
22-fact demo through the new conditioned-transformer pipeline and confirm it
is measurably better than the template baseline it replaces.

### What to Build

- `manas eval --generation` command producing:
  - fluency score (perplexity-style proxy, since no external LM is available)
  - repetition rate (n-gram repetition within generated output)
  - next-token accuracy on held-out word positions from taught sentences

### Tests

```rust
#[test]
fn generation_beats_stage19_template_baseline() {
    // recorded Stage 19 fluency/repetition numbers vs new transformer numbers
}

#[test]
fn twenty_two_fact_demo_passes_with_transformer_generation() {
    // same test as v2's core demo, but manas generate replaces manas ask --fluent
}
```

### Done When

- [ ] `BENCHMARKS_V3.md` records baseline generation metrics
- [ ] Stage 19 template code path removed once this stage's tests pass
- [ ] `manas generate` is the default fluent path

---

## Stage 26 — Multi-Fact Reasoning and Composition

**Goal:** Solve SELF_LEARNING v2 Problem 3. Let a single query activate and
combine multiple bound memories into one answer, instead of retrieving one
fact at a time.

### What to Build

```rust
impl Trainer {
    pub fn query_multi(&self, prompt: &str, k: usize) -> Vec<ConceptVector> {
        // return top-k activated concept vectors, not just the single best match
    }
}

impl ConditionedGenerator {
    pub fn generate_composed(&self, prompt: &str) -> String {
        let concepts = self.memory.query_multi(prompt, 2);
        // transformer cross-attends over the concatenated/merged concept set
        // rather than a single concept vector
    }
}
```

### Tests

```rust
#[test]
fn two_fact_prompt_activates_two_memories() {
    // "What did Einstein develop and when did the Roman Empire fall?"
    // both facts' concept vectors must appear in query_multi's top-k
}

#[test]
fn composed_answer_contains_both_facts() { /* generated text mentions both */ }
```

### Done When

- [ ] Multi-fact prompts produce answers referencing all activated facts
- [ ] Single-fact prompts are unaffected (no regression on the v2 demo)

---

## Stage 27 — Conversational Context Window

**Goal:** Carry the last N exchanges into the query encoding so follow-up
questions resolve without restating the subject.

### What to Build

```rust
pub struct ConversationContext {
    history: VecDeque<(String, String)>, // (prompt, answer) pairs, capped at N
}

impl Trainer {
    pub fn query_with_context(&self, prompt: &str, ctx: &ConversationContext) -> ConceptVector {
        // fold recent history into the query encoding before matching memory
    }
}
```

### Tests

```rust
#[test]
fn followup_pronoun_resolves_to_prior_subject() {
    // ask("What is a cat?") then ask("What does it eat?")
    // second query must resolve "it" using context, not fail or hallucinate
}

#[test]
fn context_window_is_bounded() { /* history never exceeds N entries */ }
```

### Done When

- [ ] `manas ask` maintains context across a session by default
- [ ] `--no-context` flag available for the old stateless behavior

---

## Stage 28 — Adaptive Growth Policy

**Goal:** Solve Problem 1. Distinguish a capacity problem (needs new neurons
or vocab slots) from an optimization problem (needs more training iterations)
instead of using a single fixed loss threshold.

### What to Build

```rust
impl Trainer {
    pub fn diagnose_stall(&self, loss_history: &[f32]) -> StallType {
        // Plateau: loss stable but not decreasing over N steps -> more iterations
        // Ceiling: loss stable at a high value regardless of steps -> grow
    }
}
```

### Tests

```rust
#[test]
fn plateau_does_not_trigger_growth() { /* loss oscillating low -> no growth */ }

#[test]
fn ceiling_triggers_growth() { /* loss stuck high across many steps -> growth */ }
```

### Done When

- [ ] Growth decisions logged with the diagnosed stall type
- [ ] Fewer spurious neuron/layer growths on the 22-fact demo vs v2's fixed threshold

---

## Stage 29 — Learned Importance Scoring

**Goal:** Solve Problem 2. Replace the fixed heuristic formula with a scorer
trained against actual post-compression recall outcomes.

### What to Build

```rust
pub struct ImportanceScorer {
    weights: [f32; 4], // learned, replaces the fixed 0.40/0.30/0.20/0.10 formula
}

impl ImportanceScorer {
    pub fn update_from_compression_outcome(&mut self, neuron: &Neuron, recall_survived: bool) {
        // adjust internal weights based on whether compressing this neuron
        // broke recall of a taught fact
    }
}
```

### Tests

```rust
#[test]
fn scorer_adapts_after_bad_compression() {
    // simulate a compression that broke recall, verify scorer weights shift
}
```

### Done When

- [ ] Importance formula weights are inspectable and shown to diverge from the
      v2 fixed defaults after enough compression cycles
- [ ] Recall regression rate after compression drops vs v2 baseline

---

## Stage 30 — Validated Compression

**Goal:** Solve Problem 4. Before committing a compression pass, verify all
previously-taught facts still recall correctly; back off automatically if any
regress.

### What to Build

```rust
impl Trainer {
    pub fn compress_validated(&mut self, threshold: f32) -> CompressionReport {
        let snapshot = self.snapshot();
        self.compress(threshold);
        let regressions = self.check_all_taught_facts_recall();
        if !regressions.is_empty() {
            self.restore(snapshot); // roll back
        }
        CompressionReport { regressions, committed: regressions.is_empty() }
    }
}
```

### Tests

```rust
#[test]
fn compression_rolls_back_on_regression() { /* forced bad threshold -> rollback */ }

#[test]
fn compression_commits_when_safe() { /* conservative threshold -> commits, shrinks */ }
```

### Done When

- [ ] `manas forget --validate` is the new default compression path
- [ ] No compression pass in the test suite ever leaves a previously-correct
      fact unanswerable

---

## Stage 31 — Depth-vs-Width Growth Decision

**Goal:** Let the network choose between growing a layer (depth) or growing
neurons (width) based on which historically reduced loss faster for similar
saturation patterns, rather than v2's fixed width-then-depth order.

### What to Build

```rust
impl Trainer {
    pub fn choose_growth_strategy(&self, saturation_pattern: &SaturationPattern) -> GrowthStrategy {
        // consult a small history of past growth outcomes for similar patterns
        // GrowthStrategy::Width or GrowthStrategy::Depth
    }
}
```

### Tests

```rust
#[test]
fn strategy_choice_uses_historical_outcomes() { /* seeded history biases choice */ }

#[test]
fn anti_forgetting_holds_regardless_of_strategy_chosen() { /* Stage 2 test still passes */ }
```

### Done When

- [ ] Growth strategy choice logged with the reasoning (pattern matched, outcome used)
- [ ] `manas inspect --growth-history` shows strategy choices over time

---

## Stage 32 — Attention-Lite Retrieval Ranking

**Goal:** When multiple candidate memories match a query, weigh them by
attention relevance rather than pure nearest-neighbor distance. Shares
machinery with Stage 22's cross-attention.

### What to Build

```rust
impl Trainer {
    pub fn rank_candidates_by_attention(&self, query: &[f32], candidates: Vec<ConceptVector>) -> Vec<(ConceptVector, f32)> {
        // reuse the transformer's attention scoring function as a ranking signal
        // over candidate memories, not just cosine/nearest-neighbor distance
    }
}
```

### Tests

```rust
#[test]
fn attention_ranking_differs_from_nearest_neighbor_on_ambiguous_query() {
    // construct a case where nearest-neighbor picks the wrong fact
    // and attention ranking picks correctly
}
```

### Done When

- [ ] Ambiguous-query test cases show measurable improvement over v2's
      nearest-neighbor-only decoding
- [ ] No regression on the unambiguous 22-fact demo

---

## Stage 33 — Full Evaluation Harness

**Goal:** Extend `BENCHMARKS.md` with generation-specific metrics so every
stage from 20 onward can be measured against a real baseline, not just
"does it compile and pass the demo."

### What to Build

- `manas eval --generation` — fluency, repetition rate, next-token accuracy
- `manas eval --reasoning` — multi-fact composition accuracy (Stage 26)
- `manas eval --context` — context-resolution accuracy (Stage 27)
- `manas eval --growth` — spurious growth rate, strategy accuracy (Stages 28, 31)
- `manas eval --compression` — regression rate under validated compression (Stage 30)
- `BENCHMARKS_V3.md`, generated the same way v2's `BENCHMARKS.md` was

### Tests

```rust
#[test]
fn eval_harness_reports_all_five_metric_categories() { }

#[test]
fn eval_numbers_are_reproducible_across_runs() { /* deterministic given seeded rand */ }
```

### Done When

- [ ] `BENCHMARKS_V3.md` exists with baseline numbers for every stage 20-32
- [ ] CI gate compares new runs against recorded baselines and fails on regression

---

## Principles (carried from v2, unchanged)

1. **Knowledge lives in weights** — `ask` never reads a text file to answer a taught fact.
2. **Never forget** — once a neuron or transformer weight is Frozen, it never changes.
3. **Grow when needed, not by default** — growth decisions are diagnosed, not automatic.
4. **Prove before building** — every stage has a test that must pass before the next begins.
5. **From scratch** — no Candle, no HuggingFace, no burn, no external ML framework.
6. **One file** — the entire brain, including transformer weights and vocab, lives in `brain.manas`.
7. **Honest claims** — Manas v3 is a research project exploring few-shot conditioned
   generation, not a ChatGPT replacement.
8. **Local first** — runs on any laptop CPU, no internet required for taught facts.

---

## Version History (v3 additions)

| Version | Stage | What it proves |
|---|---|---|
| v3.0.0-alpha | Stage 25 | Conditioned transformer generation replaces template realization |
| v3.1.0 | Stage 27 | Multi-fact reasoning and conversational context work together |
| v3.2.0 | Stage 30 | Growth and compression are diagnosed and validated, not fixed-threshold |
| v3.0.0 | All v3 stages | Stable Manas v3 release: few-shot conditioned generation with reasoning |
