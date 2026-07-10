# SELF_LEARNING.md — What Building Manas v3 Teaches

> This file extends v2's SELF_LEARNING.md. It documents the research concepts
> introduced in v3 and — importantly — reports what happened to v2's four
> open problems. Some are solved. Some are only partially solved. That's
> reported honestly below, not glossed over.

---

## The v3 Research Question

> Can a tiny transformer, conditioned on facts stored in an associative memory,
> generate fluent language from a handful of examples — instead of the millions
> of examples standard transformers require?

This is the core bet of v3. Standard transformers are data-hungry because they
learn language structure *and* world knowledge simultaneously, from raw text,
via attention alone. Manas v3 separates these two jobs: the associative memory
(v2, unchanged) supplies *what* to say; the transformer only has to learn *how*
to phrase it, conditioned on a concept vector it doesn't have to derive itself.

---

## New Concepts This Version Teaches

### 1. Conditioned Generation (Cross-Attention on a Concept Vector)

Standard decoder-only transformers predict the next token from the sequence
alone. Manas v3's decoder cross-attends to a **fixed concept vector** supplied
by the associative memory at every layer, every generated token.

```
standard:   logits = decoder(tokens)
conditioned: logits = decoder(tokens, cross_attend_to = concept_vector)
```

**Key insight:** this is the same architectural trick used by image captioning
models (condition a decoder on a fixed image feature vector). It works with
small data because the decoder is not responsible for *knowing* the answer —
only for producing grammatically coherent English around it.

### 2. Teacher-Forced Online Next-Token Training

Every `teach` call trains the transformer immediately on that one sentence,
same as v2's online associative learning — there is no separate "training
run" or held-out dataset.

**Challenge:** with so little data per fact, the transformer must rely on
*cross-fact* regularities (sentence templates like "X is located in Y," "X
was created by Y") to generalize phrasing. Early facts effectively teach
grammar; later facts mostly teach content, since the phrasing patterns are
already partially learned.

### 3. Weight Tying

Tying the input embedding matrix to the output projection matrix halves the
parameters the transformer needs to learn — critical when training data is a
few dozen sentences, not billions of tokens. This is standard practice in
larger LLMs (GPT-2 onward) but matters proportionally more here, where every
parameter has to earn its keep from very few examples.

### 4. Anti-Forgetting Extended Beyond Neurons

v2's Frozen/Guarded/Open protection levels were designed for associative
memory neurons. v3 discovered the same catastrophic forgetting problem exists
in the transformer's weights: teaching fact #23 can measurably degrade the
fluency of fact #1's generated answer if the transformer's weights are not
also protected.

**Key insight:** catastrophic forgetting is not a property of a specific
architecture (associative memory vs transformer) — it is a property of
gradient-based learning in general. Any component trained via backprop on a
stream of examples needs an explicit protection mechanism, or it will forget.

### 5. Multi-Fact Composition

v2 could only answer with one bound fact per query. v3's `query_multi`
retrieves the top-k activated concept vectors and merges them before
generation, so a query touching two facts produces one answer referencing
both — rather than picking one fact arbitrarily or failing.

**Key insight:** composition in v3.0 is a simple weighted merge, not a learned
reasoning step. It is a proof that multi-fact retrieval is possible at all,
not a claim that Manas can perform novel inference across facts it wasn't
taught together. That distinction matters — see Open Problem 5 below.

### 6. Conversational Context as Ephemeral State

Unlike taught facts, conversational history is deliberately **not** persisted
into `.manas`. It exists only for the duration of a session. This is a
design choice, not an oversight: conflating "what Manas knows" with "what was
recently discussed" would blur the project's central claim that knowledge
lives in weights, not in transient state.

### 7. Diagnosing Growth Triggers: Plateau vs Ceiling

v2 grew the network whenever loss stayed above a fixed threshold — a single
signal doing two jobs. v3 distinguishes:

- **Plateau** — loss is not decreasing right now, but the network likely has
  enough capacity; more training iterations should resolve it.
- **Ceiling** — loss is stable at a high value regardless of iteration count;
  the network genuinely lacks the capacity to represent this fact.

**Key insight:** these require opposite responses. Growing on a plateau wastes
capacity (adds neurons the network didn't need). Not growing on a ceiling
wastes time (retrains forever without improving). v2 could not tell these
apart; v3's diagnosis is a heuristic on loss *trajectory* rather than loss
*value* alone — and it is still a heuristic, not a solved problem (see below).

### 8. Validated Compression (Compress, Then Verify, Then Commit)

v2's compression (Stage 15) removed low-importance neurons at a fixed
threshold, with no feedback loop. v3 makes compression provisional: compress,
check that every previously-taught fact still recalls correctly, and only
then commit — otherwise roll back to the pre-compression snapshot.

**Key insight:** this turns an open, unmeasurable question ("is this threshold
safe?") into a per-compression, empirically checked one. It doesn't require
knowing the right threshold in advance — it requires only that regressions
are detected and cheap to undo.

---

## Status of v2's Open Problems

### Problem 1: How do you know when to grow? — **Partially solved**

Stage 28's Plateau/Ceiling diagnosis is a real improvement over a single fixed
threshold, but it is still a heuristic on loss trajectory, tuned by hand. It
does not know *why* the network is stuck — only whether more iterations have
been helping recently. A genuinely principled answer (e.g. estimating true
representational capacity directly) remains open.

### Problem 2: How do you know which neurons to protect? — **Partially solved**

Stage 29 replaces the fixed 0.40/0.30/0.20/0.10 formula with weights learned
from actual compression outcomes. This closes the loop the v2 formula lacked
— but the scorer still only has four input signals (freq, recency, magnitude,
age) to work with. It may be that importance depends on signals not captured
in this feature set at all (e.g. how *entangled* a neuron's representation is
with others). That richer notion of importance is still unexplored.

### Problem 3: How do you decode knowledge from weights? — **Extended, not solved**

v2's problem was decoding a single fact from nearest-neighbor search in
embedding space. v3's conditioned transformer and multi-fact composition
(Stages 22, 26) extend this to fluent, multi-fact decoding — but the harder
version of this problem remains: **implicit reasoning**. If Manas is taught
"Paris is the capital of France" and separately "France is in Europe," can it
answer "Is the capital of France in Europe?" without being taught that
specific composition? v3's `compose()` merges *retrieved* facts; it does not
perform novel inference across facts that were never activated together
during training. This is explicitly out of scope for v3 and is the most
likely candidate for v4's research question.

### Problem 4: When does a growing network become too big? — **Solved for safety, not for optimality**

Stage 30's validated compression guarantees compression never silently breaks
recall — that part of the problem (safety) is solved. It does not tell you
the *optimal* compression threshold, only a *safe* one at whatever threshold
you choose to attempt. A network could still be larger than necessary if the
chosen threshold is too conservative; validated compression protects against
harm, not against inefficiency.

---

## New Open Problems v3 Introduces

### Problem 5: Composition vs Inference

As noted above — v3 can retrieve and phrase multiple *separately taught*
facts together, but cannot infer a new fact from combining two unrelated
taught facts. This is the single biggest gap between v3 and "reasoning" in
the fuller sense of the word.

### Problem 6: How much does phrasing quality depend on fact ordering?

Because the transformer learns phrasing incrementally as facts are taught,
the order in which facts are taught may affect how well early phrasing
patterns generalize to later facts. v3 does not yet measure or control for
this — `manas eval --generation` (Stage 33) should eventually be run with
multiple teaching orders to check for order-sensitivity.

### Problem 7: Where is the ceiling of "small data"?

22 facts is proof of concept. It is not yet known how this conditioned
architecture behaves at 200 facts, or 2,000 — does the transformer's
generalization from phrasing patterns hold, or does it start requiring
proportionally more data per fact as vocabulary and topic diversity grow?
This is an empirical question for whichever eval work follows Stage 33.

---

## Why This Matters (extended from v2)

v2 showed that a small model can remember without a datacenter. v3's bet is
narrower but pointed at a real gap in how people think about small models:
that fluent generation *requires* massive pretraining. If a transformer
conditioned on structured, few-shot associative memory can produce coherent
language from dozens of examples rather than billions of tokens, it suggests
fluency and world-knowledge acquisition are more separable than the standard
LLM training recipe treats them — and that separability might be exactly what
makes continual, on-device learning practical.

---

## Recommended Reading (v3 additions)

- **Vinyals et al. (2015)** — "Show and Tell: A Neural Image Caption Generator"
  — the conditioning-on-a-fixed-vector pattern v3's cross-attention borrows from
- **Vaswani et al. (2017)** — "Attention Is All You Need" — still the reference
  for the attention mechanism itself, now applied at much smaller scale
- **Press & Wolf (2017)** — "Using the Output Embedding to Improve Language
  Models" — the weight-tying technique used in Stage 23
- **Kirkpatrick et al. (2017)** — as in v2, now additionally relevant to
  Stage 24's transformer-weight protection
- **Lake et al. (2017)** — "Building Machines That Learn and Think Like
  People" — relevant framing for Problem 5 (composition vs genuine inference)

---

*Manas v3 is still not trying to reproduce a modern LLM. It is trying to find
out how much of "fluent, few-shot, continually-learned generation" is
reachable from scratch, in Rust, on a laptop, once generation is conditioned
on a memory that already knows the answer.*
