# SELF_LEARNING.md — What Building Manas Teaches

> This file documents the research concepts, engineering lessons, and open problems
> that Manas is designed to explore. It exists so contributors and readers understand
> the learning journey behind the project — not just the code.

---

## The Core Research Question

> Can a neural network learn facts one at a time, store them in its own weights,
> and never forget them — running on a normal laptop CPU?

This is called **continual learning** or **lifelong learning** in academic research.
It is one of the hardest unsolved problems in AI. Large labs have not solved it
cleanly either — they just brute-force around it with massive retraining.

Manas is a practical, from-scratch Rust exploration of this problem.

---

## Concepts This Project Teaches

### 1. Associative Memory

The brain does not store facts as text. It stores them as patterns of activation
across neurons. When you think of "cat," a cluster of neurons activates — not a
sentence in a file.

Manas v2 tries to replicate this: when you teach it "cat is a small animal,"
the weights between neurons that respond to "cat" and neurons that respond to
"animal, small, fur" are strengthened. That connection is the knowledge.

**Key insight:** The network must learn to associate input patterns with output patterns,
not just predict the next word.

### 2. Catastrophic Forgetting

When a standard neural network learns something new (fact B), it overwrites
the weights that stored fact A. This is called catastrophic forgetting.

It happens because backpropagation distributes error signals across all weights
equally — it has no concept of "this weight is important, don't change it."

**Key insight:** Forgetting is a structural property of standard backprop.
Fixing it requires changing the update rule, not just training more carefully.

### 3. Elastic Weight Consolidation (EWC)

One research-backed solution to catastrophic forgetting. After learning task A,
compute how important each weight is to task A (using Fisher information matrix).
When learning task B, add a penalty for changing important weights:

```
total_loss = task_B_loss + λ × Σ importance_i × (w_i - w_i_old)²
```

Manas uses a simpler approximation of this idea through protection levels —
Frozen and Guarded neurons receive zero or clamped updates respectively.

**Paper:** Kirkpatrick et al., "Overcoming catastrophic forgetting in neural
networks" (2017). Read it to understand the research context.

### 4. Online Learning

Traditional ML trains on a fixed dataset in batches. Online learning updates
the model one sample at a time, in real time, as new data arrives.

Manas is an online learning system — every `teach` call updates the network
immediately. There is no "training run." There is no "dataset." There is just
continuous learning from experience.

**Challenge:** Online learning is more prone to instability and forgetting
than batch training. The protection system exists to counteract this.

### 5. Character N-Gram Tokenization

Most modern LLMs use Byte Pair Encoding (BPE) tokenization. Manas uses
character n-grams — a simpler approach that captures subword structure.

```
"cat"   → ["c", "ca", "cat", "#cat"]
"cats"  → ["c", "ca", "cat", "cats", "#cats"]
```

The shared n-grams ("c", "ca", "cat") mean the model automatically learns
that "cat" and "cats" are related. This is structural generalization —
learning from the shape of words, not just memorizing them.

### 6. Positional Embeddings

A naive embedding system averages token embeddings — making "cat eats dog"
identical to "dog eats cat." Positional embeddings encode order:

```
embed(token, position) = base_embed(token) + f(position)
```

Where `f(position)` uses sine/cosine functions to encode position uniquely.
This is the same idea used in the original Transformer paper (Vaswani et al., 2017).

### 7. Neural Network Growth

Standard neural networks have a fixed architecture. You choose the number of
layers and neurons before training and never change it.

Manas grows its architecture dynamically — adding neurons when it cannot
represent something well enough. This is related to research on:

- Progressive Neural Networks (Rusu et al., 2016)
- Dynamically Expandable Networks (Yoon et al., 2017)
- PackNet (Mallya & Lazebnik, 2018)

**Key challenge:** Growth must be controlled. Unconstrained growth leads to
a network that adds a new neuron for every single fact — exploding in size.

### 8. The `.manas` Binary Format

Designing a custom binary file format teaches:

- How to structure binary data with magic bytes and version fields
- How to implement CRC32 checksums for integrity verification
- How to design append-only formats for efficient growth
- How to handle forward and backward compatibility across versions

This is similar to how SQLite, WASM, and ELF binaries are structured.

---

## Open Problems Manas Is Exploring

### Problem 1: How do you know when to grow?

The current approach — grow when loss stays above threshold — is simple.
But the threshold is a hyperparameter. Too low and the network never grows.
Too high and it grows for every fact.

The ideal system would detect whether the existing network *can* represent
the new fact (capacity problem) versus just needing more training iterations
(optimization problem). These require different responses.

### Problem 2: How do you know which neurons to protect?

The importance scoring formula in Manas is a heuristic:

```
importance = 0.40 × freq + 0.30 × recency + 0.20 × magnitude + 0.10 × age_grace
```

Is this the right formula? What if a neuron was used heavily six months ago
but hasn't been touched since? Is it still important? The answer depends on
what the neuron represents — but the network doesn't know what it represents.

### Problem 3: How do you decode knowledge from weights?

Teaching "cat is a small animal" strengthens weight connections. But reading
those connections back as human language is non-trivial. The decoder in
`manas-learn` uses nearest-neighbor search in embedding space — finding
which known tokens are closest to the network's output vector.

This works for facts the network was explicitly taught. It is less clear
how it handles implicit reasoning or combining multiple facts.

### Problem 4: When does a growing network become too big?

A network that grows indefinitely will eventually become slow and unwieldy.
The compression system (Stage 15) is designed to remove low-importance neurons.
But the right compression threshold is unknown — compress too aggressively and
you lose knowledge, compress too conservatively and the brain bloats.

---

## Why This Matters

Modern AI requires:
- Thousands of GPUs consuming megawatts of power
- Millions of liters of water for cooling
- Billions of dollars in infrastructure
- Months of training time
- Retraining from scratch to learn anything new

If continual learning can be made to work reliably — even for small models —
it changes what AI can be. A model that learns continuously on your own laptop,
from your own data, without ever needing a datacenter, is a fundamentally
different kind of AI.

Manas is one small experiment in that direction.

---

## Recommended Reading

If you want to understand the research behind what Manas is building:

- **Kirkpatrick et al. (2017)** — "Overcoming catastrophic forgetting in neural networks" — the foundational paper on EWC
- **Rusu et al. (2016)** — "Progressive Neural Networks" — growing networks without forgetting
- **Vaswani et al. (2017)** — "Attention Is All You Need" — the Transformer paper, for positional embeddings
- **Yoon et al. (2017)** — "Lifelong Learning with Dynamically Expandable Networks"
- **McCloskey & Cohen (1989)** — "Catastrophic Interference in Connectionist Networks" — the original paper identifying the problem

---

*Manas is not trying to reproduce these papers exactly. It is trying to build
something practical that captures their core insights, from scratch, in Rust,
that runs on a laptop.*
