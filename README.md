# Manas

> *मनस् (manas) — mind, intellect, the seat of thought*

**Manas is an experimental self-growing local AI brain written from scratch in Rust.**

It starts with zero knowledge. It learns one fact at a time. It stores that knowledge
directly inside neural network weights. It never forgets what it learned — even as it
keeps learning new things. It runs entirely on your laptop CPU. No cloud. No GPU.
No external ML framework. Everything lives in one `.manas` file.

> **Experimental Project Notice**
>
> Manas v2 is a research project exploring continual learning, associative memory,
> dynamic neural growth, and anti-forgetting systems — all built from scratch in Rust.
>
> It is not a production-ready AI system. It is not a replacement for large language
> models. It is an honest attempt to answer one question: can a neural network learn
> facts continuously, store them in its own weights, and never forget them — running
> on anyone's laptop?
>
> Every milestone is tested before the next begins. Every claim is verified by code.

---

## The Problem with Current AI

| Problem | How It Is Today | How Manas Approaches It |
|---|---|---|
| Learning new facts | Retrain the whole model on a massive dataset | Learn one fact at a time, online |
| Forgetting old facts | Accepted as normal (catastrophic forgetting) | Protection system prevents it |
| Fixed model size | Parameter count frozen forever at training | Network grows new neurons as needed |
| Datacenter dependency | Requires GPU clusters and water cooling | Runs on any laptop CPU |
| Knowledge cutoff | Hard cutoff date, stale forever | Teach it new facts anytime |
| Ownership | Your data goes to a cloud API | Everything stays on your machine |

---

## What Makes This Different

Most "local AI" tools are wrappers around a pre-trained model that was trained in a
datacenter. They store your facts in a text file and search that file when you ask
a question. The neural network is decoration.

Manas v2 is built around one principle: **knowledge lives in the weights.**

```bash
./manas teach "A cat is a small domesticated animal with fur and whiskers."
./manas teach "The Eiffel Tower is located in Paris France."
# ... teach 20 more facts ...

# delete all text files — weights only
rm -f brain.manas.sources brain.manas.sourceindex

# ask — answered from neural weights, not a text file
./manas ask "What is a cat?"
# Answer: small domesticated animal fur whiskers
# Answered from: neural weights ✅
```

This is the test v1 failed. v2 is built to pass it.

---

## How It Works

### A Brain That Starts Empty

```
Day 1:   zero neurons, zero knowledge
Day 1:   teach "A cat is a small animal"
         → network grows neurons to represent this
         → weights now encode: cat → animal, small, fur
Day 2:   teach 50 more facts
         → new neurons grow for new knowledge
         → old neurons are protected — cat knowledge survives
Day 7:   ask "What is a cat?"
         → forward pass through neural weights
         → answer: small animal with fur
         → no text file involved
```

### The Anti-Forgetting System

Every neuron has a protection level:

```
Open    → newly created, full weight updates allowed
Guarded → frequently used, only tiny updates allowed
Frozen  → important knowledge, zero updates ever
```

When a neuron is `Frozen`, no future learning can change it. Old knowledge is preserved
by design, not by accident.

### The Growth System

The network grows exactly when needed:

```
teach new fact
  → run forward pass
  → loss too high? (network can't represent this yet)
      → try updating existing Open neurons
      → still too high? → grow a new neuron
  → loss is fine? → update existing neurons normally
```

The brain starts at ~1 KB and grows one neuron at a time.

---

## Current Status

Manas v2 is in active development. The roadmap follows a strict rule:
**every stage has a test that must pass before the next stage begins.**

| Stage | Name | Status |
|---|---|---|
| Stage 0 | Workspace and foundation | Complete |
| Stage 1 | Associative memory proof | Complete |
| Stage 2 | Anti-forgetting proof | Complete |
| Stage 3 | Crate structure | Complete |
| Stage 4 | `.manas` binary format | Complete |
| Stage 5 | Character n-gram tokenizer | Complete |
| Stage 6 | Positional embeddings | Complete |
| Stage 7 | Growth system | Complete |
| Stage 8 | Protection system hardened | Complete |
| Stage 9 | `manas teach` and `manas ask` | Next |
| Stage 10+ | File ingestion, inspect, benchmarks, layer growth | Planned |

Stages 1 and 2 are preserved as a standalone proof in
`manas-core/src/experiment.rs`. Stage 3 promotes the proven engine into
maintained crate modules in `manas-core` and `manas-learn`, with the
anti-forgetting gate covered by `manas-learn/tests/anti_forgetting.rs`.
Stage 4 persists that network in one `.manas` binary file through
`manas-store`; its vocab section is intentionally empty until the tokenizer
stages add real vocab persistence. Stage 5 replaces the word-hash encoder path
with a character prefix n-gram tokenizer. Stage 6 adds positional embeddings so
the encoder preserves token order while keeping the same fixed-width vector
interface and anti-forgetting proof. Stage 7 adds bounded hidden-neuron growth
so a brain can start empty and gain capacity only when learning needs it. Stage 8
hardens protection so frozen neurons cannot change, guarded neurons are clamped,
protection survives save/load, and frequently updated neurons promote
automatically during learning.

Run the proof:

```bash
rustc --edition=2024 -O -D warnings manas-core/src/experiment.rs -o /tmp/manas-stage2
/tmp/manas-stage2
```

Run the standalone tests:

```bash
rustc --edition=2024 --test -D warnings manas-core/src/experiment.rs -o /tmp/manas-stage2-tests
/tmp/manas-stage2-tests
```

Run the maintained crate proof:

```bash
cargo test -p manas-learn anti_forgetting
```

Run the persistence proof:

```bash
cargo test -p manas-store
```

Run the tokenizer proof:

```bash
cargo test -p manas-learn tokenizer
```

Run the positional embedding proof:

```bash
cargo test -p manas-learn embedder
```

Run the growth proof:

```bash
cargo test -p manas-core growth
cargo test -p manas-learn growth
```

Run the protection proof:

```bash
cargo test -p manas-core protection
cargo test -p manas-learn protection
cargo test -p manas-store protection
```

See [ROADMAP.md](./ROADMAP.md) for the full plan with tests.
See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full design.

---

## Build from Source

Manas v2 has no release binaries yet. Build from source:

```bash
# install Rust if you haven't
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# clone and build
git clone https://github.com/AarambhDevHub/manas.git
cd manas
cargo build --workspace --release

# run
./target/release/manas --help
```

**Requirements:**
- Rust 1.75 or later
- No GPU required
- No external ML library required
- Works on Linux, macOS, Windows

---

## Architecture

Manas v2 is built from 5 Rust crates, each with a single responsibility:

```
┌──────────────────────────────────────────┐
│              manas-cli                   │
│   teach | ask | inspect | neurons        │
│   forget | reset                         │
└───────────────────┬──────────────────────┘
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
┌─────────┐  ┌───────────┐  ┌───────────┐
│ manas-  │  │  manas-   │  │  manas-   │
│ ingest  │  │  learn    │  │  store    │
│         │  │           │  │           │
│ text    │  │ tokenizer │  │ .manas    │
│ files   │  │ embedder  │  │ binary    │
│ folders │  │ backprop  │  │ format    │
│ formats │  │ trainer   │  │ CRC32     │
└────┬────┘  └─────┬─────┘  └───────────┘
     │              │
     └──────┬───────┘
            ▼
┌──────────────────────────────────────────┐
│              manas-core                  │
│                                          │
│  Neuron   Layer   Network                │
│  ProtectionLevel  Source                 │
│  forward()  grow_neuron()                │
│  apply_gradients()  ← anti-forgetting    │
│  enforced HERE, structurally             │
└──────────────────────────────────────────┘
            │
            ▼
     [ brain.manas ]
     one file, everything inside
     starts: ~1 KB
     grows: one neuron at a time
```

**No `manas-language` crate.** No `manas-agent` crate. No `manas-memory` crate.
The transformer path from v1 is removed. The text sidecar from v1 is removed.
The answering system from v1 is replaced with direct neural weight retrieval.

---

## The `.manas` File Format

Everything lives in one binary file. No sidecars. No `.sources`. No `.sourceindex`.

```
[ MAGIC: MANS ] [ VERSION: 2 ] [ HEADER ]
[ VOCAB SECTION ] [ LAYER + NEURON SECTION ]
[ CRC32 CHECKSUM ]
```

Every neuron stores:
- Weights and bias
- Protection level (Open / Guarded / Frozen)
- Importance score
- Born timestamp and last activated timestamp
- Source (raw text or local file path)
- Freshness category (Timeless / Slow / Fast / Realtime)

The entire brain — weights, vocab, metadata — is in this one file.
`ask` reads the weights directly. No text search. No keyword matching.

---

## Crate Structure

```
manas/
├── Cargo.toml              ← workspace
├── ARCHITECTURE.md
├── ROADMAP.md
├── README.md
│
├── manas-core/             ← neurons, layers, growth, anti-forgetting
│   └── src/
│       ├── activation.rs
│       ├── neuron.rs
│       ├── layer.rs
│       ├── network.rs
│       └── error.rs
│
├── manas-store/            ← .manas binary format, read/write, CRC32
│   └── src/
│       ├── format.rs
│       ├── writer.rs
│       ├── reader.rs
│       ├── patcher.rs
│       └── integrity.rs
│
├── manas-learn/            ← tokenizer, embedder, backprop, trainer
│   └── src/
│       ├── tokenizer.rs
│       ├── embedder.rs
│       ├── encoder.rs
│       ├── decoder.rs
│       ├── backprop.rs
│       ├── trainer.rs
│       └── importance.rs
│
├── manas-ingest/           ← text, files, folders → clean chunks
│   └── src/
│       ├── chunker.rs
│       ├── normalizer.rs
│       ├── file_reader.rs
│       ├── folder_walker.rs
│       └── format/
│
└── manas-cli/              ← user commands, thin layer only
    └── src/
        ├── main.rs
        └── commands/
```

---

## What Manas Is and Is Not

| Manas IS | Manas IS NOT |
|---|---|
| A local associative memory system | A ChatGPT replacement |
| A self-growing neural network | A general-purpose LLM |
| A continual learning research project | A production AI system |
| Built from scratch in Rust | A wrapper around Candle or HuggingFace |
| Knowledge stored in neural weights | A text search engine |
| Runs on your laptop CPU | Requires a GPU or cloud |
| Answers from what it was taught | Answers from the internet by default |
| One `.manas` file | Multiple sidecar files |

---

## Why From Scratch

Manas uses no external ML framework — no Candle, no HuggingFace, no burn, no tch.

The reason is simple: what Manas is building does not exist in any framework.

- No framework has a "grow a new neuron when loss is too high" API
- No framework has per-neuron protection levels built into `apply_gradients()`
- No framework has a freshness system that tracks knowledge staleness at the neuron level
- No framework stores a complete growing brain in a custom binary format with CRC32 integrity

These things have to be built from scratch because they are the invention.
The matrix multiplication underneath them is 200 lines of Rust. That is the easy part.

---

## Relation to Aarambh-AI

Manas and [Aarambh-AI](https://github.com/AarambhDevHub/aarambh-ai) are different projects
with different goals:

| | Manas | Aarambh-AI |
|---|---|---|
| Goal | Local growing brain | Decoder-only transformer LLM |
| Architecture | Associative memory + growth | Standard transformer |
| Backend | From scratch, pure Rust | HuggingFace Candle |
| Scale | Starts at 0 neurons | 25M–1.3B parameters |
| Training | Online, one fact at a time | Large dataset, GPU |
| Use case | Personal local knowledge | Language modeling |

They are not competing. Aarambh-AI is the big model you download.
Manas is the personal brain that learns from your own data locally.

---

## Contributing

Manas is an open experiment. If you are interested in:

- Continual learning and anti-forgetting research
- Building neural networks from scratch in Rust
- Local-first AI systems
- The growing brain concept

You are welcome to follow along, open issues, or contribute.

Please read [ARCHITECTURE.md](./ARCHITECTURE.md) and [ROADMAP.md](./ROADMAP.md)
before contributing. Every stage has mandatory tests. No feature is accepted
until the test for its stage passes.

---

## Support

If this project interests you, you can support it via:

- [GitHub Sponsors](https://github.com/sponsors/aarambh-darshan)
- [Buy Me a Coffee](https://buymeacoffee.com/aarambhdevhub)

---

## License

MIT — see [LICENSE](./LICENSE)

---

*Manas v2 — built by [Aarambh Dev Hub](https://github.com/AarambhDevHub)*
*A self-growing brain. From scratch. In Rust. On your laptop.*
