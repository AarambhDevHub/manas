# Changelog

All notable changes to Manas will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Manas uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Completed

- Stage 0 — Workspace and foundation.
- Stage 1 — Associative memory proof.
- Stage 2 — Anti-forgetting proof.
- Stage 3 — Crate structure.
- Stage 4 — Persistence: the `.manas` binary format.

### Added

- Standalone Stage 1 experiment at `manas-core/src/experiment.rs`.
- Deterministic proof that `cat`, `paris`, and `rust` associations can be learned
  into neural weights and retrieved by cosine similarity across five fixed seeds.
- Protection levels in the standalone experiment: `Open`, `Guarded`, and `Frozen`.
- Deterministic proof that five anchor facts survive after learning 50 unrelated
  facts, while new facts also retrieve above the required similarity threshold.
- Promoted the proven Stage 1 and Stage 2 engine into maintained crates:
  `manas-core` now owns activations, neurons, layers, networks, protection, and
  errors; `manas-learn` now owns deterministic encoding, backpropagation,
  training, and fixed anti-forgetting fixtures.
- Added the `manas-learn` integration test for the Stage 3 anti-forgetting gate.
- Added `manas-store::ManasBrain` with std-only `.manas` save/load support for
  the Stage 3 network state.
- Added CRC32 verification, magic/version validation, protection metadata
  persistence, and Stage 4 persistence integration tests.

### Next

- Stage 5 — Character n-gram tokenizer.

---

<!-- Releases will be added here as stages complete -->

<!--

## [0.1.0] — TBD

### Added
- Associative memory engine — knowledge stored in neural weights directly
- Anti-forgetting system — protection levels enforced inside apply_gradients()
- Character n-gram tokenizer — structural similarity between related words
- Positional embeddings — word order matters
- Growth system — network grows new neurons when loss stays above threshold
- .manas binary format v2 — one file, CRC32 checksum, no sidecars
- manas teach — teach raw text, files, and folders
- manas ask — answered from neural weights directly
- manas inspect — full brain state visibility

### Removed
- All v1 sidecar files (brain.manas.sources, brain.manas.sourceindex,
  brain.manas.seq, brain.manas.transformer, brain.manas.langmeta)
- Text-file-based answering system from v1
- manas-language crate (transformer path)
- manas-agent crate
- manas-memory crate

-->

---

*See [ROADMAP.md](./ROADMAP.md) for planned upcoming changes.*
*See [manas-v1-archive](https://github.com/AarambhDevHub/manas-v1-archive) for v1 history.*
