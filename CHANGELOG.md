# Changelog

All notable changes to Manas will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Manas uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Completed

- Stage 0 — Workspace and foundation.
- Stage 1 — Associative memory proof.

### Added

- Standalone Stage 1 experiment at `manas-core/src/experiment.rs`.
- Deterministic proof that `cat`, `paris`, and `rust` associations can be learned
  into neural weights and retrieved by cosine similarity across five fixed seeds.

### Next

- Stage 2 — Anti-forgetting proof.

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
