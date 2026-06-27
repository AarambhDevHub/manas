# Contributing to Manas

> *मनस् (manas) — mind, intellect, the seat of thought*

Thank you for your interest in contributing to Manas. This is an experimental
research project exploring continual learning, associative memory, and self-growing
neural networks built from scratch in Rust.

Please read this document fully before opening a PR or issue.

---

## Before You Contribute

### Read the architecture first

Every contribution must align with the core design:

- [ARCHITECTURE.md](./ARCHITECTURE.md) — how the system works
- [ROADMAP.md](./ROADMAP.md) — what is being built and in what order

Manas has seven core principles (from ARCHITECTURE.md Section 3). No PR
may violate any of them:

1. **Knowledge lives in weights** — `ask` never reads a text file to answer a taught fact
2. **Never forget** — once a neuron is Frozen, its weights never change
3. **Grow when needed** — new neurons only when loss stays above threshold
4. **Prove before building** — every stage has a test that must pass first
5. **From scratch** — no Candle, HuggingFace, burn, tch, or external ML framework
6. **One file** — the brain lives in `brain.manas`, no new sidecars
7. **Honest claims** — Manas is a research project, not a ChatGPT replacement

If your contribution conflicts with any of these, it will not be merged.

### Check the current stage

The roadmap is strictly sequential. Check which stage is currently active before
contributing a feature. A feature from Stage 10 will not be accepted if Stage 7
is not yet complete.

---

## Types of Contributions

### What is welcome

- Bug fixes with a test that reproduces the bug
- Tests that improve coverage for the current active stage
- Documentation improvements (typos, clarity, accuracy)
- Performance improvements that do not change behavior
- Features from the **current active stage** in ROADMAP.md

### What is not welcome (right now)

- Features from future stages before the current stage is complete
- External ML framework dependencies (Candle, burn, tch, HuggingFace, etc.)
- New sidecar files (everything must stay in `brain.manas`)
- Language generation features before Stage 13 is proven stable
- Internet agent features before Stage 13 is proven stable

---

## Development Setup

```bash
# clone
git clone https://github.com/AarambhDevHub/manas.git
cd manas

# build
cargo build --workspace

# test
cargo test --workspace

# format
cargo fmt --all

# lint
cargo clippy --workspace --all-targets -- -D warnings
```

**Requirements:**
- Rust stable (latest)
- No GPU required
- No external ML library required

---

## Making a Change

### 1. Open an issue first

For anything beyond a small fix, open an issue before writing code.
This avoids wasted effort if the direction doesn't align with the roadmap.

### 2. One PR per stage item

Keep PRs small and focused. One logical change per PR.
Don't combine a bug fix with a new feature.

### 3. Every PR needs a test

No untested code is accepted. If you fix a bug, add a test that would
have caught it. If you add a feature, add a test that proves it works.

### 4. Run the full check before opening a PR

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace --release
./target/release/manas --help
```

All must pass with zero errors and zero warnings.

### 5. Include the stage mandatory test output

Every stage in ROADMAP.md has a mandatory test. If your PR relates to a stage,
paste the passing output of that test in your PR description.

---

## Commit Style

Use this format:

```
type(scope): short description

Longer explanation if needed.
```

Types:
- `fix` — bug fix
- `feat` — new feature
- `test` — adding or improving tests
- `refactor` — restructuring without behavior change
- `docs` — documentation only
- `ci` — CI or tooling changes
- `chore` — maintenance

Examples:
```
fix(manas-core): frozen neuron weights changing under high LR
feat(manas-learn): character n-gram tokenizer with max_ngram=4
test(manas-store): add checksum mismatch detection test
docs(ARCHITECTURE): clarify positional embedding formula
```

---

## Code Style

- Follow standard Rust idioms
- No `.unwrap()` in library code — use `Result` and `ManasError`
- No `println!` in library code — only in `manas-cli`
- Keep functions small and focused
- Write doc comments on all public types and functions
- `cargo fmt` and `cargo clippy` must pass before any PR

---

## Anti-Forgetting Rule

This is the most important rule in the codebase:

**`apply_gradients()` in `manas-core` is the single enforcement point for protection levels.**

Never bypass it. Never add a second path that updates neuron weights directly.
If you find yourself updating `neuron.weights` anywhere outside of `apply_gradients()`,
stop — that is a bug.

---

## License

By contributing to Manas, you agree that your contributions will be licensed
under the same dual MIT / Apache 2.0 license as the project.

See [LICENSE-MIT](./LICENSE-MIT) and [LICENSE-APACHE](./LICENSE-APACHE).
