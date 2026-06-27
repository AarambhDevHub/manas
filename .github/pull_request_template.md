## Summary

<!-- What does this PR change? Which stage from ROADMAP.md does it relate to? -->

## Type of change

- [ ] Fix
- [ ] Feature
- [ ] Refactor
- [ ] Documentation
- [ ] Tests
- [ ] CI / tooling

## Stage checklist

Which ROADMAP.md stage does this PR belong to?

- [ ] I have read the relevant stage in ROADMAP.md
- [ ] The mandatory test(s) for this stage pass
- [ ] No feature from a later stage is included in this PR

## Core principles checklist

- [ ] `ask` still answers from neural weights (not a text file)
- [ ] Frozen neurons are not updated anywhere in this change
- [ ] No external ML framework dependency has been added
- [ ] The `.manas` binary format is the only persistence (no new sidecars)

## Testing

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo test --workspace`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo build --workspace --release`
- [ ] Stage mandatory test passes (paste output below)

## Stage mandatory test output

```
paste the output of the relevant stage test here
```

## Notes

<!-- Any extra context, limitations, or follow-up work. -->
