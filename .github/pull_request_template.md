## Summary

<!-- What does this change do, and why? -->

## Linked issue

Closes #

## Neuromorphic relevance

<!-- Required for any substantial new public API, critic, signal, or behavior.
     See CONTRIBUTING.md#neuromorphic-relevance for the full gate. -->

- [ ] Names the neuromorphic/SNN mechanism this represents or supports
- [ ] Identifies the modulator / plasticity-control signal it affects
- [ ] Belongs in `limbic-critic` rather than an app, orchestration, generic-ML, or RL layer
- [ ] Preserves the `Environment` → critic → `ModulatorVector` output contract
- [ ] Stays independent of sibling Cargo dependencies (no `neuromod`, `plasticity-lab`, etc.)
- [ ] N/A — this is not a substantial new API, critic, signal, or behavior (e.g. docs, chore, CI)

## Verification

- [ ] `cargo fmt --check`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo test --all-features`
- [ ] `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`
- [ ] `cargo build --all-features`
