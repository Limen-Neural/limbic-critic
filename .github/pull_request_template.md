## Summary

<!-- What does this change do, and why? -->

## Linked issue

Closes #

## Neuromorphic relevance

<!-- See CONTRIBUTING.md#neuromorphic-relevance for the full gate. -->

Pick exactly ONE of the two blocks below.

- [ ] **Not substantial** — this is docs, chore, CI, tests, or a refactor that adds
      no new public API, critic, signal, or behavior. Say which in one line:

  >

- [ ] **Substantial** — this adds or changes a public API, critic, signal, or
      behavior. Answer all five in prose; a checkbox alone is not an answer.

  1. What neuromorphic or SNN mechanism does this represent or support?

     >

  2. Which modulator, plasticity-control signal, or spiking-system behavior does it affect?

     >

  3. Why does this belong in `limbic-critic` rather than an application,
     orchestration layer, generic ML crate, or RL framework?

     >

  4. How does it preserve the `Environment` → critic → `ModulatorVector` contract?

     >

  5. How does it stay independent of sibling Cargo dependencies
     (no `neuromod`, `plasticity-lab`, etc.)?

     >

## Verification

- [ ] `cargo fmt --check`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo test --all-features`
- [ ] `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`
- [ ] `cargo build --all-features`
- [ ] `cargo package --list --allow-dirty` (the `package` CI job also asserts contents)
