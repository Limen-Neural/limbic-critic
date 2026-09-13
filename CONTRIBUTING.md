# Contributing to limbic-critic

`limbic-critic` is a reward-shaping and modulator-mapping primitive for SNNs:
it maps an environment objective into a local `ModulatorVector`. It is
intentionally **not** a full actor–critic, learned value function, RL
training framework, or generic ML library — see [README.md](README.md) and
[`docs/BOUNDARY_MATRIX.md`](docs/BOUNDARY_MATRIX.md) for the crate's mission
and ownership boundaries, and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
for the full scope contract.

That narrow scope is deliberate. It keeps the crate small, decoupled from
sibling SNN crates, and easy to reuse. The tradeoff is that scope drift is
easy to miss: generic reward-shaping or RL features can look superficially
relevant even when they don't belong here. The gate below exists to make
that judgment call explicit and reviewable rather than a matter of taste.

## Neuromorphic relevance

For any substantial new public API, critic, signal, or behavior, the
PR or issue should answer these five questions:

1. **What neuromorphic or SNN mechanism does this represent or support?**
2. **Which modulator, plasticity-control signal, or spiking-system behavior does it affect?**
3. **Why does this belong in `limbic-critic` instead of an application, orchestration layer, generic ML crate, or RL framework?**
4. **Does the feature preserve the crate's output-oriented contract (`Environment` → critic → `ModulatorVector` or a clearly related neuromodulatory primitive)?**
5. **Can the feature remain independent of sibling Cargo dependencies?**

If a proposal cannot answer these questions convincingly, it should be
redirected to a more appropriate repo (an application crate, an orchestration
layer, or a sibling crate under [Limen-Neural](https://github.com/Limen-Neural))
rather than merged here.

### Belongs here

- A critic that maps prediction surprise into acetylcholine-like modulation for plasticity
- A bounded stress signal intended to gate SNN adaptation
- A reward-delta transformation whose output is consumed by reward-modulated plasticity
- Validation / normalization logic required to keep modulator outputs meaningful

### Does not belong here

- PPO / DQN / SAC implementations
- Policy/value networks
- Replay buffers or generic episode runners
- ANN training loops
- Game/mining/trading-specific reward calculators
- Hardware or deployment supervisors

## Development workflow

Run these checks locally before opening a PR — they match what CI runs:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo build --all-features
```

CI runs two more checks that are easy to miss locally:

```bash
# `package` job: the published tarball must contain only intended artifacts.
# CI additionally asserts the contents, so an accidental dev file fails the build.
cargo package --list --allow-dirty
cargo package --allow-dirty

# Coverage job (a single nextest pass through llvm-cov):
cargo llvm-cov nextest --all-features --profile ci --lcov --output-path lcov.info
```

If you add a development-only file, check whether it belongs in `package.exclude`
in `Cargo.toml` so it does not ship to crates.io.

**MSRV:** Rust 1.98.1. `Cargo.toml` `rust-version`, `rust-toolchain.toml`,
and both CI workflows (`ci.yml`, `coverage.yml`) are pinned to this version
in lockstep — CI actively fails if any of them disagree, so update all four
together if the MSRV ever changes.

## Licensing

This project is dual-licensed under MIT OR Apache-2.0 and is
[REUSE](https://reuse.software/)-compliant (see [`REUSE.toml`](REUSE.toml)).
New Rust source files need an SPDX header:

```rust
// SPDX-License-Identifier: MIT OR Apache-2.0
```

## Further reading

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — scope contract
- [`docs/BOUNDARY_MATRIX.md`](docs/BOUNDARY_MATRIX.md) — ownership boundary matrix
