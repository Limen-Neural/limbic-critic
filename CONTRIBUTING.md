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

# no_std surface (RM-2068 / GH#89) — CI's `no_std` job runs these and also
# cross-builds thumbv7em-none-eabihf (`rustup target add thumbv7em-none-eabihf`):
cargo build --no-default-features --features libm
cargo clippy --no-default-features --features libm -- -D warnings
cargo test --no-default-features --features libm
```

CI runs three more checks that are easy to miss locally:

```bash
# `package` job: the published tarball must contain only intended artifacts.
# CI additionally asserts the contents, so an accidental dev file fails the build.
cargo package --list --allow-dirty
cargo package --allow-dirty

# Coverage job (a single nextest pass through llvm-cov):
cargo llvm-cov nextest --all-features --profile ci --lcov --output-path lcov.info

# `Audit` workflow (also weekly): supply-chain policy in deny.toml —
# RustSec advisories, MIT/Apache-2.0 licenses, crates.io-only sources.
cargo deny --all-features --locked check
```

`Cargo.lock` is tracked. After changing dependencies, commit the updated
lockfile; CI's `package` and `Audit` jobs run with `--locked` and fail on a
stale one. A new dependency whose license or source is outside `deny.toml`
needs that policy widened in the same, reviewed PR.

If you add a development-only file, check whether it belongs in `package.exclude`
in `Cargo.toml` so it does not ship to crates.io.

**MSRV:** Rust 1.85. This is `Cargo.toml` `rust-version`; CI's `msrv` job
builds and tests on it, and CI fails if README.md or this file disagree with it.
The dev toolchain (`rust-toolchain.toml`, `ci.yml`, `coverage.yml`, currently
1.99.0) is separate and must stay identical across those three files.

CI's **Build & Test** job runs on `ubuntu-latest`, `macos-latest`, and
`windows-latest` with `fail-fast: false`. `cargo fmt --check` and `cargo doc`
run on Linux only; the `package` job stays on `ubuntu-latest`.

## Public API semver gate

CI's `semver-checks` job runs on pull requests only. It uses
`cargo semver-checks` to diff the proposed public API against the latest
normal (not yanked or pre-release) crates.io release no newer than the
version in the checked-out `Cargo.toml`, so no API baseline file is
maintained in this repository. After a version bump is published, the
registry baseline advances automatically for subsequent PRs.

The job checks two semver surfaces explicitly: the default `std` feature
set and the maximal `--all-features` surface (currently `std` + `libm`).
These are not every build permutation. In particular, `--all-features` is
not the embedded `--no-default-features --features libm` configuration;
that configuration is compile- and test-covered by the `no_std` job. It is
not a separate semver surface today because the published 0.3.0 baseline
predates the `libm` feature, so the same feature cannot be enabled on both
sides of the comparison. Keep the public API feature-independent; if a
feature-gated API is ever introduced, extend this gate deliberately.
`cargo-semver-checks` derives the required release type from the version
in `Cargo.toml`, so an under-declared version bump fails the job.

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
