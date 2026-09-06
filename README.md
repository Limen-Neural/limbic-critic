# limbic-critic

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Neuromodulatory reward shaping and RL critic functions for SNNs.

## Mission

This crate is a generalized engine that translates any external objective into
biological neuromodulator concentrations. Its sole purpose is to compute the
scalar values that feed into reward-modulated learning (e.g. `neuromod::rm_stdp`).

**Architecture (condensed):**

* **Environment trait** — abstract interface for any measurable external system
  (simulation score, trading PnL, LLM loss, etc.)
* **Reward functions** — temporal difference error, curiosity-driven intrinsic
  reward, moving-average baselines
* **Modulator mapping** — maps mathematical errors into constrained `f32`
  vectors for dopamine (reward), serotonin (risk/volatility), and norepinephrine
  (stress/telemetry)

**MSRV:** Rust 1.85 (`rust-version` in `Cargo.toml`).

## Getting Started

Until published on crates.io, depend on the git repository:

```toml
[dependencies]
limbic-critic = { git = "https://github.com/rmems/limbic-critic" }
```

After publish:

```bash
cargo add limbic-critic
```

Run the generic environment example:

```bash
cargo run --example generic_environment
```

## Ecosystem

| Layer | Role | Crate / repo |
|-------|------|----------------|
| Application | Implements `Environment` for your domain | your app / adapters |
| **limbic-critic** | Produces local `ModulatorVector` via `SimpleCritic` / `TDCritic` | [limbic-critic](https://github.com/rmems/limbic-critic) |
| Bridge | Maps `ModulatorVector` → neuromod `NeuroModulators` | [plasticity-lab](https://github.com/Limen-Neural/plasticity-lab) |
| Plasticity | Consumes modulators in `rm_stdp` | [neuromod](https://github.com/Limen-Neural/neuromod) |

Sibling crates live under the [Limen-Neural](https://github.com/Limen-Neural)
organization. This crate does **not** take Cargo dependencies on those
siblings; integration happens in application or bridge crates.

## Scope and Ownership Boundaries

See the full matrix: [`docs/BOUNDARY_MATRIX.md`](docs/BOUNDARY_MATRIX.md)
(LIM-9 / [GH#9](https://github.com/rmems/limbic-critic/issues/9)).

**Owns:**

* Reward shaping and credit-assignment algorithms
* The `Environment` trait
* Local `ModulatorVector` output structure

**Does not own:**

* Training loops or SNN model definitions
* Domain-specific rewards (mining, trading, games)
* Environment implementations (belong in apps/adapters)
* Neuromodulator dynamics / decay (upstream SNN crates)

**Forbidden:**

* Inter-repo Cargo dependencies on sibling crates such as `neuromod`,
  `plasticity-lab`, or other SNN primitives (keeps the crate modular and
  decoupled)

## Development

```bash
cargo check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check

# Coverage (matches CI)
cargo install cargo-llvm-cov
cargo llvm-cov --all-features --lcov --output-path lcov.info
# HTML report: cargo llvm-cov --all-features --html
```

These development commands, hygiene improvements, test assertions, MSRV
declaration, artifact cleanup, and Codecov integration were contributed by the
following GitHub issues (bundled as beads lc-r97 / PR #28):

* [GH-16](https://github.com/rmems/limbic-critic/issues/16): Remove tracked CI log artifacts
* [GH-17](https://github.com/rmems/limbic-critic/issues/17): Remove unused serde dependency
* [GH-18](https://github.com/rmems/limbic-critic/issues/18): Add serotonin/volatility assertions for critic assess()
* [GH-24](https://github.com/rmems/limbic-critic/issues/24): Add rust-version MSRV to Cargo.toml
* [GH-27](https://github.com/rmems/limbic-critic/issues/27): Add Codecov coverage reporting (cargo-llvm-cov + badge)

## Coverage

[![codecov](https://codecov.io/gh/rmems/limbic-critic/branch/main/graph/badge.svg)](https://codecov.io/gh/rmems/limbic-critic)

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or [Apache-2.0](http://www.apache.org/licenses/LICENSE-2.0))
* MIT license ([LICENSE-MIT](LICENSE-MIT) or [MIT](http://opensource.org/licenses/MIT))

at your option.
