# limbic-critic

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Reward-shaping and modulator-mapping primitive for SNNs: map an environment
objective into a local `ModulatorVector`. This is **not** a full actor–critic,
learned value function, or RL training framework.

## Mission

`limbic-critic` turns a scalar observation (plus optional risk / stress /
surprise signals) into constrained neuromodulator concentrations. Downstream
plasticity rules such as `neuromod::rm_stdp` can consume those scalars — but
this crate does **not** depend on `neuromod`. Convert `ModulatorVector` →
`NeuroModulators` in [plasticity-lab](https://github.com/Limen-Neural/plasticity-lab)
(the bridge) or in your application.

**What “TD” means here:** `TDCritic` keeps an exponential
moving average of *successive objective deltas* (`objective − prev_objective`).
That EMA is `tanh`-mapped into dopamine; `|delta|.tanh()` becomes acetylcholine.
It is **not** a learned `V(s)` / advantage estimator and does not run an
actor–critic loop.

**Architecture (condensed):**

* **Environment trait** — abstract interface for any measurable external system
  (simulation score, trading PnL, LLM loss, etc.)
* **Critics (shapers)** — `SimpleCritic` clamps the current observation;
  `TDCritic` maps an EMA of objective deltas into signed dopamine
* **Modulator mapping** — constrained `f32` fields for dopamine (shaped
  reward / delta), serotonin (risk/volatility), acetylcholine (surprise /
  |delta|), and norepinephrine (stress/telemetry)

**MSRV:** Rust 1.98.1 (`rust-version` in `Cargo.toml`, `rust-toolchain.toml`, and CI).

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
| **limbic-critic** | Reward-shaping primitive: `SimpleCritic` / `TDCritic` → local `ModulatorVector` | [limbic-critic](https://github.com/rmems/limbic-critic) |
| Bridge | Maps `ModulatorVector` → neuromod `NeuroModulators` | [plasticity-lab](https://github.com/Limen-Neural/plasticity-lab) |
| Plasticity | Consumes modulators in `rm_stdp` | [neuromod](https://github.com/Limen-Neural/neuromod) |

Sibling crates live under the [Limen-Neural](https://github.com/Limen-Neural)
organization. This crate does **not** take Cargo dependencies on those
siblings; integration happens in application or bridge crates.

## Scope and Ownership Boundaries

See the full matrix: [`docs/BOUNDARY_MATRIX.md`](docs/BOUNDARY_MATRIX.md)
(LIM-9 / [GH#9](https://github.com/rmems/limbic-critic/issues/9)).

**Owns:**

* Reward shaping / modulator mapping (`SimpleCritic`, `TDCritic`)
* The `Environment` trait
* Local `ModulatorVector` output structure

**Does not own:**

* Actor–critic loops, policy gradients, or learned value networks
* Training loops or SNN model definitions
* Domain-specific rewards (mining, trading, games)
* Environment implementations (belong in apps/adapters)
* Neuromodulator dynamics / decay (upstream SNN crates)
* Conversion of `ModulatorVector` → `neuromod::NeuroModulators` (bridge:
  `plasticity-lab`)

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
