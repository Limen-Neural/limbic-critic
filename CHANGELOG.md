# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Pin Rust **1.98.1** in lockstep across `rust-version`, `rust-toolchain.toml`, and CI/coverage workflows.
- Position the crate as a reward-shaping / modulator-mapping primitive (not a full actor–critic).
- Cargo.toml publish metadata: authors, homepage, documentation, docs.rs, keywords (`rl` → `reward-shaping`).

### Added

- `package.exclude` for CI/agent-only paths and a `cargo package` CI job.
- Focused unit tests for SimpleCritic clamps, TD dopamine sign, ACh `|td|.tanh()`, and zero/negative objectives.

### Fixed

- `TDCritic::new(alpha)` now returns `Result<TDCritic, InvalidAlpha>` and accepts only `alpha` in `(0, 1]`.

### Removed

- Qodana Cloud scan (`qodana-rust` + token workflow); membership expired. Clippy and Codecov remain.

## [0.2.0] - 2026-07-16

### Added

- GitHub Actions CI workflow (fmt, clippy, build, test)
- Codecov coverage reporting (cargo-llvm-cov) and badge
- Qodana static analysis CI
- Dual MIT/Apache-2.0 license files and SPDX headers on Rust sources
- `docs/BOUNDARY_MATRIX.md` documenting crate ownership boundaries
- `examples/generic_environment.rs` domain-agnostic example
- Local `ModulatorVector` type replacing sibling neuromod dependency (PR #29)
- SimpleCritic acetylcholine derived from `Environment::surprise` (PR #30)

### Changed

- Rust edition 2021 → 2024
- Declared MSRV `rust-version = "1.85"`
- Critic output field `cortisol` → `norepinephrine`; wire `Environment::volatility()` → serotonin

### Removed

- Legacy mining/Qubic/Dynex documentation references
- Unused `serde` dependency
- Tracked CI log artifacts from the repository
