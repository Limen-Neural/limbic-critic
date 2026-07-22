# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
