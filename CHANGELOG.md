# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `CriticError` / `CriticField` / `NonFiniteKind` and a checked assessment
  path: `SimpleCritic::try_assess` and `TDCritic::try_assess`. Non-finite
  (NaN or ±∞) observations return a field-specific error. `TDCritic`
  validates computed TD-error and EMA intermediates **before** committing
  `prev_objective` or the delta EMA, so a failed call cannot poison later
  valid output. Successful vectors are finite and lie in the documented
  per-critic ranges. Extreme finite values still follow the existing
  clamp / `tanh` policy (overflow-scale finite TD deltas saturate to
  `±f32::MAX` rather than failing).
- CI `Build & Test` job now matrices `ubuntu-latest`, `macos-latest`, and `windows-latest` (`fail-fast: false`); `cargo fmt --check` and `cargo doc` stay Linux-only, and `cargo package` stays on ubuntu-latest.
- `docs/ARCHITECTURE.md` codifying the neuromorphic/SNN scope contract: the
  canonical `Environment → Critic → ModulatorVector → plasticity/SNN` flow,
  owns/does-not-own lists, the modulator vocabulary with per-critic clamp
  ranges, and the dependency contract (semantic interop with `neuromod` /
  `plasticity-lab`, zero runtime dependencies, sibling Cargo dependencies
  forbidden). (GH#58)
- `package.exclude` for CI/agent-only paths and a `cargo package` CI job.
- Focused unit tests for SimpleCritic clamps, TD dopamine sign, ACh `|td|.tanh()`, and zero/negative objectives.
- `.github/workflows/linear-release.yml` modeled on synaptic-mesh: sync GitHub into Linear Releases **0.2.0** on main, complete on `vMAJOR.MINOR.PATCH` tags, serialize push/tag runs (`cancel-in-progress: false`). Uses org secret `LINEAR_ACCESS_KEY`. (GH-56)
- SonarQube Cloud analysis workflow and `sonar-project.properties` (requires the `SONAR_TOKEN` repository secret; the scan is skipped on fork PRs and when the token is absent) (GH-56).
- `CONTRIBUTING.md` with a neuromorphic relevance gate (five review
  questions) so future changes preserve the crate's SNN/neuromodulatory
  identity instead of drifting into a generic RL/ML framework (#59).
- `.github/pull_request_template.md` with a scope checklist derived from
  the relevance gate and a CI-matching verification checklist.

### Changed

- `SimpleCritic::assess` and `TDCritic::assess` keep their documented IEEE
  `f32` passthrough for compatibility (NaN auxiliary signals stay NaN; a
  non-finite `TDCritic` objective still poisons `prev_objective` / the
  EMA). New callers — and any caller that must not absorb NaN into critic
  state — should migrate to `try_assess`. The two paths agree on finite
  observations.
- Pin Rust **1.98.1** in lockstep across `rust-version`, `rust-toolchain.toml`, and CI/coverage workflows.
- Position the crate as a reward-shaping / modulator-mapping primitive (not a full actor–critic).
- Cargo.toml publish metadata: authors, homepage, documentation, docs.rs, keywords (`rl` → `reward-shaping`).
- Migrate self-referencing repository URLs (`Cargo.toml`, `REUSE.toml`, `README.md`, Codecov slug) from `rmems/limbic-critic` to the canonical `Limen-Neural/limbic-critic` home following the GitHub repository transfer (#60).

### Fixed

- `TDCritic::new(alpha)` now returns `Result<TDCritic, InvalidAlpha>` and accepts only `alpha` in `(0, 1]`.

### Removed

- Stopped tracking `.beads/` (stale local issue-tracker state; it stays gitignored and is already excluded from the published crate).
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
