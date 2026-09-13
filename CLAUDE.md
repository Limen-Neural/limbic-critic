# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this crate is (and what it refuses to be)

`limbic-critic` is a **reward-shaping / modulator-mapping primitive for SNNs**: it maps an
`Environment` observation into a bounded `ModulatorVector`. It is deliberately **not** an
actor–critic, a learned value function, or an RL training framework.

This is the single most load-bearing fact about the repo. Scope drift is the main thing reviews
reject, because generic reward-shaping features look superficially relevant here. Before adding
any public API, work through the five-question relevance gate in `CONTRIBUTING.md`. The ownership
matrix is in `docs/BOUNDARY_MATRIX.md`.

It is a pure computation library: **zero runtime dependencies**, no I/O, no hardware access, no
environment implementations.

## Commands

```bash
cargo build --all-features
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo run --example generic_environment

# A single test (name substring, or --exact for one)
cargo test --all-features test_environment_defaults
cargo test --all-features -- --exact critic::tests::td_critic_acetylcholine_is_abs_td_tanh

# Doc-tests only (the module docs carry 6 of them and they are load-bearing)
cargo test --all-features --doc

# Packaging — CI gates on this; see "Package contents" below
cargo package --list --allow-dirty
cargo package --allow-dirty

# Coverage, exactly as the coverage workflow runs it
cargo llvm-cov nextest --all-features --profile ci --lcov --output-path lcov.info
```

## Architecture

Three modules, one data flow:

```
Environment (input boundary)  →  SimpleCritic | TDCritic  →  ModulatorVector (output contract)
     src/environment.rs              src/critic.rs              src/modulators.rs
```

Everything downstream of `ModulatorVector` — converting modulators into plasticity updates,
driving an SNN runtime — belongs to sibling crates, not here.

**`Environment`** has one required method, `objective()`, plus three defaulted to `0.0`
(`volatility`, `surprise`, `stress`). Each optional method feeds exactly one modulator channel,
so adding a channel means touching all three modules plus the docs table in `environment.rs`.

**Two critics with deliberately different behavior.** Do not "harmonize" these — the asymmetries
are documented, tested invariants:

| | `SimpleCritic` | `TDCritic` |
|---|---|---|
| State | stateless (associated fn) | stateful: `prev_objective`, `ema_reward`, `alpha` |
| `dopamine` | `[0.0, 1.0]`, non-positive objective → `0.0` | **signed** `[-1.0, 1.0]`, `tanh` of an EMA of the TD error |
| `acetylcholine` | from `Environment::surprise` | from `abs(td_error).tanh()` — **ignores `surprise` entirely** |

`TDCritic`'s "TD error" is the successive objective delta (`objective − prev_objective`), not a
learned `V(s)` or advantage estimate. `TDCritic::new(alpha)` returns `Result<_, InvalidAlpha>` and
accepts only finite `alpha` in `(0, 1]`.

**README is compiled.** `src/lib.rs` starts with `#![doc = include_str!("../README.md")]`, so a
broken link or malformed code fence in `README.md` fails `cargo doc` under `-D warnings`, and
`README.md` must stay inside the published tarball.

## Invariants CI enforces (these fail the build, not just lint)

- **MSRV lockstep.** `Cargo.toml` `rust-version`, `rust-toolchain.toml` `channel`, and the
  `toolchain:` string in **both** `ci.yml` and `coverage.yml` must be byte-identical (`1.98.1`).
  A dedicated CI step parses all four and fails if any disagree. Change them together.
- **Package contents.** The `package` job in `ci.yml` asserts `cargo package --list` against an
  explicit required/forbidden path list. A new development-only file at the repo root will leak
  into the crate unless you add it to `package.exclude` in `Cargo.toml`.
- **No sibling Cargo dependencies.** `neuromod`, `plasticity-lab`, and other Limen-Neural SNN
  crates are forbidden as dependencies — the relationship is semantic only. The
  `ModulatorVector` → `neuromod::NeuroModulators` conversion lives in `plasticity-lab`.
- **REUSE compliance.** New Rust sources need `// SPDX-License-Identifier: MIT OR Apache-2.0` as
  the first line; path annotations live in `REUSE.toml`.

## Release state

The crate is **unpublished**; `0.2.0` is the pending first crates.io release. The repository was
recently transferred to `Limen-Neural/limbic-critic`, so treat any `rmems/limbic-critic` URL as
stale. Keep `CHANGELOG.md` in Keep a Changelog form — one heading per category under
`[Unreleased]`, in the order Added, Changed, Deprecated, Removed, Fixed, Security.

## Codacy (from `.cursor/rules/codacy.mdc`, which is gitignored and local-only)

When the Codacy MCP server is available: after editing a file, run `codacy_cli_analyze` with
`rootPath` set to the workspace and `file` set to that file, and fix what it reports. After any
dependency change, run it with `tool: "trivy"` and resolve vulnerabilities before continuing.
Do not use it to chase duplication, complexity metrics, or coverage. Never install the Codacy CLI
manually via brew/npm.
