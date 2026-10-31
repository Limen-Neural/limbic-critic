// SPDX-License-Identifier: MIT OR Apache-2.0

// no_std (RM-2068 / GH#89): non-test code is `core`-portable since
// `core::error::Error` stabilized in Rust 1.81 (below the 1.85 MSRV). The
// one `std`-only API is `f32::tanh` (TDCritic dopamine / acetylcholine);
// `no_std` builds get the same math from the optional `libm` dependency.
// Test builds keep the `std` prelude so the full suite can run on the host
// against the `libm` math path (`cargo test --no-default-features
// --features libm`); the real `no_std` compile is CI's thumbv7em-none-eabihf
// cross-build.
#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![doc = include_str!("../README.md")]

// A `tanh` provider is mandatory: `core` has no float transcendentals at
// MSRV 1.85. Fires in test builds too — doctests link the non-test lib.
#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!(
    "limbic-critic needs a `tanh` provider: keep the default `std` feature, \
     or enable `libm` for no_std targets \
     (default-features = false, features = [\"libm\"])"
);

pub mod critic;
pub mod environment;
pub mod error;
pub mod modulators;

pub use critic::{InvalidAlpha, SimpleCritic, TDCritic};
pub use environment::Environment;
pub use error::{CriticError, CriticField, NonFiniteKind};
pub use modulators::ModulatorVector;
