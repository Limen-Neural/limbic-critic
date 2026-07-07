// SPDX-License-Identifier: MIT OR Apache-2.0

//! Modulator Vector
//!
//! Defines the structure for neuromodulatory signals.

/// A vector of neuromodulator concentrations.
///
/// This is a local type decoupled from the `neuromod` crate to avoid
/// git-dependency coupling in core libraries. If `neuromod` adds fields
/// or changes semantics, this struct won't track those changes
/// automatically. Downstream consumers using both `limbic-critic` and
/// `neuromod` should maintain a manual conversion layer.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ModulatorVector {
    /// Dopamine level (typically reward/error).
    pub dopamine: f32,
    /// Serotonin level (typically risk/volatility).
    pub serotonin: f32,
    /// Acetylcholine level (typically focus/surprise).
    pub acetylcholine: f32,
    /// Norepinephrine level (typically stress/instability).
    pub norepinephrine: f32,
}
