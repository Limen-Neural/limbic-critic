// SPDX-License-Identifier: MIT OR Apache-2.0

//! Modulator Vector
//!
//! Defines the structure for neuromodulatory signals.

/// A vector of neuromodulator concentrations.
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
