// SPDX-License-Identifier: MIT OR Apache-2.0

//! Neuromodulator concentrations produced by a critic.
//!
//! A [`ModulatorVector`] is the primary output of
//! [`SimpleCritic::assess`](crate::SimpleCritic::assess) and
//! [`TDCritic::assess`](crate::TDCritic::assess). Each field maps to a
//! classical neuromodulator role used for reward-modulated learning and
//! plasticity control in SNN systems.
//!
//! # Example
//!
//! ```rust
//! use limbic_critic::ModulatorVector;
//!
//! let mods = ModulatorVector {
//!     dopamine: 0.8,
//!     serotonin: 0.2,
//!     acetylcholine: 0.1,
//!     norepinephrine: 0.0,
//! };
//! assert!(mods.dopamine >= 0.0);
//! ```

/// A vector of neuromodulator concentrations.
///
/// Critics populate these fields from an [`Environment`](crate::Environment)
/// observation (and, for temporal critics, from internal state). Consumers
/// typically feed the values into synaptic plasticity rules, attention
/// gating, or other neuromodulatory pathways.
///
/// Field ranges depend on which critic produced the vector; see each field's
/// documentation and the critic-level docs for exact mappings.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ModulatorVector {
    /// Dopamine — reward / prediction-error drive.
    ///
    /// Semantic role: reinforces or weakens recent activity proportional to
    /// how good (or how better-than-expected) the outcome was.
    ///
    /// Ranges by critic:
    /// - [`SimpleCritic`](crate::SimpleCritic): clamped to **`[0.0, 1.0]`**
    ///   (non-negative objective mapped through; negative objective → `0.0`).
    /// - [`TDCritic`](crate::TDCritic): signed, clamped to **`[-1.0, 1.0]`**
    ///   after an EMA of the temporal-difference error is passed through
    ///   `tanh`.
    pub dopamine: f32,

    /// Serotonin — risk / volatility.
    ///
    /// Semantic role: tracks environmental uncertainty or risk load. Both
    /// critics read this from [`Environment::volatility`](crate::Environment::volatility)
    /// and clamp it to **`[0.0, 1.0]`**.
    pub serotonin: f32,

    /// Acetylcholine — focus / surprise.
    ///
    /// Semantic role: elevates attention or learning rate when the world is
    /// novel or outcomes are unexpected.
    ///
    /// Ranges by critic (both clamp to **`[0.0, 1.0]`**):
    /// - [`SimpleCritic`](crate::SimpleCritic): taken from
    ///   [`Environment::surprise`](crate::Environment::surprise).
    /// - [`TDCritic`](crate::TDCritic): derived as `abs(td_error).tanh()`,
    ///   independent of `Environment::surprise`.
    pub acetylcholine: f32,

    /// Norepinephrine — stress / instability.
    ///
    /// Semantic role: reflects system pressure (error rates, thermal load,
    /// latency spikes, etc.). Both critics map
    /// [`Environment::stress`](crate::Environment::stress) into this field
    /// and clamp it to **`[0.0, 1.0]`**.
    pub norepinephrine: f32,
}
