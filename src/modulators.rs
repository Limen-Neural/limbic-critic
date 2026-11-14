// SPDX-License-Identifier: MIT OR Apache-2.0

//! Neuromodulator concentrations produced by a critic.
//!
//! A [`ModulatorVector`] is the primary output of
//! [`SimpleCritic::assess`](crate::SimpleCritic::assess) /
//! [`try_assess`](crate::SimpleCritic::try_assess) and
//! [`TDCritic::assess`](crate::TDCritic::assess) /
//! [`try_assess`](crate::TDCritic::try_assess). Each field maps to a
//! classical neuromodulator role used for reward-modulated learning and
//! plasticity control in SNN systems.
//!
//! # Example
//!
//! ```rust
//! use limbic_critic::ModulatorVector;
//!
//! let mods = ModulatorVector::new(0.8, 0.2, 0.1, 0.0);
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
/// documentation and the critic-level docs for exact mappings. Vectors from
/// `try_assess` are finite and inside those ranges; `assess` may yield NaN
/// when given non-finite observations (IEEE passthrough).
///
/// The struct is `#[non_exhaustive]`: downstream crates cannot use struct
/// literals or functional-update syntax, and struct patterns must include
/// `..`. Construct it with [`new`](Self::new) or [`Default`], then assign
/// public fields when a partial update is clearer.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub struct ModulatorVector {
    /// Dopamine — shaped reward / objective-delta drive.
    ///
    /// Semantic role: a bounded scalar that downstream plasticity can use
    /// to reinforce or weaken recent activity. [`TDCritic`](crate::TDCritic)
    /// produces a signed value from an EMA of objective deltas, not from a
    /// learned prediction-error network.
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

impl ModulatorVector {
    /// Create a vector with explicit values for the four canonical channels.
    ///
    /// This is the preferred construction path outside this crate because the
    /// struct is `#[non_exhaustive]`. The four arguments cover the canonical
    /// channels and the signature is intended to remain stable: future fields
    /// are added with documented defaults rather than by changing this
    /// constructor. [`Default`] remains available for a zeroed vector;
    /// callers may assign public fields afterward.
    #[must_use]
    pub const fn new(
        dopamine: f32,
        serotonin: f32,
        acetylcholine: f32,
        norepinephrine: f32,
    ) -> Self {
        Self {
            dopamine,
            serotonin,
            acetylcholine,
            norepinephrine,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ModulatorVector;

    #[test]
    fn new_sets_canonical_fields() {
        let vector = ModulatorVector::new(0.8, 0.2, 0.1, 0.0);
        assert_eq!(
            vector,
            ModulatorVector {
                dopamine: 0.8,
                serotonin: 0.2,
                acetylcholine: 0.1,
                norepinephrine: 0.0,
            }
        );
    }
}
