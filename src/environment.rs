// SPDX-License-Identifier: MIT OR Apache-2.0

//! Environment interface for critic evaluation.
//!
//! The [`Environment`] trait abstracts the external system that a critic
//! observes — a trading bot, game AI, hardware controller, LLM training loop,
//! or any other process that can expose a scalar objective (and optional
//! risk / stress / surprise signals).
//!
//! Critics such as [`SimpleCritic`](crate::SimpleCritic) and
//! [`TDCritic`](crate::TDCritic) depend only on this trait, remaining
//! agnostic to domain-specific details.
//!
//! # Implementing `Environment`
//!
//! Only [`objective`](Environment::objective) is required. The other methods
//! default to `0.0` and can be overridden when the corresponding modulator
//! channel is useful:
//!
//! | Method | Default | Used by critics as |
//! |--------|---------|--------------------|
//! | [`objective`](Environment::objective) | *(required)* | reward / TD target |
//! | [`volatility`](Environment::volatility) | `0.0` | serotonin |
//! | [`surprise`](Environment::surprise) | `0.0` | acetylcholine (`SimpleCritic` only) |
//! | [`stress`](Environment::stress) | `0.0` | norepinephrine |
//!
//! # Example
//!
//! ```rust
//! use limbic_critic::{Environment, SimpleCritic};
//!
//! struct TradingBot {
//!     pnl: f32,
//!     market_vol: f32,
//! }
//!
//! impl Environment for TradingBot {
//!     fn objective(&self) -> f32 {
//!         self.pnl
//!     }
//!     fn volatility(&self) -> f32 {
//!         self.market_vol
//!     }
//! }
//!
//! let bot = TradingBot {
//!     pnl: 0.6,
//!     market_vol: 0.25,
//! };
//! let mods = SimpleCritic::assess(&bot);
//! assert_eq!(mods.dopamine, 0.6);
//! assert_eq!(mods.serotonin, 0.25);
//! ```

/// Interface for any external system that a limbic critic can evaluate.
///
/// Implementors provide at least a scalar objective; optional methods supply
/// secondary signals that map onto serotonin, acetylcholine, and
/// norepinephrine channels.
///
/// See the [module-level documentation](self) for the full mapping table and
/// a worked example.
pub trait Environment {
    /// Returns the current scalar objective value from the environment.
    ///
    /// This is the primary metric the critic optimizes — profit-and-loss,
    /// cross-entropy loss (negated), game score, accuracy, or any other
    /// performance indicator.
    ///
    /// Prefer a stable, domain-normalized scale when possible. Critics also
    /// apply their own clamps / nonlinearities (`clamp`, `tanh`) so raw
    /// unnormalized values are accepted, but extreme magnitudes will saturate
    /// the resulting modulators.
    ///
    /// # Used by
    ///
    /// - [`SimpleCritic`](crate::SimpleCritic): positive values → dopamine in
    ///   `[0, 1]`; non-positive → dopamine `0`.
    /// - [`TDCritic`](crate::TDCritic): difference from the previous objective
    ///   drives the TD error and thus dopamine / acetylcholine.
    fn objective(&self) -> f32;

    /// Returns environmental volatility or risk.
    ///
    /// Optional. Mapped to **serotonin** (clamped to `[0.0, 1.0]`) by both
    /// critics. Defaults to `0.0` if not overridden.
    ///
    /// Domain examples:
    /// - Trading: realized or implied market volatility.
    /// - Games: number of threats / enemy density.
    /// - Training: loss variance over a recent window.
    fn volatility(&self) -> f32 {
        0.0
    }

    /// Returns environmental surprise or novelty.
    ///
    /// Optional. Used by [`SimpleCritic`](crate::SimpleCritic) as the source
    /// of **acetylcholine** (clamped to `[0.0, 1.0]`).
    /// [`TDCritic`](crate::TDCritic) ignores this method and instead derives
    /// acetylcholine from `abs(td_error).tanh()`. Defaults to `0.0` if not
    /// overridden.
    ///
    /// Domain examples:
    /// - Trading: anomaly or regime-change score.
    /// - Games: unexpected state transitions or newly discovered entities.
    /// - Sensors: prediction residual / novelty detector output.
    fn surprise(&self) -> f32 {
        0.0
    }

    /// Returns system stress or instability.
    ///
    /// Optional. Mapped to **norepinephrine** (clamped to `[0.0, 1.0]`) by
    /// both critics. Defaults to `0.0` if not overridden.
    ///
    /// Domain examples:
    /// - Hardware: temperature, power draw, thermal throttling.
    /// - Software: error rate, p99 latency, queue depth.
    /// - Agents: resource depletion or constraint violation severity.
    fn stress(&self) -> f32 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestEnv {
        value: f32,
        vol: f32,
        stress_val: f32,
    }

    impl Environment for TestEnv {
        fn objective(&self) -> f32 {
            self.value
        }
        fn volatility(&self) -> f32 {
            self.vol
        }
        fn stress(&self) -> f32 {
            self.stress_val
        }
    }

    #[test]
    fn test_environment_defaults() {
        struct MinimalEnv;
        impl Environment for MinimalEnv {
            fn objective(&self) -> f32 {
                42.0
            }
        }
        let env = MinimalEnv;
        assert_eq!(env.objective(), 42.0);
        assert_eq!(env.volatility(), 0.0);
        assert_eq!(env.surprise(), 0.0);
        assert_eq!(env.stress(), 0.0);
    }

    #[test]
    fn test_environment_with_values() {
        let env = TestEnv {
            value: 0.5,
            vol: 0.3,
            stress_val: 0.8,
        };
        assert_eq!(env.objective(), 0.5);
        assert_eq!(env.volatility(), 0.3);
        assert_eq!(env.stress(), 0.8);
    }

    #[test]
    fn test_environment_trait_object() {
        let env: Box<dyn Environment> = Box::new(TestEnv {
            value: -1.0,
            vol: 0.0,
            stress_val: 0.0,
        });
        assert_eq!(env.objective(), -1.0);
    }
}
