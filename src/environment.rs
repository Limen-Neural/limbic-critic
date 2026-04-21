//! Environment Trait
//!
//! Defines the interface for any external system that the `limbic-critic`
//! needs to evaluate. This trait abstracts the source of the objective
//! function, allowing the critic to be agnostic to whether it's evaluating
//! a trading bot, a game AI, or a hardware system.

pub trait Environment {
    /// Returns the current scalar objective value from the environment.
    ///
    /// This value represents the primary metric that the critic should
    /// optimize. It could be profit-and-loss, cross-entropy loss,
    /// game score, or any other performance indicator.
    ///
    /// The value should be normalized to a consistent range if possible,
    /// although the critic's reward shaping functions should also be
    /// robust to unnormalized inputs.
    fn objective(&self) -> f32;

    /// Returns a scalar value representing environmental volatility or risk.
    ///
    /// This is optional and can be used to modulate serotonin levels.
    /// For a trading bot, this might be market volatility.
    /// For a game, it could be the number of enemies on screen.
    /// Defaults to 0.0 if not implemented.
    fn volatility(&self) -> f32 {
        0.0
    }

    /// Returns a scalar value representing system stress or instability.
    ///
    /// This is optional and can be used to modulate cortisol levels.
    /// For a hardware system, this might be temperature or power draw.
    /// For a software system, it could be error rates or latency.
    /// Defaults to 0.0 if not implemented.
    fn stress(&self) -> f32 {
        0.0
    }
}
