// SPDX-License-Identifier: MIT OR Apache-2.0

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

    /// Returns a scalar value representing environmental surprise or novelty.
    ///
    /// This is optional and can be used to modulate acetylcholine levels in
    /// stateless critics. For a trading bot, this might be anomaly score.
    /// For a game, it could be unexpected state changes or newly discovered
    /// entities. Defaults to 0.0 if not implemented.
    ///
    /// Values are not required to be in `[0.0, 1.0]`; critics that consume this
    /// signal clamp it to the valid modulator range as needed.
    fn surprise(&self) -> f32 {
        0.0
    }

    /// Returns a scalar value representing system stress or instability.
    ///
    /// This is optional and can be used to modulate norepinephrine levels.
    /// For a hardware system, this might be temperature or power draw.
    /// For a software system, it could be error rates or latency.
    /// Defaults to 0.0 if not implemented.
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
