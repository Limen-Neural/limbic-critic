// SPDX-License-Identifier: MIT OR Apache-2.0

//! RL Critic and Reward Shaping
//!
//! This module contains the core logic for translating environmental
//! observations into neuromodulatory signals.

use crate::environment::Environment;
use crate::modulators::ModulatorVector;

/// A stateless critic that calculates neuromodulator levels from the
/// environment's immediate signals.
///
/// `SimpleCritic` intentionally does not infer acetylcholine from temporal
/// objective deltas because it stores no previous state. Instead,
/// acetylcholine is read directly from [`Environment::surprise`] and clamped
/// to the valid modulator range. Use [`TDCritic`] when acetylcholine should be
/// derived from temporal-difference surprise (`abs(td_error).tanh()`).
pub struct SimpleCritic;

impl SimpleCritic {
    /// Calculates neuromodulator concentrations based on the current
    /// state of the environment.
    pub fn assess(env: &impl Environment) -> ModulatorVector {
        let objective = env.objective();

        // Simple mapping: positive objective -> dopamine, negative -> nothing
        let dopamine = if objective > 0.0 {
            objective.clamp(0.0, 1.0)
        } else {
            0.0
        };

        let stress = env.stress().clamp(0.0, 1.0);
        let serotonin = env.volatility().clamp(0.0, 1.0);
        let acetylcholine = env.surprise().clamp(0.0, 1.0);

        ModulatorVector {
            dopamine,
            serotonin,
            acetylcholine,
            norepinephrine: stress,
        }
    }
}

/// A critic that calculates reward based on the Temporal Difference (TD) error.
pub struct TDCritic {
    prev_objective: f32,
    ema_reward: f32,
    alpha: f32, // Learning rate for the EMA
}

impl TDCritic {
    pub fn new(alpha: f32) -> Self {
        Self {
            prev_objective: 0.0,
            ema_reward: 0.0,
            alpha,
        }
    }

    /// Calculates neuromodulator concentrations based on the TD error.
    pub fn assess(&mut self, env: &impl Environment) -> ModulatorVector {
        let objective = env.objective();
        let td_error = objective - self.prev_objective;
        self.prev_objective = objective;

        // Surprise / Focus calculation
        let acetylcholine = td_error.abs().tanh().clamp(0.0, 1.0);

        // Update the EMA of the reward
        self.ema_reward = (1.0 - self.alpha) * self.ema_reward + self.alpha * td_error;

        // Map the smoothed reward to dopamine
        let dopamine = self.ema_reward.tanh().clamp(-1.0, 1.0);

        let stress = env.stress().clamp(0.0, 1.0);
        let serotonin = env.volatility().clamp(0.0, 1.0);

        ModulatorVector {
            dopamine,
            serotonin,
            acetylcholine,
            norepinephrine: stress,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstEnv(f32);

    impl Environment for ConstEnv {
        fn objective(&self) -> f32 {
            self.0
        }
    }

    struct VolatileEnv {
        steps: Vec<f32>,
        index: usize,
    }

    impl VolatileEnv {
        fn new(values: Vec<f32>) -> Self {
            Self {
                steps: values,
                index: 0,
            }
        }
    }

    impl Environment for VolatileEnv {
        fn objective(&self) -> f32 {
            self.steps[self.index]
        }
    }

    #[test]
    fn test_simple_critic_positive_objective() {
        let env = ConstEnv(0.8);
        let mods = SimpleCritic::assess(&env);
        assert_eq!(mods.dopamine, 0.8);
        assert_eq!(mods.norepinephrine, 0.0);
    }

    #[test]
    fn test_simple_critic_negative_objective() {
        let env = ConstEnv(-0.5);
        let mods = SimpleCritic::assess(&env);
        assert_eq!(mods.dopamine, 0.0);
    }

    #[test]
    fn test_simple_critic_clamping() {
        let env = ConstEnv(5.0);
        let mods = SimpleCritic::assess(&env);
        assert_eq!(mods.dopamine, 1.0);
    }

    #[test]
    fn test_simple_critic_stress() {
        struct StressedEnv;
        impl Environment for StressedEnv {
            fn objective(&self) -> f32 {
                0.5
            }
            fn stress(&self) -> f32 {
                0.9
            }
        }
        let mods = SimpleCritic::assess(&StressedEnv);
        assert_eq!(mods.norepinephrine, 0.9);
    }

    #[test]
    fn test_simple_critic_volatility_serotonin() {
        struct SimpleVolatileEnv;
        impl Environment for SimpleVolatileEnv {
            fn objective(&self) -> f32 {
                0.5
            }
            fn volatility(&self) -> f32 {
                0.6
            }
        }
        let mods = SimpleCritic::assess(&SimpleVolatileEnv);
        assert_eq!(mods.serotonin, 0.6);
    }

    #[test]
    fn test_simple_critic_surprise_acetylcholine() {
        struct SurprisingEnv {
            surprise: f32,
        }
        impl Environment for SurprisingEnv {
            fn objective(&self) -> f32 {
                0.5
            }
            fn surprise(&self) -> f32 {
                self.surprise
            }
        }

        let mods = SimpleCritic::assess(&SurprisingEnv { surprise: 0.7 });
        assert_eq!(mods.acetylcholine, 0.7);
    }

    #[test]
    fn test_simple_critic_surprise_acetylcholine_clamping() {
        struct SurprisingEnv(f32);
        impl Environment for SurprisingEnv {
            fn objective(&self) -> f32 {
                0.0
            }
            fn surprise(&self) -> f32 {
                self.0
            }
        }

        assert_eq!(
            SimpleCritic::assess(&SurprisingEnv(-0.2)).acetylcholine,
            0.0
        );
        assert_eq!(SimpleCritic::assess(&SurprisingEnv(1.5)).acetylcholine, 1.0);
    }

    #[test]
    fn test_td_critic_no_change() {
        let env = ConstEnv(0.5);
        let mut td = TDCritic::new(0.1);
        let mods = td.assess(&env);
        // First call: td_error = 0.5 - 0.0 = 0.5
        // ema starts at 0.0, so ema = 0.9*0.0 + 0.1*0.5 = 0.05
        // dopamine = 0.05.tanh()
        assert!((mods.dopamine - 0.05f32.tanh()).abs() < 1e-6);
    }

    #[test]
    fn test_td_critic_improvement() {
        let mut td = TDCritic::new(0.1);
        let mut env = VolatileEnv::new(vec![0.0, 1.0]);

        let first = td.assess(&env);
        env.index = 1;
        let second = td.assess(&env);

        // Second call has positive improvement
        assert!(second.dopamine > first.dopamine);
    }

    #[test]
    fn test_td_critic_degradation() {
        let mut td = TDCritic::new(0.1);
        let mut env = VolatileEnv::new(vec![1.0, 0.0]);

        let first = td.assess(&env);
        env.index = 1;
        let second = td.assess(&env);

        // Second call has negative td_error
        assert!(second.dopamine < first.dopamine);
    }

    #[test]
    fn test_td_critic_surprise() {
        let mut td = TDCritic::new(0.1);
        let env = ConstEnv(0.5);

        let mods = td.assess(&env);
        // First td_error = 0.5, acetylcholine = 0.5.abs().tanh()
        assert!((mods.acetylcholine - 0.5f32.tanh()).abs() < 1e-6);
    }

    #[test]
    fn test_td_critic_volatility_serotonin() {
        struct TdVolatileEnv {
            v: f32,
        }
        impl Environment for TdVolatileEnv {
            fn objective(&self) -> f32 {
                0.0
            }
            fn volatility(&self) -> f32 {
                self.v
            }
        }
        let mut td = TDCritic::new(0.1);
        let env = TdVolatileEnv { v: 0.6 };
        let mods = td.assess(&env);
        assert_eq!(mods.serotonin, 0.6);
    }
}
