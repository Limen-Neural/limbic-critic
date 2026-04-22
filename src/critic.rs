//! RL Critic and Reward Shaping
//!
//! This module contains the core logic for translating environmental
//! observations into neuromodulatory signals.

use crate::environment::Environment;
use crate::modulators::NeuroModulators;

/// A simple critic that calculates reward based on the immediate
/// objective value.
pub struct SimpleCritic;

impl SimpleCritic {
    /// Calculates neuromodulator concentrations based on the current
    /// state of the environment.
    pub fn assess(env: &impl Environment) -> NeuroModulators {
        let objective = env.objective();

        // Simple mapping: positive objective -> dopamine, negative -> nothing
        let dopamine = if objective > 0.0 {
            objective.clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Map volatility to serotonin and stress to cortisol
        let serotonin = (1.0 - env.volatility()).clamp(0.0, 1.0);
        let cortisol = env.stress().clamp(0.0, 1.0);

        NeuroModulators {
            dopamine,
            serotonin,
            cortisol,
            acetylcholine: 0.5, // Placeholder value
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
    pub fn assess(&mut self, env: &impl Environment) -> NeuroModulators {
        let objective = env.objective();
        let td_error = objective - self.prev_objective;
        self.prev_objective = objective;

        // Update the EMA of the reward
        self.ema_reward = (1.0 - self.alpha) * self.ema_reward + self.alpha * td_error;

        // Map the smoothed reward to dopamine
        let dopamine = (self.ema_reward.tanh()).clamp(-1.0, 1.0);

        let serotonin = (1.0 - env.volatility()).clamp(0.0, 1.0);
        let cortisol = env.stress().clamp(0.0, 1.0);

        NeuroModulators {
            dopamine,
            serotonin,
            cortisol,
            acetylcholine: 0.5, // Placeholder
        }
    }
}
