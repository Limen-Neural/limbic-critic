use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct NeuroModulators {
    pub dopamine: f32,      // Reward / Learning Rate (0.0 - 1.0)
    pub serotonin: f32,     // Risk / Patience (0.0 - 1.0)
    pub cortisol: f32,      // Stress / Inhibition (0.0 - 1.0)
    pub acetylcholine: f32, // Focus / Signal-to-Noise (0.0 - 1.0)
}
