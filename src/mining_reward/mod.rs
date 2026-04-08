//! Mining Efficiency Dopamine Reward Framework
//!
//! Computes a multi-dimensional reward signal from hardware telemetry,
//! treating mining efficiency as a biological survival signal for the SNN.
//!
//! Core equation: R = α·MiningEfficiency − β·ThermalStress − γ·EnergyWaste
//!
//! The output `mining_dopamine` is EMA-smoothed and fed into the STDP
//! learning rate blend in `engine.rs`, gating synaptic plasticity based
//! on whether the system is operating within its homeostatic envelope.

mod config;
mod q8;
mod state;

pub use config::{HomeostasisSpecs, ThermalSetpoint};
pub use q8::reward_to_q8_8;
pub use state::MiningRewardState;
