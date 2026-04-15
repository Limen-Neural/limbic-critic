//! # spikenaut-reward
//!
//! Homeostatic reward computation for cyber-physical systems in the Spikenaut ecosystem.
//!
//! This crate provides:
//! - **MiningRewardState** — EMA-smoothed multi-dimensional reward from hardware telemetry
//!   (R = alpha*efficiency - beta*thermal - gamma*waste)
//! - **NeuroModulators** — 7-system neuromodulator bank (dopamine, cortisol, acetylcholine,
//!   tempo, fpga_stress, market_volatility, mining_dopamine)
//! - **ThermalSetpoint** — Bell-curve homeostatic reward function
//! - **Q8.8 helpers** — Fixed-point reward export for FPGA deployment
//!
//! Complements the `neuromod` crate (v0.2.1) which provides the SNN neurons and STDP learning.
//!
//! ## Provenance
//!
//! Extracted from Eagle-Lander, the author's own private neuromorphic GPU supervisor
//! repository (closed-source). The reward and neuromodulator system ran in production
//! driving a 16-neuron LIF SNN for Dynex/Quai/Qubic mining optimization before being
//! open-sourced as a standalone crate.
//!
//! ## Quick Start
//!
//! ```rust
//! use spikenaut_reward::{MiningRewardState, RewardableState, HomeostasisSpecs};
//!
//! struct MyHardware { hashrate: f32, power: f32 }
//! impl RewardableState for MyHardware {
//!     fn hashrate_mh(&self) -> f32 { self.hashrate }
//!     fn power_w(&self) -> f32 { self.power }
//!     fn gpu_temp_c(&self) -> f32 { 70.0 }
//!     fn gpu_clock_mhz(&self) -> f32 { 2640.0 }
//!     fn vddcr_gfx_v(&self) -> f32 { 1.0 }
//!     fn ocean_intel(&self) -> f32 { 0.0 }
//! }
//!
//! let mut state = MiningRewardState::new();
//! let specs = HomeostasisSpecs::default();
//! let telem = MyHardware { hashrate: 0.012, power: 340.0 };
//! let reward = state.compute(&telem, &specs, Some(68.0));
//! println!("Mining dopamine: {reward:.4}");
//! ```

pub mod telemetry;
pub mod mining_reward;
pub mod modulators;

/// The contract required to compute a mining/homeostasis reward.
/// Any hardware snapshot passed to the reward state MUST implement this.
pub trait RewardableState {
    fn power_w(&self) -> f32;
    fn gpu_temp_c(&self) -> f32;
    fn gpu_clock_mhz(&self) -> f32;
    fn vddcr_gfx_v(&self) -> f32;
    fn hashrate_mh(&self) -> f32;
    fn ocean_intel(&self) -> f32;
}

// Re-export public API
pub use telemetry::{PoolEvent, CryptoAsset};
pub use mining_reward::{MiningRewardState, ThermalSetpoint, HomeostasisSpecs, reward_to_q8_8};
pub use modulators::NeuroModulators;
