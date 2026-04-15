<h1 align="center">spikenaut-reward</h1>
<p align="center">Homeostatic reward computation for cyber-physical spiking neural networks</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-GPL--3.0-orange" alt="GPL-3.0">
</p>

---

Zero-allocation, EMA-smoothed, multi-dimensional reward framework that transforms
real-time hardware telemetry into dopaminergic reward signals for SNNs or RL agents.
Treats hardware health as a biological survival signal.

## Features

- **Trait-based Contract** — Decoupled from specific hardware via the `RewardableState` trait.
- **Homeostasis Specs** — Configurable targets for different mining assets (Dynex, Quai, Qubic, Kaspa, Monero, Verus).
- `MiningRewardState` — Composite reward: `R = α·efficiency - β·thermal - γ·waste`.
- `homeostatic_reward(value, setpoint, tolerance)` — Bell-curve around optimal setpoint.
- `reward_to_q8_8(reward)` — Q8.8 fixed-point conversion for FPGA reward injection.
- `NeuroModulators` — 7-system neuromodulator state (dopamine, cortisol, acetylcholine, tempo, fpga_stress, etc.).

## Quick Start

```rust
use spikenaut_reward::{MiningRewardState, RewardableState, HomeostasisSpecs, CryptoAsset};

// Define your own hardware telemetry source
struct MyHardware { 
    hashrate: f32, 
    power: f32, 
    temp: f32 
}

impl RewardableState for MyHardware {
    fn hashrate_mh(&self) -> f32 { self.hashrate }
    fn power_w(&self) -> f32 { self.power }
    fn gpu_temp_c(&self) -> f32 { self.temp }
    fn gpu_clock_mhz(&self) -> f32 { 2640.0 }
    fn vddcr_gfx_v(&self) -> f32 { 1.0 }
    fn ocean_intel(&self) -> f32 { 0.0 }
}

let mut reward_state = MiningRewardState::new();

// Select homeostasis specs for your crypto asset
let specs = HomeostasisSpecs::for_asset(CryptoAsset::Kaspa);

let telem = MyHardware { hashrate: 1000.0, power: 400.0, temp: 72.0 };

// Compute reward (returns f32 in [-0.8, 0.8])
let reward = reward_state.compute(&telem, &specs, Some(65.0));
```

### Neuromodulator Mapping

```rust
use spikenaut_reward::{NeuroModulators, HomeostasisSpecs, CryptoAsset};

let specs = HomeostasisSpecs::for_asset(CryptoAsset::Dynex);
let mods = NeuroModulators::from_telemetry(&telem, &specs);

// mods.dopamine    — Reward for hashrate efficiency (calibrated to asset specs)
// mods.cortisol    — Stress from heat or power spikes
// mods.acetylcholine — Event-driven signal (e.g. pool events)
```

## Reward Formula

```
R(t) = α · efficiency(t) - β · thermal_penalty(t) - γ · energy_waste(t)
     = α · (hashrate / power) - β · max(thermal_stress) - γ · waste_term

Smoothed: R_ema(t) = (1-λ) · R_ema(t-1) + λ · R(t)
```

## Modular Structure

The crate is organized into focused submodules for clarity and maintainability:

```
src/
├── lib.rs                     # Public re-exports
├── mining_reward/
│   ├── mod.rs                 # Module root
│   ├── config.rs              # HomeostasisSpecs & CryptoAsset targets
│   ├── state.rs               # MiningRewardState & RewardableState trait
│   └── q8.rs                  # Q8.8 fixed-point conversion
├── modulators.rs              # NeuroModulators bank & telemetry mapping
└── telemetry.rs               # Common types & trait definitions
```

- **`mining_reward`**: Core reward state machine, homeostasis configuration, and Q8.8 fixed-point math.
- **`modulators`**: Neuromodulator bank (dopamine, cortisol, acetylcholine, etc.) with asset-calibrated telemetry mapping.
- **`telemetry`**: Shared types, the `RewardableState` trait, and `CryptoAsset` enum.

## License

GPL-3.0-or-later
