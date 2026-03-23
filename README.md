<p align="center">
  <img src="docs/logo.png" width="220" alt="Spikenaut">
</p>

<h1 align="center">spikenaut-reward</h1>
<p align="center">Homeostatic reward computation for cyber-physical spiking neural networks</p>

<p align="center">
  <a href="https://crates.io/crates/spikenaut-reward"><img src="https://img.shields.io/crates/v/spikenaut-reward" alt="crates.io"></a>
  <a href="https://docs.rs/spikenaut-reward"><img src="https://docs.rs/spikenaut-reward/badge.svg" alt="docs.rs"></a>
  <img src="https://img.shields.io/badge/license-GPL--3.0-orange" alt="GPL-3.0">
</p>

---

Zero-allocation, EMA-smoothed, multi-dimensional reward framework that transforms
real-time hardware telemetry into dopaminergic reward signals for SNNs or RL agents.
Treats hardware health as a biological survival signal.

## Features

- `MiningRewardState` — composite reward: `R = α·efficiency - β·thermal - γ·waste`
- `homeostatic_reward(value, setpoint, tolerance)` — bell-curve around optimal setpoint
- `reward_to_q8_8(reward)` — Q8.8 fixed-point conversion for FPGA reward injection
- `NeuroModulators` — 7-system neuromodulator state (dopamine, cortisol, serotonin, …)
- `NeuroModulators::from_telemetry(telem)` — maps GPU hardware readings to modulator levels
- `apply_event(event, magnitude)` — discrete event injection with phasic decay
- EMA smoothing with configurable alpha and clamp range

## Installation

```toml
spikenaut-reward = "0.1"
```

## Quick Start

```rust
use spikenaut_reward::{MiningRewardState, GpuTelemetry};

let mut reward_state = MiningRewardState::new();
let telem = GpuTelemetry {
    gpu_temp_c:    72.0,
    hashrate_mh:   95.0,
    power_w:       180.0,
    ..Default::default()
};

let reward = reward_state.compute(&telem);
// reward ∈ [-1.0, 1.0]  (positive = good, negative = thermal stress)
```

### Neuromodulator Mapping

```rust
use spikenaut_reward::NeuroModulators;

let mods = NeuroModulators::from_telemetry(&telem);
// mods.dopamine    — proportional to hashrate efficiency
// mods.cortisol    — proportional to thermal stress (temp > setpoint)
// mods.serotonin   — stability / low variance reward
```

## Reward Formula

```
R(t) = α · efficiency(t) - β · thermal_penalty(t) - γ · power_waste(t)
     = α · (hashrate / power) - β · max(0, T - T_setpoint) - γ · (1 - η)

Smoothed: R_ema(t) = λ · R_ema(t-1) + (1-λ) · R(t)
```

*Schultz (1998) — dopaminergic prediction error; Arnsten (2009) — cortisol stress model*

## Extracted from Production

Extracted from [Eagle-Lander](https://github.com/rmems/Eagle-Lander), a private
neuromorphic GPU supervisor for crypto mining. The reward computation was decoupled
from mining-specific assets so it works with any hardware telemetry source.

## Part of the Spikenaut Ecosystem

| Library | Purpose |
|---------|---------|
| [spikenaut-encoder](https://github.com/rmems/spikenaut-encoder) | Telemetry → spikes (afferent arm) |
| [neuromod](https://crates.io/crates/neuromod) | Neuromodulator dynamics |
| [spikenaut-backend](https://github.com/rmems/spikenaut-backend) | SNN backend abstraction |

## License

GPL-3.0-or-later
