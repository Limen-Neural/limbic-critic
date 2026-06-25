# limbic-critic

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)

Neuromodulatory reward shaping and RL critic functions for SNNs.

This crate provides a generalized engine that translates any external objective into biological neuromodulator concentrations. Its sole purpose is to compute the scalar values that feed into `neuromod::rm_stdp` (Reward-Modulated STDP).

## The New Mission: Global Reward Shaping

Instead of hardcoding Qubic/Dynex mining logic, this crate should become a generalized engine that translates any external objective into biological neuromodulator concentrations.

Its sole purpose is to compute the scalar values that feed into `neuromod::rm_stdp` (Reward-Modulated STDP).

### How to generalize it:

*   **Abstract the Mining Logic**: Replace mining_reward/ with a generic `Environment` trait. The crate shouldn't know if it's evaluating a cryptocurrency hash rate, a high-frequency trading bot's PnL, or an LLM's cross-entropy loss.
*   **Reward Functions**: Implement standard RL reward shaping functions (e.g., Temporal Difference error, Curiosity-driven intrinsic reward, or moving-average baselines).
*   **Modulator Mapping**: Map these mathematical errors into constrained `f32` vectors representing Dopamine (reward), Serotonin (risk/patience), and Cortisol (stress/telemetry).

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
