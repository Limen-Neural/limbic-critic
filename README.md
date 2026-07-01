# limbic-critic

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Neuromodulatory reward shaping and RL critic functions for SNNs.

This crate provides a generalized engine that translates any external objective into biological neuromodulator concentrations. Its sole purpose is to compute the scalar values that feed into `neuromod::rm_stdp` (Reward-Modulated STDP).

## Mission: Generic Reward Shaping

This crate is a generalized engine that translates any external objective into biological neuromodulator concentrations.

Its sole purpose is to compute the scalar values that feed into `neuromod::rm_stdp` (Reward-Modulated STDP).

### Architecture

* **Environment Trait**: Abstract interface via the `Environment` trait. The crate is agnostic to whether it's evaluating a simulation score, a trading bot's PnL, an LLM's cross-entropy loss, or any other performance indicator.
* **Reward Functions**: Standard RL reward shaping functions — Temporal Difference error, curiosity-driven intrinsic reward, moving-average baselines.
* **Modulator Mapping**: Maps mathematical errors into constrained `f32` vectors representing Dopamine (reward), Serotonin (risk/volatility), and Norepinephrine (stress/telemetry).

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or [Apache-2.0](http://www.apache.org/licenses/LICENSE-2.0))
* MIT license ([LICENSE-MIT](LICENSE-MIT) or [MIT](http://opensource.org/licenses/MIT))

at your option.
