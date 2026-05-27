# limbic-critic

Neuromodulatory reward shaping and RL critic functions for SNNs.

This crate provides a generalized engine that translates any external objective into biological neuromodulator concentrations. Its sole purpose is to compute the scalar values that feed into `neuromod::rm_stdp` (Reward-Modulated STDP).

## Mission: Generic Reward Shaping

This crate is a generalized engine that translates any external objective into biological neuromodulator concentrations.

Its sole purpose is to compute the scalar values that feed into `neuromod::rm_stdp` (Reward-Modulated STDP).

### Architecture

*   **Environment Trait**: Abstract interface via the `Environment` trait. The crate is agnostic to whether it's evaluating a simulation score, a trading bot's PnL, an LLM's cross-entropy loss, or any other performance indicator.
*   **Reward Functions**: Standard RL reward shaping functions — Temporal Difference error, curiosity-driven intrinsic reward, moving-average baselines.
*   **Modulator Mapping**: Maps mathematical errors into constrained `f32` vectors representing Dopamine (reward), Serotonin (risk/patience), and Cortisol (stress/telemetry).
