# limbic-critic

Neuromodulatory reward shaping and RL critic functions for SNNs.

This crate provides a generalized engine that translates any external objective into biological neuromodulator concentrations. Its sole purpose is to compute the scalar values that feed into `neuromod::rm_stdp` (Reward-Modulated STDP).

## The New Mission: Global Reward Shaping

Instead of hardcoding Qubic/Dynex mining logic, this crate should become a generalized engine that translates any external objective into biological neuromodulator concentrations.

Its sole purpose is to compute the scalar values that feed into `neuromod::rm_stdp` (Reward-Modulated STDP).

### How to generalize it:

*   **Abstract the Mining Logic**: Replace mining_reward/ with a generic `Environment` trait. The crate shouldn't know if it's evaluating a cryptocurrency hash rate, a high-frequency trading bot's PnL, or an LLM's cross-entropy loss.
*   **Reward Functions**: Implement standard RL reward shaping functions (e.g., Temporal Difference error, Curiosity-driven intrinsic reward, or moving-average baselines).
*   **Modulator Mapping**: Map these mathematical errors into constrained `f32` vectors representing Dopamine (reward), Serotonin (risk/patience), and Cortisol (stress/telemetry).
