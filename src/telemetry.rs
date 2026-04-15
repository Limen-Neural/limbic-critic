//! Hardware Telemetry Types
//!
//! Lightweight telemetry structs consumed by the reward and modulator modules.

use serde::{Deserialize, Serialize};

/// Events emitted by the mining pool client.
///
/// These drive phasic dopamine/cortisol spikes in the neuromodulator system.
#[derive(Debug, Clone)]
pub enum PoolEvent {
    /// A submitted share was accepted by the pool.
    ShareAccepted { latency_ms: u64 },
    /// A block was found (rare — strongest dopamine burst).
    BlockFound { block_height: u64, reward_dnx: f64 },
    /// Pool connection switched (cortisol spike).
    PoolSwitch { reason: String },
    /// A share was rejected (mild cortisol).
    ShareRejected { reason: String },
}

/// Supported crypto assets for reward calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CryptoAsset {
    Dynex,
    Quai,
    Qubic,
    Kaspa,
    Monero,
    Verus,
}
