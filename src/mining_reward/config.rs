use crate::telemetry::CryptoAsset;

// ── Reward component weights ─────────────────────────────────────────────────

/// Weight for the positive mining-efficiency term.
pub const ALPHA_EFFICIENCY: f32 = 0.6;

/// Weight for the thermal/power stress penalty.
pub const BETA_THERMAL: f32 = 0.3;

/// Weight for the energy-waste penalty.
pub const GAMMA_WASTE: f32 = 0.1;

// ── EMA smoothing ────────────────────────────────────────────────────────────

/// EMA blending factor for the instant reward (0.1 = 10-tick window ≈ 1 s).
pub const EMA_ALPHA: f32 = 0.1;

/// Hard clamp on the final EMA-smoothed reward.  Prevents saturation so that
/// the STDP blend always retains some influence from the event-driven dopamine.
pub const REWARD_CLAMP: f32 = 0.8;

// ── Thermal / power thresholds ───────────────────────────────────────────────

/// GPU temperature at which the thermal penalty kicks in (°C).
pub const GPU_THERMAL_PENALTY_ONSET: f32 = 85.0;

/// GPU thermal penalty divisor — maps [onset, onset+10] → [0, 1].
pub const GPU_THERMAL_DIVISOR: f32 = 10.0;

/// CPU temperature at which the thermal penalty kicks in (°C).
pub const CPU_THERMAL_PENALTY_ONSET: f32 = 85.0;

/// CPU thermal penalty divisor — maps [onset, onset+10] → [0, 1].
pub const CPU_THERMAL_DIVISOR: f32 = 10.0;

/// GPU power draw at which the power-excess penalty starts (W).
pub const POWER_PENALTY_ONSET: f32 = 400.0;

/// Power penalty divisor — maps [onset, onset+50] → [0, 1].
pub const POWER_PENALTY_DIVISOR: f32 = 50.0;

// ── Homeostasis configuration ──────────────────────────────────────────────

/// Configuration for homeostasis reward normalization.
#[derive(Debug, Clone, Copy)]
pub struct HomeostasisSpecs {
    /// Target hashrate (MH/s).
    pub target_hashrate_mh: f32,
    /// Target efficiency (MH/s per watt).
    pub target_efficiency: f32,
}

impl Default for HomeostasisSpecs {
    fn default() -> Self {
        Self::for_asset(CryptoAsset::Dynex)
    }
}

impl HomeostasisSpecs {
    /// Get default homeostasis specs for a supported asset.
    pub fn for_asset(asset: CryptoAsset) -> Self {
        match asset {
            CryptoAsset::Dynex => Self {
                target_hashrate_mh: 0.015,
                target_efficiency: 0.015 / 350.0,
            },
            CryptoAsset::Quai => Self {
                target_hashrate_mh: 10.0,
                target_efficiency: 10.0 / 300.0,
            },
            CryptoAsset::Qubic => Self {
                target_hashrate_mh: 0.1,
                target_efficiency: 0.1 / 300.0,
            },
            CryptoAsset::Kaspa => Self {
                target_hashrate_mh: 1000.0, // 1 GH/s
                target_efficiency: 1000.0 / 400.0,
            },
            CryptoAsset::Monero => Self {
                target_hashrate_mh: 0.02, // 20 kH/s
                target_efficiency: 0.02 / 100.0,
            },
            CryptoAsset::Verus => Self {
                target_hashrate_mh: 10.0,
                target_efficiency: 10.0 / 150.0,
            },
        }
    }
}

/// Nominal RTX 5080 boost clock (MHz).  Used to detect thermal throttling.
pub const NOMINAL_CLOCK_MHZ: f32 = 2640.0;

/// Clock floor below which we consider the GPU severely throttled.
pub const THROTTLE_CLOCK_FLOOR: f32 = 2000.0;

/// Throttle clock range for proportional penalty.
pub const THROTTLE_CLOCK_RANGE: f32 = NOMINAL_CLOCK_MHZ - THROTTLE_CLOCK_FLOOR;

// ── Homeostatic setpoints ────────────────────────────────────────────────────

/// Optimal operating-point targets for the RTX 5080 + Ryzen 9 9950X system.
#[derive(Debug, Clone, Copy)]
pub struct ThermalSetpoint {
    /// GPU junction temperature sweet-spot (°C).
    pub optimal_gpu_temp_c: f32,
    /// CPU Tctl sweet-spot (°C).  Zen 5 runs hotter by design.
    pub optimal_cpu_temp_c: f32,
    /// Target board power (W) — below TDP but above idle.
    pub optimal_power_w: f32,
    /// Temperature tolerance band (°C) — beyond optimal ± tolerance the
    /// homeostatic reward goes negative.
    pub temp_tolerance_c: f32,
    /// Power tolerance band (W).
    pub power_tolerance_w: f32,
}

impl Default for ThermalSetpoint {
    fn default() -> Self {
        Self {
            optimal_gpu_temp_c: 75.0,
            optimal_cpu_temp_c: 70.0, // Zen 5 — designed to boost high
            optimal_power_w: 350.0,
            temp_tolerance_c: 15.0,
            power_tolerance_w: 80.0,
        }
    }
}
