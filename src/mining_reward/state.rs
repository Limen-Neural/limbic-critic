use crate::RewardableState;

use super::config::{
    ALPHA_EFFICIENCY, BETA_THERMAL, GAMMA_WASTE, EMA_ALPHA, REWARD_CLAMP,
    GPU_THERMAL_PENALTY_ONSET, GPU_THERMAL_DIVISOR, CPU_THERMAL_PENALTY_ONSET, CPU_THERMAL_DIVISOR,
    POWER_PENALTY_ONSET, POWER_PENALTY_DIVISOR, NOMINAL_CLOCK_MHZ, THROTTLE_CLOCK_FLOOR,
    THROTTLE_CLOCK_RANGE, HomeostasisSpecs, ThermalSetpoint,
};

// ── Reward state machine ─────────────────────────────────────────────────────

/// EMA-smoothed mining-efficiency reward computer.
///
/// All state is in fixed-size scalar fields.  `compute()` performs only
/// stack arithmetic — **zero heap allocation** on the hot path.
#[derive(Debug, Clone)]
pub struct MiningRewardState {
    /// Homeostatic operating-point targets.
    pub setpoint: ThermalSetpoint,
    /// Exponential moving average of the composite reward.
    pub ema_reward: f32,
    /// Previous-tick normalised hashrate (for stability delta).
    prev_hashrate_norm: f32,
    /// Ticks where the GPU was NOT throttled (clock ≥ THROTTLE_CLOCK_FLOOR).
    uptime_ticks: u64,
    /// Total ticks observed (denominator for uptime ratio).
    total_ticks: u64,
}

impl Default for MiningRewardState {
    fn default() -> Self {
        Self::new()
    }
}

impl MiningRewardState {
    pub fn new() -> Self {
        Self {
            setpoint: ThermalSetpoint::default(),
            ema_reward: 0.0,
            prev_hashrate_norm: 0.0,
            uptime_ticks: 0,
            total_ticks: 0,
        }
    }

    /// Compute the mining-efficiency dopamine signal for this tick.
    ///
    /// Accepts the current GPU telemetry and an optional `SystemTelemetry`
    /// reference for CPU thermal data.  Returns the EMA-smoothed reward
    /// in `[-0.8, 0.8]`, suitable for direct assignment to
    /// `NeuroModulators.mining_dopamine`.
    ///
    /// # Hot-path guarantee
    ///
    /// This function touches only stack temporaries and `self` scalars.
    /// No `Vec`, `String`, `Box`, or any heap allocation.
    pub fn compute(
        &mut self,
        telem: &impl RewardableState,
        specs: &HomeostasisSpecs,
        cpu_temp_c: Option<f32>,
    ) -> f32 {
        // ── bookkeeping ──────────────────────────────────────────────
        self.total_ticks += 1;
        if telem.gpu_clock_mhz() >= THROTTLE_CLOCK_FLOOR {
            self.uptime_ticks += 1;
        }

        // ── 1. Mining Efficiency (positive term) ─────────────────────
        let hashrate_norm =
            (telem.hashrate_mh() / specs.target_hashrate_mh).clamp(0.0, 1.0);
        let hashrate_stability =
            1.0 - (hashrate_norm - self.prev_hashrate_norm).abs();
        self.prev_hashrate_norm = hashrate_norm;

        let hash_per_watt = (telem.hashrate_mh() / telem.power_w().max(1.0))
            / specs.target_efficiency;
        let hash_per_watt_clamped = hash_per_watt.clamp(0.0, 1.0);

        let uptime_ratio = if self.total_ticks > 0 {
            self.uptime_ticks as f32 / self.total_ticks as f32
        } else {
            1.0
        };

        let mining_efficiency =
            hashrate_stability * hash_per_watt_clamped * uptime_ratio;

        // ── 2. Thermal Stress (negative term) ────────────────────────
        let gpu_thermal = ((telem.gpu_temp_c() - GPU_THERMAL_PENALTY_ONSET)
            / GPU_THERMAL_DIVISOR)
            .clamp(0.0, 1.0);

        let power_excess = ((telem.power_w() - POWER_PENALTY_ONSET)
            / POWER_PENALTY_DIVISOR)
            .clamp(0.0, 1.0);

        let cpu_thermal = cpu_temp_c
            .map(|t| {
                ((t - CPU_THERMAL_PENALTY_ONSET) / CPU_THERMAL_DIVISOR)
                    .clamp(0.0, 1.0)
            })
            .unwrap_or(0.0);

        // Worst-case drives the penalty — we want ANY thermal breach to
        // suppress learning, not average them away.
        let thermal_stress =
            gpu_thermal.max(power_excess).max(cpu_thermal);

        // ── 3. Energy Waste (negative term) ──────────────────────────
        let throttle_penalty = if telem.gpu_clock_mhz() < NOMINAL_CLOCK_MHZ {
            ((NOMINAL_CLOCK_MHZ - telem.gpu_clock_mhz()) / THROTTLE_CLOCK_RANGE)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };

        let power_inefficiency = (1.0 - hash_per_watt_clamped).clamp(0.0, 1.0);

        let energy_waste =
            throttle_penalty * 0.7 + power_inefficiency * 0.3;

        // ── Composite ────────────────────────────────────────────────
        let raw_reward = ALPHA_EFFICIENCY * mining_efficiency
            - BETA_THERMAL * thermal_stress
            - GAMMA_WASTE * energy_waste;

        // ── EMA smoothing ────────────────────────────────────────────
        self.ema_reward =
            (1.0 - EMA_ALPHA) * self.ema_reward + EMA_ALPHA * raw_reward;

        self.ema_reward.clamp(-REWARD_CLAMP, REWARD_CLAMP)
    }

    /// Bell-curve homeostatic reward.
    ///
    /// Returns 1.0 at `optimal`, decays quadratically toward 0.0 at
    /// `optimal ± tolerance`, and goes negative beyond.
    #[inline]
    pub fn homeostatic_reward(
        value: f32,
        optimal: f32,
        tolerance: f32,
    ) -> f32 {
        let deviation = ((value - optimal) / tolerance).powi(2);
        (1.0 - deviation).clamp(-0.5, 1.0)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTelem {
        hashrate_mh: f32,
        power_w: f32,
        gpu_temp_c: f32,
        gpu_clock_mhz: f32,
    }

    impl RewardableState for MockTelem {
        fn hashrate_mh(&self) -> f32 { self.hashrate_mh }
        fn power_w(&self) -> f32 { self.power_w }
        fn gpu_temp_c(&self) -> f32 { self.gpu_temp_c }
        fn gpu_clock_mhz(&self) -> f32 { self.gpu_clock_mhz }
        fn vddcr_gfx_v(&self) -> f32 { 1.0 }
        fn ocean_intel(&self) -> f32 { 0.0 }
    }

    /// Helper: build a mock telemetry with the given overrides.
    fn telem(
        hashrate_mh: f32,
        power_w: f32,
        gpu_temp_c: f32,
        gpu_clock_mhz: f32,
    ) -> MockTelem {
        MockTelem {
            hashrate_mh,
            power_w,
            gpu_temp_c,
            gpu_clock_mhz,
        }
    }

    #[test]
    fn optimal_conditions_positive_reward() {
        let mut state = MiningRewardState::new();
        let specs = HomeostasisSpecs::default();
        // Prime the EMA with one tick so prev_hashrate_norm is set.
        let t = telem(0.012, 340.0, 72.0, 2640.0);
        state.compute(&t, &specs, Some(68.0));

        // Second tick at same conditions — stability = 1.0.
        let r = state.compute(&t, &specs, Some(68.0));
        assert!(r > 0.0, "optimal conditions should yield positive reward, got {r}");
    }

    #[test]
    fn thermal_stress_negative() {
        let mut state = MiningRewardState::new();
        let specs = HomeostasisSpecs::default();
        // Thermal emergency: 95°C GPU + throttled clock + excess power.
        // When the GPU is thermally throttling, hashrate drops AND temp is high.
        let t = telem(0.003, 440.0, 95.0, 1900.0);
        // Warm up EMA.
        for _ in 0..20 {
            state.compute(&t, &specs, Some(68.0));
        }
        let r = state.compute(&t, &specs, Some(68.0));
        assert!(r < 0.0, "thermal emergency should produce negative reward, got {r}");
    }

    #[test]
    fn power_excess_penalty() {
        let mut state = MiningRewardState::new();
        let specs = HomeostasisSpecs::default();
        let t = telem(0.012, 450.0, 72.0, 2640.0); // 450W
        for _ in 0..20 {
            state.compute(&t, &specs, Some(68.0));
        }
        let r_high_power = state.compute(&t, &specs, Some(68.0));

        let mut state2 = MiningRewardState::new();
        let t2 = telem(0.012, 300.0, 72.0, 2640.0); // 300W
        for _ in 0..20 {
            state2.compute(&t2, &specs, Some(68.0));
        }
        let r_low_power = state2.compute(&t2, &specs, Some(68.0));

        assert!(
            r_low_power > r_high_power,
            "300W should reward higher than 450W: {r_low_power} vs {r_high_power}"
        );
    }

    #[test]
    fn ema_smoothing_dampens_spikes() {
        let mut state = MiningRewardState::new();
        let specs = HomeostasisSpecs::default();
        let good = telem(0.012, 340.0, 72.0, 2640.0);
        // Build up positive EMA.
        for _ in 0..50 {
            state.compute(&good, &specs, Some(68.0));
        }
        let before = state.ema_reward;

        // Sudden bad tick.
        let bad = telem(0.001, 450.0, 95.0, 1800.0);
        let after = state.compute(&bad, &specs, Some(92.0));

        // EMA should not crash all the way to the raw bad value.
        assert!(
            after > -0.5,
            "EMA should dampen a single bad tick, got {after}"
        );
        assert!(
            after < before,
            "bad tick should still pull EMA down: {before} -> {after}"
        );
    }

    #[test]
    fn zero_hashrate_near_zero_efficiency() {
        let mut state = MiningRewardState::new();
        let specs = HomeostasisSpecs::default();
        let t = telem(0.0, 150.0, 55.0, 2640.0); // idle GPU
        for _ in 0..20 {
            state.compute(&t, &specs, None);
        }
        let r = state.compute(&t, &specs, None);
        // Should be slightly negative (energy waste from power_inefficiency)
        // but not a severe penalty.
        assert!(
            r > -0.3,
            "zero hashrate at idle power should not be severely penalised, got {r}"
        );
    }

    #[test]
    fn cpu_thermal_zen5_no_chronic_pain() {
        let mut state = MiningRewardState::new();
        let specs = HomeostasisSpecs::default();
        let t = telem(0.012, 340.0, 72.0, 2640.0);
        // Zen 5 at 80°C — normal all-core boost.  Should NOT trigger penalty.
        for _ in 0..20 {
            state.compute(&t, &specs, Some(80.0));
        }
        let r = state.compute(&t, &specs, Some(80.0));
        assert!(
            r > 0.0,
            "Zen 5 at 80°C should not drag reward negative, got {r}"
        );
    }

    #[test]
    fn homeostatic_bell_curve() {
        // At optimal → 1.0.
        let r = MiningRewardState::homeostatic_reward(75.0, 75.0, 15.0);
        assert!((r - 1.0).abs() < 1e-6, "at optimal: {r}");

        // At tolerance boundary → 0.0.
        let r = MiningRewardState::homeostatic_reward(90.0, 75.0, 15.0);
        assert!(r.abs() < 1e-6, "at tolerance boundary: {r}");

        // Beyond tolerance → negative.
        let r = MiningRewardState::homeostatic_reward(100.0, 75.0, 15.0);
        assert!(r < 0.0, "beyond tolerance should be negative: {r}");
    }

    #[test]
    fn reward_clamp_prevents_saturation() {
        let mut state = MiningRewardState::new();
        let specs = HomeostasisSpecs::default();
        let perfect = telem(0.015, 300.0, 65.0, 2700.0);
        for _ in 0..500 {
            state.compute(&perfect, &specs, Some(60.0));
        }
        assert!(
            state.ema_reward <= REWARD_CLAMP,
            "EMA should never exceed clamp: {}",
            state.ema_reward
        );
        assert!(
            state.ema_reward >= -REWARD_CLAMP,
            "EMA should never go below -clamp: {}",
            state.ema_reward
        );
    }

    #[test]
    fn kaspa_high_hashrate_normalization() {
        let mut state = MiningRewardState::new();
        let specs = HomeostasisSpecs::for_asset(CryptoAsset::Kaspa);
        // 1 GH/s is the target.
        let t = MockTelem {
            hashrate_mh: 1000.0, // 1 GH/s
            power_w: 400.0,
            gpu_temp_c: 70.0,
            gpu_clock_mhz: 2640.0,
        };
        // Warm up EMA.
        for _ in 0..50 {
            state.compute(&t, &specs, Some(65.0));
        }
        let r = state.compute(&t, &specs, Some(65.0));
        assert!(r > 0.5, "Kaspa at target should have high positive reward, got {r}");

        // Half the target.
        let t_half = MockTelem { hashrate_mh: 500.0, ..t };
        let r_half = state.compute(&t_half, &specs, Some(65.0));
        assert!(r_half < r, "Lower hashrate on Kaspa should lower reward: {r_half} < {r}");
    }

    #[test]
    fn monero_low_hashrate_normalization() {
        let mut state = MiningRewardState::new();
        let specs = HomeostasisSpecs::for_asset(CryptoAsset::Monero);
        // 20 kH/s is the target.
        let t = MockTelem {
            hashrate_mh: 0.02, // 20 kH/s
            power_w: 100.0,
            gpu_temp_c: 60.0,
            gpu_clock_mhz: 2640.0,
        };
        // Warm up EMA.
        for _ in 0..50 {
            state.compute(&t, &specs, Some(50.0));
        }
        let r = state.compute(&t, &specs, Some(50.0));
        assert!(r > 0.5, "Monero at target should have high positive reward, got {r}");
    }
}
