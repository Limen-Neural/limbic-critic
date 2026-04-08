// ── Q8.8 helpers ─────────────────────────────────────────────────────────────

/// Maximum representable value in unsigned Q8.8 (0xFF.FF = 255 + 255/256).
pub const Q8_8_MAX: u16 = 0xFFFF;

/// Convert a `[0.0, 1.0]` reward to Q8.8 fixed-point (unsigned).
///
/// Clamps to `[0.0, 1.0]` before conversion and caps the result at
/// `Q8_8_MAX` to prevent bit-overflow during reward spikes.
#[inline]
pub fn reward_to_q8_8(reward: f32) -> u16 {
    let clamped = reward.clamp(0.0, 1.0);
    let raw = (clamped * 256.0) as u32; // u32 intermediate prevents u16 overflow
    (raw.min(Q8_8_MAX as u32)) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_8_conversion_roundtrip() {
        assert_eq!(reward_to_q8_8(0.0), 0);
        assert_eq!(reward_to_q8_8(1.0), 256);
        assert_eq!(reward_to_q8_8(0.5), 128);
        // Overflow protection.
        assert_eq!(reward_to_q8_8(1.5), 256); // clamped to 1.0
        assert_eq!(reward_to_q8_8(-0.5), 0);  // clamped to 0.0
    }
}
