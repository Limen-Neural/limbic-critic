// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reward-shaping critics: map an [`Environment`] observation into a
//! [`ModulatorVector`].
//!
//! These types are **modulator-mapping primitives**, not a full actor–critic
//! or a learned value function. Two shapers are provided:
//!
//! - [`SimpleCritic`] — stateless clamp of the immediate objective and
//!   optional environment signals.
//! - [`TDCritic`] — stateful mapper of successive objective *deltas*
//!   (`objective − prev_objective`): dopamine is derived from the EMA of
//!   those deltas; acetylcholine is derived from the raw absolute objective
//!   delta before the EMA update. “TD” here means that delta, not
//!   `r + γV(s′) − V(s)`.
//!
//! # Quick start
//!
//! ```rust
//! use limbic_critic::{Environment, SimpleCritic, TDCritic, ModulatorVector};
//!
//! /// Minimal stub environment used only for documentation examples.
//! struct StubEnv {
//!     objective: f32,
//!     surprise: f32,
//! }
//!
//! impl Environment for StubEnv {
//!     fn objective(&self) -> f32 {
//!         self.objective
//!     }
//!     fn surprise(&self) -> f32 {
//!         self.surprise
//!     }
//! }
//!
//! let env = StubEnv {
//!     objective: 0.75,
//!     surprise: 0.3,
//! };
//!
//! // Stateless mapping: dopamine ∈ [0, 1], ACh from Environment::surprise
//! let simple: ModulatorVector = SimpleCritic::assess(&env);
//! assert!((0.0..=1.0).contains(&simple.dopamine));
//! assert_eq!(simple.acetylcholine, 0.3);
//!
//! // Temporal-difference critic: dopamine ∈ [-1, 1] after tanh of EMA(TD)
//! let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
//! let first = td.assess(&env);
//! assert!((-1.0..=1.0).contains(&first.dopamine));
//!
//! // Checked path: reject NaN / ±∞ before producing modulators or
//! // committing TDCritic state. Prefer this when observations may be invalid.
//! let checked = SimpleCritic::try_assess(&env).expect("finite observation");
//! assert_eq!(checked.dopamine, simple.dopamine);
//! ```
//!
//! # Finite observations
//!
//! [`SimpleCritic::try_assess`] and [`TDCritic::try_assess`] reject NaN and
//! ±∞ on every channel they read, and [`TDCritic::try_assess`] does not
//! update `prev_objective` / the TD-error EMA unless the whole assessment
//! succeeds. Successful vectors are finite and lie in the documented
//! per-critic ranges.
//!
//! The compatibility [`SimpleCritic::assess`] / [`TDCritic::assess`]
//! methods keep their IEEE `f32` passthrough (NaN auxiliary signals stay
//! NaN; a non-finite `TDCritic` objective poisons subsequent state). New
//! callers should use `try_assess`.

use crate::environment::Environment;
use crate::error::{CriticError, CriticField, NonFiniteKind, require_finite};
use crate::modulators::ModulatorVector;

/// A stateless reward-shaping map from the current observation to modulators.
///
/// `SimpleCritic` stores no history and does not estimate a value function.
/// Acetylcholine is read directly from [`Environment::surprise`] and clamped
/// to `[0.0, 1.0]`. Use [`TDCritic`] when acetylcholine should be derived
/// from the absolute objective delta (`abs(td_error).tanh()`).
///
/// # Mapping
///
/// | Field | Source | Range |
/// |-------|--------|-------|
/// | `dopamine` | `env.objective()` if positive, else `0.0` | `[0.0, 1.0]` |
/// | `serotonin` | `env.volatility()` | `[0.0, 1.0]` |
/// | `acetylcholine` | `env.surprise()` | `[0.0, 1.0]` |
/// | `norepinephrine` | `env.stress()` | `[0.0, 1.0]` |
///
/// # Example
///
/// ```rust
/// use limbic_critic::{Environment, SimpleCritic};
///
/// struct StubEnv;
/// impl Environment for StubEnv {
///     fn objective(&self) -> f32 { 0.8 }
///     fn surprise(&self) -> f32 { 0.4 }
/// }
///
/// let mods = SimpleCritic::assess(&StubEnv);
/// assert_eq!(mods.dopamine, 0.8);
/// assert_eq!(mods.acetylcholine, 0.4);
/// ```
pub struct SimpleCritic;

impl SimpleCritic {
    /// Calculate neuromodulator concentrations from the current environment.
    ///
    /// # Dopamine
    ///
    /// Positive [`Environment::objective`] values are clamped to
    /// **`[0.0, 1.0]`**. Negative or zero objectives produce `dopamine = 0.0`
    /// (no negative reward signal).
    ///
    /// # Acetylcholine
    ///
    /// Taken from [`Environment::surprise`] and clamped to **`[0.0, 1.0]`**.
    /// This critic does **not** infer ACh from objective deltas.
    ///
    /// # Other fields
    ///
    /// - `serotonin` ← [`Environment::volatility`] clamped to `[0.0, 1.0]`
    /// - `norepinephrine` ← [`Environment::stress`] clamped to `[0.0, 1.0]`
    ///
    /// # Non-finite observations
    ///
    /// This method is the **compatibility** path: current `f32` behavior is
    /// documented rather than sanitized. Prefer [`try_assess`](Self::try_assess)
    /// when observations may be non-finite — that path returns a field-specific
    /// [`CriticError`] and never produces a non-finite vector.
    ///
    /// - A **NaN** objective is not `> 0.0`, so `dopamine` is `0.0`.
    /// - `+∞` objective clamps to `1.0`; `-∞` is treated as non-positive (`0.0`).
    /// - NaN auxiliary signals (`surprise` / `volatility` / `stress`) pass
    ///   through `clamp` as **NaN** (IEEE: NaN comparisons are false).
    /// - `±∞` auxiliary signals clamp to the nearest bound (`0.0` or `1.0`).
    #[must_use]
    pub fn assess(env: &impl Environment) -> ModulatorVector {
        simple_modulators(
            env.objective(),
            env.volatility(),
            env.surprise(),
            env.stress(),
        )
    }

    /// Checked mapping from a finite observation to modulators.
    ///
    /// Reads every [`Environment`] channel this critic uses (`objective`,
    /// `volatility`, `surprise`, `stress`) and returns
    /// [`CriticError::NonFinite`] on the first non-finite value, in that
    /// order. Extreme **finite** values are still clamped as in [`assess`]:
    /// positive objectives and auxiliary signals saturate at `1.0`;
    /// non-positive objectives yield `dopamine = 0.0`.
    ///
    /// On success every [`ModulatorVector`] field is finite and in
    /// `[0.0, 1.0]`.
    ///
    /// # Errors
    ///
    /// Returns [`CriticError::NonFinite`] naming the failing channel
    /// ([`CriticField::Objective`], [`CriticField::Volatility`],
    /// [`CriticField::Surprise`], or [`CriticField::Stress`]) and whether
    /// the value was NaN or ±∞.
    ///
    /// # Example
    ///
    /// ```rust
    /// use limbic_critic::{CriticField, Environment, NonFiniteKind, SimpleCritic};
    ///
    /// struct InfVol;
    /// impl Environment for InfVol {
    ///     fn objective(&self) -> f32 { 0.2 }
    ///     fn volatility(&self) -> f32 { f32::INFINITY }
    /// }
    ///
    /// let err = SimpleCritic::try_assess(&InfVol).unwrap_err();
    /// assert_eq!(err.field(), CriticField::Volatility);
    /// assert_eq!(err.kind(), NonFiniteKind::PositiveInfinity);
    /// ```
    pub fn try_assess(env: &impl Environment) -> Result<ModulatorVector, CriticError> {
        let objective = require_finite(env.objective(), CriticField::Objective)?;
        let volatility = require_finite(env.volatility(), CriticField::Volatility)?;
        let surprise = require_finite(env.surprise(), CriticField::Surprise)?;
        let stress = require_finite(env.stress(), CriticField::Stress)?;
        Ok(simple_modulators(objective, volatility, surprise, stress))
    }
}

/// Stateless SimpleCritic mapping used by both `assess` and `try_assess`.
fn simple_modulators(
    objective: f32,
    volatility: f32,
    surprise: f32,
    stress: f32,
) -> ModulatorVector {
    // Positive objective -> dopamine; non-positive (and NaN) -> nothing.
    let dopamine = if objective > 0.0 {
        objective.clamp(0.0, 1.0)
    } else {
        0.0
    };

    ModulatorVector {
        dopamine,
        serotonin: volatility.clamp(0.0, 1.0),
        acetylcholine: surprise.clamp(0.0, 1.0),
        norepinephrine: stress.clamp(0.0, 1.0),
    }
}

/// A stateful reward-shaping map based on successive objective deltas.
///
/// “TD” here is an exponential moving average (EMA) of
/// `objective − prev_objective`, **not** a learned `V(s)` or actor–critic
/// backup. Dopamine reflects *change* in the raw objective rather than
/// absolute level. Acetylcholine is derived from surprise in that delta
/// (`abs(td_error).tanh()`), not from [`Environment::surprise`].
///
/// # Internal state
///
/// | Field | Meaning |
/// |-------|---------|
/// | `prev_objective` | Objective observed on the previous successful [`assess`](Self::assess) / [`try_assess`](Self::try_assess) call; starts at `0.0`. |
/// | `ema_reward` | EMA of successive TD errors (`objective - prev_objective`); starts at `0.0`. |
/// | `alpha` | EMA learning rate, **enforced** in `(0, 1]` by [`TDCritic::new`]. Higher values weight recent TD errors more heavily. |
///
/// # Mapping
///
/// | Field | Source | Range |
/// |-------|--------|-------|
/// | `dopamine` | `ema_reward.tanh()` | **`[-1.0, 1.0]`** |
/// | `serotonin` | `env.volatility()` | `[0.0, 1.0]` |
/// | `acetylcholine` | `abs(td_error).tanh()` | `[0.0, 1.0]` |
/// | `norepinephrine` | `env.stress()` | `[0.0, 1.0]` |
///
/// # Example
///
/// ```rust
/// use limbic_critic::{Environment, TDCritic};
///
/// struct StubEnv(f32);
/// impl Environment for StubEnv {
///     fn objective(&self) -> f32 { self.0 }
/// }
///
/// let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
/// let step1 = td.assess(&StubEnv(0.0));
/// let step2 = td.assess(&StubEnv(1.0));
/// // Improvement produces a higher (more positive) dopamine signal.
/// assert!(step2.dopamine > step1.dopamine);
/// ```
pub struct TDCritic {
    prev_objective: f32,
    ema_reward: f32,
    alpha: f32, // Learning rate for the EMA
}

/// `alpha` supplied to [`TDCritic::new`] was not in `(0, 1]`.
///
/// Rejected values include `0.0`, negatives, values greater than `1.0`,
/// `NaN`, and infinities. Finite values in `(0, 1]` (including `1.0`)
/// are accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidAlpha;

impl std::fmt::Display for InvalidAlpha {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TDCritic alpha must be in (0, 1] (finite, greater than 0, at most 1)")
    }
}

impl std::error::Error for InvalidAlpha {}

impl TDCritic {
    /// Create a new TD critic with the given EMA learning rate.
    ///
    /// `alpha` controls how quickly the internal EMA of TD errors adapts:
    ///
    /// - **Small `alpha`** (e.g. `0.05`) — smooth, slow reaction to changes.
    /// - **Large `alpha`** (e.g. `0.5`) — fast tracking of recent TD errors.
    /// - **`alpha = 1.0`** — no memory; the EMA equals the latest TD error.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidAlpha`] when `alpha` is not in **`(0, 1]`**.
    /// The predicate is `alpha > 0.0 && alpha <= 1.0`, which also rejects
    /// `NaN` and infinities (IEEE comparisons with `NaN` are false).
    ///
    /// Initial state on success:
    /// - `prev_objective = 0.0`
    /// - `ema_reward = 0.0`
    ///
    /// The first [`assess`](Self::assess) / [`try_assess`](Self::try_assess)
    /// call therefore treats the TD error as `objective - 0.0`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use limbic_critic::TDCritic;
    ///
    /// let critic = TDCritic::new(0.2).expect("alpha in (0, 1]");
    /// // critic is ready; call assess(&env) on each time step
    /// ```
    pub fn new(alpha: f32) -> Result<Self, InvalidAlpha> {
        if !is_valid_alpha(alpha) {
            return Err(InvalidAlpha);
        }
        Ok(Self {
            prev_objective: 0.0,
            ema_reward: 0.0,
            alpha,
        })
    }

    /// Calculate neuromodulator concentrations from the TD error.
    ///
    /// # Algorithm
    ///
    /// 1. `td_error = env.objective() - prev_objective`
    /// 2. Store the current objective as `prev_objective` for the next call.
    /// 3. `acetylcholine = abs(td_error).tanh()`, clamped to `[0.0, 1.0]`.
    /// 4. Update EMA: `ema_reward ← (1 - alpha) * ema_reward + alpha * td_error`.
    /// 5. `dopamine = ema_reward.tanh()`, clamped to **`[-1.0, 1.0]`**.
    /// 6. `serotonin` / `norepinephrine` from `volatility` / `stress`, each
    ///    clamped to `[0.0, 1.0]`.
    ///
    /// Unlike [`SimpleCritic::assess`], this method mutates internal state and
    /// can produce **negative dopamine** when recent TD errors are negative
    /// (worsening outcomes).
    ///
    /// # Non-finite observations
    ///
    /// This method is the **compatibility** path: current `f32` behavior is
    /// documented rather than sanitized. Prefer [`try_assess`](Self::try_assess)
    /// when observations may be non-finite. That path returns a field-specific
    /// [`CriticError`] and **does not** update `prev_objective` or `ema_reward`
    /// on failure, so a later valid observation matches a critic that never
    /// saw the invalid sample.
    ///
    /// - A **NaN** objective produces a NaN TD error. `tanh` / `clamp` then
    ///   leave **NaN** dopamine and acetylcholine (NaN comparisons are false).
    ///   That NaN is stored in `prev_objective` and folded into `ema_reward`,
    ///   so the state is **absorbing**: every later `assess`, even with a
    ///   finite objective, yields NaN TD / dopamine / ACh. There is no reset.
    /// - A **±∞** objective on a finite EMA produces an infinite TD error.
    ///   `tanh` saturates (`tanh(∞) = 1`, `tanh(-∞) = -1`), so that step's
    ///   dopamine is `±1.0` and acetylcholine is `1.0`.
    /// - After non-finite history, a later finite (or opposite-signed
    ///   infinite) objective can make the EMA `∞ + -∞` or `0 * ∞`, so
    ///   **dopamine becomes NaN**. Acetylcholine still saturates from
    ///   `|td_error|.tanh()` when `td_error` is infinite.
    /// - Same-signed successive infinities (`+∞` then `+∞`, or `-∞` then
    ///   `-∞`) produce `td_error = ∞ − ∞ = NaN`, so **both** dopamine and
    ///   acetylcholine are NaN on that step (ACh does not saturate).
    /// - Auxiliary `volatility` / `stress` follow the same clamp rules as
    ///   [`SimpleCritic::assess`]: NaN stays NaN; infinities clamp to bounds.
    ///
    /// # Parameters
    ///
    /// - `env` — environment providing the current objective (and optional
    ///   stress / volatility signals).
    pub fn assess(&mut self, env: &impl Environment) -> ModulatorVector {
        let objective = env.objective();
        let td_error = objective - self.prev_objective;
        self.prev_objective = objective;

        // Update the EMA of the reward
        self.ema_reward = (1.0 - self.alpha) * self.ema_reward + self.alpha * td_error;

        td_modulators(td_error, self.ema_reward, env.volatility(), env.stress())
    }

    /// Checked mapping that rejects non-finite observations without
    /// committing critic state.
    ///
    /// Shared input channels (`objective`, `volatility`, `stress`) are
    /// validated in that order. [`Environment::surprise`] is **not**
    /// read — acetylcholine comes from `|td_error|.tanh()`, matching
    /// [`assess`](Self::assess).
    ///
    /// Intermediates are computed from the candidate observation **before**
    /// `prev_objective` or `ema_reward` are overwritten:
    ///
    /// 1. `td_error = objective − prev_objective`. A NaN delta (only
    ///    possible if prior compatibility [`assess`](Self::assess) already
    ///    poisoned state) is rejected. An overflow `±∞` delta from extreme
    ///    **finite** objectives is saturated to `±f32::MAX` so `tanh`
    ///    still saturates acetylcholine / dopamine.
    /// 2. Candidate EMA `(1 − alpha) * ema_reward + alpha * td_error`.
    ///    NaN is rejected; overflow `±∞` is saturated to `±f32::MAX`.
    /// 3. Only then are `prev_objective` and `ema_reward` updated.
    ///
    /// Extreme finite auxiliary signals are clamped to `[0.0, 1.0]` as in
    /// [`assess`](Self::assess). On success dopamine is finite in
    /// `[-1.0, 1.0]` and the other fields are finite in `[0.0, 1.0]`.
    ///
    /// # Errors
    ///
    /// Returns [`CriticError::NonFinite`] for a non-finite input channel or
    /// a NaN intermediate. A failed call leaves `prev_objective` and
    /// `ema_reward` unchanged.
    ///
    /// # Example
    ///
    /// ```rust
    /// use limbic_critic::{CriticField, Environment, NonFiniteKind, TDCritic};
    ///
    /// struct NanThenValid {
    ///     objective: f32,
    /// }
    /// impl Environment for NanThenValid {
    ///     fn objective(&self) -> f32 { self.objective }
    /// }
    ///
    /// let mut td = TDCritic::new(0.2).expect("alpha in (0, 1]");
    /// let _ = td.try_assess(&NanThenValid { objective: 0.4 }).unwrap();
    /// let err = td
    ///     .try_assess(&NanThenValid { objective: f32::NAN })
    ///     .unwrap_err();
    /// assert_eq!(err.field(), CriticField::Objective);
    /// assert_eq!(err.kind(), NonFiniteKind::Nan);
    /// // State is unchanged: the next finite step matches a fresh critic
    /// // that only saw the valid samples.
    /// let retry = td.try_assess(&NanThenValid { objective: 0.9 }).unwrap();
    /// let mut control = TDCritic::new(0.2).expect("alpha in (0, 1]");
    /// let _ = control.try_assess(&NanThenValid { objective: 0.4 }).unwrap();
    /// let expected = control.try_assess(&NanThenValid { objective: 0.9 }).unwrap();
    /// assert_eq!(retry, expected);
    /// ```
    pub fn try_assess(&mut self, env: &impl Environment) -> Result<ModulatorVector, CriticError> {
        let objective = require_finite(env.objective(), CriticField::Objective)?;
        let volatility = require_finite(env.volatility(), CriticField::Volatility)?;
        let stress = require_finite(env.stress(), CriticField::Stress)?;

        let td_error = shaped_td_error(objective, self.prev_objective)?;
        let ema_reward = shaped_ema(self.ema_reward, self.alpha, td_error)?;
        let mods = td_modulators(td_error, ema_reward, volatility, stress);
        debug_assert!(
            mods.dopamine.is_finite()
                && mods.serotonin.is_finite()
                && mods.acetylcholine.is_finite()
                && mods.norepinephrine.is_finite()
        );

        self.prev_objective = objective;
        self.ema_reward = ema_reward;
        Ok(mods)
    }
}

/// Stateless TD mapping from a (possibly saturated) delta and EMA.
fn td_modulators(td_error: f32, ema_reward: f32, volatility: f32, stress: f32) -> ModulatorVector {
    ModulatorVector {
        dopamine: ema_reward.tanh().clamp(-1.0, 1.0),
        serotonin: volatility.clamp(0.0, 1.0),
        acetylcholine: td_error.abs().tanh().clamp(0.0, 1.0),
        norepinephrine: stress.clamp(0.0, 1.0),
    }
}

/// Finite-input TD error: reject NaN, saturate overflow ±∞ to `±f32::MAX`.
fn shaped_td_error(objective: f32, prev_objective: f32) -> Result<f32, CriticError> {
    saturate_or_reject(objective - prev_objective, CriticField::TdError)
}

/// Finite-input EMA update: reject NaN, saturate overflow ±∞ to `±f32::MAX`.
fn shaped_ema(prev_ema: f32, alpha: f32, td_error: f32) -> Result<f32, CriticError> {
    saturate_or_reject(
        (1.0 - alpha) * prev_ema + alpha * td_error,
        CriticField::EmaReward,
    )
}

/// Reject NaN; replace `±∞` with `±f32::MAX` so `tanh` still saturates.
fn saturate_or_reject(value: f32, field: CriticField) -> Result<f32, CriticError> {
    if value.is_nan() {
        return Err(CriticError::NonFinite {
            field,
            kind: NonFiniteKind::Nan,
        });
    }
    if value.is_infinite() {
        return Ok(value.signum() * f32::MAX);
    }
    Ok(value)
}

/// `alpha` is accepted only in the open-closed interval `(0, 1]`.
///
/// `NaN` and infinities fail the comparison and are rejected.
fn is_valid_alpha(alpha: f32) -> bool {
    alpha > 0.0 && alpha <= 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstEnv(f32);

    impl Environment for ConstEnv {
        fn objective(&self) -> f32 {
            self.0
        }
    }

    struct VolatileEnv {
        steps: Vec<f32>,
        index: usize,
    }

    impl VolatileEnv {
        fn new(values: Vec<f32>) -> Self {
            Self {
                steps: values,
                index: 0,
            }
        }
    }

    impl Environment for VolatileEnv {
        fn objective(&self) -> f32 {
            self.steps[self.index]
        }
    }

    #[test]
    fn test_simple_critic_positive_objective() {
        let env = ConstEnv(0.8);
        let mods = SimpleCritic::assess(&env);
        assert_eq!(mods.dopamine, 0.8);
        assert_eq!(mods.norepinephrine, 0.0);
    }

    #[test]
    fn test_simple_critic_negative_objective() {
        let env = ConstEnv(-0.5);
        let mods = SimpleCritic::assess(&env);
        assert_eq!(mods.dopamine, 0.0);
    }

    #[test]
    fn test_simple_critic_clamping() {
        let env = ConstEnv(5.0);
        let mods = SimpleCritic::assess(&env);
        assert_eq!(mods.dopamine, 1.0);
    }

    #[test]
    fn test_simple_critic_stress() {
        struct StressedEnv;
        impl Environment for StressedEnv {
            fn objective(&self) -> f32 {
                0.5
            }
            fn stress(&self) -> f32 {
                0.9
            }
        }
        let mods = SimpleCritic::assess(&StressedEnv);
        assert_eq!(mods.norepinephrine, 0.9);
    }

    #[test]
    fn test_simple_critic_volatility_serotonin() {
        struct SimpleVolatileEnv;
        impl Environment for SimpleVolatileEnv {
            fn objective(&self) -> f32 {
                0.5
            }
            fn volatility(&self) -> f32 {
                0.6
            }
        }
        let mods = SimpleCritic::assess(&SimpleVolatileEnv);
        assert_eq!(mods.serotonin, 0.6);
    }

    #[test]
    fn test_simple_critic_surprise_acetylcholine() {
        struct SurprisingEnv {
            surprise: f32,
        }
        impl Environment for SurprisingEnv {
            fn objective(&self) -> f32 {
                0.5
            }
            fn surprise(&self) -> f32 {
                self.surprise
            }
        }

        let mods = SimpleCritic::assess(&SurprisingEnv { surprise: 0.7 });
        assert_eq!(mods.acetylcholine, 0.7);
    }

    #[test]
    fn test_simple_critic_surprise_acetylcholine_clamping() {
        struct SurprisingEnv(f32);
        impl Environment for SurprisingEnv {
            fn objective(&self) -> f32 {
                0.0
            }
            fn surprise(&self) -> f32 {
                self.0
            }
        }

        assert_eq!(
            SimpleCritic::assess(&SurprisingEnv(-0.2)).acetylcholine,
            0.0
        );
        assert_eq!(SimpleCritic::assess(&SurprisingEnv(1.5)).acetylcholine, 1.0);
    }

    #[test]
    fn test_td_critic_no_change() {
        let env = ConstEnv(0.5);
        let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
        let mods = td.assess(&env);
        // First call: td_error = 0.5 - 0.0 = 0.5
        // ema starts at 0.0, so ema = 0.9*0.0 + 0.1*0.5 = 0.05
        // dopamine = 0.05.tanh()
        assert!((mods.dopamine - 0.05f32.tanh()).abs() < 1e-6);
    }

    #[test]
    fn test_td_critic_improvement() {
        let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
        let mut env = VolatileEnv::new(vec![0.0, 1.0]);

        let first = td.assess(&env);
        env.index = 1;
        let second = td.assess(&env);

        // Second call has positive improvement
        assert!(second.dopamine > first.dopamine);
    }

    #[test]
    fn test_td_critic_degradation() {
        let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
        let mut env = VolatileEnv::new(vec![1.0, 0.0]);

        let first = td.assess(&env);
        env.index = 1;
        let second = td.assess(&env);

        // Second call has negative td_error
        assert!(second.dopamine < first.dopamine);
    }

    #[test]
    fn test_td_critic_surprise() {
        let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
        let env = ConstEnv(0.5);

        let mods = td.assess(&env);
        // First td_error = 0.5, acetylcholine = 0.5.abs().tanh()
        assert!((mods.acetylcholine - 0.5f32.tanh()).abs() < 1e-6);
    }

    #[test]
    fn test_td_critic_volatility_serotonin() {
        struct TdVolatileEnv {
            v: f32,
        }
        impl Environment for TdVolatileEnv {
            fn objective(&self) -> f32 {
                0.0
            }
            fn volatility(&self) -> f32 {
                self.v
            }
        }
        let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
        let env = TdVolatileEnv { v: 0.6 };
        let mods = td.assess(&env);
        assert_eq!(mods.serotonin, 0.6);
    }

    #[test]
    fn td_critic_rejects_non_positive_or_above_one_alpha() {
        assert!(matches!(TDCritic::new(0.0), Err(InvalidAlpha)));
        assert!(matches!(TDCritic::new(-0.1), Err(InvalidAlpha)));
        assert!(matches!(TDCritic::new(-0.0), Err(InvalidAlpha)));
        assert!(matches!(TDCritic::new(1.0001), Err(InvalidAlpha)));
        assert!(matches!(TDCritic::new(2.0), Err(InvalidAlpha)));
    }

    #[test]
    fn td_critic_rejects_nonfinite_alpha() {
        assert!(matches!(TDCritic::new(f32::NAN), Err(InvalidAlpha)));
        assert!(matches!(TDCritic::new(f32::INFINITY), Err(InvalidAlpha)));
        assert!(matches!(
            TDCritic::new(f32::NEG_INFINITY),
            Err(InvalidAlpha)
        ));
    }

    #[test]
    fn invalid_alpha_display_and_error() {
        assert!(matches!(TDCritic::new(0.0), Err(InvalidAlpha)));
        assert_eq!(
            InvalidAlpha.to_string(),
            "TDCritic alpha must be in (0, 1] (finite, greater than 0, at most 1)"
        );
        let as_error: &dyn std::error::Error = &InvalidAlpha;
        assert!(as_error.source().is_none());
    }

    #[test]
    fn td_critic_accepts_unit_interval_alpha() {
        assert!(TDCritic::new(1.0).is_ok());
        assert!(TDCritic::new(f32::EPSILON).is_ok());
        assert!(TDCritic::new(0.5).is_ok());
    }

    struct AuxEnv {
        objective: f32,
        surprise: f32,
        volatility: f32,
        stress: f32,
    }

    impl Environment for AuxEnv {
        fn objective(&self) -> f32 {
            self.objective
        }
        fn surprise(&self) -> f32 {
            self.surprise
        }
        fn volatility(&self) -> f32 {
            self.volatility
        }
        fn stress(&self) -> f32 {
            self.stress
        }
    }

    #[test]
    fn simple_critic_nonfinite_and_negative_objectives() {
        // NaN / non-positive → dopamine 0 (the `objective > 0` branch).
        assert_eq!(SimpleCritic::assess(&ConstEnv(f32::NAN)).dopamine, 0.0);
        assert_eq!(
            SimpleCritic::assess(&ConstEnv(f32::NEG_INFINITY)).dopamine,
            0.0
        );
        assert_eq!(SimpleCritic::assess(&ConstEnv(-1.5)).dopamine, 0.0);
        assert_eq!(SimpleCritic::assess(&ConstEnv(0.0)).dopamine, 0.0);
        // +∞ clamps to 1.0
        assert_eq!(SimpleCritic::assess(&ConstEnv(f32::INFINITY)).dopamine, 1.0);
    }

    #[test]
    fn simple_critic_nonfinite_aux_signals_follow_clamp() {
        let nan = SimpleCritic::assess(&AuxEnv {
            objective: 0.2,
            surprise: f32::NAN,
            volatility: f32::NAN,
            stress: f32::NAN,
        });
        assert!(nan.acetylcholine.is_nan());
        assert!(nan.serotonin.is_nan());
        assert!(nan.norepinephrine.is_nan());

        let inf = SimpleCritic::assess(&AuxEnv {
            objective: 0.2,
            surprise: f32::INFINITY,
            volatility: f32::NEG_INFINITY,
            stress: f32::INFINITY,
        });
        assert_eq!(inf.acetylcholine, 1.0);
        assert_eq!(inf.serotonin, 0.0);
        assert_eq!(inf.norepinephrine, 1.0);
    }

    #[test]
    fn td_critic_nan_objective_propagates_through_td() {
        let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
        let mods = td.assess(&ConstEnv(f32::NAN));
        assert!(mods.dopamine.is_nan());
        assert!(mods.acetylcholine.is_nan());
    }

    #[test]
    fn td_critic_infinite_objective_saturates_tanh() {
        let mut td = TDCritic::new(1.0).expect("alpha in (0, 1]");
        let pos = td.assess(&ConstEnv(f32::INFINITY));
        assert_eq!(pos.dopamine, 1.0);
        assert_eq!(pos.acetylcholine, 1.0);

        let mut td = TDCritic::new(1.0).expect("alpha in (0, 1]");
        let neg = td.assess(&ConstEnv(f32::NEG_INFINITY));
        assert_eq!(neg.dopamine, -1.0);
        assert_eq!(neg.acetylcholine, 1.0);
    }

    #[test]
    fn td_critic_infinite_history_then_finite_yields_nan_dopamine() {
        let mut td = TDCritic::new(1.0).expect("alpha in (0, 1]");
        let _ = td.assess(&ConstEnv(f32::INFINITY));
        // td_error = finite - ∞ = -∞; ACh saturates, but EMA is 0*∞ + 1*(-∞) → NaN.
        let after = td.assess(&ConstEnv(0.0));
        assert!(after.dopamine.is_nan());
        assert_eq!(after.acetylcholine, 1.0);
    }

    #[test]
    fn td_critic_nan_history_then_finite_stays_nan() {
        let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
        let _ = td.assess(&ConstEnv(f32::NAN));
        let after = td.assess(&ConstEnv(0.5));
        assert!(after.dopamine.is_nan());
        assert!(after.acetylcholine.is_nan());
    }

    #[test]
    fn td_critic_same_signed_infinities_yield_nan_ach() {
        let mut td = TDCritic::new(1.0).expect("alpha in (0, 1]");
        let _ = td.assess(&ConstEnv(f32::INFINITY));
        // td_error = ∞ − ∞ = NaN; ACh does not saturate.
        let after = td.assess(&ConstEnv(f32::INFINITY));
        assert!(after.dopamine.is_nan());
        assert!(after.acetylcholine.is_nan());
    }

    #[test]
    fn simple_critic_clamps_all_modulator_channels() {
        let high = SimpleCritic::assess(&AuxEnv {
            objective: 8.0,
            surprise: 3.0,
            volatility: 4.0,
            stress: 5.0,
        });
        assert_eq!(high.dopamine, 1.0);
        assert_eq!(high.acetylcholine, 1.0);
        assert_eq!(high.serotonin, 1.0);
        assert_eq!(high.norepinephrine, 1.0);

        let low = SimpleCritic::assess(&AuxEnv {
            objective: -2.0,
            surprise: -0.5,
            volatility: -1.0,
            stress: -3.0,
        });
        assert_eq!(low.dopamine, 0.0);
        assert_eq!(low.acetylcholine, 0.0);
        assert_eq!(low.serotonin, 0.0);
        assert_eq!(low.norepinephrine, 0.0);
    }

    #[test]
    fn simple_critic_zero_objective_is_no_dopamine() {
        let mods = SimpleCritic::assess(&AuxEnv {
            objective: 0.0,
            surprise: 0.25,
            volatility: 0.4,
            stress: 0.1,
        });
        assert_eq!(mods.dopamine, 0.0);
        assert_eq!(mods.acetylcholine, 0.25);
        assert_eq!(mods.serotonin, 0.4);
        assert_eq!(mods.norepinephrine, 0.1);
    }

    #[test]
    fn td_critic_dopamine_positive_on_objective_improvement() {
        let mut td = TDCritic::new(0.5).expect("alpha in (0, 1]");
        let baseline = td.assess(&ConstEnv(0.2));
        let improved = td.assess(&ConstEnv(0.9));
        assert!(
            improved.dopamine > baseline.dopamine,
            "improvement should raise dopamine ({baseline:?} -> {improved:?})"
        );
        assert!(improved.dopamine > 0.0);
    }

    #[test]
    fn td_critic_acetylcholine_is_abs_td_tanh() {
        let mut td = TDCritic::new(0.3).expect("alpha in (0, 1]");
        // First assess seeds prev_objective = 0.4; td_error = 0.4.
        let first = td.assess(&ConstEnv(0.4));
        assert!((first.acetylcholine - 0.4f32.abs().tanh()).abs() < 1e-6);

        // Drop to -0.6: td_error = -1.0; ACh = tanh(1.0), independent of sign.
        let drop = td.assess(&ConstEnv(-0.6));
        assert!((drop.acetylcholine - 1.0f32.tanh()).abs() < 1e-6);

        let mut td_up = TDCritic::new(0.3).expect("alpha in (0, 1]");
        let _ = td_up.assess(&ConstEnv(0.4));
        let rise = td_up.assess(&ConstEnv(1.4)); // td_error = +1.0
        assert!((rise.acetylcholine - drop.acetylcholine).abs() < 1e-6);
    }

    #[test]
    fn td_critic_negative_and_zero_objectives() {
        let mut td = TDCritic::new(1.0).expect("alpha in (0, 1]");
        // alpha = 1: EMA equals the latest delta. First call: 0.0 - 0.0 = 0.
        let zero = td.assess(&ConstEnv(0.0));
        assert!((zero.dopamine - 0.0f32.tanh()).abs() < 1e-6);
        assert!((zero.acetylcholine - 0.0f32.tanh()).abs() < 1e-6);

        // Worsening into negative territory: td_error = -0.8.
        let worse = td.assess(&ConstEnv(-0.8));
        assert!(worse.dopamine < 0.0);
        assert!((worse.dopamine - (-0.8f32).tanh()).abs() < 1e-6);
        assert!((worse.acetylcholine - 0.8f32.tanh()).abs() < 1e-6);

        // Recovery toward zero is still an improvement: td_error = +0.8.
        let recover = td.assess(&ConstEnv(0.0));
        assert!(recover.dopamine > 0.0);
        assert!((recover.dopamine - 0.8f32.tanh()).abs() < 1e-6);
    }

    #[test]
    fn td_critic_alpha_one_tracks_latest_delta_only() {
        let mut td = TDCritic::new(1.0).expect("alpha in (0, 1]");
        let _ = td.assess(&ConstEnv(2.0));
        let second = td.assess(&ConstEnv(2.5));
        // td_error = 0.5; alpha = 1 ⇒ ema = 0.5; dopamine = tanh(0.5).
        assert!((second.dopamine - 0.5f32.tanh()).abs() < 1e-6);
    }

    fn finite_env() -> AuxEnv {
        AuxEnv {
            objective: 0.4,
            surprise: 0.1,
            volatility: 0.2,
            stress: 0.3,
        }
    }

    fn assert_non_finite(err: CriticError, field: CriticField, kind: NonFiniteKind) {
        assert_eq!(err.field(), field);
        assert_eq!(err.kind(), kind);
    }

    fn assert_simple_bounds(mods: ModulatorVector) {
        assert!(mods.dopamine.is_finite() && (0.0..=1.0).contains(&mods.dopamine));
        assert!(mods.serotonin.is_finite() && (0.0..=1.0).contains(&mods.serotonin));
        assert!(mods.acetylcholine.is_finite() && (0.0..=1.0).contains(&mods.acetylcholine));
        assert!(mods.norepinephrine.is_finite() && (0.0..=1.0).contains(&mods.norepinephrine));
    }

    fn assert_td_bounds(mods: ModulatorVector) {
        assert!(mods.dopamine.is_finite() && (-1.0..=1.0).contains(&mods.dopamine));
        assert!(mods.serotonin.is_finite() && (0.0..=1.0).contains(&mods.serotonin));
        assert!(mods.acetylcholine.is_finite() && (0.0..=1.0).contains(&mods.acetylcholine));
        assert!(mods.norepinephrine.is_finite() && (0.0..=1.0).contains(&mods.norepinephrine));
    }

    fn non_finite_cases() -> [(f32, NonFiniteKind); 3] {
        [
            (f32::NAN, NonFiniteKind::Nan),
            (f32::INFINITY, NonFiniteKind::PositiveInfinity),
            (f32::NEG_INFINITY, NonFiniteKind::NegativeInfinity),
        ]
    }

    #[test]
    fn try_assess_rejects_each_simple_channel_independently() {
        for (value, kind) in non_finite_cases() {
            let mut env = finite_env();
            env.objective = value;
            assert_non_finite(
                SimpleCritic::try_assess(&env).unwrap_err(),
                CriticField::Objective,
                kind,
            );

            env = finite_env();
            env.volatility = value;
            assert_non_finite(
                SimpleCritic::try_assess(&env).unwrap_err(),
                CriticField::Volatility,
                kind,
            );

            env = finite_env();
            env.surprise = value;
            assert_non_finite(
                SimpleCritic::try_assess(&env).unwrap_err(),
                CriticField::Surprise,
                kind,
            );

            env = finite_env();
            env.stress = value;
            assert_non_finite(
                SimpleCritic::try_assess(&env).unwrap_err(),
                CriticField::Stress,
                kind,
            );
        }
    }

    #[test]
    fn try_assess_rejects_each_td_shared_channel_independently() {
        for (value, kind) in non_finite_cases() {
            let mut env = finite_env();
            env.objective = value;
            let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
            assert_non_finite(
                td.try_assess(&env).unwrap_err(),
                CriticField::Objective,
                kind,
            );

            env = finite_env();
            env.volatility = value;
            let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
            assert_non_finite(
                td.try_assess(&env).unwrap_err(),
                CriticField::Volatility,
                kind,
            );

            env = finite_env();
            env.stress = value;
            let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
            assert_non_finite(td.try_assess(&env).unwrap_err(), CriticField::Stress, kind);
        }
    }

    #[test]
    fn try_assess_shared_channel_errors_match_between_critics() {
        for (value, kind) in non_finite_cases() {
            let cases = [
                (
                    AuxEnv {
                        objective: value,
                        surprise: 0.1,
                        volatility: 0.2,
                        stress: 0.3,
                    },
                    CriticField::Objective,
                ),
                (
                    AuxEnv {
                        objective: 0.4,
                        surprise: 0.1,
                        volatility: value,
                        stress: 0.3,
                    },
                    CriticField::Volatility,
                ),
                (
                    AuxEnv {
                        objective: 0.4,
                        surprise: 0.1,
                        volatility: 0.2,
                        stress: value,
                    },
                    CriticField::Stress,
                ),
            ];
            for (env, field) in cases {
                let simple = SimpleCritic::try_assess(&env).unwrap_err();
                let td = TDCritic::new(0.25)
                    .expect("alpha in (0, 1]")
                    .try_assess(&env)
                    .unwrap_err();
                assert_eq!(simple, td);
                assert_non_finite(simple, field, kind);
            }
        }
    }

    #[test]
    fn try_assess_td_ignores_nonfinite_surprise() {
        let env = AuxEnv {
            objective: 0.4,
            surprise: f32::NAN,
            volatility: 0.2,
            stress: 0.3,
        };
        let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
        let mods = td.try_assess(&env).expect("surprise is not a TD input");
        assert_td_bounds(mods);
        assert!(SimpleCritic::try_assess(&env).is_err());
    }

    #[test]
    fn try_assess_reports_first_nonfinite_channel_in_documented_order() {
        let all_bad = AuxEnv {
            objective: f32::NAN,
            surprise: f32::INFINITY,
            volatility: f32::NEG_INFINITY,
            stress: f32::NAN,
        };
        assert_eq!(
            SimpleCritic::try_assess(&all_bad).unwrap_err().field(),
            CriticField::Objective
        );

        let aux_bad = AuxEnv {
            objective: 0.1,
            surprise: f32::NAN,
            volatility: f32::INFINITY,
            stress: f32::NAN,
        };
        assert_eq!(
            SimpleCritic::try_assess(&aux_bad).unwrap_err().field(),
            CriticField::Volatility
        );
        let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
        assert_eq!(
            td.try_assess(&aux_bad).unwrap_err().field(),
            CriticField::Volatility
        );
    }

    #[test]
    fn try_assess_invalid_sample_does_not_poison_td_or_simple_output() {
        let valid = finite_env();
        let retry = AuxEnv {
            objective: 0.85,
            surprise: 0.15,
            volatility: 0.25,
            stress: 0.05,
        };

        let invalid_envs = [
            AuxEnv {
                objective: f32::NAN,
                surprise: 0.1,
                volatility: 0.2,
                stress: 0.3,
            },
            AuxEnv {
                objective: 0.4,
                surprise: 0.1,
                volatility: f32::INFINITY,
                stress: 0.3,
            },
            AuxEnv {
                objective: 0.4,
                surprise: 0.1,
                volatility: 0.2,
                stress: f32::NEG_INFINITY,
            },
        ];

        for invalid in invalid_envs {
            let mut observed = TDCritic::new(0.2).expect("alpha in (0, 1]");
            let mut control = TDCritic::new(0.2).expect("alpha in (0, 1]");
            assert_eq!(
                observed.try_assess(&valid).unwrap(),
                control.try_assess(&valid).unwrap()
            );
            let simple_err = SimpleCritic::try_assess(&invalid).unwrap_err();
            let td_err = observed.try_assess(&invalid).unwrap_err();
            assert_eq!(simple_err, td_err);
            assert_eq!(
                observed.try_assess(&retry).unwrap(),
                control.try_assess(&retry).unwrap()
            );
            assert_simple_bounds(SimpleCritic::try_assess(&retry).unwrap());
        }
    }

    #[test]
    fn try_assess_td_surprise_nan_does_not_diverge_from_control() {
        let valid = finite_env();
        let surprise_nan = AuxEnv {
            objective: 0.4,
            surprise: f32::NAN,
            volatility: 0.2,
            stress: 0.3,
        };
        let retry = AuxEnv {
            objective: 0.9,
            surprise: 0.0,
            volatility: 0.2,
            stress: 0.3,
        };

        let mut observed = TDCritic::new(0.2).expect("alpha in (0, 1]");
        let mut control = TDCritic::new(0.2).expect("alpha in (0, 1]");
        assert_eq!(
            observed.try_assess(&valid).unwrap(),
            control.try_assess(&valid).unwrap()
        );
        // Surprise is ignored, so this *is* a valid TD step and both critics take it.
        assert_eq!(
            observed.try_assess(&surprise_nan).unwrap(),
            control.try_assess(&surprise_nan).unwrap()
        );
        assert_eq!(
            observed.try_assess(&retry).unwrap(),
            control.try_assess(&retry).unwrap()
        );
    }

    #[test]
    fn try_assess_extreme_finite_values_are_clamped() {
        let high = AuxEnv {
            objective: f32::MAX,
            surprise: f32::MAX,
            volatility: 1.0e30,
            stress: 8.0,
        };
        let simple_high = SimpleCritic::try_assess(&high).unwrap();
        assert_eq!(simple_high.dopamine, 1.0);
        assert_eq!(simple_high.acetylcholine, 1.0);
        assert_eq!(simple_high.serotonin, 1.0);
        assert_eq!(simple_high.norepinephrine, 1.0);

        let low = AuxEnv {
            objective: f32::MIN,
            surprise: -1.0e20,
            volatility: f32::MIN,
            stress: -0.25,
        };
        let simple_low = SimpleCritic::try_assess(&low).unwrap();
        assert_eq!(simple_low.dopamine, 0.0);
        assert_eq!(simple_low.acetylcholine, 0.0);
        assert_eq!(simple_low.serotonin, 0.0);
        assert_eq!(simple_low.norepinephrine, 0.0);

        let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
        let first = td.try_assess(&high).unwrap();
        assert_td_bounds(first);
        assert_eq!(first.serotonin, 1.0);
        assert_eq!(first.norepinephrine, 1.0);
        assert_eq!(first.acetylcholine, 1.0);
        assert_eq!(first.dopamine, 1.0);

        // Opposite-extreme finite sample: overflow TD error is saturated, not
        // rejected, and state from the previous success remains usable.
        let opposite = AuxEnv {
            objective: f32::MIN,
            surprise: 0.0,
            volatility: 0.0,
            stress: 0.0,
        };
        let second = td.try_assess(&opposite).unwrap();
        assert_td_bounds(second);
        assert_eq!(second.dopamine, -1.0);
        assert_eq!(second.acetylcholine, 1.0);

        let modest = td.try_assess(&finite_env()).unwrap();
        assert_td_bounds(modest);
    }

    #[test]
    fn try_assess_matches_assess_on_finite_inputs() {
        let samples = [
            finite_env(),
            AuxEnv {
                objective: -2.0,
                surprise: 1.5,
                volatility: -0.4,
                stress: 0.0,
            },
            AuxEnv {
                objective: 0.0,
                surprise: 0.25,
                volatility: 0.4,
                stress: 0.1,
            },
        ];
        for env in samples {
            assert_eq!(
                SimpleCritic::assess(&env),
                SimpleCritic::try_assess(&env).unwrap()
            );
        }

        let sequence = [0.0, 0.4, -0.2, 1.2, 0.8];
        let mut via_assess = TDCritic::new(0.3).expect("alpha in (0, 1]");
        let mut via_try = TDCritic::new(0.3).expect("alpha in (0, 1]");
        for objective in sequence {
            let env = ConstEnv(objective);
            assert_eq!(via_assess.assess(&env), via_try.try_assess(&env).unwrap());
        }
    }

    #[test]
    fn try_assess_nan_intermediate_from_poisoned_assess_state() {
        let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
        let _ = td.assess(&ConstEnv(f32::NAN));
        let err = td.try_assess(&ConstEnv(0.5)).unwrap_err();
        assert_eq!(err.field(), CriticField::TdError);
        assert_eq!(err.kind(), NonFiniteKind::Nan);
        // Failed try_assess did not overwrite poisoned prev_objective.
        let still = td.try_assess(&ConstEnv(0.25)).unwrap_err();
        assert_eq!(still.field(), CriticField::TdError);
    }

    #[test]
    fn property_try_assess_outputs_are_finite_and_in_range() {
        let interesting = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
            1.0e-20,
            -1.0e10,
            42.0,
        ];
        for &objective in &interesting {
            for &surprise in &interesting {
                for &volatility in &interesting {
                    for &stress in &interesting {
                        let env = AuxEnv {
                            objective,
                            surprise,
                            volatility,
                            stress,
                        };
                        let simple = SimpleCritic::try_assess(&env).unwrap();
                        assert_simple_bounds(simple);
                    }
                }
            }
        }

        let mut td = TDCritic::new(0.15).expect("alpha in (0, 1]");
        let mut bits: u32 = 0xA5A5_5A5A;
        for _ in 0..400 {
            bits ^= bits << 13;
            bits ^= bits >> 17;
            bits ^= bits << 5;
            let objective = f32::from_bits(bits);
            if !objective.is_finite() {
                continue;
            }
            bits ^= bits << 13;
            bits ^= bits >> 17;
            bits ^= bits << 5;
            let volatility = f32::from_bits(bits);
            if !volatility.is_finite() {
                continue;
            }
            bits ^= bits << 13;
            bits ^= bits >> 17;
            bits ^= bits << 5;
            let stress = f32::from_bits(bits);
            if !stress.is_finite() {
                continue;
            }
            let env = AuxEnv {
                objective,
                surprise: 0.0,
                volatility,
                stress,
            };
            let mods = td.try_assess(&env).unwrap();
            assert_td_bounds(mods);
            let simple = SimpleCritic::try_assess(&env).unwrap();
            assert_simple_bounds(simple);
        }
    }

    #[test]
    fn try_assess_simple_surprise_invalid_does_not_affect_retry() {
        let valid = finite_env();
        let invalid = AuxEnv {
            objective: 0.4,
            surprise: f32::NAN,
            volatility: 0.2,
            stress: 0.3,
        };
        assert_eq!(
            SimpleCritic::try_assess(&invalid).unwrap_err().field(),
            CriticField::Surprise
        );
        assert_eq!(
            SimpleCritic::try_assess(&valid).unwrap(),
            SimpleCritic::assess(&valid)
        );
    }
}
