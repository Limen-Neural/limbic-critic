// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed errors for checked critic assessment.
//!
//! [`SimpleCritic::try_assess`](crate::SimpleCritic::try_assess) and
//! [`TDCritic::try_assess`](crate::TDCritic::try_assess) reject non-finite
//! environment observations before any modulator is produced and, for the
//! temporal critic, before internal state is committed. The compatibility
//! [`assess`](crate::SimpleCritic::assess) methods keep their documented IEEE
//! `f32` passthrough (including `TDCritic` state poisoning) — see those
//! methods and the crate changelog for the migration path.

/// Classification of a non-finite `f32` that failed critic validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NonFiniteKind {
    /// IEEE NaN (any payload, including negative-NaN).
    Nan,
    /// `+∞`.
    PositiveInfinity,
    /// `−∞`.
    NegativeInfinity,
}

impl NonFiniteKind {
    /// Classify `value` as non-finite, or `None` when it is finite.
    ///
    /// Finite values include subnormals and signed zero.
    #[must_use]
    pub fn classify(value: f32) -> Option<Self> {
        if value.is_nan() {
            Some(Self::Nan)
        } else if value.is_infinite() {
            if value.is_sign_positive() {
                Some(Self::PositiveInfinity)
            } else {
                Some(Self::NegativeInfinity)
            }
        } else {
            None
        }
    }
}

impl std::fmt::Display for NonFiniteKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nan => f.write_str("NaN"),
            Self::PositiveInfinity => f.write_str("+inf"),
            Self::NegativeInfinity => f.write_str("-inf"),
        }
    }
}

/// A validated field or computed intermediate in a critic assessment.
///
/// Input channels (`Objective`, `Volatility`, `Surprise`, `Stress`) are
/// read from [`Environment`](crate::Environment). Intermediates (`TdError`,
/// `EmaReward`) are computed by [`TDCritic`](crate::TDCritic) and checked
/// before state is committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CriticField {
    /// [`Environment::objective`](crate::Environment::objective).
    Objective,
    /// [`Environment::volatility`](crate::Environment::volatility).
    Volatility,
    /// [`Environment::surprise`](crate::Environment::surprise).
    ///
    /// Validated by [`SimpleCritic::try_assess`](crate::SimpleCritic::try_assess)
    /// only. [`TDCritic`](crate::TDCritic) ignores this channel.
    Surprise,
    /// [`Environment::stress`](crate::Environment::stress).
    Stress,
    /// `objective − prev_objective` inside [`TDCritic`](crate::TDCritic).
    TdError,
    /// The EMA of TD errors inside [`TDCritic`](crate::TDCritic).
    EmaReward,
}

impl std::fmt::Display for CriticField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Objective => "objective",
            Self::Volatility => "volatility",
            Self::Surprise => "surprise",
            Self::Stress => "stress",
            Self::TdError => "TD error",
            Self::EmaReward => "EMA reward",
        })
    }
}

/// A checked assessment failed because a value was not finite.
///
/// Each error names the field that failed and whether the value was NaN or
/// an infinity. Successful [`try_assess`](crate::SimpleCritic::try_assess)
/// results have finite modulator fields in the critic's documented ranges.
///
/// # Example
///
/// ```rust
/// use limbic_critic::{
///     CriticError, CriticField, Environment, NonFiniteKind, SimpleCritic,
/// };
///
/// struct NanObjective;
/// impl Environment for NanObjective {
///     fn objective(&self) -> f32 {
///         f32::NAN
///     }
/// }
///
/// let err = SimpleCritic::try_assess(&NanObjective).unwrap_err();
/// assert_eq!(
///     err,
///     CriticError::NonFinite {
///         field: CriticField::Objective,
///         kind: NonFiniteKind::Nan,
///     }
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CriticError {
    /// A raw observation or a computed intermediate was NaN or ±∞.
    NonFinite {
        /// Which input channel or intermediate failed validation.
        field: CriticField,
        /// Whether the value was NaN or an infinity.
        kind: NonFiniteKind,
    },
}

impl CriticError {
    /// Construct a [`NonFinite`](Self::NonFinite) error from a non-finite `f32`.
    ///
    /// Returns `None` when `value` is finite.
    #[must_use]
    pub fn from_non_finite(field: CriticField, value: f32) -> Option<Self> {
        NonFiniteKind::classify(value).map(|kind| Self::NonFinite { field, kind })
    }

    /// The field or intermediate that failed validation.
    #[must_use]
    pub const fn field(&self) -> CriticField {
        match *self {
            Self::NonFinite { field, .. } => field,
        }
    }

    /// Whether the rejected value was NaN or an infinity.
    #[must_use]
    pub const fn kind(&self) -> NonFiniteKind {
        match *self {
            Self::NonFinite { kind, .. } => kind,
        }
    }
}

impl std::fmt::Display for CriticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::NonFinite { field, kind } => {
                write!(f, "non-finite {field} ({kind})")
            }
        }
    }
}

impl std::error::Error for CriticError {}

/// Reject `value` unless it is finite.
pub(crate) fn require_finite(value: f32, field: CriticField) -> Result<f32, CriticError> {
    match CriticError::from_non_finite(field, value) {
        Some(err) => Err(err),
        None => Ok(value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_finite_and_nonfinite() {
        assert_eq!(NonFiniteKind::classify(0.0), None);
        assert_eq!(NonFiniteKind::classify(-0.0), None);
        assert_eq!(NonFiniteKind::classify(f32::MIN_POSITIVE), None);
        assert_eq!(NonFiniteKind::classify(f32::MAX), None);
        assert_eq!(NonFiniteKind::classify(f32::NAN), Some(NonFiniteKind::Nan));
        assert_eq!(
            NonFiniteKind::classify(f32::INFINITY),
            Some(NonFiniteKind::PositiveInfinity)
        );
        assert_eq!(
            NonFiniteKind::classify(f32::NEG_INFINITY),
            Some(NonFiniteKind::NegativeInfinity)
        );
    }

    #[test]
    fn display_and_error_trait() {
        let err = CriticError::NonFinite {
            field: CriticField::Objective,
            kind: NonFiniteKind::Nan,
        };
        assert_eq!(err.to_string(), "non-finite objective (NaN)");
        assert_eq!(
            CriticError::NonFinite {
                field: CriticField::Volatility,
                kind: NonFiniteKind::PositiveInfinity,
            }
            .to_string(),
            "non-finite volatility (+inf)"
        );
        assert_eq!(
            CriticError::NonFinite {
                field: CriticField::Stress,
                kind: NonFiniteKind::NegativeInfinity,
            }
            .to_string(),
            "non-finite stress (-inf)"
        );
        let as_error: &dyn std::error::Error = &err;
        assert!(as_error.source().is_none());
        assert_eq!(err.field(), CriticField::Objective);
        assert_eq!(err.kind(), NonFiniteKind::Nan);
        assert_eq!(CriticField::Surprise.to_string(), "surprise");
        assert_eq!(CriticField::TdError.to_string(), "TD error");
        assert_eq!(CriticField::EmaReward.to_string(), "EMA reward");
    }

    #[test]
    fn from_non_finite_skips_finite() {
        assert!(CriticError::from_non_finite(CriticField::Objective, 1.0).is_none());
        assert_eq!(require_finite(1.5, CriticField::Surprise).unwrap(), 1.5);
        assert!(require_finite(f32::NAN, CriticField::Surprise).is_err());
    }
}
