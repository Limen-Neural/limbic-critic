// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]

pub mod critic;
pub mod environment;
pub mod error;
pub mod modulators;

pub use critic::{InvalidAlpha, SimpleCritic, TDCritic};
pub use environment::Environment;
pub use error::{CriticError, CriticField, NonFiniteKind};
pub use modulators::ModulatorVector;
