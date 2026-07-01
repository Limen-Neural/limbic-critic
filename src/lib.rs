// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]

pub mod critic;
pub mod environment;

pub use critic::{SimpleCritic, TDCritic};
pub use environment::Environment;
