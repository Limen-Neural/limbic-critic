// SPDX-License-Identifier: MIT OR Apache-2.0

//! Example: using limbic-critic with a generic `Environment`.
//!
//! This demonstrates how any external system (trading bot, game, LLM training, etc.)
//! can supply an objective value and stress/volatility signals, and receive
//! neuromodulator concentrations for reward-modulated learning.

use limbic_critic::{Environment, SimpleCritic, TDCritic};

/// A minimal environment that reports a scalar objective (e.g. PnL, accuracy, score).
struct GenericEnv {
    objective: f32,
    stress: f32,
    volatility: f32,
}

impl Environment for GenericEnv {
    fn objective(&self) -> f32 {
        self.objective
    }
    fn stress(&self) -> f32 {
        self.stress
    }
    fn volatility(&self) -> f32 {
        self.volatility
    }
}

fn main() {
    // Example 1: positive objective, low stress
    let env = GenericEnv {
        objective: 0.75,
        stress: 0.1,
        volatility: 0.2,
    };
    let mods = SimpleCritic::assess(&env);
    println!(
        "SimpleCritic: dopamine={:.3} serotonin={:.3} acetylcholine={:.3} norepinephrine={:.3}",
        mods.dopamine, mods.serotonin, mods.acetylcholine, mods.norepinephrine
    );

    // Example 2: TD critic over a volatile sequence
    let mut td = TDCritic::new(0.2);
    for &obj in &[0.0, 0.4, 0.3, 0.9] {
        let e = GenericEnv {
            objective: obj,
            stress: 0.0,
            volatility: 0.5,
        };
        let m = td.assess(&e);
        println!("TD step obj={:.1} -> dopamine={:.3}", obj, m.dopamine);
    }
}
