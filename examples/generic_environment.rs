/// Domain-agnostic example: a simple simulation environment.
///
/// This demonstrates how any external system can implement the `Environment`
/// trait for use with `limbic-critic` reward shaping.
use limbic_critic::{Environment, SimpleCritic, TDCritic};
use std::f32::consts::PI;

/// A simple pendulum swing-up simulation.
///
/// The objective is the negative angle error: 0 = upright, -PI = hanging down.
struct PendulumEnv {
    angle: f32,    // radians, 0 = upright
    velocity: f32, // radians per step
}

impl PendulumEnv {
    fn new() -> Self {
        Self {
            angle: PI, // start hanging down
            velocity: 0.0,
        }
    }

    fn step(&mut self, torque: f32) {
        let dt = 0.05;
        let gravity = 9.81;
        let length = 1.0;
        let damping = 0.1;

        let accel = (gravity / length) * self.angle.sin() + torque - damping * self.velocity;
        self.velocity += accel * dt;
        self.angle += self.velocity * dt;
        self.angle = self.angle.rem_euclid(2.0 * PI);
    }
}

impl Environment for PendulumEnv {
    fn objective(&self) -> f32 {
        // Reward: 1.0 when upright (angle=0), -1.0 when hanging (angle=PI)
        // Using cos ensures smooth, continuous, symmetric signal around upright,
        // avoiding discontinuities at the angle wrap boundary (2π ≡ 0).
        self.angle.cos()
    }

    fn volatility(&self) -> f32 {
        // Volatility proportional to angular velocity
        self.velocity.abs() * 0.1
    }

    fn stress(&self) -> f32 {
        // Stress proportional to absolute angle from upright.
        // cos = 1 at upright (0 stress), cos = -1 at hanging (max stress).
        (1.0 - self.angle.cos()) * 0.5
    }
}

fn main() {
    let mut env = PendulumEnv::new();

    println!("=== SimpleCritic assessment ===");
    for step in 0..20 {
        env.step(0.5 * (1.0 - env.objective())); // simple proportional control
        let mods = SimpleCritic::assess(&env);
        println!(
            "step {:2}: objective={:+.3}, dopamine={:.3}, cortisol={:.3}",
            step,
            env.objective(),
            mods.dopamine,
            mods.cortisol,
        );
    }

    let mut env2 = PendulumEnv::new();
    let mut td = TDCritic::new(0.1);

    println!("\n=== TDCritic assessment ===");
    for step in 0..20 {
        env2.step(0.5 * (1.0 - env2.objective()));
        let mods = td.assess(&env2);
        println!(
            "step {:2}: objective={:+.3}, dopamine={:.3}, acetylcholine={:.3}",
            step,
            env2.objective(),
            mods.dopamine,
            mods.acetylcholine,
        );
    }
}
