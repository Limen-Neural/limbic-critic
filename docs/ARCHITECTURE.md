# limbic-critic: Architecture Contract

> Codifies [GH#58](https://github.com/Limen-Neural/limbic-critic/issues/58): the
> long-term scope contract that keeps `limbic-critic` in the
> **neuromorphic / spiking-neural-network (SNN)** domain rather than
> drifting into a generic reinforcement-learning or ML utility crate.

## Purpose

This is a **scope-governance document**, not a design proposal. It exists so
a cold reader — and every future contributor — can tell why a change belongs
in `limbic-critic` and why another change does not, without re-litigating the
question each time.

## Core contract

`limbic-critic` exists to transform environment feedback into bounded,
biologically inspired modulatory signals intended for **spiking / plasticity
systems**. It is a reward-shaping / modulator-mapping primitive, not a
learned value function, a policy, or a training loop.

## Canonical flow

```text
Environment
    ↓
SimpleCritic / TDCritic / future critic
    ↓
ModulatorVector
(dopamine / serotonin / acetylcholine / norepinephrine)
    ↓
plasticity / neuromodulatory bridge
    ↓
SNN / neuromorphic runtime
```

`limbic-critic` owns everything up to and including the `ModulatorVector`
output. Everything downstream of that arrow — converting modulators into
plasticity updates, and driving an SNN runtime — belongs outside this crate:
in an application crate, a bridge such as `plasticity-lab`, or a sibling
runtime crate.

## Owns

- Reward shaping / objective-delta shaping for SNN-oriented systems
- Mapping environment signals into neuromodulator-like bounded scalars
- `Environment` as the observation boundary
- `ModulatorVector` as the crate-local output contract
- Critic algorithms whose output is a neuromodulatory control signal

## Does not own

- Generic actor–critic loops
- Learned policy/value networks
- Policy gradients / PPO / SAC / DQN or other general RL algorithms
- Full training loops
- ANN model definitions
- Gym-style environment orchestration
- Domain-specific controllers or reward implementations
- Synapse / neuron / STDP implementations owned by sibling neuromorphic crates

## Future-feature rule

A new critic or public abstraction belongs in this crate only if its primary
output or role can be expressed as a **neuromodulatory / plasticity control
signal for spiking systems**.

Features that instead produce actions, policies, value estimates, generic
episode management, or ANN-specific training behavior should live elsewhere
— in an application crate, an adapter, or a sibling Limen-Neural repo.

## Modulator vocabulary

Each [`ModulatorVector`](../src/modulators.rs) field carries a fixed semantic
role. Critics may differ in *how* they compute a field, but not in *what it
means*:

| Field | Semantic role | `SimpleCritic` range | `TDCritic` range |
|-------|----------------|----------------------|-------------------|
| `dopamine` | Reward / prediction-error drive | `[0.0, 1.0]` (positive objective clamped; non-positive → `0.0`) | `[-1.0, 1.0]` — signed, `tanh` of an EMA of the TD error (`objective − prev_objective`) |
| `serotonin` | Risk / volatility | `[0.0, 1.0]` (`Environment::volatility`) | `[0.0, 1.0]` (`Environment::volatility`) |
| `acetylcholine` | Focus / surprise | `[0.0, 1.0]` (`Environment::surprise`) | `[0.0, 1.0]` (`abs(td_error).tanh()`, independent of `Environment::surprise`) |
| `norepinephrine` | Stress / instability | `[0.0, 1.0]` (`Environment::stress`) | `[0.0, 1.0]` (`Environment::stress`) |

A future critic may compute these fields differently, but a new field, or a
field whose role diverges from this table, is a signal that the change is
out of scope for this crate — see the future-feature rule above.

## Dependency contract

- **Semantic interop, not Cargo coupling.** Sibling Limen-Neural crates —
  `neuromod` (plasticity rules such as `rm_stdp`) and `plasticity-lab` (the
  `ModulatorVector` → `neuromod::NeuroModulators` bridge) — are the intended
  downstream consumers of this crate's output. That relationship is
  intentional and documented.
- **Sibling Cargo dependencies are forbidden.** `limbic-critic` does not, and
  must not, take a Cargo dependency on `neuromod`, `plasticity-lab`, or any
  other sibling SNN primitive crate. Integration happens one layer up, in
  application or bridge crates.
- **Current dependency footprint:** zero runtime dependencies (see
  `[dependencies]` in [`Cargo.toml`](../Cargo.toml)). Any proposal to add a
  runtime dependency — sibling or otherwise — is a scope change and should be
  weighed against this contract, not added incidentally.

## Relationship to other documents

- [`BOUNDARY_MATRIX.md`](BOUNDARY_MATRIX.md) covers the detailed ownership
  boundaries (owns / does not own / allowed / forbidden dependencies) and the
  core-library vs. supervisor/app vs. deployment/hardware layering. This
  document is the higher-level architectural contract that the boundary
  matrix implements.
- Contribution-time enforcement of this contract (e.g. a PR checklist or
  reviewer guidance for judging whether a new feature fits) lives in
  `CONTRIBUTING.md`.
- [`../README.md`](../README.md) carries a condensed version of this scope
  statement for first-time readers.
