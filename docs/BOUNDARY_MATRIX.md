# limbic-critic: Runtime/Deployment Boundary Matrix

> Part of the [LIM-9](https://linear.app/saaq-spiking-adaptive-activity/issue/LIM-9/plan-rust-runtime-and-deployment-repo-boundary-matrix) Rust runtime boundary planning work.

## Purpose

`limbic-critic` provides reward shaping and credit assignment mechanisms for SNN (Spiking Neural Network) training. It translates external objective signals from any environment into biological neuromodulator concentrations (dopamine, norepinephrine, acetylcholine, etc.).

It is a **pure computation library** with no I/O, no hardware access, and no application-specific environment implementations.

## Owns

- Reward shaping algorithms (temporal difference, advantage estimation)
- Credit assignment mechanisms (eligibility traces, reward propagation)
- `Environment` abstraction trait — the interface for any measurable external system
- Reward normalization and scaling
- Definition of local `ModulatorVector` output structure

## Does Not Own

- Domain-specific reward calculations (mining efficiency, trading PnL, game scores)
- Environment implementations (those belong in application crates or adapters)
- I/O, networking, or hardware access
- SNN model definitions or training loops
- Neuromodulator dynamics or decay profiles (owned by upstream SNN crates)

## Allowed Dependencies

- `serde` — optional serialization for checkpoint/debug
- Math and statistics libraries (e.g., `nalgebra` for matrix operations)
- `rand` — for stochastic reward shaping where needed

## Forbidden Dependencies

- `neuromod` — or any other sibling SNN primitive crates (decoupled for modularity)
- Mining-specific or cryptocurrency libraries
- I/O or networking crates (tokio, reqwest, hyper)
- Hardware abstraction crates
- Domain-specific reward calculators
- GUI or visualization libraries

## Core-Library vs Supervisor/App vs Deployment/Hardware Boundaries

| Layer | Responsibility | Example Repos |
|-------|---------------|---------------|
| **Core Library** | Reward shaping algorithms, trait definitions, modulator mapping | `limbic-critic`, `neuromod` |
| **Supervisor/App** | Environment implementations, training orchestration, experiment config | `brainstem-daemon` |
| **Deployment/Hardware** | FPGA integration, sensor I/O, real-time control loops | `Spikenaut-Hardware`, `silicon-bridge` |

## Domain Leaks

1. **Naming history**: The crate was originally conceived for mining reward shaping. The `Environment` trait is now fully generic, but legacy references may remain in documentation.
2. **Example environments**: Any example environments shipped with this crate risk becoming domain-specific; they should use deliberately generic examples (simulation scores, not mining hashrates).

## Migration Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| ModulatorVector field mapping | Low | Use a bridge adapter in the app/training crate to map local ModulatorVector to `neuromod::NeuroModulators` |
| Example drift toward specific domain | Low | Keep examples deliberately abstract; ship domain adapters in separate repos |
| Feature creep from app-layer concerns | Low | Enforce via crate-level linting and review |

## Sequencing Questions

1. Should `limbic-critic` ship built-in example environments behind feature flags, or should all environments live in `brainstem-daemon`?
2. Does `silicon-bridge` need a `limbic-critic` adapter for its hardware reward sources, or should it use the `Environment` trait directly?

## Related Boundary Issues

- [neuromod #11](https://github.com/Limen-Neural/neuromod/issues/11)
- [brainstem-daemon #4](https://github.com/Limen-Neural/brainstem-daemon/issues/4)
- [silicon-bridge #3](https://github.com/Limen-Neural/silicon-bridge/issues/3)
- [Spikenaut-Hardware #3](https://github.com/Limen-Neural/Spikenaut-Hardware/issues/3)
