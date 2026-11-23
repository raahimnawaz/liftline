# liftline

[![CI](https://github.com/raahimnawaz/liftline/actions/workflows/ci.yml/badge.svg)](https://github.com/raahimnawaz/liftline/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

liftline trains small ML models of airfoil aerodynamics and judges them against the closed-form thin-airfoil theory. It then lifts those 2D predictions to a full finite wing with a validated nonlinear lifting-line solver, ported to Rust for real-time use.

> Planned as the wing model through stall for [wildfire-vtol](https://github.com/raahimnawaz/wildfire-vtol).

![NeuralFoil-coupled 3D wing analysis](figures/llt_neuralfoil_wing.png)

**Full method, math and every table: [docs/WRITEUP.md](docs/WRITEUP.md).**

## Results

**2D: ML beats the theory only where the theory's assumptions fail.** The models are trained on 15,334 NeuralFoil-labelled samples from 107 airfoils. The test set is *whole airfoils the model never saw*. Lift R² by flight regime:

| Regime | α | Thin-airfoil theory | Gradient boosting |
|---|---|---:|---:|
| Linear | −4° to 4° | 0.950 | 0.952 |
| Pre-stall | 4° to 10° | 0.800 | **0.892** |
| Stall | 10° to 14° | **−1.766** | **0.771** |

At low α, a formula with zero learned parameters ties a 600-tree model. The model earns its keep only toward stall, where potential flow breaks down. Across all regimes, the best model (Poly-2 Ridge) reaches R² 0.986, against 0.910 for the theory.

**3D: the wing solver matches the analytical identities it should.** It runs Newton iteration on Prandtl's lifting-line equation and accepts any 2D polar:

| Check | Agreement |
|---|---:|
| Elliptic wing: $C_{D_i} = C_L^2/(\pi\,AR)$ | ~10⁻¹⁵ relative |
| Independent Glauert Fourier-series solver ($C_L$, $C_{D_i}$) | ≤ 10⁻⁴ relative |
| Finite-wing lift slope $2\pi\,AR/(AR+2)$ (Helmbold) | ≤ 6×10⁻⁵ relative |
| Through stall to α = 55° | converges in ≤ 4 iterations |

Coupled with NeuralFoil on a NACA 2412 wing (AR 8), **induced drag is 69% of total drag at $C_L$ = 0.5**. A 2D-only pipeline cannot see that drag.

**Rust: 9.8× faster, identical answers.** A 25-point α-sweep using the baked-in Ridge surrogate takes 3.9 ms in Rust against 38 ms in Python. That is fast enough for a 100 Hz control loop, with no Python at runtime. 28 parity tests show the two implementations agree to 10⁻¹⁰.

## Limits

- **It does not beat NeuralFoil.** The 2D labels come from NeuralFoil, so NeuralFoil's accuracy is the ceiling. The 3D solver then adds induced drag on top of it.
- **No extrapolation.** The held-out airfoils come from the same NACA and UIUC families. Transonic sections are untested.
- **Planar, unswept wings only.** Swept wings need a vortex-lattice method instead.

## Quickstart

```bash
git clone https://github.com/raahimnawaz/liftline && cd liftline
pip install -e ".[dev]"
pytest -q                       # 64 tests; Rust parity tests skip without the wheel
python scripts/train_eval.py    # reproduce the 2D tables
```

```python
import math
from aerosurrogate.lifting_line import Wing, ThinAirfoilSection, solve_lifting_line

wing = Wing.elliptic(span=10.0, root_chord=4 * 10.0 / (math.pi * 8), n_sections=80)  # AR 8
res = solve_lifting_line(wing, alpha_deg=5.0, section=ThinAirfoilSection())
print(res.CL, res.CDi, res.span_efficiency)  # e = 1.000000 for an elliptic wing
```

Rust build and benchmarks: [rust/README.md](rust/README.md).
