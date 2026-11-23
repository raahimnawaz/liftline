# liftline

[![CI](https://github.com/raahimnawaz/liftline/actions/workflows/ci.yml/badge.svg)](https://github.com/raahimnawaz/liftline/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Where does machine learning beat textbook aerodynamics, and what does it take to go from an airfoil to a whole wing?**

> Planned as the wing model through stall for [wildfire-vtol](https://github.com/raahimnawaz/wildfire-vtol). The full method, every table and all figures are in [docs/WRITEUP.md](docs/WRITEUP.md).

## Abstract

We train small regression models to predict the lift, drag and pitching moment of airfoils. We test them on 21 airfoils they never saw during training, against thin-airfoil theory, a closed-form result from 1926 with no learned parameters.

In the linear range (−4° to 4°) the two are tied: lift R² is 0.952 for the best tree model and 0.950 for the theory. The models pull ahead only as the flow approaches stall, where the theory's assumptions break: in stall the theory's R² falls to −1.77, while the model holds 0.77.

We then lift these 2D predictions to a finite wing with a nonlinear lifting-line solver. The solver reproduces the elliptic-wing drag identity to about 10⁻¹⁵ and agrees with an independent classical solver to 10⁻⁴. It also shows that induced drag, which a 2D model cannot see, is 69% of the total at a typical cruise lift.

A Rust port of the solver runs 9.8× faster than Python and matches it to 10⁻¹⁰, which is fast enough for a 100 Hz control loop.

## 1. Problem

Designing a wing means knowing, for every airfoil cross-section, three numbers that depend on the angle of attack α (the tilt of the airfoil into the oncoming air) and the Reynolds number Re (the flow's scale and speed). They are the lift coefficient $C_L$, the drag coefficient $C_D$ and the pitching-moment coefficient $C_M$.

Accurate values come from XFOIL (a panel method with a boundary-layer model) or from CFD, at seconds to hours per query. Two questions follow:

1. Can a cheap model replace that solver, and **in which flight regime** is it actually better than the classical formula?
2. A real wing has finite span, so air spills around the tips and creates **induced drag**, which no 2D section model includes. Can the 2D predictions be turned into trustworthy whole-wing numbers?

## 2. Method

### 2.1 Data

- **Airfoils:** 107 in total. They are parametric NACA 4-digit shapes (several cambers, camber positions and thicknesses 8–21%) plus curated airfoils from the UIUC database.
- **Conditions:** α from −6° to 14° and Re from 10⁵ to 10⁷, at two turbulence levels.
- **Labels:** each condition is labelled with [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil), a neural surrogate of XFOIL. Low-confidence rows are dropped, leaving **15,334 samples**.
- **Features:** each sample has 11 inputs:
  - 8 shape features: camber, camber position, thickness, leading-edge radius, trailing-edge angle, and the thickness at 25, 50 and 75% chord.
  - 3 flow features: α, log Re and turbulence level.

**The split holds out whole airfoils**: 86 for training and 21 for testing, so 3,055 test rows. A random row split would let the model see α = 5° of an airfoil and then be tested on α = 6° of the same one, which measures interpolation, not generalization.

### 2.2 Baseline: thin-airfoil theory

Thin-airfoil theory treats the airfoil as a thin curved plate in frictionless flow. For a parabolic camber line with maximum camber $m$ it gives

$$
C_L = 2\pi\,(\alpha - \alpha_{L=0}), \qquad \alpha_{L=0} \approx -2m \;\text{(radians)}
$$

The baseline uses each test airfoil's own camber. It ignores viscosity, thickness and Reynolds number, so it cannot predict stall.

### 2.3 Models

Four scikit-learn models, in increasing flexibility:

- **Ridge:** linear regression.
- **Poly-2 Ridge:** linear regression on the features plus all their pairwise products.
- **Random forest:** 400 trees.
- **Gradient boosting (GBM):** 600 trees of depth 4.

The results are reported per α regime as well as overall, because one global score hides where each method fails.

### 2.4 From airfoil to wing: nonlinear lifting-line theory

The solver turns any 2D polar (theory, a trained model, or NeuralFoil) into whole-wing coefficients:

1. **Discretize.** Split the span into N strips, packed more tightly toward the tips. Each strip carries a horseshoe vortex with circulation $\Gamma_i$, which is the strip's lift per unit span divided by air density and speed.
2. **Downwash.** The trailing legs of all the vortices push air down at every strip: $w = W\Gamma$, where $W$ comes from the Biot–Savart law. This lowers each strip's effective angle: $\alpha_{\text{eff},i} = \alpha + \text{twist}_i - w_i / V_\infty$.
3. **Consistency.** Each strip's circulation must equal the lift its own 2D polar gives at that effective angle:

$$
F_i(\Gamma) = \Gamma_i - \tfrac{1}{2} V_\infty\, c_i\, C_l(\alpha_{\text{eff},i}) = 0
$$

4. **Solve.** Use Newton's method with Jacobian $J = I + \mathrm{diag}(\tfrac{1}{2} c\, a)\,W$. The local lift slope $a = dC_l/d\alpha$ comes from a finite difference, so even a black-box polar works. A backtracking line search keeps it stable past stall, where the slope turns negative.
5. **Integrate.** From the converged circulation, compute the lift $C_L$, the induced drag $C_{D_i}$ (from $\Gamma$ and $w$), the profile drag, and the span efficiency $e = C_L^2/(\pi\,AR\,C_{D_i})$.

To check the solver, a second one implements Glauert's 1926 method. It writes the circulation as a Fourier sine series and solves a single linear system, sharing no discretization with the Newton solver.

### 2.5 Rust port

The solver is ported function for function to Rust, with the trained Poly-2 Ridge model compiled in as constant arrays, so it needs no Python at runtime. A PyO3 wrapper lets the Python test suite run both implementations on identical wings.

## 3. Results

### 3.1 Airfoils: ML helps only where the theory's assumptions fail

![CL error vs angle of attack: thin-airfoil theory against gradient boosting](figures/ml_vs_physics_cl.png)

*Figure 1. Lift error on the 21 held-out airfoils. Left: every test point. Right: median error with the interquartile band, in 2° bins. The shaded band is stall (α ≥ 10°).*

Through the linear range both errors are small: from −4° to 4° the median error is 0.045 for the theory and 0.024 for GBM. From about 6° on, the theory's error climbs steeply because real flow starts to separate from the upper surface, which frictionless theory cannot represent. The model's error stays flat. In stall the median error is 0.246 for the theory against 0.020 for GBM.

| Regime | α | Theory R² | GBM R² |
|---|---|---:|---:|
| Linear | −4° to 4° | 0.950 | 0.952 |
| Pre-stall | 4° to 10° | 0.800 | 0.892 |
| Stall | 10° to 14° | −1.766 | 0.771 |
| All | −6° to 14° | 0.910 | 0.981 (best: Poly-2 Ridge, 0.986) |

**The shared outliers** are the diagonal string of points that both methods get wrong at 0–9° in the left panel. They are one held-out airfoil, the Wortmann FX 72-LS-160, a high-camber, high-lift section. All of its largest errors are at the lowest Reynolds number (10⁵), where both models under-predict its lift by 0.4 to 0.7. A held-out split is meant to expose cases like this. A row split would have put this airfoil's other conditions into training.

For drag and moment, which the theory does not predict, the best models reach R² 0.886 and 0.875.

### 3.2 Wings: the solver is exact where theory says it should be

| Check | Expected | Measured |
|---|---|---:|
| Elliptic wing (AR 4, 8, 16) | $C_{D_i} = C_L^2/(\pi\,AR)$, so $e = 1$ | relative error ~10⁻¹⁵ |
| Independent Glauert solver | same $C_L$ and $C_{D_i}$ | agree to ≤ 10⁻⁴ |
| Finite-wing lift slope (AR 6, 10, 20) | $2\pi\,AR/(AR+2)$ (Helmbold) | within 6×10⁻⁵ |
| Rectangular wing | e slightly below 1 | 0.97 (AR 4) to 0.86 (AR 20) |
| Sweep to α = 55° through stall | converges | ≤ 4 Newton steps per angle |

![2D NeuralFoil against the 3D lifting-line wing](figures/llt_neuralfoil_wing.png)

*Figure 2. Rectangular NACA 2412 wing, aspect ratio 8, Re 3×10⁶, with NeuralFoil as the section polar. Left: the finite wing makes less lift than the airfoil alone at the same α (1.34 against 1.6 at 14°), because downwash lowers the effective angle. Right: the 2D drag curve is almost vertical, because it has profile drag only. The 3D curve adds induced drag (blue), which is **69% of total drag at $C_L$ = 0.5**.*

### 3.3 Speed

For a 25-angle sweep with the Ridge polar, Python takes 38 ms and Rust 3.9 ms (**9.8×**). For a single angle the times are 3.9 ms and 0.45 ms. 28 parity tests show the two implementations agree to 10⁻¹⁰ on every wing in the suite.

## 4. Limitations

- **It does not beat NeuralFoil.** The 2D labels come from NeuralFoil, so its accuracy is the ceiling. The gain is speed in 2D, and induced drag plus stall behavior in 3D.
- **No extrapolation.** The test airfoils are held out but come from the same NACA and UIUC families. Transonic and supercritical sections are untested.
- **Planar, unswept wings only.** Sweep, dihedral or multiple lifting surfaces need a vortex-lattice method.

## 5. Reproduce

```bash
git clone https://github.com/raahimnawaz/liftline && cd liftline
pip install -e ".[dev]"
pytest -q                       # 64 tests; the 28 Rust parity tests skip without the wheel
python scripts/train_eval.py    # every 2D number above
python scripts/llt_demo.py      # 3D figures (needs: pip install -e ".[build]")
```

```python
import math
from aerosurrogate.lifting_line import Wing, ThinAirfoilSection, solve_lifting_line

wing = Wing.elliptic(span=10.0, root_chord=4 * 10.0 / (math.pi * 8), n_sections=80)  # AR 8
res = solve_lifting_line(wing, alpha_deg=5.0, section=ThinAirfoilSection())
print(res.CL, res.CDi, res.span_efficiency)  # e = 1.000000 for an elliptic wing
```

Rust build and benchmarks are in [rust/README.md](rust/README.md). The full writeup, with every table and figure and the derivations, is [docs/WRITEUP.md](docs/WRITEUP.md).
