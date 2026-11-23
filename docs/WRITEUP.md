# liftline — full writeup

The [README](../README.md) gives the results. This file has the method, the math, every table, and the limits.

1. [Problem](#1-problem)
2. [Data and split](#2-data-and-split)
3. [Models and baseline](#3-models-and-baseline)
4. [2D results](#4-2d-results)
5. [3D: nonlinear lifting-line theory](#5-3d-nonlinear-lifting-line-theory)
6. [3D validation](#6-3d-validation)
7. [Rust port](#7-rust-port)
8. [Limits](#8-limits)
9. [Reproducing](#9-reproducing)

---

## 1. Problem

A wing designer needs three coefficients for every airfoil section: lift $C_L$, drag $C_D$, and pitching moment $C_M$. Getting them accurately means running XFOIL (a panel method with a boundary layer) or CFD, which takes seconds to hours per query. Across thousands of airfoils and dozens of flight conditions, that cost is the bottleneck.

This project asks two questions:

- **2D:** how close does a cheap regression get to a panel-method solver, and *where* does it stop beating the closed-form theory that already exists?
- **3D:** a real wing has finite span, so tip vortices add induced drag that no 2D model can see. Can the 2D polars be lifted into a validated finite-wing solver?

## 2. Data and split

| Stage | Detail |
|---|---|
| Shapes | Parametric NACA 4-digit (camber 0/2/4/6%, several camber positions, thickness 8/10/12/15/18/21%) plus curated UIUC airfoils (Selig, Eppler, Wortmann, AeroLab). **107 airfoils.** |
| Conditions | α ∈ [−6°, +14°], 16 points; Re ∈ [10⁵, 10⁷], 5 log-spaced points; n_crit ∈ {5, 9} |
| Labels | [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) (MIT), a neural surrogate of XFOIL. Rows with `analysis_confidence < 0.90` are dropped. **15,334 samples.** |
| Features | 8 geometric (max camber, camber position, max thickness, leading-edge radius, trailing-edge wedge angle, thickness at 25/50/75% chord) + 3 flow (α, log₁₀ Re, n_crit) |

**The split holds out whole airfoils.** `split_by_airfoil()` puts every row of a given shape in either train or test. With a row-level split, NACA 0012 at α = 5° could be in train and α = 6° in test, and the model would score well by interpolating along a curve it has already seen. Holding out shapes measures what a design loop actually needs: accuracy on a geometry the model has never seen.

## 3. Models and baseline

| Model | Capacity |
|---|---|
| Thin-airfoil theory | closed form, 0 learned parameters |
| Ridge | linear in the 11 features |
| Poly-2 + Ridge | linear in the 11 features plus all pairwise products |
| Random forest | 400 trees |
| Gradient boosting | 600 trees, depth 4, learning rate 0.05 |

**Thin-airfoil theory** (Glauert, 1926) linearizes potential flow around a thin cambered plate. For a parabolic camber line $z(x) = 4m\,x(1-x)$:

$$
C_L = 2\pi\,(\alpha - \alpha_{L=0}), \qquad \alpha_{L=0} \approx -2m \quad [\text{rad}]
$$

It has no learned parameters, only π, α and camber. Comparing against it shows where ML actually earns something (viscosity, separation, thickness) and where it doesn't (low α, where the theory is already within a few percent).

As a check on the data: fitting the lift slope at |α| < 4° on the held-out airfoils gives a **median 0.105 /deg**, against the theoretical 2π/rad = 0.110 /deg (within 5%).

## 4. 2D results

All numbers are on held-out airfoils. Reproduce with `python scripts/train_eval.py`.

### Lift, all regimes

| Model | CV R² | Test R² | Test MAE |
|---|---:|---:|---:|
| **Poly-2 Ridge** | 0.989 | **0.986** | 0.055 |
| Gradient boosting | 0.998 | 0.981 | 0.046 |
| Random forest | 0.998 | 0.978 | 0.049 |
| Ridge | 0.968 | 0.960 | 0.092 |
| Thin-airfoil theory | — | 0.910 | 0.117 |

### Lift by flight regime

| Regime | α band | Theory R² | Theory MAE | GBM R² | GBM MAE |
|---|---|---:|---:|---:|---:|
| Linear | [−4°, +4°) | 0.950 | 0.058 | 0.952 | **0.047** |
| Pre-stall | [+4°, +10°) | 0.800 | 0.080 | **0.892** | **0.041** |
| Stall | [+10°, +14°] | −1.766 | 0.282 | **0.771** | **0.052** |
| Negative | [−6°, −4°) | 0.888 | 0.066 | **0.928** | **0.042** |

A single global R² hides the main finding. In the linear regime the textbook formula ties a 600-tree model (R² ≈ 0.95 for both). ML earns its place only at higher α: it lifts R² from 0.80 to 0.89 through pre-stall. In stall, the potential-flow assumption fails and the theory's R² goes negative, while gradient boosting holds 0.77.

### Drag and moment (ML only, since the theory predicts neither)

| Target | Best model | Test R² | Test MAE |
|---|---|---:|---:|
| C_D | Random forest (GBM 0.885) | 0.886 | 0.0016 |
| C_M | Gradient boosting | 0.875 | 0.013 |

The feature importances match the physics. Drag is driven by α (47%) and log₁₀ Re (39%), so the viscous regime matters. Moment is driven by max camber (70%), which is what sets the moment.

### Figures

Predicted vs actual on held-out airfoils:

![Predicted vs actual](../figures/predicted_vs_actual.png)

Residuals vs α, with the stall band shaded:

![Residuals vs α](../figures/residuals_vs_alpha.png)

Feature importance (random forest, MDI):

![Feature importance](../figures/feature_importance.png)

### Physics property tests (`tests/test_physics.py`, 9 tests)

- $C_L(\alpha=0, m=0) = 0$: a symmetric airfoil makes no lift at zero α.
- $\partial C_L/\partial\alpha = 2\pi$ /rad for the closed form.
- Odd symmetry $C_L(-\alpha) = -C_L(\alpha)$ for symmetric airfoils.
- The empirical lift slope in the NeuralFoil data lies in [0.08, 0.13] /deg (actual 0.105).
- Symmetric airfoils have $|C_L| < 0.2$ at $|\alpha| \le 1.5°$ (actual mean 0.07).
- $C_D > 0$ on every row.
- The drag bucket of symmetric airfoils is centered near α = 0.

## 5. 3D: nonlinear lifting-line theory

The 2D models describe an infinite-span section. On a finite wing, the trailing vortex sheet rolls up into tip vortices, which tilt each section's lift vector backward and create **induced drag**, typically 30–50% of total drag at cruise.

`aerosurrogate.lifting_line` pairs any 2D polar (thin-airfoil theory, the trained surrogate, or NeuralFoil) with Prandtl's induced-downwash kernel. It solves the nonlinear system in the style of Phillips & Snyder (2000), restricted to planar, unswept wings.

At each spanwise station $i$, the bound circulation must satisfy 2D Kutta–Joukowski at the section's effective angle:

$$
F_i(\Gamma) \equiv \Gamma_i - \tfrac{1}{2} V_\infty\, c_i\, C_l\!\left(\alpha_{\text{eff},i},\, Re_i\right) = 0,
\qquad \alpha_{\text{eff},i} = \alpha + \theta_i - \frac{(W\Gamma)_i}{V_\infty}
$$

$W$ is the horseshoe-vortex downwash matrix and $\theta_i$ is the local twist. Newton iteration uses the Jacobian

$$
J_{ij} = \delta_{ij} + \tfrac{1}{2}\, c_i\, a_i\, W_{ij}, \qquad a_i = \frac{dC_l}{d\alpha}\bigg|_{\alpha_{\text{eff},i}}
$$

where $a_i$ comes from a one-sided finite difference on the polar, so the solver works with any polar, including a black-box network. Each iteration is one N×N linear solve, with a backtracking line search for stability through stall, where the local slope flattens or turns negative. An α-sweep warm-starts each solve from the previous Γ.

Wing outputs: $C_L = \frac{2}{S V_\infty}\int\Gamma\,dy$, $C_{D_i} = \frac{2}{S V_\infty^2}\int \Gamma\, w\,dy$, profile drag $\frac{1}{S}\int C_d\, c\,dy$, and span efficiency $e = C_L^2/(\pi\,AR\,C_{D_i})$.

**Independent second solver.** `classical.py` implements Glauert's 1926 Fourier-series lifting line: it expands Γ as a half-span sine series and solves one linear system. It shares no discretization with the Newton solver, which is why agreement between the two is a meaningful check.

## 6. 3D validation

`tests/test_lifting_line.py` has 17 tests. The *measured* column is the actual agreement. The *gate* column is the test tolerance.

| Identity | Measured | Gate |
|---|---:|---:|
| Elliptic wing $C_{D_i} = C_L^2/(\pi AR)$, AR 4/8/16 | ~10⁻¹⁵ relative | 5×10⁻³ |
| Newton vs Glauert, $C_L$ and $C_{D_i}$, elliptic / rectangular / tapered | ≤ 1×10⁻⁴ relative | 5×10⁻³ |
| Glauert on an elliptic wing: $e = 1$, higher modes $A_{n\ge3}$ vanish | to machine precision | 10⁻¹² |
| Helmbold slope $a = 2\pi\,AR/(AR+2)$, AR 6/10/20 | ≤ 6×10⁻⁵ relative | 10⁻² |
| Elliptic loading $\Gamma(y) \propto \sqrt{1-(2y/b)^2}$ | — | 5×10⁻³ |
| Rectangular wing span efficiency | 0.86 (AR 20) to 0.97 (AR 4) | $e \in (0.85, 1)$ |
| Convergence through stall, α 0°→55°, flat-plate polar | ≤ 4 Newton iterations per α (warm start) | converges |
| −3° washout shifts loading inboard | tip Γ down 53% | tip Γ drops |

The elliptic solve converges in 2 Newton iterations at every aspect ratio tested.

### Figures (`python scripts/llt_demo.py`)

| Figure | What it shows |
|---|---|
| ![](../figures/llt_neuralfoil_wing.png) | NACA 2412 wing, AR 8, Re 3×10⁶, with NeuralFoil as the section polar. Left: the 3D wing reaches $C_L$ 1.34 at 14° where the 2D section reaches 1.6. Right: the 2D drag polar is almost vertical (profile drag only), while the 3D polar picks up induced drag. **At $C_L$ = 0.5, induced drag is 69% of the total.** |
| ![](../figures/llt_elliptic_identity.png) | The solver trace lies on the analytical line $C_{D_i} = C_L^2/(\pi AR)$. |
| ![](../figures/llt_lift_curve_stall.png) | Rectangular AR 10 wing through stall with a Hoerner flat-plate polar, a case linear lifting-line theory cannot handle. |
| ![](../figures/llt_span_efficiency.png) | Span efficiency vs aspect ratio for three planforms: elliptic stays at $e \approx 1$, rectangular falls to 0.86 at AR 20. |

## 7. Rust port

`rust/aerosurrogate-core` ports the lifting-line package function by function, for use inside a control loop where the Python version's millisecond latency is too slow. `rust/aerosurrogate-py` wraps it with PyO3, so the Python test suite can drive both implementations.

The Python solver pays for the interpreter, NumPy dispatch, and a sklearn `predict()` call inside every Newton step. The Rust port uses nalgebra's LU and evaluates the trained Poly-2 Ridge surrogate from `pub const` f64 arrays compiled into the binary (`surrogate/ridge.rs`, generated by `rust/scripts/export_ridge_weights.py`). A single surrogate query is allocation-free scalar arithmetic. The Newton loop itself still allocates its working vectors each iteration.

**Speed** (M-series MacBook, single thread, `cargo bench --bench solver`):

| Call | Polar | Python | Rust | Speed-up |
|---|---|---:|---:|---:|
| single α | thin-airfoil | 120 µs | 40 µs | 3.0× |
| single α | Ridge surrogate (NACA 2412) | 3.9 ms | 0.45 ms | 8.7× |
| 25-point α-sweep, warm start | thin-airfoil | 3.0 ms | 1.0 ms | 3.1× |
| 25-point α-sweep, warm start | Ridge surrogate (NACA 2412) | 38 ms | 3.9 ms | **9.8×** |

A full 25-point drag build-up in 3.9 ms fits inside a 100 Hz loop with no Python or sklearn at runtime.

**Correctness.**

- 32 Rust unit and integration tests cover the same identities as the Python suite, plus Ridge-vs-sklearn agreement to 10⁻¹².
- 28 Python↔Rust parity tests (`tests/test_rust_parity.py`) run identical wings through both implementations and assert agreement to 10⁻¹⁰. For example, the AR 8 elliptic wing at α = 5° gives $C_L$ = 0.438627 in both, after the same 2 Newton iterations.

**Not ported:** the gradient-boosting surrogate (Ridge is the embedded polar), NeuralFoil itself, and swept or dihedral wings.

## 8. Limits

- **It does not beat NeuralFoil.** The 2D labels come from NeuralFoil, so NeuralFoil's accuracy is the surrogate's ceiling. The 2D claim is how close a cheap model gets, at much lower inference cost. In 3D, NeuralFoil is an *input*, and the wing solver adds induced drag and stall behavior that NeuralFoil alone does not provide.
- **It does not extrapolate.** The test airfoils are held out, but they come from the same NACA + UIUC families. Transonic and supercritical sections are not measured.
- **The wings are planar and unswept.** Sweep, dihedral, or multiple lifting surfaces need a vortex-lattice method. The current scope is exactly the case where classical lifting-line theory is valid.

## 9. Reproducing

```bash
pip install -e ".[dev]"
pytest -q                        # 64 tests (the 28 parity tests skip without the Rust wheel)
python scripts/train_eval.py     # every 2D table above
python scripts/make_figures.py   # 2D figures
python scripts/llt_demo.py       # 3D figures (needs neuralfoil: pip install -e ".[build]")
python scripts/build_dataset.py  # rebuild data/aero_dataset.csv from scratch (slow, needs [build])
```

Rust:

```bash
cd rust && cargo test --workspace && cargo bench --bench solver
# Python wheel (enables the parity tests): see rust/README.md
```

### Layout

```
src/aerosurrogate/
├── airfoils.py      NACA + UIUC loaders
├── features.py      geometric features
├── dataset.py       build / load / split_by_airfoil
├── models.py        sklearn pipelines
├── physics.py       thin-airfoil baseline
├── eval.py          per-model and per-regime metrics
├── plot.py          2D figures
└── lifting_line/
    ├── geometry.py     Wing (rectangular / elliptic / tapered)
    ├── sections.py     section polars: thin-airfoil, flat-plate, NeuralFoil
    ├── biot_savart.py  downwash matrix
    ├── solver.py       Newton solver + α-sweep
    └── classical.py    Glauert Fourier-series solver
rust/
├── aerosurrogate-core/  Rust library, tests, benches
├── aerosurrogate-py/    PyO3 wheel (aerosurrogate_rs)
└── scripts/             sklearn → Rust const-array export
```
