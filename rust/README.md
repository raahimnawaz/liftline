# aerosurrogate-rs

A Rust port of the Python lifting-line solver, built for real-time use. It runs a 25-point α-sweep with the baked-in Ridge surrogate in 3.9 ms, against 38 ms in Python (9.8× faster). It matches the Python results to 10⁻¹⁰.

- **`aerosurrogate-core`**: the library. It contains the wing geometry, section polars, the downwash matrix, the Newton solver, an independent Glauert solver, and the trained Ridge surrogate compiled in as `const` arrays.
- **`aerosurrogate-py`**: a PyO3 wrapper, imported as `aerosurrogate_rs`. The Python parity tests use it.
- **`scripts/export_ridge_weights.py`**: regenerates `surrogate/ridge.rs` from the trained sklearn pipeline.

Design, benchmarks and validation: [docs/WRITEUP.md §7](../docs/WRITEUP.md#7-rust-port).

## Build and test

```bash
cd rust
cargo test --workspace          # 32 tests
cargo bench --bench solver      # report in target/criterion/
```

To build the Python wheel and run the parity tests from the repo root:

```bash
pip install maturin
maturin build --release --manifest-path rust/aerosurrogate-py/Cargo.toml --out dist
pip install --force-reinstall dist/*.whl
pytest -q tests/test_rust_parity.py   # 28 tests
```

## Use

```python
import math
import aerosurrogate_rs as rs

wing = rs.Wing.elliptic(span=10.0, root_chord=4 * 10 / (math.pi * 8), n_sections=80)
res = rs.solve_lifting_line(wing, 5.0, rs.RidgeSurrogateSection.naca4(2.0, 4.0, 12.0, n_crit=9.0), re_ref=3e6)
print(res.CL, res.CD, res.span_efficiency)
```

```rust
use aerosurrogate_core::{solve_lifting_line, SolverOptions, ThinAirfoilSection, Wing};

let wing = Wing::elliptic(10.0, 1.5915, 0.0, 80)?;
let res = solve_lifting_line(&wing, 5.0, &ThinAirfoilSection::default(), &SolverOptions::default())?;
```
