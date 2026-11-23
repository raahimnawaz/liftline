"""aerosurrogate: ML surrogates for 2D airfoil coefficients, plus a nonlinear
lifting-line solver that lifts them to finite 3D wings.
"""
from . import lifting_line
from .dataset import DATASET_PATH, load_dataset, split_by_airfoil
from .eval import (
    evaluate,
    evaluate_cl_with_physics,
    feature_importance,
    regime_eval,
    regime_eval_cl_with_physics,
)
from .lifting_line import (
    FlatPlatePostStall,
    GlauertResult,
    LiftingLineResult,
    NeuralFoilSection,
    SectionalAero,
    ThinAirfoilSection,
    Wing,
    alpha_sweep,
    glauert_fourier_llt,
    solve_lifting_line,
)
from .models import FEATURES, TARGETS, build_models
from .physics import (
    thin_airfoil_cl,
    thin_airfoil_cl_slope_per_deg,
    thin_airfoil_zero_lift_alpha,
)

__all__ = [
    "DATASET_PATH",
    "FEATURES",
    "FlatPlatePostStall",
    "GlauertResult",
    "LiftingLineResult",
    "NeuralFoilSection",
    "SectionalAero",
    "TARGETS",
    "ThinAirfoilSection",
    "Wing",
    "alpha_sweep",
    "build_models",
    "evaluate",
    "evaluate_cl_with_physics",
    "feature_importance",
    "glauert_fourier_llt",
    "lifting_line",
    "load_dataset",
    "regime_eval",
    "regime_eval_cl_with_physics",
    "solve_lifting_line",
    "split_by_airfoil",
    "thin_airfoil_cl",
    "thin_airfoil_cl_slope_per_deg",
    "thin_airfoil_zero_lift_alpha",
]
