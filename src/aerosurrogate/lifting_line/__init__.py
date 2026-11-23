"""3D finite-wing analysis by nonlinear lifting-line theory.

Pairs any 2D section polar (thin-airfoil theory, a trained surrogate, or
NeuralFoil) with Prandtl's induced-downwash kernel to give wing lift,
profile drag and induced drag. See docs/WRITEUP.md §5.
"""
from __future__ import annotations

from .biot_savart import downwash_matrix
from .classical import GlauertResult, glauert_fourier_llt
from .geometry import Wing
from .sections import (
    FlatPlatePostStall,
    NeuralFoilSection,
    SectionalAero,
    ThinAirfoilSection,
)
from .solver import LiftingLineResult, alpha_sweep, solve_lifting_line

__all__ = [
    "FlatPlatePostStall",
    "GlauertResult",
    "LiftingLineResult",
    "NeuralFoilSection",
    "SectionalAero",
    "ThinAirfoilSection",
    "Wing",
    "alpha_sweep",
    "downwash_matrix",
    "glauert_fourier_llt",
    "solve_lifting_line",
]
