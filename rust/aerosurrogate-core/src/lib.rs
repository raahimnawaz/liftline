//! # aerosurrogate-core
//!
//! Nonlinear lifting-line solver for planar, unswept wings: any 2D section
//! polar plus Prandtl's downwash kernel, solved by Newton iteration
//! (Phillips & Snyder, 2000). A port of Python `aerosurrogate.lifting_line`
//! that agrees with it to 1e-10; see `docs/WRITEUP.md` §7.
//!
//! * [`geometry`]: [`Wing`] and its rectangular / elliptic / tapered factories.
//! * [`sections`]: the [`SectionalAero`] trait, analytical polars and the
//!   baked-in Ridge surrogate ([`surrogate::ridge`]).
//! * [`solver`]: Newton solver and α-sweep.
//! * [`classical`]: independent Glauert Fourier-series solver for cross-checks.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod biot_savart;
pub mod classical;
pub mod geometry;
pub mod sections;
pub mod solver;
pub mod surrogate;

pub use biot_savart::downwash_matrix;
pub use classical::{glauert_fourier_llt, GlauertResult};
pub use geometry::Wing;
pub use sections::{FlatPlatePostStall, RidgeSurrogateSection, SectionalAero, ThinAirfoilSection};
pub use solver::{
    alpha_sweep, solve_lifting_line, AlphaSweepResult, LiftingLineResult, SolverOptions,
};

/// Library error type.
///
/// Returned by constructors and the solver when invariants are violated
/// (e.g. negative span, collocation point coincident with a segment edge,
/// inconsistent input shapes).
#[derive(Debug, thiserror::Error)]
pub enum AeroError {
    /// A geometric input violated an invariant (negative span, taper ratio
    /// ≤ 0, fewer than 4 sections, etc.).
    #[error("invalid geometry: {0}")]
    Geometry(String),

    /// A grid was passed to a routine that requires monotonicity or
    /// non-overlap with control points; the routine cannot proceed.
    #[error("invalid grid: {0}")]
    Grid(String),

    /// The solver received an initial guess of the wrong shape.
    #[error("invalid initial guess: {0}")]
    Init(String),
}

/// Crate-wide [`Result`](::std::result::Result) alias using [`AeroError`].
pub type Result<T> = ::std::result::Result<T, AeroError>;
