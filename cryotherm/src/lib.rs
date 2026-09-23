//! Heat transport in frozen and thawing ground.
//!
//! `cryotherm` provides an implicit finite-difference solver for the heat
//! equation on a 3D lattice with cavities, plus the numerical kernels the
//! solver is built from.
//!
//! # Layers
//!
//! - [`math`] — pure numerical kernels. No domain knowledge. Currently the
//!   Thomas tridiagonal solver.
//! - [`domain`] — the 3D lattice, its shape, and the per-point state mask.
//! - [`solvers`] — time-step drivers that combine a domain with kernels.
//!
//! The layering is strict: [`math`] depends on nothing, [`domain`] depends
//! on nothing, and [`solvers`] depends on both.
//!
//! # Example
//!
//! ```
//! use cryotherm::{Lattice, Shape, HeatParams, HeatSolver};
//!
//! let shape = Shape::cube(4);
//! let mut lattice = Lattice::new(shape);
//!
//! // Hot spot in the centre of an insulated domain.
//! lattice.set(2, 2, 2, 100.0);
//!
//! let params = HeatParams { dt: 1.0, alpha: 0.1, h: 1.0 };
//! let mut solver = HeatSolver::new(shape, params);
//! for _ in 0..10 {
//!     solver.step(&mut lattice).unwrap();
//! }
//!
//! // Total heat is conserved: every boundary is insulated.
//! let total: f64 = lattice.data().iter().sum();
//! assert!((total - 100.0).abs() < 1e-9);
//!
//! // The centre has cooled but is still above the mean.
//! let centre = lattice.value(2, 2, 2);
//! let mean = 100.0 / 64.0;
//! assert!(centre < 30.0, "centre = {centre}");
//! assert!(centre > mean, "centre = {centre}");
//!
//! // Its nearest neighbours have warmed.
//! assert!(lattice.value(1, 2, 2) > 1.0);
//! assert!(lattice.value(2, 1, 2) > 1.0);
//! assert!(lattice.value(2, 2, 1) > 1.0);
//! ```

mod error;

pub mod domain;
pub mod math;
pub mod solvers;

pub use domain::{Axis, Lattice, Point, Shape};
pub use error::Error;
pub use solvers::heat::{HeatParams, HeatSolver};
