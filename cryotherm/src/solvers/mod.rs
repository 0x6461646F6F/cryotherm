//! Time-step drivers.
//!
//! Each submodule implements a scheme that advances a [`crate::domain::Lattice`]
//! forward in time using the numerical kernels in [`crate::math`].

pub mod heat;
