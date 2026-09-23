//! The 3D domain: shape, axes, and the lattice that holds state.

mod lattice;
mod layout;

pub use lattice::{Lattice, Point};
pub use layout::{Axis, Shape};
