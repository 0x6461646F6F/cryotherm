//! Extents and axes of a 3D lattice.
//!
//! A lattice is stored row-major with `x` fastest, then `y`, then `z`.
//! [`Shape`] describes the extents and provides the index arithmetic that
//! maps `(x, y, z)` to a flat buffer position. [`Axis`] names one of the
//! three coordinate axes and knows which two are transverse to it, which
//! the sweep drivers use to iterate lines.

/// One of the three coordinate axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// The x-axis.
    X,
    /// The y-axis.
    Y,
    /// The z-axis.
    Z,
}

impl Axis {
    /// All three axes, in sweep order.
    pub const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

    /// The two axes orthogonal to `self`, in cyclic order.
    ///
    /// For [`Axis::X`] this returns `[Y, Z]`; for [`Axis::Y`] it returns
    /// `[Z, X]`; for [`Axis::Z`] it returns `[X, Y]`.
    pub const fn orthogonal(self) -> [Axis; 2] {
        match self {
            Axis::X => [Axis::Y, Axis::Z],
            Axis::Y => [Axis::Z, Axis::X],
            Axis::Z => [Axis::X, Axis::Y],
        }
    }
}

/// Extent of a 3D lattice along each axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    /// Extent along the x-axis.
    pub nx: usize,
    /// Extent along the y-axis.
    pub ny: usize,
    /// Extent along the z-axis.
    pub nz: usize,
}

impl Shape {
    /// Constructs a shape from three extents.
    pub const fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self { nx, ny, nz }
    }

    /// Constructs a cube of side `n`.
    pub const fn cube(n: usize) -> Self {
        Self {
            nx: n,
            ny: n,
            nz: n,
        }
    }

    /// Total number of points.
    pub const fn len(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Largest of the three extents.
    pub const fn max_axis(&self) -> usize {
        let m = if self.nx > self.ny { self.nx } else { self.ny };
        if m > self.nz {
            m
        } else {
            self.nz
        }
    }

    /// Extent along `axis`.
    pub const fn dim(&self, axis: Axis) -> usize {
        match axis {
            Axis::X => self.nx,
            Axis::Y => self.ny,
            Axis::Z => self.nz,
        }
    }

    /// Stride of `axis` in flat row-major storage.
    ///
    /// `x` has stride 1, `y` has stride `nx`, `z` has stride `nx * ny`.
    pub const fn stride(&self, axis: Axis) -> usize {
        match axis {
            Axis::X => 1,
            Axis::Y => self.nx,
            Axis::Z => self.nx * self.ny,
        }
    }

    /// Flat index of `(x, y, z)` in row-major storage.
    pub const fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        x * self.stride(Axis::X) + y * self.stride(Axis::Y) + z * self.stride(Axis::Z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_all_order() {
        assert_eq!(Axis::ALL, [Axis::X, Axis::Y, Axis::Z]);
    }

    #[test]
    fn axis_orthogonal() {
        assert_eq!(Axis::X.orthogonal(), [Axis::Y, Axis::Z]);
        assert_eq!(Axis::Y.orthogonal(), [Axis::Z, Axis::X]);
        assert_eq!(Axis::Z.orthogonal(), [Axis::X, Axis::Y]);
    }

    #[test]
    fn shape_new_and_len() {
        let s = Shape::new(2, 3, 4);
        assert_eq!(s.len(), 24);
        assert_eq!((s.nx, s.ny, s.nz), (2, 3, 4));
    }

    #[test]
    fn shape_cube() {
        let s = Shape::cube(5);
        assert_eq!(s.len(), 125);
        assert_eq!((s.nx, s.ny, s.nz), (5, 5, 5));
    }

    #[test]
    fn shape_max_axis() {
        assert_eq!(Shape::new(3, 5, 4).max_axis(), 5);
        assert_eq!(Shape::new(7, 2, 4).max_axis(), 7);
        assert_eq!(Shape::new(2, 3, 9).max_axis(), 9);
        assert_eq!(Shape::cube(5).max_axis(), 5);
    }

    #[test]
    fn shape_dim() {
        let s = Shape::new(2, 3, 4);
        assert_eq!(s.dim(Axis::X), 2);
        assert_eq!(s.dim(Axis::Y), 3);
        assert_eq!(s.dim(Axis::Z), 4);
    }

    #[test]
    fn shape_stride() {
        let s = Shape::new(2, 3, 4);
        assert_eq!(s.stride(Axis::X), 1);
        assert_eq!(s.stride(Axis::Y), 2);
        assert_eq!(s.stride(Axis::Z), 6);
    }

    #[test]
    fn shape_stride_matches_row_major_layout() {
        let s = Shape::new(3, 4, 5);
        let idx = 1 * s.stride(Axis::X) + 2 * s.stride(Axis::Y) + 3 * s.stride(Axis::Z);
        assert_eq!(idx, 1 + 2 * 3 + 3 * 12);
        assert!(idx < s.len());
    }

    #[test]
    fn shape_idx() {
        let s = Shape::new(3, 4, 5);
        let (x, y, z) = (1, 0, 2);
        let idx = s.idx(x, y, z);
        assert_eq!(idx, 25);
    }
}
