//! The 3D lattice: field values and per-point state.
//!
//! A [`Lattice`] owns field buffer and a per-point state mask. The
//! state mask marks each point as either [`Point::Solid`] or
//! [`Point::Void`]; the solver's line traversal splits at voids.

use super::Shape;

/// State of a lattice point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Point {
    /// Material is present; the point participates in the solve.
    Solid,
    /// Cavity; segments are split around this point.
    Void,
}

/// A 3D lattice with a scalar field and a per-point state mask.
///
/// Flat storage is row-major with `x` fastest, then `y`, then `z`. Use
/// [`idx`](Self::idx) to map `(x, y, z)` to a flat position.
#[derive(Debug)]
pub struct Lattice {
    shape: Shape,
    data: Vec<f64>,
    points: Vec<Point>,
}

impl Lattice {
    /// Constructs a lattice with a zeroed field and all points solid.
    ///
    /// # Panics
    ///
    /// Panics if the shape has zero extent along any axis.
    pub fn new(shape: Shape) -> Self {
        let n = shape.len();
        assert!(n > 0, "shape must have positive dimensions");

        Self {
            shape,
            data: vec![0.0; n],
            points: vec![Point::Solid; n],
        }
    }

    /// The lattice's shape.
    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// Flat index of `(x, y, z)`.
    #[inline]
    pub fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        self.shape.idx(x, y, z)
    }

    /// Field value at `(x, y, z)`.
    pub fn value(&self, x: usize, y: usize, z: usize) -> f64 {
        self.data[self.idx(x, y, z)]
    }

    /// Sets the field value at `(x, y, z)`.
    pub fn set(&mut self, x: usize, y: usize, z: usize, value: f64) {
        let i = self.idx(x, y, z);
        self.data[i] = value;
    }

    /// Fills the field with `value`.
    pub fn set_all(&mut self, value: f64) {
        self.data.fill(value);
    }

    /// Fills the field from a closure over `(x, y, z)`.
    pub fn set_from<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, usize, usize) -> f64,
    {
        let (nx, ny, nz) = (self.shape.nx, self.shape.ny, self.shape.nz);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = self.idx(x, y, z);
                    self.data[i] = f(x, y, z);
                }
            }
        }
    }

    /// The field, indexed as [`idx`](Self::idx).
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// State of the point at `(x, y, z)`.
    pub fn point(&self, x: usize, y: usize, z: usize) -> Point {
        self.points[self.idx(x, y, z)]
    }

    /// Marks `(x, y, z)` as void.
    pub fn void(&mut self, x: usize, y: usize, z: usize) {
        let i = self.idx(x, y, z);
        self.points[i] = Point::Void;
    }

    /// Marks `(x, y, z)` as solid.
    pub fn fill(&mut self, x: usize, y: usize, z: usize) {
        let i = self.idx(x, y, z);
        self.points[i] = Point::Solid;
    }

    /// Two disjoint borrows of the internal buffers, in the order
    /// `(data, points)`.
    pub(crate) fn split(&mut self) -> (&mut [f64], &[Point]) {
        (&mut self.data, &self.points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_zeroed_and_solid() {
        let l = Lattice::new(Shape::new(2, 3, 4));
        assert_eq!(l.shape(), Shape::new(2, 3, 4));
        assert_eq!(l.data().len(), 24);
        assert!(l.data().iter().all(|&v| v == 0.0));
        for z in 0..4 {
            for y in 0..3 {
                for x in 0..2 {
                    assert_eq!(l.point(x, y, z), Point::Solid);
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "shape must have positive dimensions")]
    fn new_rejects_empty_shape() {
        let _ = Lattice::new(Shape::new(0, 1, 1));
    }

    #[test]
    fn idx_matches_row_major_layout() {
        let l = Lattice::new(Shape::new(3, 4, 5));
        assert_eq!(l.idx(0, 0, 0), 0);
        assert_eq!(l.idx(1, 0, 0), 1);
        assert_eq!(l.idx(0, 1, 0), 3);
        assert_eq!(l.idx(0, 0, 1), 12);
        assert_eq!(l.idx(2, 3, 4), 2 + 3 * 3 + 4 * 12);
    }

    #[test]
    fn set_and_value_round_trip() {
        let mut l = Lattice::new(Shape::cube(3));
        l.set(1, 2, 0, 42.0);
        assert_eq!(l.value(1, 2, 0), 42.0);
        assert_eq!(l.value(0, 0, 0), 0.0);
    }

    #[test]
    fn set_all_fills_every_point() {
        let mut l = Lattice::new(Shape::new(2, 3, 4));
        l.set_all(7.5);
        assert!(l.data().iter().all(|&v| v == 7.5));
    }

    #[test]
    fn set_from_matches_coordinate_formula() {
        let mut l = Lattice::new(Shape::new(3, 4, 2));
        l.set_from(|x, y, z| (x + 10 * y + 100 * z) as f64);
        for z in 0..2 {
            for y in 0..4 {
                for x in 0..3 {
                    let expected = (x + 10 * y + 100 * z) as f64;
                    assert_eq!(l.value(x, y, z), expected);
                }
            }
        }
    }

    #[test]
    fn void_and_fill_toggle_point_state() {
        let mut l = Lattice::new(Shape::cube(2));
        assert_eq!(l.point(1, 1, 1), Point::Solid);
        l.void(1, 1, 1);
        assert_eq!(l.point(1, 1, 1), Point::Void);
        l.fill(1, 1, 1);
        assert_eq!(l.point(1, 1, 1), Point::Solid);
    }

    #[test]
    fn split_returns_disjoint_views() {
        let mut l = Lattice::new(Shape::cube(2));
        l.set(0, 0, 0, 1.0);

        let (data, points) = l.split();
        assert_eq!(data[0], 1.0);
        assert_eq!(points[0], Point::Solid);
    }
}
