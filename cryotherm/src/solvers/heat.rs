//! Implicit heat-equation driver.
//!
//! Time integration is backward Euler — chosen for unconditional
//! stability and L-stability, which is what keeps a freezing front from
//! oscillating when phase change is added later.
//!
//! Each time step performs three directional sweeps (X, then Y, then Z).
//! Within a sweep, lines parallel to the sweep axis are independent
//! tridiagonal systems, split into separate systems at
//! [`Point::Void`]. Segment ends are insulated.

use crate::domain::{Axis, Lattice, Point, Shape};
use crate::error::Error;
use crate::math::tridiagonal::Workspace;

/// Physical parameters for the heat equation.
#[derive(Debug, Clone, Copy)]
pub struct HeatParams {
    /// Time step size.
    pub dt: f64,
    /// Thermal diffusivity.
    pub alpha: f64,
    /// Uniform grid spacing.
    pub h: f64,
}

impl HeatParams {
    /// Dimensionless stencil weight `λ = α · Δt / h²`.
    pub fn lambda(&self) -> f64 {
        self.alpha * self.dt / (self.h * self.h)
    }
}

/// A contiguous run of solid points along a lattice line.
#[derive(Debug, Clone, Copy)]
struct Segment {
    base: usize,
    start: usize,
    len: usize,
    stride: usize,
}

/// Implicit heat-equation stepper with sequential X → Y → Z LOD splitting.
///
/// Construct once for a given shape, then call [`step`](Self::step) in a
/// loop. No allocation happens inside `step`.
#[derive(Debug)]
pub struct HeatSolver {
    shape: Shape,
    lambda: f64,
    ws: Workspace,
}

impl HeatSolver {
    /// Constructs a stepper for the given shape and physical parameters.
    ///
    /// The internal workspace is sized for the largest axis of `shape`,
    /// no allocations happen during [`step`](Self::step).
    pub fn new(shape: Shape, params: HeatParams) -> Self {
        Self {
            shape,
            lambda: params.lambda(),
            ws: Workspace::new(shape.max_axis()),
        }
    }

    /// Advances `lattice` by one time step.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidDimensions`] if `lattice.shape()` does not match
    ///   the shape given to [`new`](Self::new).
    /// - [`Error::UnstableSystem`] if any segment's tridiagonal system
    ///   fails the diagonal-dominance check.
    pub fn step(&mut self, lattice: &mut Lattice) -> Result<(), Error> {
        if lattice.shape() != self.shape {
            return Err(Error::InvalidDimensions);
        }

        for axis in Axis::ALL {
            self.sweep(lattice, axis)?;
        }

        Ok(())
    }

    fn sweep(&mut self, lattice: &mut Lattice, axis: Axis) -> Result<(), Error> {
        let shape = lattice.shape();

        let stride = shape.stride(axis);
        let len = shape.dim(axis);

        let [b_axis, c_axis] = axis.orthogonal();

        let b_len = shape.dim(b_axis);
        let b_stride = shape.stride(b_axis);

        let c_len = shape.dim(c_axis);
        let c_stride = shape.stride(c_axis);

        let (data, points) = lattice.split();

        for c in 0..c_len {
            for b in 0..b_len {
                let base = b * b_stride + c * c_stride;
                let mut seg_start = 0usize;
                let mut seg_len = 0usize;

                for k in 0..len {
                    let i = base + k * stride;

                    if points[i] == Point::Solid {
                        if seg_len == 0 {
                            seg_start = k;
                        }
                        seg_len += 1;
                    } else if seg_len > 0 {
                        let seg = Segment {
                            base,
                            start: seg_start,
                            len: seg_len,
                            stride,
                        };
                        self.solve_segment(seg, data)?;
                        seg_len = 0;
                    }
                }

                if seg_len > 0 {
                    let seg = Segment {
                        base,
                        start: seg_start,
                        len: seg_len,
                        stride,
                    };
                    self.solve_segment(seg, data)?;
                }
            }
        }

        Ok(())
    }

    fn solve_segment(&mut self, seg: Segment, data: &mut [f64]) -> Result<(), Error> {
        let n = seg.len;
        let lambda = self.lambda;

        self.ws.fill(n, |a, b, c, d| {
            a.fill(-lambda);
            b.fill(1.0 + 2.0 * lambda);
            c.fill(-lambda);
            // Insulated ends: fold the ghost point into the diagonal.
            // TODO: actual logic here, if any.
            b[0] -= lambda;
            b[n - 1] -= lambda;

            for k in 0..n {
                d[k] = data[seg.base + (seg.start + k) * seg.stride];
            }
        })?;

        let sol = self.ws.solve(n)?;

        for k in 0..n {
            data[seg.base + (seg.start + k) * seg.stride] = sol[k];
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn params(lambda: f64) -> HeatParams {
        HeatParams {
            dt: 1.0,
            alpha: lambda,
            h: 1.0,
        }
    }

    #[test]
    fn params_lambda() {
        let p = HeatParams {
            dt: 2.0,
            alpha: 3.0,
            h: 0.5,
        };

        assert!((p.lambda() - 24.0).abs() < EPS);
    }

    #[test]
    fn decays_eigenmode_at_correct_rate() {
        let n = 11;
        let lambda = 0.5;
        let steps = 5;
        let shape = Shape::new(n, 1, 1);

        let mut l = Lattice::new(shape);
        l.set_from(|i, _, _| {
            let theta = std::f64::consts::PI * (i as f64 + 0.5) / n as f64;
            theta.cos()
        });

        let mut s = HeatSolver::new(shape, params(lambda));
        for _ in 0..steps {
            s.step(&mut l).unwrap();
        }

        let mu = 2.0 - 2.0 * (std::f64::consts::PI / n as f64).cos();
        let factor = (1.0 / (1.0 + lambda * mu)).powi(steps);

        for i in 0..n {
            let theta = std::f64::consts::PI * (i as f64 + 0.5) / n as f64;
            let expected = factor * theta.cos();
            let got = l.value(i, 0, 0);
            assert!(
                (got - expected).abs() < 1e-10,
                "i = {i}: got {got}, expected {expected}",
            );
        }
    }

    #[test]
    fn approaches_mean_in_insulated_3d_domain() {
        let n = 4;
        let mut l = Lattice::new(Shape::cube(n));
        l.set(1, 1, 1, 100.0);
        l.set(2, 2, 2, 50.0);

        let mean = l.data().iter().sum::<f64>() / l.data().len() as f64;
        assert!((mean - 150.0 / 64.0).abs() < 1e-12);

        let mut s = HeatSolver::new(l.shape(), params(1.0));
        for _ in 0..100 {
            s.step(&mut l).unwrap();
        }

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let v = l.value(i, j, k);
                    assert!(
                        (v - mean).abs() < EPS,
                        "({i},{j},{k}): got {v}, expected {mean}",
                    );
                }
            }
        }
    }

    #[test]
    fn single_solid_point_is_unchanged() {
        let mut l = Lattice::new(Shape::new(3, 1, 1));
        l.set(1, 0, 0, 42.0);
        l.void(0, 0, 0);
        l.void(2, 0, 0);

        let mut s = HeatSolver::new(l.shape(), params(0.5));
        s.step(&mut l).unwrap();

        assert!((l.value(1, 0, 0) - 42.0).abs() < EPS);
    }

    #[test]
    fn uniform_field_is_preserved() {
        let mut l = Lattice::new(Shape::cube(4));
        l.set_all(10.0);

        let mut s = HeatSolver::new(l.shape(), params(0.1));
        for _ in 0..5 {
            s.step(&mut l).unwrap();
        }

        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    assert!((l.value(x, y, z) - 10.0).abs() < EPS);
                }
            }
        }
    }

    #[test]
    fn total_heat_conserved_in_insulated_domain() {
        let mut l = Lattice::new(Shape::cube(4));
        l.set(1, 1, 1, 50.0);
        l.set(2, 2, 2, 30.0);

        let initial: f64 = l.data().iter().sum();

        let mut s = HeatSolver::new(l.shape(), params(0.1));
        for _ in 0..30 {
            s.step(&mut l).unwrap();
        }

        let final_sum: f64 = l.data().iter().sum();
        assert!((final_sum - initial).abs() < EPS);
    }

    #[test]
    fn rejects_mismatched_shape() {
        let mut l = Lattice::new(Shape::cube(3));
        let mut s = HeatSolver::new(Shape::cube(4), params(0.1));
        assert_eq!(s.step(&mut l), Err(Error::InvalidDimensions));
    }

    #[test]
    fn single_element_grid() {
        let mut l = Lattice::new(Shape::cube(1));
        l.set(0, 0, 0, 7.0);

        let mut s = HeatSolver::new(l.shape(), params(0.1));
        s.step(&mut l).unwrap();

        assert!((l.value(0, 0, 0) - 7.0).abs() < EPS);
    }

    #[test]
    fn solver_works_across_different_lattices_of_same_shape() {
        let shape = Shape::cube(3);
        let mut solver = HeatSolver::new(shape, params(0.1));

        let mut l1 = Lattice::new(shape);
        l1.set(1, 1, 1, 100.0);
        solver.step(&mut l1).unwrap();

        let mut l2 = Lattice::new(shape);
        l2.set(0, 0, 0, 50.0);
        solver.step(&mut l2).unwrap();

        assert!(l1.value(1, 1, 1) < 100.0);
        assert!(l2.value(0, 0, 0) < 50.0);
    }
}
