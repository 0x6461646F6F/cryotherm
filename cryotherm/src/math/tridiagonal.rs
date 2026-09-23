//! Tridiagonal solver for `A·x = d`.
//!
//! Implements the Thomas algorithm with a strict diagonal dominance check
//! on entry. Two entry points are provided:
//!
//! - [`solve`] — one-shot, takes all buffers from the caller.
//! - [`Workspace`] — preallocated buffers for repeated solves.

use crate::error::Error;

/// Preallocated buffers for repeated solves.
///
/// Construct once for the longest system you expect, then per solve:
///
/// 1. Write coefficients and RHS through [`fill`](Self::fill).
/// 2. Call [`solve`](Self::solve), which returns the solution slice.
///
/// # Examples
///
/// ```
/// use cryotherm::math::tridiagonal::Workspace;
///
/// // Solve [4 1 0; 1 4 1; 0 1 4]·x = [6, 12, 14] => x = [1, 2, 3]
/// let mut ws = Workspace::new(3);
/// ws.fill(3, |a, b, c, d| {
///     a.copy_from_slice(&[1.0, 1.0]);
///     b.copy_from_slice(&[4.0, 4.0, 4.0]);
///     c.copy_from_slice(&[1.0, 1.0]);
///     d.copy_from_slice(&[6.0, 12.0, 14.0]);
/// }).unwrap();
///
/// let sol = ws.solve(3).unwrap();
/// assert!((sol[0] - 1.0).abs() < 1e-12);
/// assert!((sol[1] - 2.0).abs() < 1e-12);
/// assert!((sol[2] - 3.0).abs() < 1e-12);
/// ```
#[derive(Debug)]
pub struct Workspace {
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
    x: Vec<f64>,
    scratch: Vec<f64>,
}

impl Workspace {
    /// Allocates buffers for systems of up to `max_len` unknowns.
    ///
    /// `max_len == 0` is permitted but every subsequent call to
    /// [`fill`](Self::fill) or [`solve`](Self::solve) will return
    /// [`Error::InvalidDimensions`].
    pub fn new(max_len: usize) -> Self {
        let off = max_len.saturating_sub(1);

        Self {
            a: vec![0.0; off],
            b: vec![0.0; max_len],
            c: vec![0.0; off],
            d: vec![0.0; max_len],
            x: vec![0.0; max_len],
            scratch: vec![0.0; off],
        }
    }

    /// Largest system this workspace can solve.
    pub fn max_len(&self) -> usize {
        self.b.len()
    }

    /// Writes the four input buffers for a system of length `n`.
    ///
    /// The closure receives `a`, `b`, `c`, `d` with lengths `n-1`, `n`,
    /// `n-1`, `n` respectively. The slices are disjoint and borrowed only
    /// for the duration of the call.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidDimensions`] if `n == 0` or
    /// `n > max_len`. In that case the closure is not called and the
    /// internal buffers are left untouched.
    pub fn fill<F>(&mut self, n: usize, f: F) -> Result<(), Error>
    where
        F: FnOnce(&mut [f64], &mut [f64], &mut [f64], &mut [f64]),
    {
        if n == 0 || n > self.max_len() {
            return Err(Error::InvalidDimensions);
        }

        let off = n - 1;
        f(
            &mut self.a[..off],
            &mut self.b[..n],
            &mut self.c[..off],
            &mut self.d[..n],
        );

        Ok(())
    }

    /// Solves the system currently held in the input buffers, returning
    /// the solution slice of length `n`.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidDimensions`] if `n == 0` or `n > max_len`.
    /// - [`Error::UnstableSystem`] if the system is not strictly
    ///   diagonally dominant. See [`solve`] for details.
    pub fn solve(&mut self, n: usize) -> Result<&[f64], Error> {
        if n == 0 || n > self.max_len() {
            return Err(Error::InvalidDimensions);
        }

        let off = n - 1;
        solve(
            &self.a[..off],
            &self.b[..n],
            &self.c[..off],
            &self.d[..n],
            &mut self.x[..n],
            &mut self.scratch[..off],
        )?;

        Ok(&self.x[..n])
    }
}

/// Solves a tridiagonal system `A·x = d` in place.
///
/// `a`, `b`, `c` are the sub-, main-, and super-diagonals: `a.len() == n-1`,
/// `c.len() == n-1`, `d.len() == n`, where `n == b.len()`. `x` receives the
/// solution and must have length `n`. `scratch` must be at least `n-1`.
///
/// # Errors
///
/// - [`Error::InvalidDimensions`] if any slice has the wrong length, or if
///   `n == 0`.
/// - [`Error::UnstableSystem`] if `A` is not strictly diagonally dominant
///   by row or by column.
///
/// On either error, `x` and `scratch` are left unchanged.
///
/// # Algorithm
///
/// Forward elimination followed by back-substitution. The dominance check
/// runs first, in O(n), so no division by a degenerate pivot can occur —
/// strict dominance guarantees every denominator during elimination is
/// bounded away from zero.
///
/// Systems that are nonsingular but not strictly dominant are rejected.
/// This is intentional: the Thomas algorithm is not stable in that regime,
/// and a partial-pivoting variant is out of scope.
///
/// # Examples
///
/// ```
/// use cryotherm::math::tridiagonal::solve;
///
/// // [4 1 0; 1 4 1; 0 1 4]·x = [6, 12, 14] => x = [1, 2, 3]
/// let a = [1.0, 1.0];
/// let b = [4.0, 4.0, 4.0];
/// let c = [1.0, 1.0];
/// let d = [6.0, 12.0, 14.0];
///
/// let mut x = [0.0; 3];
/// let mut scratch = [0.0; 2];
/// solve(&a, &b, &c, &d, &mut x, &mut scratch).unwrap();
///
/// assert!((x[0] - 1.0).abs() < 1e-12);
/// assert!((x[1] - 2.0).abs() < 1e-12);
/// assert!((x[2] - 3.0).abs() < 1e-12);
/// ```
pub fn solve(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
    x: &mut [f64],
    scratch: &mut [f64],
) -> Result<(), Error> {
    let n = d.len();

    if n == 0
        || a.len() != n - 1
        || b.len() != n
        || c.len() != n - 1
        || x.len() != n
        || scratch.len() < n - 1
    {
        return Err(Error::InvalidDimensions);
    }

    let mut row_dominance = true;
    let mut col_dominance = true;

    for i in 0..n {
        let bi = b[i].abs();

        if row_dominance {
            let left = if i == 0 { 0.0 } else { a[i - 1].abs() };
            let right = if i == n - 1 { 0.0 } else { c[i].abs() };
            row_dominance = bi > left + right;
        }

        if col_dominance {
            let up = if i == 0 { 0.0 } else { c[i - 1].abs() };
            let down = if i == n - 1 { 0.0 } else { a[i].abs() };
            col_dominance = bi > up + down;
        }

        if !row_dominance && !col_dominance {
            return Err(Error::UnstableSystem);
        }
    }

    x[0] = d[0] / b[0];
    if n == 1 {
        return Ok(());
    }

    scratch[0] = c[0] / b[0];

    for i in 1..n {
        let denom = b[i] - a[i - 1] * scratch[i - 1];

        x[i] = (d[i] - a[i - 1] * x[i - 1]) / denom;
        if i < n - 1 {
            scratch[i] = c[i] / denom;
        }
    }

    for i in (0..n - 1).rev() {
        x[i] -= scratch[i] * x[i + 1];
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn noop(_: &mut [f64], _: &mut [f64], _: &mut [f64], _: &mut [f64]) {}

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < EPS, "index {i}: got {a}, expected {e}");
        }
    }

    #[test]
    fn solve_n1() {
        let a: [f64; 0] = [];
        let b = [4.0];
        let c: [f64; 0] = [];
        let d = [8.0];
        let mut x = [0.0];
        let mut scratch: [f64; 0] = [];
        solve(&a, &b, &c, &d, &mut x, &mut scratch).unwrap();
        assert_close(&x, &[2.0]);
    }

    #[test]
    fn solve_n2() {
        let a = [1.0];
        let b = [3.0, 3.0];
        let c = [1.0];
        let d = [5.0, 7.0];
        let mut x = [0.0; 2];
        let mut scratch = [0.0; 1];
        solve(&a, &b, &c, &d, &mut x, &mut scratch).unwrap();
        assert_close(&x, &[1.0, 2.0]);
    }

    #[test]
    fn solve_n3() {
        let a = [1.0, 1.0];
        let b = [4.0, 4.0, 4.0];
        let c = [1.0, 1.0];
        let d = [6.0, 12.0, 14.0];
        let mut x = [0.0; 3];
        let mut scratch = [0.0; 2];
        solve(&a, &b, &c, &d, &mut x, &mut scratch).unwrap();
        assert_close(&x, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn solve_n5() {
        let a = [1.0; 4];
        let b = [4.0; 5];
        let c = [1.0; 4];
        let d = [5.0, 6.0, 6.0, 6.0, 5.0];
        let mut x = [0.0; 5];
        let mut scratch = [0.0; 4];
        solve(&a, &b, &c, &d, &mut x, &mut scratch).unwrap();
        assert_close(&x, &[1.0; 5]);
    }

    #[test]
    fn solve_rejects_n_zero() {
        let empty: [f64; 0] = [];
        let mut x: [f64; 0] = [];
        let mut scratch: [f64; 0] = [];
        assert_eq!(
            solve(&empty, &empty, &empty, &empty, &mut x, &mut scratch),
            Err(Error::InvalidDimensions),
        );
    }

    #[test]
    fn solve_rejects_wrong_lengths() {
        let a = [1.0, 1.0];
        let b = [4.0, 4.0, 4.0];
        let c = [1.0, 1.0];
        let d = [1.0, 1.0, 1.0];
        let mut x = [0.0; 3];
        let mut scratch = [0.0; 2];

        assert_eq!(
            solve(&a[..1], &b, &c, &d, &mut x, &mut scratch),
            Err(Error::InvalidDimensions),
        );
        assert_eq!(
            solve(&a, &b[..2], &c, &d, &mut x, &mut scratch),
            Err(Error::InvalidDimensions),
        );
        assert_eq!(
            solve(&a, &b, &c[..1], &d, &mut x, &mut scratch),
            Err(Error::InvalidDimensions),
        );
        assert_eq!(
            solve(&a, &b, &c, &d[..2], &mut x, &mut scratch),
            Err(Error::InvalidDimensions),
        );

        let mut x_short = [0.0; 2];
        assert_eq!(
            solve(&a, &b, &c, &d, &mut x_short, &mut scratch),
            Err(Error::InvalidDimensions),
        );

        let mut scratch_short = [0.0; 1];
        assert_eq!(
            solve(&a, &b, &c, &d, &mut x, &mut scratch_short),
            Err(Error::InvalidDimensions),
        );
    }

    #[test]
    fn solve_rejects_non_dominant() {
        let a = [1.0, 1.0];
        let b = [1.0, 1.0, 1.0];
        let c = [1.0, 1.0];
        let d = [1.0, 1.0, 1.0];
        let mut x = [0.0; 3];
        let mut scratch = [0.0; 2];
        assert_eq!(
            solve(&a, &b, &c, &d, &mut x, &mut scratch),
            Err(Error::UnstableSystem),
        );
    }

    #[test]
    fn solve_row_fails_then_loop_continues() {
        let a = [1.0, 0.1];
        let b = [10.0, 1.0, 10.0];
        let c = [0.1, 0.1];
        let d = [1.0, 1.0, 1.0];
        let mut x = [0.0; 3];
        let mut scratch = [0.0; 2];
        solve(&a, &b, &c, &d, &mut x, &mut scratch).unwrap();

        let x1 = 9.0 / 98.9;
        let x2 = 89.0 / 98.9;
        assert!((x[0] - x1).abs() < 1e-12);
        assert!((x[1] - x2).abs() < 1e-12);
        assert!((x[2] - x1).abs() < 1e-12);
    }

    #[test]
    fn solve_col_fails_then_loop_continues() {
        let a = [0.1, 0.1];
        let b = [10.0, 1.0, 10.0];
        let c = [1.0, 0.1];
        let d = [1.0, 1.0, 1.0];
        let mut x = [0.0; 3];
        let mut scratch = [0.0; 2];
        solve(&a, &b, &c, &d, &mut x, &mut scratch).unwrap();

        let x1 = 0.9 / 989.0;
        let x2 = 98.0 / 98.9;
        let x3 = 89.1 / 989.0;
        assert!((x[0] - x1).abs() < 1e-12);
        assert!((x[1] - x2).abs() < 1e-12);
        assert!((x[2] - x3).abs() < 1e-12);
    }

    #[test]
    fn solve_leaves_buffers_unchanged_on_unstable() {
        let a = [1.0, 1.0];
        let b = [1.0, 1.0, 1.0];
        let c = [1.0, 1.0];
        let d = [1.0, 1.0, 1.0];
        let mut x = [7.0; 3];
        let mut scratch = [8.0; 2];
        let _ = solve(&a, &b, &c, &d, &mut x, &mut scratch);
        assert_eq!(x, [7.0; 3]);
        assert_eq!(scratch, [8.0; 2]);
    }

    #[test]
    fn solve_leaves_buffers_unchanged_on_bad_dims() {
        let a = [1.0, 1.0];
        let b = [4.0, 4.0, 4.0];
        let c = [1.0, 1.0];
        let d = [1.0, 1.0, 1.0];
        let mut x = [7.0; 3];
        let mut scratch = [8.0; 1];
        let _ = solve(&a, &b, &c, &d, &mut x, &mut scratch);
        assert_eq!(x, [7.0; 3]);
        assert_eq!(scratch, [8.0]);
    }

    #[test]
    fn workspace_max_len() {
        assert_eq!(Workspace::new(10).max_len(), 10);
        assert_eq!(Workspace::new(0).max_len(), 0);
    }

    #[test]
    fn workspace_solves_and_reuses() {
        let mut ws = Workspace::new(3);

        ws.fill(3, |a, b, c, d| {
            a.copy_from_slice(&[1.0, 1.0]);
            b.copy_from_slice(&[4.0, 4.0, 4.0]);
            c.copy_from_slice(&[1.0, 1.0]);
            d.copy_from_slice(&[6.0, 12.0, 14.0]);
        })
        .unwrap();

        assert_close(ws.solve(3).unwrap(), &[1.0, 2.0, 3.0]);

        ws.fill(3, |a, b, c, d| {
            a.copy_from_slice(&[1.0, 1.0]);
            b.copy_from_slice(&[4.0, 4.0, 4.0]);
            c.copy_from_slice(&[1.0, 1.0]);
            d.copy_from_slice(&[4.0, 4.0, 4.0]);
        })
        .unwrap();

        assert_close(ws.solve(3).unwrap(), &[6.0 / 7.0, 4.0 / 7.0, 6.0 / 7.0]);
    }

    #[test]
    fn workspace_fill_rejects_out_of_range() {
        let mut ws = Workspace::new(3);
        assert_eq!(ws.fill(0, noop), Err(Error::InvalidDimensions));
        assert_eq!(ws.fill(4, noop), Err(Error::InvalidDimensions));
    }

    #[test]
    fn workspace_solve_rejects_out_of_range() {
        let mut ws = Workspace::new(3);
        assert!(matches!(ws.solve(0), Err(Error::InvalidDimensions)));
        assert!(matches!(ws.solve(4), Err(Error::InvalidDimensions)));
    }
}
