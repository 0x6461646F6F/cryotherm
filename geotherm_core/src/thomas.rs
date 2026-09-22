//! Solver for tridiagonal linear systems using the Thomas algorithm.

/// Errors returned by [`thomas_algorithm`].
#[derive(Debug, PartialEq, Eq)]
pub enum ThomasError {
    /// Slice lengths don't match the system size. Required for size `n`:
    /// `a`/`c` = `n-1`, `b`/`d`/`x` = `n`, `scratch` >= `n-1`, `n > 0`.
    InvalidDimensions,
    /// The matrix is not strictly diagonally dominant (by row or by column),
    /// so the algorithm is not guaranteed to be numerically stable.
    UnstableSystem,
}

/// Solves a tridiagonal system `A·x = d` in place, without allocating.
///
/// The matrix `A` is given by its three diagonals: `a` (sub), `b` (main),
/// `c` (super). The right-hand side is `d`; the solution is written to `x`.
/// `scratch` is workspace of length `>= n - 1`; its contents on entry are
/// ignored and are unspecified after the call.
///
/// Requires the matrix to be strictly diagonally dominant (either by row or
/// by column). This holds for typical finite-difference / finite-volume
/// discretizations of the heat equation with Dirichlet or mixed boundary
/// conditions, but not for all-Neumann problems (which are singular anyway).
///
/// # Errors
///
/// - [`ThomasError::InvalidDimensions`] — slices have wrong length.
///   Nothing is written in this case.
/// - [`ThomasError::UnstableSystem`] — the diagonal dominance precondition
///   is not met. This check runs before any output is written, so `x` is
///   left untouched.
///
/// # Example
///
/// ```
/// use geotherm_core::thomas::thomas_algorithm;
///
/// // [4 1 0; 1 4 1; 0 1 4] x = [6, 12, 14]  =>  x = [1, 2, 3]
/// let (a, b, c, d) = ([1.0, 1.0], [4.0; 3], [1.0, 1.0], [6.0, 12.0, 14.0]);
/// let (mut x, mut scratch) = ([0.0; 3], [0.0; 2]);
/// thomas_algorithm(&a, &b, &c, &d, &mut x, &mut scratch).unwrap();
/// assert!((x[1] - 2.0).abs() < 1e-12);
/// ```
pub fn thomas_algorithm(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
    x: &mut [f64],
    scratch: &mut [f64],
) -> Result<(), ThomasError> {
    let n = d.len();

    if n == 0
        || a.len() != n - 1
        || b.len() != n
        || c.len() != n - 1
        || x.len() != n
        || scratch.len() < n - 1
    {
        return Err(ThomasError::InvalidDimensions);
    }

    let mut row_dominance = true;
    let mut column_dominance = true;

    for i in 0..n {
        let bi_abs = b[i].abs();

        if row_dominance {
            let row_left = if i == 0 { 0.0 } else { a[i - 1].abs() };
            let row_right = if i == n - 1 { 0.0 } else { c[i].abs() };
            let row = row_left + row_right;

            row_dominance = bi_abs > row;
        }

        if column_dominance {
            let column_up = if i == 0 { 0.0 } else { c[i - 1].abs() };
            let column_down = if i == n - 1 { 0.0 } else { a[i].abs() };
            let column = column_up + column_down;

            column_dominance = bi_abs > column;
        }

        if !row_dominance && !column_dominance {
            return Err(ThomasError::UnstableSystem);
        }
    }

    x[0] = d[0] / b[0];

    if n == 1 {
        return Ok(());
    }

    scratch[0] = c[0] / b[0];

    for i in 1..n {
        let denominator = b[i] - a[i - 1] * scratch[i - 1];

        x[i] = (d[i] - a[i - 1] * x[i - 1]) / denominator;
        if i < n - 1 {
            scratch[i] = c[i] / denominator;
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

    const EPS: f64 = 1e-9;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < EPS, "idx {i}: got {a}, want {e}");
        }
    }

    #[test]
    fn solves_n1() {
        let mut x = [0.0];
        let mut s: [f64; 0] = [];
        thomas_algorithm(&[], &[2.0], &[], &[4.0], &mut x, &mut s).unwrap();
        assert_close(&x, &[2.0]);
    }

    #[test]
    fn solves_n3_both_dominant() {
        let mut x = [0.0; 3];
        let mut s = [0.0; 3];
        thomas_algorithm(
            &[1.0, 1.0],
            &[4.0; 3],
            &[1.0, 1.0],
            &[6.0, 12.0, 14.0],
            &mut x,
            &mut s,
        )
        .unwrap();
        assert_close(&x, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn solves_row_dominant_only() {
        let mut x = [0.0; 3];
        let mut s = [0.0; 3];
        thomas_algorithm(
            &[0.5, 2.0],
            &[3.0; 3],
            &[2.0, 0.5],
            &[5.0, 4.0, 5.0],
            &mut x,
            &mut s,
        )
        .unwrap();
        assert_close(&x, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn solves_column_dominant_only() {
        let mut x = [0.0; 3];
        let mut s = [0.0; 3];
        thomas_algorithm(
            &[2.0, 0.5],
            &[3.0; 3],
            &[0.5, 2.0],
            &[3.5, 7.0, 3.5],
            &mut x,
            &mut s,
        )
        .unwrap();
        assert_close(&x, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn rejects_n_zero() {
        let mut x: [f64; 0] = [];
        let mut s: [f64; 0] = [];
        assert!(matches!(
            thomas_algorithm(&[], &[], &[], &[], &mut x, &mut s),
            Err(ThomasError::InvalidDimensions)
        ));
    }

    #[test]
    fn rejects_wrong_dimensions() {
        let mut x = [0.0; 3];
        let mut s = [0.0; 3];

        assert!(matches!(
            thomas_algorithm(&[1.0], &[4.0; 3], &[1.0; 2], &[1.0; 3], &mut x, &mut s),
            Err(ThomasError::InvalidDimensions)
        ));
        assert!(matches!(
            thomas_algorithm(&[1.0; 2], &[4.0; 2], &[1.0; 2], &[1.0; 3], &mut x, &mut s),
            Err(ThomasError::InvalidDimensions)
        ));
        assert!(matches!(
            thomas_algorithm(&[1.0; 2], &[4.0; 3], &[1.0; 3], &[1.0; 3], &mut x, &mut s),
            Err(ThomasError::InvalidDimensions)
        ));
        let mut x_bad = [0.0; 4];
        assert!(matches!(
            thomas_algorithm(
                &[1.0; 2], &[4.0; 3], &[1.0; 2], &[1.0; 3], &mut x_bad, &mut s
            ),
            Err(ThomasError::InvalidDimensions)
        ));
        let mut s_bad = [0.0; 1];
        assert!(matches!(
            thomas_algorithm(
                &[1.0; 2], &[4.0; 3], &[1.0; 2], &[1.0; 3], &mut x, &mut s_bad
            ),
            Err(ThomasError::InvalidDimensions)
        ));
    }

    #[test]
    fn rejects_both_fail_same_iteration() {
        let mut x = [0.0; 3];
        let mut s = [0.0; 3];
        let r = thomas_algorithm(
            &[0.5, 0.5],
            &[1.0, 1.0, 1.0],
            &[0.5, 0.5],
            &[1.0, 2.0, 1.0],
            &mut x,
            &mut s,
        );
        assert!(matches!(r, Err(ThomasError::UnstableSystem)));
        assert_eq!(x, [0.0; 3]);
    }

    #[test]
    fn rejects_column_fails_first_then_row() {
        let mut x = [0.0; 2];
        let mut s = [0.0; 2];
        let r = thomas_algorithm(&[3.0], &[3.0, 3.0], &[1.0], &[1.0, 1.0], &mut x, &mut s);
        assert!(matches!(r, Err(ThomasError::UnstableSystem)));
        assert_eq!(x, [0.0; 2]);
    }

    #[test]
    fn rejects_row_fails_first_then_column() {
        let mut x = [0.0; 3];
        let mut s = [0.0; 3];
        let r = thomas_algorithm(
            &[0.5, 0.5],
            &[1.0, 5.0, 1.0],
            &[3.0, 1.0],
            &[1.0, 1.0, 1.0],
            &mut x,
            &mut s,
        );
        assert!(matches!(r, Err(ThomasError::UnstableSystem)));
        assert_eq!(x, [0.0; 3]);
    }

    #[test]
    fn rejects_zero_diagonal_n1() {
        let mut x = [0.0];
        let mut s: [f64; 0] = [];
        let r = thomas_algorithm(&[], &[0.0], &[], &[1.0], &mut x, &mut s);
        assert!(matches!(r, Err(ThomasError::UnstableSystem)));
        assert_eq!(x, [0.0]);
    }
}
