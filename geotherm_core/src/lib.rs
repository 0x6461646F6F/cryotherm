//! temp geotherm_core doc

/// hehehe error
#[derive(Debug, PartialEq, Eq)]
pub enum ThomasError {
    /// hehe dimensions
    InvalidDimensions,
    /// heheh stability
    UnstableSystem,
}

/// hehehe thomas
pub fn thomas_algorithm(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
    x: &mut [f64],
    scratch: &mut [f64],
) -> Result<(), ThomasError> {
    let x_num = d.len();

    if x_num == 0
        || a.len() != x_num - 1
        || b.len() != x_num
        || c.len() != x_num - 1
        || x.len() != x_num
        || scratch.len() < x_num
    {
        return Err(ThomasError::InvalidDimensions);
    }

    let mut row_dominance = true;
    let mut column_dominance = true;

    for i in 0..x_num {
        let bi_abs = b[i].abs();

        if row_dominance {
            let row_left = if i == 0 { 0.0 } else { a[i - 1].abs() };
            let row_right = if i == x_num - 1 { 0.0 } else { c[i].abs() };
            let row = row_left + row_right;

            row_dominance &= bi_abs > row;
        }

        if column_dominance {
            let column_up = if i == 0 { 0.0 } else { c[i - 1].abs() };
            let column_down = if i == x_num - 1 { 0.0 } else { a[i].abs() };
            let column = column_up + column_down;

            column_dominance &= bi_abs > column;
        }

        if !row_dominance && !column_dominance {
            return Err(ThomasError::UnstableSystem);
        }
    }

    x[0] = d[0] / b[0];

    if x_num == 1 {
        return Ok(());
    }

    scratch[0] = c[0] / b[0];

    for i in 1..x_num {
        let denominator = b[i] - a[i - 1] * scratch[i - 1];

        if i < x_num - 1 {
            scratch[i] = c[i] / denominator;
        }
        x[i] = (d[i] - a[i - 1] * x[i - 1]) / denominator;
    }

    for i in (0..x_num - 1).rev() {
        x[i] -= scratch[i] * x[i + 1];
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-12;

    #[test]
    fn test_empty() {
        let a = [];
        let b = [];
        let c = [];
        let d = [];

        let mut scratch = vec![];
        let mut x = vec![];

        let result = thomas_algorithm(&a, &b, &c, &d, &mut x, &mut scratch);
        assert_eq!(result, Err(ThomasError::InvalidDimensions));
    }

    #[test]
    fn test_scalar() {
        let a = [];
        let b = [1.0];
        let c = [];
        let d = [1.0];

        let mut scratch = vec![0.0; 1];
        let mut x = vec![0.0; 1];

        let result = thomas_algorithm(&a, &b, &c, &d, &mut x, &mut scratch);
        assert!(result.is_ok());

        assert!(
            (x[0] - 1.0).abs() < EPSILON,
            "x0 failed: expected 1.0, got {}",
            x[0]
        );
    }

    #[test]
    fn test_zero_scalar() {
        let a = [];
        let b = [0.0];
        let c = [];
        let d = [0.0];

        let mut scratch = vec![0.0; 1];
        let mut x = vec![0.0; 1];

        let result = thomas_algorithm(&a, &b, &c, &d, &mut x, &mut scratch);
        assert_eq!(result, Err(ThomasError::UnstableSystem));
    }

    #[test]
    fn test_proper_matrix() {
        let a = [1.0, 1.0];
        let b = [3.0, 3.0, 3.0];
        let c = [1.0, 1.0];
        let d = [5.0, 10.0, 11.0];

        let mut scratch = vec![0.0; 3];
        let mut x = vec![0.0; 3];

        assert!(thomas_algorithm(&a, &b, &c, &d, &mut x, &mut scratch).is_ok());

        assert!(
            (x[0] - 1.0).abs() < EPSILON,
            "x0 failed: expected 1.0, got {}",
            x[0]
        );
        assert!(
            (x[1] - 2.0).abs() < EPSILON,
            "x1 failed: expected 2.0, got {}",
            x[1]
        );
        assert!(
            (x[2] - 3.0).abs() < EPSILON,
            "x2 failed: expected 3.0, got {}",
            x[2]
        );
    }

    #[test]
    fn test_row_dominant_only() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0, 5.0];
        let c = [1.0, 2.0];
        let d = [5.0, 15.0, 19.0];

        let mut scratch = vec![0.0; 3];
        let mut x = vec![0.0; 3];

        assert!(thomas_algorithm(&a, &b, &c, &d, &mut x, &mut scratch).is_ok());

        assert!(
            (x[0] - 1.0).abs() < EPSILON,
            "x0 failed: expected 1.0, got {}",
            x[0]
        );
        assert!(
            (x[1] - 2.0).abs() < EPSILON,
            "x1 failed: expected 2.0, got {}",
            x[1]
        );
        assert!(
            (x[2] - 3.0).abs() < EPSILON,
            "x2 failed: expected 3.0, got {}",
            x[2]
        );
    }

    #[test]
    fn test_mixed_unstable_matrix() {
        let a = [1.0, 3.0];
        let b = [4.0, 4.0, 2.0];
        let c = [2.0, 1.0];
        let d = [1.0, 1.0, 1.0];

        let mut scratch = vec![0.0; 3];
        let mut x = vec![0.0; 3];

        let result = thomas_algorithm(&a, &b, &c, &d, &mut x, &mut scratch);
        assert_eq!(result, Err(ThomasError::UnstableSystem));
    }

    #[test]
    fn test_invalid_dimensions_catch() {
        let a = [1.0];
        let b = [2.0, 2.0, 2.0];
        let c = [1.0, 1.0];
        let d = [4.0, 8.0, 8.0];

        let mut scratch = vec![0.0; 3];
        let mut x = vec![0.0; 3];

        let result = thomas_algorithm(&a, &b, &c, &d, &mut x, &mut scratch);
        assert_eq!(result, Err(ThomasError::InvalidDimensions));
    }
}
