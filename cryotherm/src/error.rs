//! The crate-wide error type.
//!
//! Every fallible function in `cryotherm` returns [`Error`]. Variants are
//! split by origin:
//!
//! - [`Error::InvalidDimensions`] — the caller passed arguments whose
//!   lengths or shapes don't fit together.
//! - [`Error::UnstableSystem`] — the input is well-formed but numerically
//!   unsuitable for the requested computation.
//!
//! The enum is `#[non_exhaustive]`: new variants may be added in minor
//! versions, so downstream `match` expressions must include a wildcard
//! arm.

/// The error type returned by fallible functions in this crate.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// A slice length or lattice shape did not match what the function
    /// required.
    ///
    /// Raised before any computation begins, so no output buffer is
    /// modified when this variant is returned. Sources include
    /// [`crate::math::tridiagonal::solve`] and
    /// `crate::solvers::heat::HeatSolver::step`.
    InvalidDimensions,

    /// The tridiagonal system is not strictly diagonally dominant, by row
    /// or by column, so the Thomas algorithm is not guaranteed to be
    /// numerically stable.
    ///
    /// Returned before any buffer is written. With the stencils currently
    /// used by the crate, this variant is unreachable — strict dominance
    /// holds for any positive stencil weight — but the underlying
    /// [`crate::math::tridiagonal::solve`] kernel is generic and will report
    /// it for arbitrary systems.
    UnstableSystem,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidDimensions => f.write_str("incompatible dimensions"),
            Error::UnstableSystem => {
                f.write_str("tridiagonal system is not strictly diagonally dominant")
            }
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_stable() {
        assert_eq!(
            Error::InvalidDimensions.to_string(),
            "incompatible dimensions"
        );
        assert_eq!(
            Error::UnstableSystem.to_string(),
            "tridiagonal system is not strictly diagonally dominant",
        );
    }

    #[test]
    fn implements_std_error() {
        fn assert_is_error<E: std::error::Error>(_: &E) {}
        assert_is_error(&Error::InvalidDimensions);
        assert_is_error(&Error::UnstableSystem);
    }

    #[test]
    fn variants_are_copy_and_eq() {
        let e = Error::InvalidDimensions;
        let f = e;
        assert_eq!(e, f);
        assert_ne!(Error::InvalidDimensions, Error::UnstableSystem);
    }
}
