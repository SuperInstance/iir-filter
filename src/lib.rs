//! # IIR Filter Design
//!
//! Infinite impulse response filter design using Butterworth and Chebyshev
//! approximations. Supports biquad cascade (second-order sections) form for
//! numerical stability.

pub mod butterworth;
pub mod chebyshev;
pub mod biquad;
pub mod cascade;
pub mod response;

pub use butterworth::ButterworthFilter;
pub use chebyshev::{ChebyshevType1Filter, ChebyshevType2Filter};
pub use biquad::Biquad;
pub use cascade::CascadeFilter;
pub use response::{freqz, freqz_range};

/// IIR filter coefficients in transfer function form: H(z) = B(z)/A(z)
#[derive(Debug, Clone)]
pub struct IirCoeffs {
    /// Numerator coefficients (b0, b1, b2, ...)
    pub b: Vec<f64>,
    /// Denominator coefficients (a0=1, a1, a2, ...)
    pub a: Vec<f64>,
}

impl IirCoeffs {
    /// Create new IIR coefficients. a[0] is normalized to 1.0.
    pub fn new(b: Vec<f64>, mut a: Vec<f64>) -> Self {
        let a0 = a[0];
        if a0.abs() > 1e-15 {
            for v in &mut a {
                *v /= a0;
            }
        }
        Self { b, a }
    }

    /// Apply the IIR filter to a signal using direct form II transposed
    pub fn filter(&self, signal: &[f64]) -> Vec<f64> {
        let nb = self.b.len();
        let na = self.a.len();
        let nf = nb.max(na);
        let n = signal.len();
        let mut output = vec![0.0; n];
        let mut w = vec![0.0; nf];

        for i in 0..n {
            let mut acc = if i < nb { self.b[0] * signal[i] } else { 0.0 };
            for j in 1..nf {
                let x_delayed = if i >= j && j < nb { signal[i - j] } else { 0.0 };
                acc += if j < nb { self.b[j] * x_delayed } else { 0.0 };
                acc -= if j < na { self.a[j] * w[j - 1] } else { 0.0 };
            }
            // Shift delay line
            for j in (1..nf).rev() {
                w[j] = if j > 0 { w[j - 1] } else { 0.0 };
            }
            w[0] = acc;
            output[i] = acc;
        }
        output
    }
}
