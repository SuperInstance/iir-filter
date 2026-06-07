//! Biquad (second-order IIR) filter section.

/// A single biquad filter section: H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)
#[derive(Debug, Clone, Copy)]
pub struct Biquad {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
    pub z1: f64,
    pub z2: f64,
}

impl Biquad {
    /// Create a new biquad section.
    pub fn new(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Self { b0, b1, b2, a1, a2, z1: 0.0, z2: 0.0 }
    }

    /// Process a single sample through the biquad (Direct Form II Transposed).
    pub fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Process a slice of samples, returning filtered output.
    pub fn filter(&mut self, signal: &[f64]) -> Vec<f64> {
        signal.iter().map(|&x| self.process(x)).collect()
    }

    /// Compute the magnitude response at normalized frequency omega (0 to π).
    pub fn magnitude(&self, omega: f64) -> f64 {
        let (re, im) = self.response(omega);
        re.hypot(im)
    }

    /// Compute the complex frequency response at omega.
    pub fn response(&self, omega: f64) -> (f64, f64) {
        let cos_w = omega.cos();
        let sin_w = omega.sin();
        let cos_2w = (2.0 * omega).cos();
        let sin_2w = (2.0 * omega).sin();

        // Numerator: b0 + b1·e^{-jω} + b2·e^{-j2ω}
        let num_re = self.b0 + self.b1 * cos_w + self.b2 * cos_2w;
        let num_im = -(self.b1 * sin_w + self.b2 * sin_2w);

        // Denominator: 1 + a1·e^{-jω} + a2·e^{-j2ω}
        let den_re = 1.0 + self.a1 * cos_w + self.a2 * cos_2w;
        let den_im = -(self.a1 * sin_w + self.a2 * sin_2w);

        // Complex division
        let den_mag_sq = den_re * den_re + den_im * den_im;
        let h_re = (num_re * den_re + num_im * den_im) / den_mag_sq;
        let h_im = (num_im * den_re - num_re * den_im) / den_mag_sq;

        (h_re, h_im)
    }

    /// Check if the biquad is stable (poles inside the unit circle).
    pub fn is_stable(&self) -> bool {
        self.a2.abs() < 1.0 && self.a1.abs() < 1.0 + self.a2
    }

    /// Reset the internal state.
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Get internal state z1 (for testing).
    pub fn state_z1(&self) -> f64 {
        self.z1
    }

    /// Get internal state z2 (for testing).
    pub fn state_z2(&self) -> f64 {
        self.z2
    }
}

impl Default for Biquad {
    fn default() -> Self {
        Self {
            b0: 1.0, b1: 0.0, b2: 0.0,
            a1: 0.0, a2: 0.0,
            z1: 0.0, z2: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_biquad() {
        let mut bq = Biquad::default();
        let signal = [1.0, 2.0, 3.0, 4.0, 5.0];
        let output = bq.filter(&signal);
        for i in 0..signal.len() {
            assert!((output[i] - signal[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn test_biquad_dc_gain() {
        let bq = Biquad::new(0.5, 0.5, 0.0, 0.0, 0.0);
        let dc = bq.magnitude(0.0);
        assert!((dc - 1.0).abs() < 1e-14, "DC gain should be 1.0");
    }

    #[test]
    fn test_biquad_stability() {
        let stable = Biquad::new(1.0, 0.0, 0.0, 0.5, 0.3);
        assert!(stable.is_stable());

        let unstable = Biquad::new(1.0, 0.0, 0.0, 0.0, 1.5);
        assert!(!unstable.is_stable());
    }

    #[test]
    fn test_biquad_reset() {
        let mut bq = Biquad::new(1.0, 1.0, 0.0, -0.5, 0.0);
        bq.filter(&[1.0, 2.0, 3.0]);
        bq.reset();
        assert!((bq.state_z1()).abs() < 1e-15);
        assert!((bq.state_z2()).abs() < 1e-15);
    }

    #[test]
    fn test_biquad_magnitude_range() {
        let bq = Biquad::new(1.0, 0.5, 0.25, -0.3, 0.1);
        for i in 0..100 {
            let omega = std::f64::consts::PI * i as f64 / 100.0;
            let mag = bq.magnitude(omega);
            assert!(mag >= 0.0, "Magnitude should be non-negative");
        }
    }

    #[test]
    fn test_biquad_response_symmetry() {
        let bq = Biquad::new(0.3, 0.5, 0.2, -0.4, 0.15);
        for i in 1..50 {
            let w = std::f64::consts::PI * i as f64 / 100.0;
            let m1 = bq.magnitude(w);
            let m2 = bq.magnitude(2.0 * std::f64::consts::PI - w);
            assert!((m1 - m2).abs() < 1e-12);
        }
    }
}
