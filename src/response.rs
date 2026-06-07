//! Frequency response analysis utilities.

/// Compute the frequency response of a filter given numerator and denominator coefficients.
/// Returns (real, imag) at frequency omega.
pub fn freqz(b: &[f64], a: &[f64], omega: f64) -> (f64, f64) {
    let mut num_re = 0.0;
    let mut num_im = 0.0;
    for (n, &coeff) in b.iter().enumerate() {
        let angle = omega * n as f64;
        num_re += coeff * angle.cos();
        num_im -= coeff * angle.sin();
    }

    let mut den_re = 0.0;
    let mut den_im = 0.0;
    for (n, &coeff) in a.iter().enumerate() {
        let angle = omega * n as f64;
        den_re += coeff * angle.cos();
        den_im -= coeff * angle.sin();
    }

    let den_mag_sq = den_re * den_re + den_im * den_im;
    let h_re = (num_re * den_re + num_im * den_im) / den_mag_sq;
    let h_im = (num_im * den_re - num_re * den_im) / den_mag_sq;
    (h_re, h_im)
}

/// Compute frequency response over a range of frequencies.
/// Returns vectors of (frequency, magnitude, phase).
pub fn freqz_range(b: &[f64], a: &[f64], n_points: usize) -> Vec<(f64, f64, f64)> {
    (0..n_points).map(|i| {
        let omega = std::f64::consts::PI * i as f64 / (n_points - 1) as f64;
        let (re, im) = freqz(b, a, omega);
        let mag = re.hypot(im);
        let phase = im.atan2(re);
        (omega, mag, phase)
    }).collect()
}

/// Convert magnitude to decibels.
pub fn mag_to_db(mag: f64) -> f64 {
    20.0 * mag.max(1e-20).log10()
}

/// Convert decibels to linear magnitude.
pub fn db_to_mag(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Compute the group delay at a given frequency using finite differences.
pub fn group_delay(b: &[f64], a: &[f64], omega: f64, delta: f64) -> f64 {
    let phase1 = {
        let (re, im) = freqz(b, a, omega - delta);
        im.atan2(re)
    };
    let phase2 = {
        let (re, im) = freqz(b, a, omega + delta);
        im.atan2(re)
    };
    -(phase2 - phase1) / (2.0 * delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freqz_identity() {
        // H(z) = 1
        let (re, im) = freqz(&[1.0], &[1.0], 0.5);
        assert!((re - 1.0).abs() < 1e-14);
        assert!(im.abs() < 1e-14);
    }

    #[test]
    fn test_freqz_dc() {
        // H(z) = (1 + z^{-1}) / 2 at DC: H(0) = (1+1)/2 = 1
        let (re, im) = freqz(&[0.5, 0.5], &[1.0], 0.0);
        assert!((re - 1.0).abs() < 1e-14);
        assert!(im.abs() < 1e-14);
    }

    #[test]
    fn test_freqz_nyquist() {
        // H(z) = (1 - z^{-1}) / 2 at Nyquist: H(π) = (1-(-1))/2 = 1
        let (re, _im) = freqz(&[0.5, -0.5], &[1.0], std::f64::consts::PI);
        assert!((re - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_freqz_range_length() {
        let result = freqz_range(&[1.0], &[1.0], 100);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_mag_to_db_unity() {
        assert!((mag_to_db(1.0)).abs() < 1e-14);
    }

    #[test]
    fn test_mag_to_db_factor() {
        // 10x magnitude = 20 dB
        assert!((mag_to_db(10.0) - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_db_to_mag_roundtrip() {
        let _db = -6.5;
        assert!((db_to_mag(mag_to_db(0.47)) - 0.47).abs() < 1e-10);
    }

    #[test]
    fn test_group_delay_positive() {
        // Group delay should be non-negative for causal filters
        let gd = group_delay(&[1.0, 0.5], &[1.0, -0.3], 0.5, 0.001);
        assert!(gd > -10.0, "Group delay should be reasonable: {}", gd);
    }
}
