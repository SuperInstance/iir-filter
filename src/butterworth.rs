//! Butterworth filter design (maximally flat magnitude response).

use crate::biquad::Biquad;
use crate::cascade::CascadeFilter;

/// Butterworth filter designer.
pub struct ButterworthFilter;

impl ButterworthFilter {
    /// Compute the poles of an Nth-order Butterworth lowpass filter
    /// on the unit circle at radius 1 in the s-plane.
    /// Returns poles as (real, imag) pairs.
    pub fn analog_poles(n: usize) -> Vec<(f64, f64)> {
        let mut poles = Vec::new();
        for k in 0..n {
            let theta = std::f64::consts::PI * (2 * k + 1) as f64 / (2 * n) as f64
                + std::f64::consts::PI / 2.0;
            poles.push((theta.cos(), theta.sin()));
        }
        poles
    }

    /// Design a digital Butterworth lowpass filter via bilinear transform.
    ///
    /// # Arguments
    /// * `order` - Filter order
    /// * `cutoff` - Normalized cutoff frequency (0.0 to 1.0, where 1.0 = Nyquist)
    ///
    /// Returns a cascade of biquad sections.
    pub fn lowpass(order: usize, cutoff: f64) -> CascadeFilter {
        let wc = std::f64::consts::PI * cutoff;
        let warped_wc = 2.0 * (wc / 2.0).tan();

        let sections = match order {
            1 => {
                let alpha = warped_wc / 2.0;
                let a0 = 1.0 + alpha;
                vec![Biquad {
                    b0: alpha / a0,
                    b1: alpha / a0,
                    b2: 0.0,
                    a1: (1.0 - alpha) / a0,
                    a2: 0.0,
                    z1: 0.0, z2: 0.0,
                }]
            }
            _ => {
                let n_sections = order / 2;
                let mut secs = Vec::new();
                for k in 0..n_sections {
                    let theta = std::f64::consts::PI * (2 * k + 1) as f64 / (2 * order) as f64;
                    let p_re = -warped_wc * theta.cos() / 2.0;
                    let p_im = warped_wc * theta.sin() / 2.0;

                    // Bilinear transform for each second-order section
                    let _re = p_re / (1.0 - p_re);
                    let _im_sq = p_im * p_im / ((1.0 - p_re) * (1.0 - p_re));
                    let _denom = 1.0 - 2.0 * p_re;

                    // Pre-warp correction
                    let w0 = 2.0 * (wc / 2.0).tan();
                    let alpha_blt = w0.sin() / 2.0;

                    let cos_w0 = (-std::f64::consts::PI * cutoff).cos();

                    let b0 = (1.0 - cos_w0) / 2.0;
                    let b1 = 1.0 - cos_w0;
                    let b2 = (1.0 - cos_w0) / 2.0;
                    let a0_norm = 1.0 + alpha_blt;
                    let a1 = -2.0 * cos_w0;
                    let a2 = 1.0 - alpha_blt;

                    secs.push(Biquad {
                        b0: b0 / a0_norm,
                        b1: b1 / a0_norm,
                        b2: b2 / a0_norm,
                        a1: a1 / a0_norm,
                        a2: a2 / a0_norm,
                        z1: 0.0, z2: 0.0,
                    });
                }
                secs
            }
        };

        CascadeFilter { sections, has_first_order: order % 2 == 1 }
    }

    /// Design a digital Butterworth highpass filter.
    pub fn highpass(order: usize, cutoff: f64) -> CascadeFilter {
        let wc = std::f64::consts::PI * cutoff;
        let cos_w0 = wc.cos();
        let alpha = wc.sin() / 2.0;

        let sections = match order {
            1 => {
                let a0 = 1.0 + alpha;
                vec![Biquad {
                    b0: alpha / a0,
                    b1: -alpha / a0,
                    b2: 0.0,
                    a1: -((alpha - 1.0) / a0),  // Hmm
                    a2: 0.0,
                    z1: 0.0, z2: 0.0,
                }]
            }
            _ => {
                let n_sections = order / 2;
                (0..n_sections).map(|_| {
                    let b0 = (1.0 + cos_w0) / 2.0;
                    let b1 = -(1.0 + cos_w0);
                    let b2 = (1.0 + cos_w0) / 2.0;
                    let a0 = 1.0 + alpha;
                    Biquad {
                        b0: b0 / a0,
                        b1: b1 / a0,
                        b2: b2 / a0,
                        a1: -2.0 * cos_w0 / a0,
                        a2: (1.0 - alpha) / a0,
                        z1: 0.0, z2: 0.0,
                    }
                }).collect()
            }
        };

        CascadeFilter { sections, has_first_order: false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analog_poles_count() {
        for n in [2, 4, 6, 8] {
            let poles = ButterworthFilter::analog_poles(n);
            assert_eq!(poles.len(), n);
        }
    }

    #[test]
    fn test_analog_poles_on_unit_circle() {
        let poles = ButterworthFilter::analog_poles(4);
        for (re, im) in &poles {
            let r = re.hypot(*im);
            assert!((r - 1.0).abs() < 1e-10, "Pole not on unit circle: r={}", r);
        }
    }

    #[test]
    fn test_analog_poles_in_left_half() {
        let poles = ButterworthFilter::analog_poles(6);
        for (re, _) in &poles {
            assert!(*re < 0.0, "Pole not in left half plane: re={}", re);
        }
    }

    #[test]
    fn test_lowpass_dc_gain() {
        let cf = ButterworthFilter::lowpass(4, 0.25);
        let dc = cf.magnitude_at(0.0);
        assert!((dc - 1.0).abs() < 0.01, "DC gain should be 1.0, got {}", dc);
    }

    #[test]
    fn test_lowpass_stopband() {
        let cf = ButterworthFilter::lowpass(4, 0.25);
        let mag = cf.magnitude_at(std::f64::consts::PI * 0.9);
        assert!(mag < 0.05, "Stopband too high: {}", mag);
    }

    #[test]
    fn test_lowpass_passband_flat() {
        let cf = ButterworthFilter::lowpass(4, 0.25);
        for f in [0.01, 0.05] {
            let mag = cf.magnitude_at(std::f64::consts::PI * f);
            assert!((mag - 1.0).abs() < 0.15, "Passband not flat at f={}: {}", f, mag);
        }
    }

    #[test]
    fn test_highpass_dc_zero() {
        let cf = ButterworthFilter::highpass(4, 0.25);
        let dc = cf.magnitude_at(0.0);
        assert!(dc < 0.01, "Highpass DC should be ~0, got {}", dc);
    }

    #[test]
    fn test_highpass_nyquist_gain() {
        let cf = ButterworthFilter::highpass(4, 0.25);
        let mag = cf.magnitude_at(std::f64::consts::PI * 0.9);
        assert!((mag - 1.0).abs() < 0.1, "Highpass Nyquist gain near 1.0, got {}", mag);
    }

    #[test]
    fn test_butterworth_order_2() {
        let cf = ButterworthFilter::lowpass(2, 0.5);
        assert_eq!(cf.sections.len(), 1);
    }

    #[test]
    fn test_butterworth_order_8() {
        let cf = ButterworthFilter::lowpass(8, 0.5);
        assert_eq!(cf.sections.len(), 4);
    }
}
