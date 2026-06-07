//! Cascade filter (series of biquad sections).

use crate::biquad::Biquad;

/// A cascade of biquad sections for numerically stable IIR filtering.
#[derive(Debug, Clone)]
pub struct CascadeFilter {
    /// Biquad sections (second-order stages)
    pub sections: Vec<Biquad>,
    /// Whether there's an additional first-order section
    pub has_first_order: bool,
}

impl CascadeFilter {
    /// Create a new cascade filter from biquad sections.
    pub fn new(sections: Vec<Biquad>) -> Self {
        Self { sections, has_first_order: false }
    }

    /// Apply the cascade filter to a signal.
    pub fn filter(&self, signal: &[f64]) -> Vec<f64> {
        let mut data = signal.to_vec();
        for bq in &self.sections {
            let mut section = *bq;
            data = section.filter(&data);
        }
        data
    }

    /// Compute the magnitude response at a normalized frequency omega (0 to π).
    pub fn magnitude_at(&self, omega: f64) -> f64 {
        let mut mag = 1.0;
        for section in &self.sections {
            mag *= section.magnitude(omega);
        }
        mag
    }

    /// Compute the complex frequency response at omega.
    pub fn response_at(&self, omega: f64) -> (f64, f64) {
        let mut h_re = 1.0;
        let mut h_im = 0.0;
        for section in &self.sections {
            let (s_re, s_im) = section.response(omega);
            let new_re = h_re * s_re - h_im * s_im;
            let new_im = h_re * s_im + h_im * s_re;
            h_re = new_re;
            h_im = new_im;
        }
        (h_re, h_im)
    }

    /// Check stability of all sections.
    pub fn is_stable(&self) -> bool {
        self.sections.iter().all(|s| s.is_stable())
    }

    /// Get the number of sections.
    pub fn num_sections(&self) -> usize {
        self.sections.len()
    }

    /// Compute the effective filter order.
    pub fn order(&self) -> usize {
        let base = self.sections.len() * 2;
        if self.has_first_order { base + 1 } else { base }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cascade_identity() {
        let cf = CascadeFilter::new(vec![Biquad::new(1.0, 0.0, 0.0, 0.0, 0.0)]);
        let signal = [1.0, 2.0, 3.0];
        let output = cf.filter(&signal);
        for i in 0..signal.len() {
            assert!((output[i] - signal[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn test_cascade_dc_gain() {
        let bq = Biquad::new(0.5, 0.5, 0.0, 0.0, 0.0);
        let cf = CascadeFilter::new(vec![bq]);
        let dc = cf.magnitude_at(0.0);
        assert!((dc - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_cascade_stability() {
        let bq1 = Biquad::new(1.0, 0.0, 0.0, 0.3, 0.1);
        let bq2 = Biquad::new(1.0, 0.0, 0.0, 0.2, 0.05);
        let cf = CascadeFilter::new(vec![bq1, bq2]);
        assert!(cf.is_stable());
    }

    #[test]
    fn test_cascade_order() {
        let cf = CascadeFilter::new(vec![Biquad::new(1.0, 0.0, 0.0, 0.0, 0.0); 3]);
        assert_eq!(cf.order(), 6);
    }

    #[test]
    fn test_cascade_num_sections() {
        let cf = CascadeFilter::new(vec![Biquad::new(1.0, 0.0, 0.0, 0.0, 0.0); 4]);
        assert_eq!(cf.num_sections(), 4);
    }

    #[test]
    fn test_cascade_magnitude_product() {
        let bq1 = Biquad::new(1.0, 0.5, 0.0, -0.3, 0.0);
        let bq2 = Biquad::new(1.0, 0.0, 0.0, -0.2, 0.0);
        let cf = CascadeFilter::new(vec![bq1, bq2]);
        let omega = std::f64::consts::PI * 0.25;
        let expected = bq1.magnitude(omega) * bq2.magnitude(omega);
        let actual = cf.magnitude_at(omega);
        assert!((actual - expected).abs() < 1e-12);
    }
}
