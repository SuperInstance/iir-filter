# iir-filter

Infinite impulse response (IIR) filter design in pure Rust.

## Features

- **Butterworth** — Maximally flat magnitude response
- **Chebyshev Type I** — Equiripple in passband
- **Chebyshev Type II** — Equiripple in stopband
- **Biquad sections** — Second-order stages for numerical stability
- **Cascade form** — Chain biquad sections for higher-order filters
- **Frequency response** — Complex response, magnitude, phase, group delay

## Modules

| Module | Description |
|--------|-------------|
| `butterworth` | Butterworth filter design |
| `chebyshev` | Chebyshev Type I and II filters |
| `biquad` | Second-order IIR sections |
| `cascade` | Cascade (SOS) filter form |
| `response` | Frequency response analysis |

## Quick Start

```rust
use iir_filter::{ButterworthFilter, CascadeFilter};

// Design a 4th-order Butterworth lowpass
let filter = ButterworthFilter::lowpass(4, 0.25);
let signal = vec![1.0, 0.5, -0.3, 0.8, 0.2];
let filtered = filter.filter(&signal);
```

## License

MIT OR Apache-2.0
