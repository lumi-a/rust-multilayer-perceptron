use num::complex::Complex32;

const MAX_ITERATIONS: usize = 1000;

/// Computes the escape time of a point in the Mandelbrot set
/// with smoothing, up to a limit of [`MAXIMUM_ESCAPETIME`]
/// iterations.
pub fn escape_time(c: Complex32) -> f32 {
    let mut z = Complex32::new(0.0, 0.0);
    let mut z_norm_sqr = 0.0;
    for i in 0..MAX_ITERATIONS {
        z = z * z + c;
        z_norm_sqr = z.norm_sqr();
        if z_norm_sqr > 4.0 {
            return i as f32 - z_norm_sqr.log2();
        }
    }
    MAX_ITERATIONS as f32 - z_norm_sqr.log2()
}
