use num::complex::Complex;

const MAXIMUM_ESCAPETIME: usize = 1000;

/// Computes the escape time of a point in the Mandelbrot set,
/// up to a limit of [`MAXIMUM_ESCAPETIME`].
pub fn escape_time(c: Complex<f32>) -> usize {
    let mut z = Complex { re: 0.0, im: 0.0 };
    for i in 0..MAXIMUM_ESCAPETIME {
        z = z * z + c;
        if z.norm_sqr() > 4.0 {
            return i;
        }
    }
    MAXIMUM_ESCAPETIME
}
