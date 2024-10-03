mod mlp;
use mlp::Mlp;

mod mandelbrot;
use mandelbrot::escape_time;

fn main() {
    let mlp = Mlp::<16, 16>::new(&[3, 7, 15]);
    println!(
        "{:?}",
        mlp.forward([
            1.0, 2.0, 7.0, 5.0, 1.0, 2.0, 7.0, 5.0, 1.0, 2.0, 7.0, 5.0, 1.0, 2.0, 7.0, 5.0
        ])
    )
}
