mod mlp;
use itertools::iproduct;
use mlp::Mlp;

use num::complex::{c32, Complex32};
mod mandelbrot;
use mandelbrot::{escape_time, MAX_ITERATIONS};

use core::f32;
use image::ImageBuffer;
use rand::prelude::*;
use std::io::Cursor;

fn render_image(
    filename: &str,
    center: Complex32,
    zoom: f32,
    width: u32,
    height: u32,
    fun: impl Fn(Complex32) -> f32,
) {
    let mut values = Vec::new();
    let zoom = zoom * (width + height) as f32 / 8.0;
    let halfwidth = width as f32 / 2.0;
    let halfheight = height as f32 / 2.0;
    let mut minfun = f32::MAX;
    let mut maxfun = f32::MIN;
    for x in 0..width {
        for y in 0..height {
            let c = center + c32(x as f32 - halfwidth, y as f32 - halfheight) / zoom;
            let v = fun(c);
            minfun = minfun.min(v);
            maxfun = maxfun.max(v);
            values.push((x, y, v));
        }
    }

    let mut img = ImageBuffer::new(width, height);
    for (x, y, v) in values {
        // Normalize to {0,...,255}
        let n = (255.0 * (v - minfun) / (maxfun - minfun)).floor() as u8;
        img.put_pixel(x, y, image::Rgb([n, n, n]));
    }
    img.save(filename).unwrap();
}

/// Given a center and zoom, compute the topleft and bottomright corners of the image,
/// as used in [`render_image`].
fn image_corners(center: Complex32, zoom: f32, width: u32, height: u32) -> (Complex32, Complex32) {
    let zoom = zoom * (width + height) as f32 / 8.0;
    let halfwidth = width as f32 / 2.0;
    let halfheight = height as f32 / 2.0;
    let offset = c32(halfwidth, halfheight) / zoom;
    (center - offset, center + offset)
}

fn main() {
    let width = 128;
    let height = 128;
    let center = c32(0.4, 0.2);
    let zoom = 20.0;
    let (topleft, bottomright) = image_corners(center, zoom, width, height);

    let f = escape_time;
    let f = |c: Complex32| c.re * c.re + ((c.im * 100.0).sin() + 1.0) / 30.0;
    render_image("mandelbrot.png", center, zoom, width, height, f);

    let mut mlp: Mlp<2, 1> = Mlp::new(&[32, 32, 32]);
    let mut dataset: Vec<([f32; 2], [f32; 1])> = Vec::new();
    let halfwidth = width as f32 / 2.0;
    let halfheight = height as f32 / 2.0;
    for x in 0..width {
        for y in 0..height {
            let zoom = zoom * (width + height) as f32 / 8.0;
            let c = center + c32(x as f32 - halfwidth, y as f32 - halfheight) / zoom;
            let v = f(c);
            dataset.push(([c.re, c.im], [v]));
        }
    }

    println!("Training the MLP...");
    for epoch in 0..4000 {
        mlp.train(&dataset, 0.001);
        if epoch % 10 == 0 {
            let loss = dataset
                .iter()
                .map(|(i, [v])| (mlp.forward(*i)[0] - v).powi(2))
                .sum::<f32>();
            println!("Epoch {epoch}: loss = {loss}");
        }
    }
    let mlp_function = |c: Complex32| mlp.forward([c.re, c.im])[0];
    render_image(
        &format!("mandelbrot-mlp.png"),
        center,
        zoom,
        width,
        height,
        &mlp_function,
    );
}
