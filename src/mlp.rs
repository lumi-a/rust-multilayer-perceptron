use ndarray::prelude::*;

struct Layer {
    weights: Array2<f32>,
    biases: Array1<f32>,
}
struct MLP<const INPUT: usize, const OUTPUT: usize> {
    layers: Vec<Layer>, // Includes output layer
}

fn activation(x: f32) -> f32 {
    x.max(0.0)
}

impl<const INPUT: usize, const OUTPUT: usize> MLP<INPUT, OUTPUT> {
    pub fn forward(&self, input: [f32; INPUT]) -> [f32; OUTPUT] {
        let mut x = Array1::from(input.to_vec());
        for layer in &self.layers {
            x = (layer.weights.dot(&x) + &layer.biases).map(|x| activation(*x));
        }
        x.to_vec().try_into().unwrap()
    }
}
