use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

struct Layer {
    weights: Array2<f32>,
    biases: Array1<f32>,
}
pub struct Mlp<const INPUT: usize, const OUTPUT: usize> {
    layers: Vec<Layer>, // Includes output-layer but not input-layer
}

fn activation(x: f32) -> f32 {
    x.max(0.0)
}
fn activation_derivative(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

impl<const INPUT: usize, const OUTPUT: usize> Mlp<INPUT, OUTPUT> {
    // layer_shape only includes hidden layers
    pub fn new(layer_shape: &[usize]) -> Self {
        let mut layers = Vec::new();
        let mut input_size = INPUT;
        for &size in layer_shape {
            let weights = Array2::random((size, input_size), Uniform::new(-1.0, 1.0));
            let biases = Array1::zeros(size);
            layers.push(Layer { weights, biases });
            input_size = size;
        }
        let output_weights = Array2::random((OUTPUT, input_size), Uniform::new(-1.0, 1.0));
        let output_biases = Array1::zeros(OUTPUT);
        layers.push(Layer {
            weights: output_weights,
            biases: output_biases,
        });

        Mlp { layers }
    }

    pub fn forward(&self, input: [f32; INPUT]) -> [f32; OUTPUT] {
        let mut x = Array1::from(input.to_vec());
        for layer in &self.layers {
            x = (layer.weights.dot(&x) + &layer.biases).map(|x| activation(*x));
        }
        x.to_vec().try_into().unwrap()
    }

    pub fn train(
        &mut self,
        inputs_and_targets: &[([f32; INPUT], [f32; OUTPUT])],
        learning_rate: f32,
    ) {
        for &(input, target) in inputs_and_targets {
            let output = Array1::from(self.forward(input).to_vec());
            let target = Array1::from(target.to_vec());
            let error = target - output;

            for layer in self.layers.iter_mut().rev() {
                todo!()
            }
        }
    }
}
