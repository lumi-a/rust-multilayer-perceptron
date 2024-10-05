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

fn activation(xs: &Array1<f32>) -> Array1<f32> {
    xs.map(|x| x.max(0.0))
}
fn activation_derivative(xs: &Array1<f32>) -> Array1<f32> {
    xs.map(|x| if *x > 0.0 { 1.0 } else { 0.0 })
}

// SIGH https://github.com/rust-ndarray/ndarray/issues/1148
fn outer(x: &Array<f32, Ix1>, y: &Array<f32, Ix1>) -> Array<f32, Ix2> {
    let (size_x, size_y) = (x.shape()[0], y.shape()[0]);
    let x_reshaped = x.view().into_shape_with_order((size_x, 1)).unwrap();
    let y_reshaped = y.view().into_shape_with_order((1, size_y)).unwrap();
    x_reshaped.dot(&y_reshaped)
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
        let output_biases = Array1::random(OUTPUT, Uniform::new(0.0, 0.01));
        layers.push(Layer {
            weights: output_weights,
            biases: output_biases,
        });

        Mlp { layers }
    }

    pub fn forward(&self, input: [f32; INPUT]) -> [f32; OUTPUT] {
        let mut x = Array1::from(input.to_vec());
        for layer in &self.layers {
            let unactivated = layer.weights.dot(&x) + &layer.biases;
            x = activation(&unactivated);
        }
        x.to_vec().try_into().unwrap()
    }

    /// Perform backpropagation, computing gradients for weights and biases
    fn backpropagation(
        &self,
        input: &Array1<f32>,
        target: &Array1<f32>,
        learning_rate: f32,
    ) -> Vec<Layer> {
        // Manually do a forward-pass
        let (activations, pre_activations) = {
            let mut activations = Vec::with_capacity(self.layers.len());
            let mut pre_activations = Vec::with_capacity(self.layers.len());
            activations.push(input.clone());
            pre_activations.push(input.clone());
            let mut current_activation = input.clone(); // Start with input

            for layer in &self.layers {
                let z = layer.weights.dot(&current_activation) + &layer.biases;
                let a = activation(&z);
                pre_activations.push(z);
                activations.push(a.clone());
                current_activation = a;
            }
            (activations, pre_activations)
        };

        let mut delta_reverse_layers = Vec::with_capacity(self.layers.len());

        // Output-layer:
        let mut delta = (activations.last().unwrap() - target)
            * activation_derivative(pre_activations.last().unwrap());

        // Backpropagate through layers in reverse order
        for (l, layer) in self.layers.iter().enumerate().rev() {
            let prev_activation = activations[l].clone();
            let weight_gradient = outer(&delta, &prev_activation);
            let bias_gradient = delta.clone();

            delta_reverse_layers.push(Layer {
                weights: -learning_rate * weight_gradient,
                biases: -learning_rate * bias_gradient,
            });

            // Prepare previous layer
            if l > 0 {
                let weights_transposed = layer.weights.t();
                delta = weights_transposed.dot(&delta);
                delta = delta * activation_derivative(&pre_activations[l]);
            }
        }

        delta_reverse_layers.reverse();
        delta_reverse_layers
    }

    pub fn train(
        &mut self,
        inputs_and_targets: &[([f32; INPUT], [f32; OUTPUT])],
        learning_rate: f32,
    ) {
        // Don't train on an empty set, silly
        if inputs_and_targets.is_empty() {
            return;
        }
        let adjusted_learning_rate = learning_rate / inputs_and_targets.len() as f32;
        let mut delta_mlps = Vec::new();
        for &(input, target) in inputs_and_targets {
            let target = Array1::from(target.to_vec());
            let input = Array1::from(input.to_vec());
            let delta_layers = self.backpropagation(&input, &target, adjusted_learning_rate);
            delta_mlps.push(delta_layers);
        }
        // Only after all the updates are collected, do a batch-update
        for delta_layers in delta_mlps {
            for (layer, delta_layer) in self.layers.iter_mut().zip(delta_layers) {
                layer.weights = layer.weights.clone() + delta_layer.weights;
                layer.biases = layer.biases.clone() + delta_layer.biases;
            }
        }
    }
}
