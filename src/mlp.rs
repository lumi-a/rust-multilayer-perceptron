use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

/// A layer of neurons of an MLP
#[derive(Debug)]
struct Layer {
    /// The weights of this layers' neurons,
    /// each column corresponding to one neuron of the previous layer
    /// and each row corresponding to one neuron of this layer, so
    /// that:
    /// ```
    /// new_activations = self.weights.dot(previous_activations) + self.biases
    /// ```
    weights: Array2<f32>,
    /// The biases of this layers' neurons
    biases: Array1<f32>,
}

/// A multilayer-perceptron (MLP)
#[derive(Debug)]
pub struct Mlp<const INPUT: usize, const OUTPUT: usize> {
    /// The layers of the MLP, including the output-layer
    /// but not the input-layer.
    layers: Vec<Layer>,
}

/// The activation function of all MLPs
fn activation(xs: &Array1<f32>) -> Array1<f32> {
    // Leaky ReLU
    // xs.map(|x| x.max(x * 0.1))

    // tanh
    xs.map(|x| x.tanh())
}
/// The derivative of [`activation`].
fn activation_derivative(xs: &Array1<f32>) -> Array1<f32> {
    // Leaky ReLU
    // xs.map(|x| if *x > 0.0 { 1.0 } else { 0.1 })

    // tanh
    xs.map(|x| 1.0 / (x.cosh().powi(2)))
}

/// An outer product of two vectors.
/// Stolen from <https://github.com/rust-ndarray/ndarray/issues/1148>
fn outer(x: &Array<f32, Ix1>, y: &Array<f32, Ix1>) -> Array<f32, Ix2> {
    let (size_x, size_y) = (x.shape()[0], y.shape()[0]);
    let x_reshaped = x.view().into_shape_with_order((size_x, 1)).unwrap();
    let y_reshaped = y.view().into_shape_with_order((1, size_y)).unwrap();
    x_reshaped.dot(&y_reshaped)
}

impl<const INPUT: usize, const OUTPUT: usize> Mlp<INPUT, OUTPUT> {
    /// Return a new randomly-initialized MLP.
    /// The hidden layer sizes are given by `layer_shape`.
    #[must_use]
    pub fn new(layer_shape: &[usize]) -> Self {
        let mut layers = Vec::new();
        let mut input_size = INPUT;
        for &size in layer_shape {
            let weights = Array2::random(
                (size, input_size),
                Uniform::new(0.0, (2.0 / input_size as f32).sqrt()),
            );

            let biases = Array1::random(size, Uniform::new(0.0, 0.01));
            layers.push(Layer { weights, biases });
            input_size = size;
        }
        let output_weights = Array2::random(
            (OUTPUT, input_size),
            Uniform::new(0.0, (2.0 / input_size as f32).sqrt()),
        );
        let output_biases = Array1::random(OUTPUT, Uniform::new(0.0, 0.01));
        layers.push(Layer {
            weights: output_weights,
            biases: output_biases,
        });

        Self { layers }
    }

    /// Compute the output of the MLP for the given input
    #[must_use]
    pub fn forward(&self, input: [f32; INPUT]) -> [f32; OUTPUT] {
        let mut x = Array1::from(input.to_vec());
        for (i, layer) in self.layers.iter().enumerate() {
            let unactivated = layer.weights.dot(&x) + &layer.biases;
            if i == self.layers.len() - 1 {
                x = unactivated;
            } else {
                x = activation(&unactivated);
            }
        }
        x.to_vec().try_into().unwrap()
    }

    /// Perform backpropagation, computing gradients for weights and biases
    fn backpropagation(&self, input: &Array1<f32>, target: &Array1<f32>) -> Vec<Layer> {
        // Forward pass (same as before)
        let (prev_activations, unactivations) = {
            let mut prev_activations = Vec::with_capacity(self.layers.len() + 1);
            let mut unactivations = Vec::with_capacity(self.layers.len() + 1);
            prev_activations.push(input.clone());
            let mut current_activation = input.clone();

            for (i, layer) in self.layers.iter().enumerate() {
                let z = layer.weights.dot(&current_activation) + &layer.biases;
                unactivations.push(z.clone());

                if i == self.layers.len() - 1 {
                    current_activation = z;
                } else {
                    current_activation = activation(&z);
                }
                prev_activations.push(current_activation.clone());
            }
            (prev_activations, unactivations)
        };

        let mut delta_reverse_layers = Vec::with_capacity(self.layers.len());

        // W = self.layers[l].weights
        // b = self.layers[l].biases
        // x = prev_activations[l]
        // C = 1/2 * ‖Wx+b - target‖²
        // ∂C/∂W =   [Wx+b - target] * x
        // ∂C/∂b =   [Wx+b - target]
        // delta =:  ^^^^^^^^^^^^^^^
        let mut delta = prev_activations.last().unwrap() - target;
        // Note that  Wx+b == unactivations[l] == prev_activations[l+1]
        // because we don't have an activation-function on the final layer.
        // let l = self.layers.len() - 1;
        // assert_eq!(prev_activations[l + 1], unactivations[l]);

        for l in (0..self.layers.len()).rev() {
            // W  = self.layers[l].weights
            // b  = self.layers[l].biases
            // x  = prev_activations[l]
            // W" = self.layers[l-1].weights
            // b" = self.layers[l-1].biases
            // x" = prev_activations[l-1]
            // C = 1/2 * ‖Wσ(W"x"+b")+b - target‖²
            // ∂C/∂W" = delta * W * ∂x/∂W" = delta * W * σ'(W"x"+b") * x"
            // ∂C/∂b" = delta * W * ∂x/∂b" = delta * W * σ'(W"x"+b")
            // delta <- delta * W * σ'(W"x"+b")
            // W"x"+b" == unactivations[l-1]

            let weight_gradient = outer(&delta, &prev_activations[l]);
            let bias_gradient = delta.clone();

            delta_reverse_layers.push(Layer {
                weights: weight_gradient,
                biases: bias_gradient,
            });

            if l > 0 {
                delta = self.layers[l].weights.t().dot(&delta);
                delta = delta * activation_derivative(&unactivations[l - 1]);
            }
        }

        delta_reverse_layers.reverse();
        delta_reverse_layers
    }

    /// Train the MLP on the given inputs and targets
    pub fn train(
        &mut self,
        inputs_and_targets: &[([f32; INPUT], [f32; OUTPUT])],
        learning_rate: f32,
    ) {
        // Don't train on an empty set, silly
        if inputs_and_targets.is_empty() {
            return;
        }

        // Initialize accumulators for gradients of weights and biases to 0
        let mut accumulated_deltas = self
            .layers
            .iter()
            .map(|layer| Layer {
                weights: Array2::zeros(layer.weights.dim()),
                biases: Array1::zeros(layer.biases.len()),
            })
            .collect::<Vec<_>>();

        // For each input-target pair, perform backpropagation and accumulate the gradients
        for &(input, target) in inputs_and_targets {
            let target = Array1::from(target.to_vec());
            let input = Array1::from(input.to_vec());

            let delta_layers = self.backpropagation(&input, &target);

            // Accumulate the gradients for weights and biases
            for (accum_layer, delta_layer) in accumulated_deltas.iter_mut().zip(delta_layers) {
                accum_layer.weights = &accum_layer.weights + &delta_layer.weights;
                accum_layer.biases = &accum_layer.biases + &delta_layer.biases;
            }
        }

        // Now update the weights and biases by applying the accumulated (and averaged) gradients
        let batch_size = inputs_and_targets.len() as f32;
        for (layer, accum_layer) in self.layers.iter_mut().zip(accumulated_deltas) {
            layer.weights = &layer.weights - learning_rate / batch_size * &accum_layer.weights;
            layer.biases = &layer.biases - learning_rate / batch_size * &accum_layer.biases;
        }
    }
}
