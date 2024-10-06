//! Unoptimized MLPs in Rust.

/// MLPs will be structs with private fields,
/// mostly interacted with via [`mlp::Mlp::new`],
/// [`mlp::Mlp::forward`], and [`mlp::Mlp::train`].
pub mod mlp;
use mlp::Mlp;

fn main() {
    let mut mlp = Mlp::<1, 1>::new(&[16, 16, 16]);

    // Generate training data
    let num_samples = 100u32;
    let training_data: Vec<([f32; 1], [f32; 1])> = (0..num_samples)
        .map(|i| {
            let x = i as f32 / num_samples as f32;
            let y = ((x * 10.0).sin() + 1.0) / 2.0;
            ([x], [y])
        })
        .collect();

    // Training loop
    let num_epochs = 10000u32;
    let learning_rate = 0.2;
    for epoch in 0..num_epochs {
        mlp.train(&training_data, learning_rate);

        if epoch % 500 == 0 {
            let loss: f32 = training_data
                .iter()
                .map(|&(input, target)| {
                    let output = mlp.forward(input)[0];
                    (output - target[0]).powi(2)
                })
                .sum::<f32>()
                / num_samples as f32;
            println!("Epoch {epoch}: Loss = {loss}");
        }
    }

    // Test the trained network
    println!("\nTesting the trained network:");
    for &x in &training_data {
        let predicted = mlp.forward(x.0)[0];
        // let actual = x.1[0];
        print!("{}", " ".repeat((predicted * 100.0).floor() as usize));
        println!("{predicted:.2}");
    }
}
