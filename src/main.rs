// Add these dependencies to your Cargo.toml
// prettytable = "0.11"
// colored = "2.0"

use crate::model::ConvNet;
use candle_core;
use candle_nn;
use colored::*;
use prettytable::{Cell, Row, Table};
use std::path::Path;

// Import your utils module
mod model;
mod utils;

fn main() -> candle_core::Result<()> {
    // Initialize device
    let dev = candle_core::Device::cuda_if_available(0).expect("Failed to get any device");

    // Initialize VarMap
    let mut vm = candle_nn::VarMap::new();

    // Load model from existing file
    let model = ConvNet::new_from_file(&mut vm, "model.safetensors")?;

    // Load dataset
    println!("");
    println!("Loading MNIST dataset...");
    let time_start_loading_dataset = std::time::Instant::now();
    let dataset = utils::get_mnist_dataset().unwrap();
    println!(
        "Loaded MNIST dataset with {:?} training samples - took {:.2} seconds\n",
        dataset.train_images.dims(),
        time_start_loading_dataset.elapsed().as_secs_f64()
    );

    // Test the model
    println!("Testing the model on the MNIST test dataset...");
    model.test(&dev, &dataset, 64)?;
    println!("");

    // Define test images and their correct labels
    let test_images = vec![
        ("0.png", 0),
        ("1.png", 1),
        ("2.png", 2),
        ("3.png", 3),
        ("4.png", 4),
        ("5.png", 5),
        ("6.png", 6),
        ("7.png", 7),
        ("8.png", 8),
        ("9.png", 9),
    ];

    // Initialize table
    let mut table = Table::new();
    table.add_row(Row::new(vec![
        Cell::new(&"Image"),
        Cell::new(&"Prediction"),
        Cell::new(&"Correct"),
        Cell::new(&"Result"),
    ]));

    // Iterate over test images and make predictions
    println!("Making predictions on personal test set...\n");
    std::thread::sleep(std::time::Duration::from_secs(1));
    let model_predict_start_time = std::time::Instant::now();

    for (filename, correct_label) in test_images {
        let image_path = format!("../../personal_test_set/{}", filename);
        let image_tensor = utils::image_to_formatted_tensor(&image_path, &dev).unwrap();
        let prediction = model.predict(&image_tensor, &dev).unwrap();

        // Determine if the prediction is correct
        let result = if prediction == correct_label {
            "Correct".green()
        } else {
            "Incorrect".red()
        };

        // Add row to the table
        table.add_row(Row::new(vec![
            Cell::new(
                &Path::new(&image_path)
                    .file_name()
                    .unwrap()
                    .to_string_lossy(),
            ),
            Cell::new(&prediction.to_string()),
            Cell::new(&correct_label.to_string()),
            Cell::new(&result.to_string()),
        ]));
    }

    // Print the table
    table.printstd();

    println!(
        "\nModel predictions completed in {:.2} seconds.",
        model_predict_start_time.elapsed().as_secs_f64()
    );
    println!("");

    Ok(())
}
