mod model;
use crate::model::ConvNet;

mod utils;

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};

fn main() -> candle_core::Result<()> {
    let dev = candle_core::Device::cuda_if_available(0).unwrap();

    let var_map = VarMap::new();

    let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &dev);

    let model = ConvNet::new(var_builder).unwrap();

    let training_args = model::TrainingArgs {
        epochs: 15,
        learning_rate: 0.0005,
        load: None,
        save: None,
    };

    println!("Training model...");
    let model_train_start_time = std::time::Instant::now();
    model.train(&utils::get_mnist_dataset().unwrap(), &training_args)?;
    println!(
        "Model training took: {} seconds",
        model_train_start_time.elapsed().as_secs()
    );

    let dataset = utils::get_mnist_dataset().unwrap();

    println!("Testing model...");
    let model_test_start_time = std::time::Instant::now();
    model.test(&dev, &dataset, 64)?;
    println!(
        "Model testing took: {} seconds",
        model_test_start_time.elapsed().as_secs()
    );

    let image_0_tensor =
        utils::image_to_formatted_tensor("../../personal_test_set/0.png", &dev).unwrap();

    let image_1_tensor =
        utils::image_to_formatted_tensor("../../personal_test_set/1.png", &dev).unwrap();

    println!("Predicting...");
    let model_predict_start_time = std::time::Instant::now();
    let image_0_guess = model.predict(&image_0_tensor, &dev).unwrap();
    let image_1_guess = model.predict(&image_1_tensor, &dev).unwrap();
    println!("Predicted: {} - Correct: 0", image_0_guess);
    println!("Predicted: {} - Correct: 1", image_1_guess);
    println!(
        "Model prediction took: {} seconds",
        model_predict_start_time.elapsed().as_secs()
    );

    Ok(())
}
