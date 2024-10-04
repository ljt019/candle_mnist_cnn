mod model;
use crate::model::ConvNet;

mod utils;

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};

fn main() -> candle_core::Result<()> {
    let dev = candle_core::Device::cuda_if_available(0).expect("Failed to get any device");

    let mut var_map = VarMap::new();

    let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &dev);

    let model = ConvNet::new(var_builder).expect("Failed to create model");

    let training_args = model::TrainingArgs {
        epochs: 5,
        learning_rate: 0.002,
        load: None,
        save: Some("model.safetensors".to_string()),
    };

    println!("Training model...");
    let model_train_start_time = std::time::Instant::now();
    model.train(
        &utils::get_mnist_dataset().unwrap(),
        &training_args,
        &mut var_map,
    )?;
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

    let image_2_tensor =
        utils::image_to_formatted_tensor("../../personal_test_set/2.png", &dev).unwrap();

    let image_3_tensor =
        utils::image_to_formatted_tensor("../../personal_test_set/3.png", &dev).unwrap();

    let image_4_tensor =
        utils::image_to_formatted_tensor("../../personal_test_set/4.png", &dev).unwrap();

    let image_5_tensor =
        utils::image_to_formatted_tensor("../../personal_test_set/5.png", &dev).unwrap();

    let image_6_tensor =
        utils::image_to_formatted_tensor("../../personal_test_set/6.png", &dev).unwrap();

    let image_7_tensor =
        utils::image_to_formatted_tensor("../../personal_test_set/7.png", &dev).unwrap();

    let image_8_tensor =
        utils::image_to_formatted_tensor("../../personal_test_set/8.png", &dev).unwrap();

    let image_9_tensor =
        utils::image_to_formatted_tensor("../../personal_test_set/9.png", &dev).unwrap();

    let self_drawn_3_tensor = utils::image_to_formatted_tensor("../../3.png", &dev).unwrap();

    println!("Predicting...");
    let model_predict_start_time = std::time::Instant::now();
    let image_0_guess = model.predict(&image_0_tensor, &dev).unwrap();
    let image_1_guess = model.predict(&image_1_tensor, &dev).unwrap();
    let image_2_guess = model.predict(&image_2_tensor, &dev).unwrap();
    let image_3_guess = model.predict(&image_3_tensor, &dev).unwrap();
    let image_4_guess = model.predict(&image_4_tensor, &dev).unwrap();
    let image_5_guess = model.predict(&image_5_tensor, &dev).unwrap();
    let image_6_guess = model.predict(&image_6_tensor, &dev).unwrap();
    let image_7_guess = model.predict(&image_7_tensor, &dev).unwrap();
    let image_8_guess = model.predict(&image_8_tensor, &dev).unwrap();
    let image_9_guess = model.predict(&image_9_tensor, &dev).unwrap();
    let self_drawn_3_guess = model.predict(&self_drawn_3_tensor, &dev).unwrap();
    println!("Predicted: {} - Correct: 0", image_0_guess);
    println!("Predicted: {} - Correct: 1", image_1_guess);
    println!("Predicted: {} - Correct: 2", image_2_guess);
    println!("Predicted: {} - Correct: 3", image_3_guess);
    println!("Predicted: {} - Correct: 4", image_4_guess);
    println!("Predicted: {} - Correct: 5", image_5_guess);
    println!("Predicted: {} - Correct: 6", image_6_guess);
    println!("Predicted: {} - Correct: 7", image_7_guess);
    println!("Predicted: {} - Correct: 8", image_8_guess);
    println!("Predicted: {} - Correct: 9", image_9_guess);
    println!("Predicted: {} - Correct: 3", self_drawn_3_guess);
    println!(
        "Model prediction took: {} seconds",
        model_predict_start_time.elapsed().as_secs()
    );

    Ok(())
}
