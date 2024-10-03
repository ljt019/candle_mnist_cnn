use candle_core::Tensor;

pub fn image_to_formatted_tensor(
    path: &str,
    device: &candle_core::Device,
) -> candle_core::Result<Tensor> {
    let image = image::open(path).expect("Failed to open image");
    let image = image.resize_exact(28, 28, image::imageops::FilterType::Nearest);
    let image = image.to_luma8().into_raw();
    let image = image
        .into_iter()
        .map(|p: u8| p as f32 / 255.0)
        .collect::<Vec<f32>>();
    let image = Tensor::from_vec(image, &[784], device)?;

    let image = image.reshape((1, 784))?;

    Ok(image)
}

pub fn save_tensor_as_image(tensor: &Tensor, path: &str) -> candle_core::Result<()> {
    // Check if the tensor has shape [1, 784] and squeeze it to [784]
    let squeezed_tensor = if tensor.rank() == 2 && tensor.shape().dims() == [1, 784] {
        tensor.squeeze(0)?
    } else {
        tensor.clone()
    };

    // Convert the squeezed tensor to a vector
    let vec = squeezed_tensor.to_vec1::<f32>()?;

    // Scale pixel values back to [0, 255] and convert to u8
    let img_data: Vec<u8> = vec.iter().map(|&v| (v * 255.0) as u8).collect();

    // Create an ImageBuffer from the vector
    let img = image::ImageBuffer::from_vec(28, 28, img_data).unwrap();

    // Save the image
    image::DynamicImage::ImageLuma8(img).save(path).unwrap();

    Ok(())
}

pub fn get_mnist_dataset() -> candle_core::Result<candle_datasets::vision::Dataset> {
    let dataset = candle_datasets::vision::mnist::load();

    return dataset;
}
