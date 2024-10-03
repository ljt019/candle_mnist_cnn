use crate::utils;
use candle_core::{DType, Tensor, D};
use candle_datasets;
use candle_nn::{loss, ops, Conv2d, Linear, ModuleT, Optimizer, VarBuilder, VarMap};
use rand::prelude::*;

const LABELS: usize = 10;

#[derive(Debug)]
pub struct ConvNet {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    dropout: candle_nn::Dropout,
}

impl ConvNet {
    pub fn new(vs: VarBuilder) -> candle_core::Result<Self> {
        let conv1 = candle_nn::conv2d(1, 32, 5, Default::default(), vs.pp("c1"))?;
        let conv2 = candle_nn::conv2d(32, 64, 5, Default::default(), vs.pp("c2"))?;
        let fc1 = candle_nn::linear(1024, 1024, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(1024, LABELS, vs.pp("fc2"))?;
        let dropout = candle_nn::Dropout::new(0.5);
        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
            dropout,
        })
    }

    fn forward(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let (b_sz, _img_dim) = xs.dims2()?;
        let xs = xs
            .reshape((b_sz, 1, 28, 28))?
            .apply(&self.conv1)?
            .max_pool2d(2)?
            .apply(&self.conv2)?
            .max_pool2d(2)?
            .flatten_from(1)?
            .apply(&self.fc1)?
            .relu()?;
        self.dropout.forward_t(&xs, train)?.apply(&self.fc2)
    }

    pub fn predict(
        &self,
        image: &Tensor,
        device: &candle_core::Device,
    ) -> candle_core::Result<u32> {
        utils::save_tensor_as_image(image, "../../saved_tensors/image.png")?;
        // Forward pass through the model
        let logits = self.forward(&image.to_device(device)?, false)?;
        let label = logits.argmax(D::Minus1)?;
        let label = label.squeeze(0)?;
        let label = label.to_scalar::<u32>()?;
        Ok(label)
    }

    pub fn train(
        &self,
        m: &candle_datasets::vision::Dataset,
        args: &TrainingArgs,
    ) -> candle_core::Result<VarMap> {
        const BSIZE: usize = 64;

        let dev = candle_core::Device::cuda_if_available(0)?;

        let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
        let train_images = m.train_images.to_device(&dev)?;

        let mut varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let model = ConvNet::new(vs.clone())?;

        if let Some(load) = &args.load {
            println!("loading weights from {load}");
            varmap.load(load)?
        }

        let adamw_params = candle_nn::ParamsAdamW {
            lr: args.learning_rate,
            ..Default::default()
        };
        let mut opt = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;
        let test_images = m.test_images.to_device(&dev)?;
        let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
        let n_batches = train_images.dim(0)? / BSIZE;
        let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();
        for epoch in 1..=args.epochs {
            let mut sum_loss = 0f32;
            batch_idxs.shuffle(&mut thread_rng());
            for batch_idx in batch_idxs.iter() {
                let train_images = train_images.narrow(0, batch_idx * BSIZE, BSIZE)?;
                let train_labels = train_labels.narrow(0, batch_idx * BSIZE, BSIZE)?;
                let logits = model.forward(&train_images, true)?;
                let log_sm = ops::log_softmax(&logits, D::Minus1)?;
                let loss = loss::nll(&log_sm, &train_labels)?;
                opt.backward_step(&loss)?;
                sum_loss += loss.to_vec0::<f32>()?;
            }
            let avg_loss = sum_loss / n_batches as f32;

            let test_logits = model.forward(&test_images, false)?;
            let test_prediction = test_logits.argmax(D::Minus1)?;

            let test_prediction_vec = test_prediction.to_vec1::<u32>()?;
            let test_labels_vec = test_labels.to_vec1::<u32>()?;

            for (i, (prediction, label)) in test_prediction_vec
                .iter()
                .zip(test_labels_vec.iter())
                .enumerate()
            {
                println!("Prediction: {} - Label: {}", prediction, label);
                if i == 10 {
                    break;
                }
            }

            let sum_ok = test_prediction
                .eq(&test_labels)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;

            let test_accuracy = sum_ok / test_labels.dims1()? as f32;
            println!(
                "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
                avg_loss,
                100. * test_accuracy
            );
        }
        if let Some(save) = &args.save {
            println!("saving trained weights in {save}");
            varmap.save(save)?
        }
        Ok(varmap)
    }

    pub fn test(
        &self,
        device: &candle_core::Device,
        data: &candle_datasets::vision::Dataset,
        batch_size: usize,
    ) -> candle_core::Result<()> {
        let test_images = data.test_images.to_device(device)?;
        let test_labels = data.test_labels.to_device(device)?;
        let n_batches = test_images.dim(0)? / batch_size;
        let mut correct_predictions = 0;
        let mut total_predictions = 0;

        println!("Testing {} batches", n_batches);

        for i in 0..n_batches {
            println!("Batch {}/{}", i + 1, n_batches);

            let images = test_images.narrow(0, i * batch_size, batch_size)?;
            let labels = test_labels.narrow(0, i * batch_size, batch_size)?;

            let labels = labels.to_dtype(DType::U32)?;

            let logits = self.forward(&images, false)?;
            let predictions = logits.argmax(D::Minus1)?;

            println!("Predictions: {:?}", predictions);
            println!("Labels: {:?}", labels);

            println!("Predictions shape: {:?}", predictions.shape());
            println!("Labels shape: {:?}", labels.shape());

            // Print detailed information for each prediction
            let predictions_vec: Vec<u32> = predictions.to_vec1::<u32>()?;
            let labels_vec: Vec<u32> = labels.to_vec1::<u32>()?;
            for (i, (prediction, label)) in
                predictions_vec.iter().zip(labels_vec.iter()).enumerate()
            {
                println!("Prediction: {} - Label: {}", prediction, label);
                if i == 10 {
                    break;
                }
            }

            correct_predictions += predictions
                .eq(&labels)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()? as usize;

            total_predictions += batch_size;
        }

        let accuracy = correct_predictions as f32 / total_predictions as f32 * 100.0;
        println!("Test Accuracy: {:.2}%", accuracy);

        Ok(())
    }
}

pub struct TrainingArgs {
    pub epochs: usize,
    pub learning_rate: f64,
    pub load: Option<String>,
    pub save: Option<String>,
}
