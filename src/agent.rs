use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d
};
use burn::prelude::*;
use burn::tensor::activation::{relu, softmax};

use crate::parameters::{NUM_FILTERS, NUM_RES_BLOCKS};


#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
}

impl<B: Backend> ResidualBlock<B> {
    pub fn new() -> Self {
        let device = B::Device::default();
        let conv_config = Conv2dConfig::new([NUM_FILTERS, NUM_FILTERS], [3, 3])
            .with_padding(PaddingConfig2d::Same);
        
        Self {
            conv1: conv_config.init(&device),
            bn1: BatchNormConfig::new(NUM_FILTERS).init(&device),
            conv2: conv_config.init(&device),
            bn2: BatchNormConfig::new(NUM_FILTERS).init(&device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = x.clone();

        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = relu(x);

        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);

        let x = x + residual;
        relu(x)
    }
}

// --- Main AlphaZero Module ---
#[derive(Module, Debug)]
pub struct AlphaZero<B: Backend> {
    // Shared Body
    input_conv: Conv2d<B>,
    input_bn: BatchNorm<B, 2>,
    res_blocks: Vec<ResidualBlock<B>>,

    // Policy Head
    policy_conv_1: Conv2d<B>,
    policy_bn: BatchNorm<B, 2>,
    policy_conv_2: Conv2d<B>,
    
    // Value Head
    value_conv: Conv2d<B>,
    value_bn: BatchNorm<B, 2>,
    value_linear_1: Linear<B>,
    value_linear_2: Linear<B>,
}

impl<B: Backend> AlphaZero<B> {
    pub fn new() -> Self {
        let device = B::Device::default();

        // --- Initialize Shared Body ---
        // Input layer is 19 channels * 8 ranks * 8 files (=1216)
        let conv_config = Conv2dConfig::new([19, NUM_FILTERS], [3, 3])
            .with_padding(PaddingConfig2d::Same);
        let input_conv = conv_config.init(&device);
        let input_bn = BatchNormConfig::new(NUM_FILTERS).init(&device);
        let mut res_blocks = Vec::with_capacity(NUM_RES_BLOCKS);
        for _ in 0..NUM_RES_BLOCKS {
            res_blocks.push(ResidualBlock::new());
        }

        // --- Initialize Policy Head ---
        // Input to final convolutional layer is 32 channels * 8 ranks * 8 files (= 2048)
        let policy_conv_1 = Conv2dConfig::new([NUM_FILTERS, 32], [1, 1]).init(&device);
        let policy_bn = BatchNormConfig::new(32).init(&device);
        let policy_conv_2 = Conv2dConfig::new([32, 64], [1, 1]).init(&device);

        // --- Initialize Value Head ---
        // Input to linear layer is 8 channels * 8 ranks * 8 files = 512
        let value_conv = Conv2dConfig::new([NUM_FILTERS, 8], [1, 1]).init(&device);
        let value_bn = BatchNormConfig::new(8).init(&device);
        let value_linear_1 = LinearConfig::new(8 * 8 * 8, 64).init(&device);
        let value_linear_2 = LinearConfig::new(64, 1).init(&device);

        Self {
            input_conv,
            input_bn,
            res_blocks,

            policy_conv_1,
            policy_bn,
            policy_conv_2,

            value_conv,
            value_bn,
            value_linear_1,
            value_linear_2,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        // --- Shared Body ---
        let mut x = self.input_conv.forward(x);
        x = self.input_bn.forward(x);

        x = relu(x);
        for block in &self.res_blocks {
            x = block.forward(x);
        }
        let body_output = x;

        // --- Policy Head ---
        let policy = self.policy_conv_1.forward(body_output.clone());
        let policy = self.policy_bn.forward(policy);
        let policy = relu(policy);
        let policy_logits = self.policy_conv_2.forward(policy);
        let [batch_size, _, _, _] = policy_logits.dims();
        let flat_logits = policy_logits.reshape([batch_size as i32, -1]);
        let policy = softmax(flat_logits, 1);

        // --- Value Head ---
        let value = self.value_conv.forward(body_output);
        let value = self.value_bn.forward(value);
        let value = relu(value);
        let [batch_size, _, _, _] = value.dims();
        let value = value.reshape([batch_size as i32, -1]);
        let value = self.value_linear_1.forward(value);
        let value = relu(value);
        let value_logit = self.value_linear_2.forward(value);
        let value = value_logit.tanh().squeeze(1);

        (policy, value)
    }
}