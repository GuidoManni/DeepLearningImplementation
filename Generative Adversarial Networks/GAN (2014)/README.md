# [Original GAN Implementation]

## Overview
This repository contains a PyTorch implementation of the original Generative Adversarial Network (GAN) as proposed by Goodfellow et al. in 2014. This implementation maintains the original architecture choices, using simple MLPs for both generator and discriminator.

## Detailed Explanation
For a comprehensive understanding of GANs and their significance, please refer to the [original paper](https://arxiv.org/abs/1406.2661).

## Major Contributions
The major contributions of the paper include:
- **Novel Framework:** Introduced the adversarial framework where two networks compete against each other
- **Simple Architecture:** Demonstrated that even simple MLPs could generate realistic images
- **Versatile Approach:** Provided a framework that could be applied to various types of data generation
- **Theoretical Foundation:** Established theoretical guarantees for the training process

## Architecture Details

### Generator Architecture
- **Input:** 100-dimensional uniform random noise z ∈ [-1, 1]
- **Hidden Layer:** 240 units with ReLU activation
- **Output Layer:** 784 units (28×28) with Sigmoid activation
- **Output Reshaping:** Flat vector reshaped to 28×28 image

### Discriminator Architecture
- **Input:** 784-dimensional flattened image (28×28)
- **Hidden Layer:** 240 units with ReLU activation
- **Output Layer:** Single unit with Sigmoid activation

## Key Implementation Features
1. **Generator Features:**
   - Flexible batch size handling
   - Built-in noise sampling
   - Automatic reshaping to image dimensions
   
2. **Discriminator Features:**
   - Integrated flattening in the architecture
   - Single probability output for real/fake classification

## Usage Example
```python
# Initialize models
generator = Generator(z_dim=100, output_dim=28)
discriminator = Discriminator(input_dim=28)

# Generate fake images
fake_images = generator(batch_size=64)  # Shape: [64, 1, 28, 28]

# Discriminate images
predictions = discriminator(fake_images)  # Shape: [64, 1]