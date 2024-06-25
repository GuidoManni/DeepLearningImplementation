# [AlexNet]

## Overview
This repository contains the implementation of AlexNet. Below you will find detailed information and resources related to this architecture.

## Detailed Explanation
For a comprehensive understanding of the paper and its contributions, please refer to the [detailed blog post](https://gvdmnni.notion.site/AlexNet-caa1c8a968a54179a1da454ff764cb5e?pvs=4).

## Major Contributions
The major contributions of the paper include:
- **Depth and Complexity:** AlexNet demonstrated that deeper networks with many layers could achieve significantly better performance on complex image classification tasks
- **ReLU Activation**: Substitute tanh activation function with **Re**ctified **L**inear **U**nit, showing lower training time.
- **Dropout:** Utilized dropout as a regularization technique to prevent overfitting.
- **GPU Implementation:** Leveraged the power of GPUs to accelerate the training process, making it feasible to train large networks on large datasets.


## Architecture Scheme
Below is a schematic representation of the architecture:

![Architecture Scheme](https://github.com/GuidoManni/DeepLearningImplementation/blob/main/Architectures/AlexNet/src/AlexNet.png)

## Reproduced Results
The following results were reproduced as per the methodology described in the paper:
- Result 1: [Description and value]
- Result 2: [Description and value]
- Result 3: [Description and value]
- ...

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/GuidoManni/DeepLearningImplementation.git
   cd DeepLearningImplementation/[Architecture Directory]
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Train the model:
    ```bash
    python train.py
    ```
4. Evaluate the model:
    ```bash
    python evaluate.py
    ```

## References
- [Original Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- [Detailed Blog Post](https://gvdmnni.notion.site/AlexNet-caa1c8a968a54179a1da454ff764cb5e?pvs=4)
