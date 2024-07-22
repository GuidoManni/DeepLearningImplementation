# [ShuffleNet]

## Overview
This repository contains the implementation of ShuffleNet. Below you will find detailed information and resources related to this architecture.

## Detailed Explanation
For a comprehensive understanding of the paper and its contributions, please refer to the [detailed blog post](https://gvdmnni.notion.site/ShuffleNet-97075c040be24950b7d1ec244a00ed4e?pvs=4).

## Major Contributions
The major contributions of the paper include:
1. Introduction of the ShuffleNet architecture, which uses depthwise separable convolutions as its core building block.
2. Proposal of two hyperparameters - width multiplier and resolution multiplier - that allow for easy adjustment of the model size and computational requirements.
3. Extensive experiments demonstrating the effectiveness of ShuffleNets across various tasks and applications, including image classification, object detection, and face attribute detection.
4. Comparison with other popular models, showing that ShuffleNets can achieve comparable accuracy with significantly reduced computational cost and model size.

## Architecture Scheme
Below a schematic representation of the ShuffleNet units that are used in the architecture:
![Image](./src/shuffleblock_no_stride.png)**ShuffleNet Unit without Stride**
![Image](./src/shuffleblock_with_stride.png)**ShuffleNet Unit with Stride**




## Reproduced Results (TBD)
The following results were reproduced as per the methodology described in the paper:
- Result 1: [Description and value]
- Result 2: [Description and value]
- Result 3: [Description and value]
- ...


## References
- [Original Paper](https://arxiv.org/abs/1707.01083)
- [Detailed Blog Post](https://gvdmnni.notion.site/ShuffleNet-97075c040be24950b7d1ec244a00ed4e?pvs=4)
