# [SqueezeNet]

## Overview
This repository contains the implementation of SqueezeNet. Below you will find detailed information and resources related to this architecture.

## Detailed Explanation
For a comprehensive understanding of the paper and its contributions, please refer to the [detailed blog post](https://gvdmnni.notion.site/SqueezeNet-6872b7d0b1b849c5956de2927a880105?pvs=4).

## Major Contributions
The major contributions of the paper include:
1. Introduction of architectural design strategies for creating small CNNs
2. Presentation of the SqueezeNet architecture, which achieves AlexNet-level accuracy with 50x fewer parameters
3. Demonstration that SqueezeNet can be compressed to 510x smaller than AlexNet while maintaining accuracy
4. Exploration of the CNN microarchitecture design space, providing insights into the impact of various design choices on model size and accuracy
5. Investigation of different CNN macroarchitecture configurations, including the use of bypass connections

## Architecture Scheme
Below a schematic representation of the SqueezeNet architecture:
![Image](./src/SqueezeNet_architecture.png)

Below a schematic representation of the FIRE modules that are used in the architecture:
![Image](./src/SqueezeNet_Fire_module.png)




## Reproduced Results (TBD)
The following results were reproduced as per the methodology described in the paper:
- Result 1: [Description and value]
- Result 2: [Description and value]
- Result 3: [Description and value]
- ...


## References
- [Original Paper](https://arxiv.org/abs/1602.07360)
- [Detailed Blog Post](https://gvdmnni.notion.site/SqueezeNet-6872b7d0b1b849c5956de2927a880105?pvs=4)
