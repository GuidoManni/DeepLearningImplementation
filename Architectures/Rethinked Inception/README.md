# [InceptionV2/V3]

## Overview
This repository contains the implementation of InceptionV2/V3. Below you will find detailed information and resources related to this architecture.

## Detailed Explanation
For a comprehensive understanding of the paper and its contributions, please refer to the [detailed blog post](https://gvdmnni.notion.site/InceptionV2-V3-a5fa66c1e34c495aabd5b8950e4389f5?pvs=4).

## Major Contributions
The major contributions of the paper include:
- Introduction of design principles for efficiently scaling up CNNs, focusing on factorized convolutions and dimension reduction.
- Development of Inception-v2 and Inception-v3 architectures, which significantly improve upon the original GoogLeNet design.
- Proposal of label smoothing as a regularization technique to prevent the network from becoming too confident in its predictions.
- Investigation of the impact of input resolution on network performance, showing that lower resolution inputs can still achieve competitive results when the network is properly adapted.
- Achievement of state-of-the-art performance on the ILSVRC 2012 classification benchmark, with a substantial reduction in computational cost compared to other top-performing networks.


## Architecture Scheme
Below a schematic representation of the Inception modules that are used in the architecture:
![Inception Module](https://github.com/GuidoManni/DeepLearningImplementation/blob/main/Architectures/Rethinked%20Inception/src/Improved%20Inception%20Module.png)
*Figure: Inception modules with factorized convolutions. The original 5x5 convolutional filter has been replaced with two consecutive 3x3 convolutional filters, reducing computational cost while maintaining the effective receptive field.*

![Inception Module](https://github.com/GuidoManni/DeepLearningImplementation/blob/main/Architectures/Rethinked%20Inception/src/Improved%20Inception%20Module%20with%20more%20factorization.png)
*Figure: Inception modules after the factorization of the nxn convolutions.*

![Inception Module](https://github.com/GuidoManni/DeepLearningImplementation/blob/main/Architectures/Rethinked%20Inception/src/Inception%20Module%20with%20expanded%20filter%20bank.png)
*Figure: Inception modules with expanded filter bank outputs to promote high dimensional representations.*


## Reproduced Results (TBD)
The following results were reproduced as per the methodology described in the paper:
- Result 1: [Description and value]
- Result 2: [Description and value]
- Result 3: [Description and value]
- ...


## References
- [Original Paper](https://arxiv.org/abs/1512.00567)
- [Detailed Blog Post](https://gvdmnni.notion.site/InceptionV2-V3-a5fa66c1e34c495aabd5b8950e4389f5?pvs=44)
