# [MobileNet]

## Overview
This repository contains the implementation of MobileNet. Below you will find detailed information and resources related to this architecture.

## Detailed Explanation
For a comprehensive understanding of the paper and its contributions, please refer to the [detailed blog post](https://gvdmnni.notion.site/MobileNets-387efa72839b4b0fa980cfc858c5052f).

## Major Contributions
The major contributions of the paper include:
- Introduction of design principles for efficiently scaling up CNNs, focusing on factorized convolutions and dimension reduction.
- Development of Inception-v2 and Inception-v3 architectures, which significantly improve upon the original GoogLeNet design.
- Proposal of label smoothing as a regularization technique to prevent the network from becoming too confident in its predictions.
- Investigation of the impact of input resolution on network performance, showing that lower resolution inputs can still achieve competitive results when the network is properly adapted.
- Achievement of state-of-the-art performance on the ILSVRC 2012 classification benchmark, with a substantial reduction in computational cost compared to other top-performing networks.


## Architecture Scheme
Below a schematic representation of the MobileNet architecture:
![Image](./src/MobileNet_architecture.png)

Below a schematic representation of the DepthWise Convolution modules that are used in the architecture:
![Image](./src/MobileNets_DepthwiseSeparableConv.png)




## Reproduced Results (TBD)
The following results were reproduced as per the methodology described in the paper:
- Result 1: [Description and value]
- Result 2: [Description and value]
- Result 3: [Description and value]
- ...


## References
- [Original Paper](https://arxiv.org/abs/1704.04861)
- [Detailed Blog Post](https://gvdmnni.notion.site/MobileNets-387efa72839b4b0fa980cfc858c5052f)
