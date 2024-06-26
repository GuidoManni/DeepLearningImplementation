# DeepLearningImplementation

Welcome to the DeepLearningImplementation repository! 
This repository is dedicated to the implementation of various seminal deep learning architectures for computer vision. 
Whether you are a researcher, student, or practitioner, you'll find comprehensive implementations, training scripts, 
and documentation for some of the most influential models in the field.

## Philosophy
The DeepLearningImplementation repository is built on a philosophy of simplicity and clarity. 
The primary goal is to offer implementations that prioritize readability and understandability over optimization and performance. 
This repository is designed to be a learning resource, helping researchers, students, and practitioners gain a deeper understanding of the inner workings of seminal deep learning architectures.

### Key Principles:
- **Simplicity**: Each implementation is crafted to be as straightforward as possible. The aim is to minimize complexity, making it easier for users to follow along and grasp the core concepts without being overwhelmed by intricate optimizations or advanced coding techniques.

- **Readability**: The code is written with a strong emphasis on readability. Clear variable names, concise comments, and structured organization are prioritized to ensure that anyone reading the code can easily understand the flow and purpose of each component.

- **Learning-Oriented**: The repository is meant to be a hands-on educational tool. By focusing on the fundamental mechanisms of each architecture, users can learn how these models work at a basic level, facilitating a deeper comprehension that can serve as a foundation for more advanced studies or applications.

- **Minimal Dependencies**: To keep things simple and focused, the project relies solely on PyTorch, one of the most widely used and accessible deep learning frameworks. This decision eliminates the need for additional external libraries, reducing setup complexity and ensuring that users can dive straight into learning.

## Contents

### Architectures
Click on the checkmarks to go to project directories.
- [AlexNet [2012]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) - [✅](./Architectures/AlexNet) 
- [ZFNet [2013]](https://arxiv.org/abs/1311.2901) - [✅](./Architectures/ZFNet)
- [GoogLeNet [2014]](https://arxiv.org/abs/1409.4842) - [✅](./Architectures/GoogLeNet)
- [VGG16 [2015]](https://arxiv.org/pdf/1409.1556) - [ ]
- [ResNet [2015]](https://arxiv.org/pdf/1512.03385) - [ ]
- [Rethinked Inception [2015]](https://arxiv.org/abs/1512.00567) - [ ]
- [DenseNet [2016]](https://arxiv.org/abs/1608.06993)
- [Xception [2016]](https://arxiv.org/abs/1610.02357) - [ ]
- [SqueezeNet [2016]](https://arxiv.org/abs/1602.07360) - [ ]
- [MobileNet [2017]](https://arxiv.org/abs/1704.04861) - [ ]
- [Residual Attention Network [2017]](https://arxiv.org/abs/1704.06904) - [ ]
- [EfficientNet [2019]](https://arxiv.org/abs/1905.11946) - [ ]
- [RegNet [2020]](https://arxiv.org/abs/2003.13678) - [ ]
- [ConvNet [2020]](https://arxiv.org/abs/2001.06268) - [ ]
- [VisionTransformer [2020]](https://arxiv.org/pdf/2010.11929) - [ ]
- [SwinTransformer [2021]](https://arxiv.org/pdf/2103.14030) - [ ]
- [VisionLSTM [2024]](https://arxiv.org/pdf/2406.04303) - [ ]

### Semantic Segmentation
- [FCN [2014]](https://arxiv.org/abs/1411.4038) - [ ]
- [SegNet [2015]](https://arxiv.org/abs/1511.00561) - [ ]
- [UNet [2015]](https://arxiv.org/abs/1505.04597) - [ ]
- [PSPNet [2016]](https://arxiv.org/abs/1612.01105) - [ ]
- [DeepLab [2016]](https://arxiv.org/abs/1606.00915) - [ ]
- [ENet [2016]](https://arxiv.org/abs/1606.02147) - [ ]
- [ICNet [2018]](https://arxiv.org/abs/1704.08545) - [ ]

### Generative Adversarial Networks
- [GAN](https://arxiv.org/abs/1406.2661) - [ ]
- [DCGAN](https://arxiv.org/abs/1511.06434) - [ ]
- [WGAN](https://arxiv.org/abs/1701.07875) - [ ]
- [Pix2Pix](https://arxiv.org/abs/1611.07004) - [ ]
- [CycleGAN](https://arxiv.org/abs/1703.10593) - [ ]

### Autoregressive Generative Networks
- [PixelRNN](https://arxiv.org/pdf/1601.06759) - [ ]

### Object Detection
- [RCNN](https://arxiv.org/abs/1311.2524) - [ ]
- [Fast-RCNN](https://arxiv.org/abs/1504.08083) - [ ]
- [Faster-RCNN](https://arxiv.org/abs/1506.01497) - [ ]
- [SSD](https://arxiv.org/abs/1512.02325) - [ ]
- [YOLO](https://arxiv.org/abs/1506.02640) - [ ]
- [YOLO9000](https://arxiv.org/abs/1612.08242) - [ ]


## Project Phases
The DeepLearningImplementation repository is structured into distinct phases to ensure a comprehensive and systematic approach to developing and refining deep learning models. 
Each phase builds upon the previous one, progressively enhancing the quality and utility of the repository.

### First Phase: Implementation and Initial Documentation
The first phase is dedicated to the implementation of various deep learning architectures. 
During this phase, the primary focus is on writing clear and understandable code for each model. 
Alongside the implementation, a raw documentation is provided to explain the basic functioning and structure of the models. 
This phase sets the foundation for further development and ensures that each model is accessible and easy to comprehend.

**Current Status**: We are currently in phase 1.

### Second Phase: Training and Performance Evaluation
In the second phase, the focus shifts to training each implemented model on relevant datasets. 
This phase involves computing the performance metrics for each model and making comparisons to understand their strengths and weaknesses.

### Third Phase: Code Refinement and Documentation Enhancement
The third and final phase involves refining the code implementations. 
This phase also includes enhancing the documentation to provide more detailed explanations, usage instructions, and best practices.
The aim is to polish the repository, making it a robust and reliable resource for learning and experimentation.


## Getting Started

Each directory contains the implementation of a specific architecture along with training scripts and detailed documentation. To get started with any architecture, navigate to the respective directory and follow the instructions in the README file.

### Installation

Each architecture has its own set of dependencies listed in the `requirements.txt` file in its directory. You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Project Structure
* [Architectures](./Architectures)
  * [AlexNet](./Architectures/AlexNet)
    * [README.md](./Architectures/AlexNet/README.md)
    * [alexnet.py](./Architectures/AlexNet/alexnet.py)
    * [requirements.txt](./Architectures/AlexNet/requirements.txt)
  * [ZFNet](./Architectures/ZFNet)
    * [README.md](./Architectures/VGG/README.md)
    * [zfnet.py](./Architectures/ZFNet/zfnet.py)
    * [requirements.txt](./Architectures/VGG/requirements.txt)
  * [...](./Architectures)
* [GANs](./GANs)
  * [GAN](./GANs/GAN)
    * [README.md](./GANs/GAN/README.md)
    * [gan.py](./GANs/GAN/gan.py)
    * [train.py](./GANs/GAN/train.py)
    * [requirements.txt](./GANs/GAN/requirements.txt)
  * [DCGAN](./GANs/DCGAN)
    * [README.md](./GANs/DCGAN/README.md)
    * [dcgan.py](./GANs/DCGAN/dcgan.py)
    * [train.py](./GANs/DCGAN/train.py)
    * [requirements.txt](./GANs/DCGAN/requirements.txt)
  * [...](./GANs)
* [SemanticSegmentation](./SemanticSegmentation)
  * [FCN](./SemanticSegmentation/FCN)
    * [README.md](./SemanticSegmentation/FCN/README.md)
    * [fcn.py](./SemanticSegmentation/FCN/fcn.py)
    * [train.py](./SemanticSegmentation/FCN/train.py)
    * [requirements.txt](./SemanticSegmentation/FCN/requirements.txt)
  * [UNet](./SemanticSegmentation/UNet)
    * [README.md](./SemanticSegmentation/UNet/README.md)
    * [unet.py](./SemanticSegmentation/UNet/unet.py)
    * [train.py](./SemanticSegmentation/UNet/train.py)
    * [requirements.txt](./SemanticSegmentation/UNet/requirements.txt)
  * [...](./SemanticSegmentation)
* [ObjectDetection](./ObjectDetection)
  * [RCNN](./ObjectDetection/RCNN)
    * [README.md](./ObjectDetection/RCNN/README.md)
    * [rcnn.py](./ObjectDetection/RCNN/rcnn.py)
    * [train.py](./ObjectDetection/RCNN/train.py)
    * [requirements.txt](./ObjectDetection/RCNN/requirements.txt)
  * [YOLO](./ObjectDetection/YOLO)
    * [README.md](./ObjectDetection/YOLO/README.md)
    * [yolo.py](./ObjectDetection/YOLO/yolo.py)
    * [train.py](./ObjectDetection/YOLO/train.py)
    * [requirements.txt](./ObjectDetection/YOLO/requirements.txt)
  * [...](./ObjectDetection)
* [LICENSE](./LICENSE)
* [README.md](./README.md)

### Contributing
Contributions are welcome! Please feel free to submit issues or pull requests to help improve the implementations and documentation.

### Contact
For any questions, please open an issue or contact the repository maintainer.
