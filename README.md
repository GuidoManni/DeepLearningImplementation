# DeepLearningImplementation

Welcome to the DeepLearningImplementation repository! 
This repository is dedicated to the implementation of various seminal deep learning architectures for computer vision. Whether you are a researcher, student, or practitioner, you'll find comprehensive implementations, training scripts, and documentation for some of the most influential models in the field.

## Contents

### Architectures
- [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) [ ]
- [ZFNet](https://arxiv.org/abs/1311.2901) - [ ]
- [VGG16](https://arxiv.org/abs/1505.06798) - [ ]
- [ResNet](https://arxiv.org/abs/1704.06904) - [ ]
- [GoogLeNet](https://arxiv.org/abs/1409.4842) - [ ]
- [Inception](https://arxiv.org/abs/1512.00567) - [ ]
- [Xception](https://arxiv.org/abs/1610.02357) - [ ]
- [MobileNet](https://arxiv.org/abs/1704.04861) - [ ]

### Semantic Segmentation
- [FCN](https://arxiv.org/abs/1411.4038) - [ ]
- [SegNet](https://arxiv.org/abs/1511.00561) - [ ]
- [UNet](https://arxiv.org/abs/1505.04597) - [ ]
- [PSPNet](https://arxiv.org/abs/1612.01105) - [ ]
- [DeepLab](https://arxiv.org/abs/1606.00915) - [ ]
- [ICNet](https://arxiv.org/abs/1704.08545) - [ ]
- [ENet](https://arxiv.org/abs/1606.02147) - [ ]

### Generative Adversarial Networks
- [GAN](https://arxiv.org/abs/1406.2661) - [ ]
- [DCGAN](https://arxiv.org/abs/1511.06434) - [ ]
- [WGAN](https://arxiv.org/abs/1701.07875) - [ ]
- [Pix2Pix](https://arxiv.org/abs/1611.07004) - [ ]
- [CycleGAN](https://arxiv.org/abs/1703.10593) - [ ]

### Object Detection
- [RCNN](https://arxiv.org/abs/1311.2524) - [ ]
- [Fast-RCNN](https://arxiv.org/abs/1504.08083) - [ ]
- [Faster-RCNN](https://arxiv.org/abs/1506.01497) - [ ]
- [SSD](https://arxiv.org/abs/1512.02325) - [ ]
- [YOLO](https://arxiv.org/abs/1506.02640) - [ ]
- [YOLO9000](https://arxiv.org/abs/1612.08242) - [ ]

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
    * [train.py](./Architectures/AlexNet/train.py)
    * [requirements.txt](./Architectures/AlexNet/requirements.txt)
  * [VGG](./Architectures/VGG)
    * [README.md](./Architectures/VGG/README.md)
    * [vgg.py](./Architectures/VGG/vgg.py)
    * [train.py](./Architectures/VGG/train.py)
    * [requirements.txt](./Architectures/VGG/requirements.txt)
  * [ResNet](./Architectures/ResNet)
    * [README.md](./Architectures/ResNet/README.md)
    * [resnet.py](./Architectures/ResNet/resnet.py)
    * [train.py](./Architectures/ResNet/train.py)
    * [requirements.txt](./Architectures/ResNet/requirements.txt)
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
