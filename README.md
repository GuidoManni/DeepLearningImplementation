# DeepLearningImplementation 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Welcome to the DeepLearningImplementation repository! This project provides clean, readable implementations of seminal deep learning architectures for computer vision. Whether you're a researcher, student, or practitioner, you'll find comprehensive implementations, training scripts, and documentation for some of the most influential models in the field.

## 🎯 Project Philosophy

We prioritize clarity and understanding over optimization. Our implementations focus on:

- **Simplicity**: Clean, straightforward code that's easy to follow
- **Readability**: Clear variable names, thorough comments, and structured organization
- **Learning-Oriented**: Focus on fundamental mechanisms for deeper understanding
- **Minimal Dependencies**: Built primarily with PyTorch for simplified setup

## 📚 Available and Planned Implementations

### Computer Vision Architectures
- [✅] [AlexNet (2012)](./Architectures/AlexNet)
- [✅] [ZFNet (2013)](./Architectures/ZFNet)
- [✅] [GoogLeNet (2014)](./Architectures/GoogLeNet)
- [✅] [VGG16 (2015)](./Architectures/VGG16)
- [✅] [ResNet (2015)](./Architectures/ResNet)
- [✅] [Rethinked Inception (2015)](./Architectures/Rethinked%20Inception)
- [✅] [DenseNet (2016)](./Architectures/DenseNet)
- [✅] [Xception (2016)](./Architectures/Xception)
- [✅] [SqueezeNet (2016)](./Architectures/SqueezeNet)
- [✅] [ResNeXt (2016)](./Architectures/ResNeXt)
- [✅] [SENet (2017)](./Architectures/SENet)
- [✅] [MobileNet (2017)](./Architectures/MobileNet)
- [✅] [ShuffleNet (2017)](./Architectures/ShuffleNet)
- [✅] [Residual Attention Network (2017)](./Architectures/ResidualAttentionNetwork)
- [✅] [MobileNetV2 (2018)](./Architectures/MobileNetV2)
- [✅] [EfficientNet (2019)](./Architectures/EfficientNet)
- [ ] [RegNet (2020)](https://arxiv.org/abs/2003.13678)
- [ ] [ConvNet (2020)](https://arxiv.org/abs/2001.06268)
- [ ] [VisionTransformer (2020)](https://arxiv.org/pdf/2010.11929)
- [ ] [SwinTransformer (2021)](https://arxiv.org/pdf/2103.14030)
- [ ] [MaxViT (2022)](https://arxiv.org/pdf/2204.01697)
- [ ] [RepVIT (2023)](https://arxiv.org/abs/2307.09283)
- [ ] [VisionLSTM (2024)](https://arxiv.org/pdf/2406.04303)


### Semantic Segmentation
- [ ] [FCN (2014)](https://arxiv.org/abs/1411.4038)
- [ ] [SegNet (2015)](https://arxiv.org/abs/1511.00561)
- [✅] [UNet (2015)](./Semantic%20Segmentation/UNet/)
- [ ] [PSPNet (2016)](https://arxiv.org/abs/1612.01105)
- [ ] [DeepLab (2016)](https://arxiv.org/abs/1606.00915)
- [ ] [ENet (2016)](https://arxiv.org/abs/1606.02147)
- [ ] [Mask R-CNN (2017)](https://arxiv.org/abs/1703.06870)
- [ ] [DeepLabV3 (2017)](https://arxiv.org/abs/1706.05587)
- [ ] [ICNet (2018)](https://arxiv.org/abs/1704.08545)
- [✅] [Attention Unet (2018)](./Semantic%20Segmentation/AttentionUnet/)
- [ ] [HRNet (2019)](https://arxiv.org/abs/1904.04514)
- [ ] [OCRNet (2019)](https://arxiv.org/abs/1909.11065)
- [✅] [U-Net++ (2019)](./Semantic%20Segmentation/UNet++/)
- [ ] [SegFormer (2021)](https://arxiv.org/abs/2105.15203)
- [ ] [Mask2Former (2022)](https://arxiv.org/abs/2204.01697)

### Object Detection
- [ ] [RCNN (2014)](https://arxiv.org/abs/1311.2524)
- [ ] [Fast-RCNN (2015)](https://arxiv.org/abs/1504.08083)
- [ ] [Faster-RCNN (2015)](https://arxiv.org/abs/1506.01497)
- [ ] [YOLO (2015)](https://arxiv.org/abs/1506.02640)
- [ ] [SSD (2016)](https://arxiv.org/abs/1512.02325)
- [ ] [YOLO9000 (2016)](https://arxiv.org/abs/1612.08242)
- [ ] [RetinaNet (2017)](https://arxiv.org/abs/1708.02002)
- [ ] [YOLOv3 (2018)](https://arxiv.org/abs/1804.02767)
- [ ] [YOLOv4 (2020)](https://arxiv.org/abs/2004.10934)

### Generative Adversarial Networks
- [✅] [GAN (2014)](./Generative%20Adversarial%20Networks/GAN%20(2014)/)
- [ ] [DCGAN (2015)](https://arxiv.org/abs/1511.06434)
- [ ] [InfoGAN (2016)](https://arxiv.org/abs/1606.03657)
- [ ] [Pix2Pix (2016)](https://arxiv.org/abs/1611.07004)
- [ ] [WGAN (2017)](https://arxiv.org/abs/1701.07875)
- [ ] [CycleGAN (2017)](https://arxiv.org/abs/1703.10593)
- [ ] [BigGAN (2018)](https://arxiv.org/abs/1809.11096)
- [ ] [StyleGAN (2018)](https://arxiv.org/abs/1812.04948)
- [ ] [StyleGAN2 (2019)](https://arxiv.org/abs/1912.04958)

### Diffusion Generative Models
- [ ] [DDPM (2020)](https://arxiv.org/abs/2006.11239)

### Autoregressive Generative Networks
- [ ] [PixelRNN (2016)](https://arxiv.org/pdf/1601.06759)
- [ ] [PixelCNN (2016)](https://arxiv.org/abs/1606.05328)
- [ ] [PixelSNAIL (2017)](https://arxiv.org/abs/1712.09763)

### 3D Reconstruction from 2D Images
- [ ] [3D-R2N2 (2016)](https://arxiv.org/abs/1604.00449)
- [ ] [3D-RecGAN (2017)](https://arxiv.org/abs/1708.07969)
- [ ] [3D-GAN (2017)](https://arxiv.org/abs/1707.09557)
- [ ] [3D-RecGAN++ (2018)](https://arxiv.org/abs/1802.00411)
- [ ] [AtlasNet (2018)](https://arxiv.org/abs/1802.05384)
- [ ] [Occupancy Networks (2018)](https://arxiv.org/abs/1812.03828)
- [ ] [DeepSDF (2019)](https://arxiv.org/abs/1901.05103)
- [ ] [NeRF (2020)](https://arxiv.org/abs/2003.08934)

### Attention Mechanism
- [✅] [SENet (2017)](./Architectures/SENet)
- [✅] [Residual Attention Network (2017)](./Architectures/ResidualAttentionNetwork)
- [✅] [Attention Unet (2018)](./Semantic%20Segmentation/AttentionUnet/)
- [✅] [CBAM (2018)](./Attention%20Mechanism/CBAM)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DeepLearningImplementation.git
cd DeepLearningImplementation
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies for specific architecture:
```bash
cd Architectures/DesiredModel
pip install -r requirements.txt
```

## 📁 Project Structure

```
DeepLearningImplementation/
├── Architectures/          # CNN architectures
│   ├── AlexNet/
│   │   ├── README.md
│   │   ├── alexnet.py
│   │   └── requirements.txt
│   └── ...
├── SemanticSegmentation/
├── ObjectDetection/
├── GANs/
├── LICENSE
└── README.md
```

## 🛠️ Project Phases

### Phase 1: Implementation and Initial Documentation (Current)
- Writing clear, understandable code for each model
- Providing basic documentation
- Setting foundation for further development

### Phase 2: Training and Performance Evaluation (Planned)
- Training models on relevant datasets
- Computing performance metrics
- Comparing model strengths and weaknesses

### Phase 3: Code Refinement and Documentation Enhancement (Planned)
- Refining code implementations
- Enhancing documentation
- Adding detailed explanations and best practices

## 👥 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to help improve the implementations and documentation.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

For any questions, please open an issue or contact the repository maintainer.

---
Made with ❤️ for the deep learning community
