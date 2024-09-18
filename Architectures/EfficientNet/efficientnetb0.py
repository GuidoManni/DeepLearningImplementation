import torch
from torchsummary import summary
from EfficientNet_Module import *


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = ConvBNAct(3, 32, stride=2)

        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, 1, 3, 1),
            MBConvBlock(16, 24, 6, 3, 2),
            MBConvBlock(24, 24, 6, 3, 1),
            MBConvBlock(24, 40, 6, 5, 2),
            MBConvBlock(40, 40, 6, 5, 1),
            MBConvBlock(40, 80, 6, 3, 2),
            MBConvBlock(80, 80, 6, 3, 1),
            MBConvBlock(80, 80, 6, 3, 1),
            MBConvBlock(80, 112, 6, 5, 1),
            MBConvBlock(112, 112, 6, 5, 1),
            MBConvBlock(112, 112, 6, 5, 1),
            MBConvBlock(112, 192, 6, 5, 2),
            MBConvBlock(192, 192, 6, 5, 1),
            MBConvBlock(192, 192, 6, 5, 1),
            MBConvBlock(192, 192, 6, 5, 1),
            MBConvBlock(192, 320, 6, 3, 1)
        )

        self.head = nn.Sequential(
            ConvBNAct(320, 1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


from torchvision.models import efficientnet_b0

if __name__ == "__main__":
    model = EfficientNetB0(num_classes=1000)
    summary(model, (3, 224, 224))

    model1 = efficientnet_b0()
    summary(model1, (3, 224, 224))