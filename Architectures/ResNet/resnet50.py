import torch
import torch.nn as nn
from residual_block import ResidualBlock2
from torchsummary import summary

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = nn.Sequential(
            ResidualBlock2(in_channels=64, out_channels1=64, out_channels2=256, stride=1),
            ResidualBlock2(in_channels=256, out_channels1=64, out_channels2=256, stride=1),
            ResidualBlock2(in_channels=256, out_channels1=64, out_channels2=256, stride=1),
        )

        self.conv3_x = nn.Sequential(
            ResidualBlock2(in_channels=256, out_channels1=128, out_channels2=512, stride=2),
            ResidualBlock2(in_channels=512, out_channels1=128, out_channels2=512, stride=1),
            ResidualBlock2(in_channels=512, out_channels1=128, out_channels2=512, stride=1),
            ResidualBlock2(in_channels=512, out_channels1=128, out_channels2=512, stride=1),
        )

        self.conv4_x = nn.Sequential(
            ResidualBlock2(in_channels=512, out_channels1=256, out_channels2=1024, stride=2),
            ResidualBlock2(in_channels=1024, out_channels1=256, out_channels2=1024, stride=1),
            ResidualBlock2(in_channels=1024, out_channels1=256, out_channels2=1024, stride=1),
            ResidualBlock2(in_channels=1024, out_channels1=256, out_channels2=1024, stride=1),
            ResidualBlock2(in_channels=1024, out_channels1=256, out_channels2=1024, stride=1),
            ResidualBlock2(in_channels=1024, out_channels1=256, out_channels2=1024, stride=1),
        )

        self.conv5_x = nn.Sequential(
            ResidualBlock2(in_channels=1024, out_channels1=512, out_channels2=2048, stride=2),
            ResidualBlock2(in_channels=2048, out_channels1=512, out_channels2=2048, stride=1),
            ResidualBlock2(in_channels=2048, out_channels1=512, out_channels2=2048, stride=1),
        )

        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear((2048), num_classes)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.output_layer(x)
        return x

if __name__ == '__main__':
    model = ResNet50(1000)
    summary(model, (3, 224, 224))