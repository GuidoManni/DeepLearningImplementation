import torch
import torch.nn as nn
from residual_block import ResidualBlock1
from torchsummary import summary

class ResNet34(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet34, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # repeat 3 times
        self.conv2_x = nn.Sequential(
            ResidualBlock1(in_channels=64, out_channels=64, stride=1),
            ResidualBlock1(in_channels=64, out_channels=64, stride=1),
            ResidualBlock1(in_channels=64, out_channels=128, stride=1),
        )

        # repeat 4 times
        self.conv3_x = nn.Sequential(
            ResidualBlock1(in_channels=128, out_channels=128, stride=2),
            ResidualBlock1(in_channels=128, out_channels=128, stride=1),
            ResidualBlock1(in_channels=128, out_channels=128, stride=1),
            ResidualBlock1(in_channels=128, out_channels=256, stride=1),
        )

        # repeat 6 times
        self.conv4_x = nn.Sequential(
            ResidualBlock1(in_channels=256, out_channels=256, stride=2),
            ResidualBlock1(in_channels=256, out_channels=256, stride=1),
            ResidualBlock1(in_channels=256, out_channels=256, stride=1),
            ResidualBlock1(in_channels=256, out_channels=256, stride=1),
            ResidualBlock1(in_channels=256, out_channels=256, stride=1),
            ResidualBlock1(in_channels=256, out_channels=512, stride=1),
        )

        # repeat 3 times
        self.conv5_x = nn.Sequential(
            ResidualBlock1(in_channels=512, out_channels=512, stride=2),
            ResidualBlock1(in_channels=512, out_channels=512, stride=1),
            ResidualBlock1(in_channels=512, out_channels=512, stride=1),
        )

        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear((512), num_classes)
        )

    def forward(self, x):
        x = self.input_layer(x)
        print(x.shape)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        print(x.shape)
        x = self.output_layer(x)
        return x

if __name__ == '__main__':
    model = ResNet34(1000)
    summary(model, (3, 224, 224))