import torch
import torch.nn as nn
from torchsummary import summary

from senet_block import *


class SENet(nn.Module):
    def __init__(self, num_classes = 1000, reduction = 16):
        super(SENet, self).__init__()

        self.input_layers = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels_bottleneck=64, out_channels=256, stride=1),
            SEBlock(in_feature=256, reduction=reduction),
            ResidualBlock(in_channels=256, out_channels_bottleneck=64, out_channels=256, stride=1),
            SEBlock(in_feature=256, reduction=reduction),
            ResidualBlock(in_channels=256, out_channels_bottleneck=64, out_channels=256, stride=1),
            SEBlock(in_feature=256, reduction=reduction),
        )



        conv_3x_list = []
        for i in range(4):
            if i == 0:
                conv_3x_list.append(ResidualBlock(in_channels=256, out_channels_bottleneck=128, out_channels=512, stride=2))
            else:
                conv_3x_list.append(ResidualBlock(in_channels=512, out_channels_bottleneck=128, out_channels=512, stride=1))

            conv_3x_list.append(SEBlock(in_feature=512, reduction=reduction))
        self.conv3_x = nn.Sequential(*conv_3x_list)


        conv_4x_list = []
        for i in range(6):
            if i == 0:
                conv_4x_list.append(ResidualBlock(in_channels=512, out_channels_bottleneck=256, out_channels=1024, stride=2))
            else:
                conv_4x_list.append(ResidualBlock(in_channels=1024, out_channels_bottleneck=256, out_channels=1024, stride=1))

            conv_4x_list.append(SEBlock(in_feature=1024, reduction=reduction))
        self.conv4_x = nn.Sequential(*conv_4x_list)


        self.conv5_x = nn.Sequential(
            ResidualBlock(in_channels=1024, out_channels_bottleneck=512, out_channels=2048, stride=2),
            SEBlock(in_feature=2048, reduction=reduction),
            ResidualBlock(in_channels=2048, out_channels_bottleneck=512, out_channels=2048, stride=1),
            SEBlock(in_feature=2048, reduction=reduction),
            ResidualBlock(in_channels=2048, out_channels_bottleneck=512, out_channels=2048, stride=1),
            SEBlock(in_feature=2048, reduction=reduction),
        )



        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.input_layers(x)
        x = self.conv2_x(x)

        x = self.conv3_x(x)

        x = self.conv4_x(x)

        x = self.conv5_x(x)

        x = self.output_layer(x)
        return x


if __name__ == '__main__':
    model = SENet(1000)
    summary(model, (3, 224, 224))

