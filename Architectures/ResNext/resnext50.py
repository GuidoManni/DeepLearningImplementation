import torch
import torch.nn as nn
from torchsummary import summary

from resnext_module import *


class ResNext50(nn.Module):
    def __init__(self, num_classes = 1000, cardinality = 32):
        super(ResNext50, self).__init__()

        self.input_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        conv_2_list = []
        for i in range(3):
            if i == 0:
                # The first blok of each stage perform downsampling using stride = 2 on the grouped conv 3x3
                conv_2_list.append(ResNextBlock(in_channels=64, out_channels_bottleneck=128, out_channels=256, stride=1, groups=cardinality))
            else:
                conv_2_list.append(ResNextBlock(in_channels=256, out_channels_bottleneck=128, out_channels=256, stride=1, groups=cardinality))

        self.conv2 = nn.Sequential(*conv_2_list)

        conv_3_list = []
        for i in range(4):
            if i == 0:
                conv_3_list.append(ResNextBlock(in_channels=256, out_channels_bottleneck=256, out_channels=512, stride=2, groups=cardinality))
            else:
                conv_3_list.append(ResNextBlock(in_channels=512, out_channels_bottleneck=256, out_channels=512, stride=1, groups=cardinality))
        self.conv3 = nn.Sequential(*conv_3_list)

        conv_4_list = []
        for i in range(6):
            if i == 0:
                conv_4_list.append(ResNextBlock(in_channels=512, out_channels_bottleneck=512, out_channels=1024, stride=2, groups=cardinality))
            else:
                conv_4_list.append(ResNextBlock(in_channels=1024, out_channels_bottleneck=512, out_channels=1024, stride=1, groups=cardinality))
        self.conv4 = nn.Sequential(*conv_4_list)

        conv_5_list = []
        for i in range(3):
            if i == 0:
                conv_5_list.append(ResNextBlock(in_channels=1024, out_channels_bottleneck=1024, out_channels=2048, stride=2, groups=cardinality))
            else:
                conv_5_list.append(ResNextBlock(in_channels=2048, out_channels_bottleneck=1024, out_channels=2048, stride=1, groups=cardinality))
        self.conv5 = nn.Sequential(*conv_5_list)

        self.output_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )
    def forward(self, x):
        x = self.input_layers(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.output_layers(x)
        return x



if __name__ == '__main__':
    model = ResNext50(1000)
    summary(model, (3, 224, 224))