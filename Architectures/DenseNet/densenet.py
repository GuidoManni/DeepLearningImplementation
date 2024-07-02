import torch
import torch.nn as nn
from torchsummary import summary

from densenet_blocks import *


class DenseNet121(nn.Module):
    def __init__(self, growthRate = 12, num_classes = 1000, n_dense_block = [6, 12, 24, 16]):
        super(DenseNet121, self).__init__()

        nChannels = 2 * growthRate

        # Input Layers
        self.input_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=nChannels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(nChannels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )


        # Dense Block 1
        first_dense_layer = []
        for i in range(n_dense_block[0]):
            first_dense_layer.append(DenseBlock(in_channels=nChannels, growth_rate=growthRate))
            nChannels += growthRate
        self.dense_block_1 = nn.Sequential(*first_dense_layer)

        # Transition Block 1
        self.transition_block_1 = TransitionBlock(in_channels=nChannels, out_1x1=nChannels)

        # Dense Block 2
        second_dense_layer = []
        for i in range(n_dense_block[1]):
            second_dense_layer.append(DenseBlock(in_channels=nChannels, growth_rate=growthRate))
            nChannels += growthRate
        self.dense_block_2 = nn.Sequential(*second_dense_layer)

        # Transition Block 2
        self.transition_block_2 = TransitionBlock(in_channels=nChannels, out_1x1=nChannels)

        # Dense Block 3
        third_dense_layer = []
        for i in range(n_dense_block[2]):
            third_dense_layer.append(DenseBlock(in_channels=nChannels, growth_rate=growthRate))
            nChannels += growthRate
        self.dense_block_3 = nn.Sequential(*third_dense_layer)

        # Transition Block 3
        self.transition_block_3 = TransitionBlock(in_channels=nChannels, out_1x1=nChannels)

        # Dense Block 4
        fourth_dense_layer = []
        for i in range(n_dense_block[3]):
            fourth_dense_layer.append(DenseBlock(in_channels=nChannels, growth_rate=growthRate))
            nChannels += growthRate
        self.dense_block_4 = nn.Sequential(*fourth_dense_layer)

        # Output Layer
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear((720 * 7 * 7), num_classes)
        )

    def forward(self, x):
        x = self.input_layers(x)
        x = self.dense_block_1(x)
        x = self.transition_block_1(x)
        x = self.dense_block_2(x)
        x = self.transition_block_2(x)
        x = self.dense_block_3(x)
        x = self.transition_block_3(x)
        x = self.dense_block_4(x)
        x = self.output_layer(x)
        return x

if __name__ == '__main__':
    n_dense_block = [6, 12, 24, 16] # DenseNet-121
    # n_dense_block = [6, 12, 32, 32] # DenseNet-169
    # n_dense_block = [6, 12, 48, 32] # DenseNet-201
    # n_dense_block = [6, 12, 64, 48] # DenseNet-264
    model = DenseNet121(n_dense_block=n_dense_block)
    summary(model, (3, 224, 224))