import torch
import torch.nn as nn


class DepthwiseSeprableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        '''
        A separable convolution block is a combination of depthwise convolution and pointwise convolution. This means
        that the input tensor is convolved with a kernel of size (kernel_size, kernel_size) and then the output of this
        operation is convolved with a 1x1 kernel. This is done to reduce the number of parameters in the model.
        '''
        super(DepthwiseSeprableConv2d, self).__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(ConvBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)