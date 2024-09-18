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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.conv_block(x)
        if not self.linear:
            x = nn.ReLU()(x)
        return x



class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion_factor=6):
        super(BottleneckResidualBlock, self).__init__()
        self.stride = stride

        self.bot_res_block = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=(in_channels * expansion_factor), kernel_size=1, linear=False),
            DepthwiseSeprableConv2d(in_channels=(in_channels * expansion_factor), out_channels=(in_channels * expansion_factor), kernel_size=3, stride=stride),
            ConvBlock(in_channels=(in_channels * expansion_factor), out_channels=out_channels, kernel_size=1, linear=True)
        )

        self.residual = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.residual = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, linear=True)
    def forward(self, x):
        x = self.bot_res_block(x)
        if self.residual == 1:
            x += self.residual(x)
        return x

