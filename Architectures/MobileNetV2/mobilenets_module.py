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


class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion_factor=6):
        super(BottleneckResidualBlock, self).__init__()
        self.stride = stride
        expanded_channels = in_channels * expansion_factor

        layers = []
        # Expansion phase
        if expansion_factor != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU6(inplace=True)
            ])

        # Depthwise phase
        layers.extend([
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True)
        ])

        # Projection phase
        layers.extend([
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)
        self.use_residual = in_channels == out_channels and stride == 1

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



