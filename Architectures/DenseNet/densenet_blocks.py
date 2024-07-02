import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        return self.conv_block(x)


class DenseBlock(nn.Module):
    '''
    This is the implementation of the variant with the bottleneck
    '''
    def __init__(self, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        out_1x1 = 4 * growth_rate

        self.conv_block1x1 = ConvBlock(in_channels=in_channels, out_channels=out_1x1, kernel_size=1)
        self.conv_block3x3 = ConvBlock(in_channels=out_1x1, out_channels=growth_rate, kernel_size=3, padding=1)


    def forward(self, x):
        out = self.conv_block1x1(x)
        out = self.conv_block3x3(out)
        out = torch.cat([x, out], 1)
        return out



class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1):
        super(TransitionBlock, self).__init__()

        self.transition_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, )
        )

    def forward(self, x):
        return self.transition_block(x)

