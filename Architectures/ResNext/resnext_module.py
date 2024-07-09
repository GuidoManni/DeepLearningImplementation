import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    '''
    This is a simple Convolutional block that consists of Conv2d, BatchNorm2d and ReLU
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(ConvBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv_block(x)


class ResNextBlock(nn.Module):
    '''
    This is the ResNext block that consists of 3 ConvBlocks, the first one is 1x1, the second one is 3x3 grouped and the last one is 1x1.
    The shortcut is projection shortcut (option B from the paper)
    '''
    def __init__(self, in_channels, out_channels_bottleneck, out_channels, groups, stride = 1):
        super(ResNextBlock, self).__init__()

        # first conv is a 1x1 (squeezing the channel)
        self.squeezeconv1x1 = ConvBlock(in_channels=in_channels, out_channels=out_channels_bottleneck, kernel_size=1, stride=1, padding=0)

        # second conv is 3x3 and it is also grouped
        self.grouped_conv3x3 = ConvBlock(in_channels=out_channels_bottleneck, out_channels=out_channels_bottleneck, kernel_size=3, stride=stride, padding=1, groups=groups)

        # third conv is again a 1x1 (expanding the channel)
        self.expandconv1x1 = ConvBlock(in_channels=out_channels_bottleneck, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        # as a shortcut we are using the option B from the paper (projection shortcut)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.squeezeconv1x1(x)
        x = self.grouped_conv3x3(x)
        x = self.expandconv1x1(x)
        x += self.shortcut(residual)
        x = self.relu(x)
        return x



