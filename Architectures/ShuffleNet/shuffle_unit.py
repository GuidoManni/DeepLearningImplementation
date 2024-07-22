import torch
import torch.nn as nn


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        # transpose
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batch_size, -1, height, width)
        return x



class ShuffleUnitNoStride(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super(ShuffleUnitNoStride, self).__init__()
        mid_channels = out_channels // 4

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, groups=groups),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            ChannelShuffle(groups),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1,),
            nn.BatchNorm2d(out_channels)
        )

        self.skip_connection = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.bottleneck(x)
        residual = self.skip_connection(residual)
        x += residual
        return x

class ShuffleUnitWithStride(nn.Module):
    def __init__(self, in_channels, out_channels, groups, stride=2):
        super(ShuffleUnitWithStride, self).__init__()
        out_channels = out_channels - in_channels
        mid_channels = out_channels // 4


        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            ChannelShuffle(groups),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, stride=stride, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.skip_connection = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        residual = x
        x = self.bottleneck(x)
        residual = self.skip_connection(residual)
        x = torch.cat((residual, x), dim=1)
        return x


