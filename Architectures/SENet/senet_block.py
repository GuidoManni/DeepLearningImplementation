import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)


class SEBlock(nn.Module):
    def __init__(self, in_feature, reduction):
        super(SEBlock, self).__init__()
        bootleneck_feature = int(in_feature / reduction)
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features= in_feature, out_features=bootleneck_feature),
            nn.ReLU(),
            nn.Linear(in_features=bootleneck_feature, out_features=in_feature),
            nn.Sigmoid(),
        )

    def forward(self, x):
        initial_feature = x
        x = self.se_block(x)
        scaled_feature = initial_feature * torch.reshape(x, (x.shape[0], x.shape[1], 1, 1))
        return scaled_feature


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels_bottleneck, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.bootleneck = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels_bottleneck, kernel_size=1),
            ConvBlock(in_channels=out_channels_bottleneck, out_channels=out_channels_bottleneck, kernel_size=3, stride=stride, padding=1),
            ConvBlock(in_channels=out_channels_bottleneck, out_channels=out_channels, kernel_size=1)
        )
        self.shortcut = nn.Identity()
        if stride!=1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
                # we do not use the ReLU activation function in the shortcut because we want to preserve the information flow
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.bootleneck(x)
        x += self.shortcut(residual)
        x = self.relu(x)
        return x
