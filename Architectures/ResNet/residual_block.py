import torch
import torch.nn as nn
class ResidualBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1, self).__init__()
        self.sub_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.sub_block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

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
        x = self.sub_block1(x)
        x = self.sub_block2(x)
        residual = self.shortcut(residual)
        x += residual
        x = self.relu(x)
        return x

class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, stride=1):
        super(ResidualBlock2, self).__init__()
        self.sub_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels1),
            nn.ReLU(inplace=True),
        )
        self.sub_block2 = nn.Sequential(
            nn.Conv2d(out_channels1, out_channels1, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels1),
        )

        self.sub_block3 = nn.Sequential(
            nn.Conv2d(out_channels1, out_channels2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels2),
        )

        # as a shortcut we are using the option B from the paper (projection shortcut)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels2, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels2),
                # we do not use the ReLU activation function in the shortcut because we want to preserve the information flow
            )

        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        x = self.sub_block1(x)
        x = self.sub_block2(x)
        x = self.sub_block3(x)
        residual = self.shortcut(residual)
        x += residual
        x = self.relu(x)
        return x