import torch
from torchsummary import summary

from shuffle_unit import *


class ShuffleNet(nn.Module):
    def __init__(self, num_classes=100, s=1, groups=1):
        super(ShuffleNet, self).__init__()
        self.s = s
        if groups == 1:
            out_channels = [144, 288, 576]
        elif groups == 2:
            out_channels = [200, 400, 800]
        elif groups == 3:
            out_channels = [240, 480, 960]
        elif groups == 4:
            out_channels = [272, 544, 1088]
        elif groups == 8:
            out_channels = [384, 768, 1536]

        out_channels = self._perform_channel_reduction(out_channels)

        self.initial_block = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage2 = nn.Sequential(
            ShuffleUnitWithStride(24, out_channels[0], groups, stride=2),
            ShuffleUnitNoStride(out_channels[0], out_channels[0], groups),
            ShuffleUnitNoStride(out_channels[0], out_channels[0], groups),
            ShuffleUnitNoStride(out_channels[0], out_channels[0], groups),
        )

        self.stage3 = nn.Sequential(
            ShuffleUnitWithStride(out_channels[0], out_channels[1], groups),
            ShuffleUnitNoStride(out_channels[1], out_channels[1], groups),
            ShuffleUnitNoStride(out_channels[1], out_channels[1], groups),
            ShuffleUnitNoStride(out_channels[1], out_channels[1], groups),
            ShuffleUnitNoStride(out_channels[1], out_channels[1], groups),
            ShuffleUnitNoStride(out_channels[1], out_channels[1], groups),
            ShuffleUnitNoStride(out_channels[1], out_channels[1], groups),
            ShuffleUnitNoStride(out_channels[1], out_channels[1], groups),
        )

        self.stage4 = nn.Sequential(
            ShuffleUnitWithStride(out_channels[1], out_channels[2], groups),
            ShuffleUnitNoStride(out_channels[2], out_channels[2], groups),
            ShuffleUnitNoStride(out_channels[2], out_channels[2], groups),
            ShuffleUnitNoStride(out_channels[2], out_channels[2], groups),
        )

        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels[2], num_classes),
        )

    def _perform_channel_reduction(self, out_channels):
        out_channels = torch.Tensor(out_channels)
        out_channels = out_channels * self.s
        return out_channels.int().tolist()



    def forward(self, x):
        x = self.initial_block(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.output_layer(x)
        return x

if __name__ == '__main__':
    model = ShuffleNet()
    summary(model, (3, 224, 224))