import torch
from mobilenets_module import *
from torchsummary import summary

class MobileNetV2(nn.Module):
    def __init__(self, in_channels, num_classes, expansion_factor_first_layer=1, expansion_factor=6):
        super(MobileNetV2, self).__init__()

        self.input_layer = ConvBlock(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1)

        self.bottleneck_1 = BottleneckResidualBlock(in_channels=32, out_channels=16, stride=1, expansion_factor=expansion_factor_first_layer)
        self.bottleneck_2 = nn.Sequential(
            BottleneckResidualBlock(in_channels=16, out_channels=24, stride=2, expansion_factor=expansion_factor),
            BottleneckResidualBlock(in_channels=24, out_channels=24, stride=1, expansion_factor=expansion_factor)
        )
        self.bottleneck_3 = nn.Sequential(
            BottleneckResidualBlock(in_channels=24, out_channels=32, stride=2, expansion_factor=expansion_factor),
            BottleneckResidualBlock(in_channels=32, out_channels=32, stride=1, expansion_factor=expansion_factor),
            BottleneckResidualBlock(in_channels=32, out_channels=32, stride=1, expansion_factor=expansion_factor)
        )
        self.bottleneck_4 = nn.Sequential(
            BottleneckResidualBlock(in_channels=32, out_channels=64, stride=2, expansion_factor=expansion_factor),
            BottleneckResidualBlock(in_channels=64, out_channels=64, stride=1, expansion_factor=expansion_factor),
            BottleneckResidualBlock(in_channels=64, out_channels=64, stride=1, expansion_factor=expansion_factor),
            BottleneckResidualBlock(in_channels=64, out_channels=64, stride=1, expansion_factor=expansion_factor)
        )
        self.bottleneck_5 = nn.Sequential(
            BottleneckResidualBlock(in_channels=64, out_channels=96, stride=1, expansion_factor=expansion_factor),
            BottleneckResidualBlock(in_channels=96, out_channels=96, stride=1, expansion_factor=expansion_factor),
            BottleneckResidualBlock(in_channels=96, out_channels=96, stride=1, expansion_factor=expansion_factor)
        )
        self.bottleneck_6 = nn.Sequential(
            BottleneckResidualBlock(in_channels=96, out_channels=160, stride=2, expansion_factor=expansion_factor),
            BottleneckResidualBlock(in_channels=160, out_channels=160, stride=1, expansion_factor=expansion_factor),
            BottleneckResidualBlock(in_channels=160, out_channels=160, stride=1, expansion_factor=expansion_factor)
        )


        # This does not follow the original implementation but it does not really matter
        self.output_layer = nn.Sequential(
            ConvBlock(in_channels=160, out_channels=1280, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
            )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.bottleneck_1(x)
        x = self.bottleneck_2(x)
        x = self.bottleneck_3(x)
        x = self.bottleneck_4(x)
        x = self.bottleneck_5(x)
        x = self.bottleneck_6(x)
        x = self.output_layer(x)

        return x

if __name__ == '__main__':
    model = MobileNetV2(in_channels=3, num_classes=1000)
    summary(model, (3, 224, 224))