import torch
import torch.nn as nn
from torchsummary import summary

from squeezenet_blocks import *

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(SqueezeNet, self).__init__()

        self.input_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fire_block2 = FireModule(in_channels=96, s1x1=16, e1x1=64, e3x3=64)
        self.fire_block3 = FireModule(in_channels=128, s1x1=16, e1x1=64, e3x3=64)
        self.fire_block4 = FireModule(in_channels=128, s1x1=32, e1x1=128, e3x3=128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fire_block5 = FireModule(in_channels=256, s1x1=32, e1x1=128, e3x3=128)
        self.fire_block6 = FireModule(in_channels=256, s1x1=48, e1x1=192, e3x3=192)
        self.fire_block7 = FireModule(in_channels=384, s1x1=48, e1x1=192, e3x3=192)
        self.fire_block8 = FireModule(in_channels=384, s1x1=64, e1x1=256, e3x3=256)

        self.maxpool8 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fire_block9 = FireModule(in_channels=512, s1x1=64, e1x1=256, e3x3=256)

        self.output_layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.input_layers(x)
        x = self.fire_block2(x)
        bypass1 = x
        x = self.fire_block3(x)
        x = x + bypass1
        x = self.fire_block4(x)
        x = self.maxpool4(x)
        bypass2 = x
        x = self.fire_block5(x)
        x = x + bypass2
        x = self.fire_block6(x)
        bypass3 = x
        x = self.fire_block7(x)
        x = x + bypass3
        x = self.fire_block8(x)
        x = self.maxpool8(x)
        bypass4 = x
        x = self.fire_block9(x)
        x = x + bypass4
        x = self.output_layers(x)
        return x

if __name__ == '__main__':
    model = SqueezeNet(num_classes=1000)
    summary(model, (3, 224, 224))
    