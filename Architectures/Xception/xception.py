import torch
import torch.nn as nn
from torchsummary import summary

from xception_blocks import *

class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()

        # Entry Flow
        self.input_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.entry_flow = nn.Sequential(
            XceptionEFlowBlock(in_channels=64, out_channels=128),
            XceptionEFlowBlock(in_channels=128, out_channels=256),
            XceptionEFlowBlock(in_channels=256, out_channels=728)
        )

        # Middle Flow
        middle_layers = []
        for _ in range(8):
            middle_layers.append(XceptionMiddleFlowBlock(in_channels=728))
        self.middle_flow = nn.Sequential(*middle_layers)

        # Exit Flow
        self.exit_flow = nn.Sequential(
            XceptionEFlowBlock(in_channels=728, out_channels=728),
            SeparableConv2d(in_channels=728, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            SeparableConv2d(in_channels=1024, out_channels=1536, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.output_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1536, num_classes)
        )

    def forward(self, x):
        x = self.input_layers(x)
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = self.output_layers(x)
        return x


if __name__ == '__main__':
    model = Xception()
    summary(model, (3, 299, 299))
