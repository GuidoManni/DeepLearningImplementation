import torch
import torch.nn as nn
from torchsummary import summary

from mobilenets_module import DepthwiseSeprableConv2d, ConvBlock

class MobileNet(nn.Module):
    def __init__(self, num_classes = 1000, alpha = 1.0):
        super(MobileNet, self).__init__()
        output_channel = int(alpha * 32)

        self.input_block = ConvBlock(in_channels = 3, out_channels = output_channel, kernel_size = 3, stride = 2)

        self.intermediate_block1 = nn.Sequential(
            DepthwiseSeprableConv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1),
            ConvBlock(in_channels=output_channel, out_channels=2*output_channel, kernel_size=1, stride=1), # This is similar to the expanding layer of the SqueezeNet
            DepthwiseSeprableConv2d(in_channels=2*output_channel, out_channels=2*output_channel, kernel_size=3, stride=2),
            ConvBlock(in_channels=2*output_channel, out_channels = 4*output_channel, kernel_size=1, stride=1),
            DepthwiseSeprableConv2d(in_channels=4*output_channel, out_channels=4*output_channel, kernel_size=3, stride=1),
            ConvBlock(in_channels=4 * output_channel, out_channels=4 * output_channel, kernel_size=1, stride=1),
            DepthwiseSeprableConv2d(in_channels=4 * output_channel, out_channels=4 * output_channel, kernel_size=3, stride=2),
            ConvBlock(in_channels=4*output_channel, out_channels=8 * output_channel, kernel_size=1, stride=1), # This is similar to the expanding layer of the SqueezeNet
            DepthwiseSeprableConv2d(in_channels=8 * output_channel, out_channels=8 * output_channel, kernel_size=3, stride=1),
            ConvBlock(in_channels=8 * output_channel, out_channels=8 * output_channel, kernel_size=1, stride=1),
            DepthwiseSeprableConv2d(in_channels=8 * output_channel, out_channels=8 * output_channel, kernel_size=3, stride=2),
            ConvBlock(in_channels=8 * output_channel, out_channels=16 * output_channel, kernel_size=1, stride=1),# This is similar to the expanding layer of the SqueezeNet
        )

        intermediate_block = []
        for i in range(5):
            intermediate_block.append(DepthwiseSeprableConv2d(in_channels=16 * output_channel, out_channels=16 * output_channel, kernel_size=3, stride=1))
            intermediate_block.append(ConvBlock(in_channels=16 * output_channel, out_channels=16 * output_channel, kernel_size=1, stride=1))
        self.intermediate_block2 = nn.Sequential(*intermediate_block)

        self.final_block = nn.Sequential(
            DepthwiseSeprableConv2d(in_channels=16 * output_channel, out_channels=16 * output_channel, kernel_size=3,stride=2),
            ConvBlock(in_channels=16 * output_channel, out_channels=32 * output_channel, kernel_size=1, stride=1), # This is similar to the expanding layer of the SqueezeNet
            DepthwiseSeprableConv2d(in_channels=32 * output_channel, out_channels=32 * output_channel, kernel_size=3, stride=2),
            ConvBlock(in_channels=32 * output_channel, out_channels=32 * output_channel, kernel_size=1, stride=1),
        )

        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(in_features=32*output_channel, out_features=1000)
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.intermediate_block1(x)
        x = self.intermediate_block2(x)
        x = self.final_block(x)
        x = self.output_layer(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNet().to(device)
    summary(model, (3, 224, 224))