import torch
import torch.nn as nn
from torchsummary import summary
from unet3D_utils import *

'''
This implementation follows the original paper "Attention U-Net: Learning Where to Look for the Pancreas" 
which uses 3D convolutions for volumetric data (CT scans).

Key differences from the original paper:

1. Deep Supervision: The original paper uses deep supervision (mentioned in Section 2) which is not implemented
   in this version for simplicity.
'''

class ContractingPath(nn.Module):
    def __init__(self, in_channels=1):  # Changed default in_channels to 1 for medical images
        super(ContractingPath, self).__init__()

        self.first_level = ConvBlock(in_channels=in_channels, out_channels=64, maxpool=True)
        self.second_level = ConvBlock(in_channels=64, out_channels=128, maxpool=True)
        self.third_level = ConvBlock(in_channels=128, out_channels=256, maxpool=True)
        self.fourth_level = ConvBlock(in_channels=256, out_channels=512, maxpool=True)

        self.bridge = ConvBlock(in_channels=512, out_channels=1024, maxpool=True)

    def forward(self, x):
        out1, flv1 = self.first_level(x)
        out2, flv2 = self.second_level(out1)
        out3, flv3 = self.third_level(out2)
        out4, flv4 = self.fourth_level(out3)
        _, brdout = self.bridge(out4)

        return [flv1, flv2, flv3, flv4, brdout]

class ExpandingPath(nn.Module):
    def __init__(self, n_classes):
        super(ExpandingPath, self).__init__()

        self.cc = CropAndConc()

        # Add attention gates before each skip connection
        self.attention1 = AttentionGate(input_channels=512, gate_channels=512, intermediate_channels=512)
        self.attention2 = AttentionGate(input_channels=256, gate_channels=256, intermediate_channels=256)
        self.attention3 = AttentionGate(input_channels=128, gate_channels=128, intermediate_channels=128)
        self.attention4 = AttentionGate(input_channels=64, gate_channels=64, intermediate_channels=64)

        self.up1 = UP(in_channels=1024, out_channels=512)
        self.fourth_level = ConvBlock(in_channels=1024, out_channels=512, maxpool=False)
        self.up2 = UP(in_channels=512, out_channels=256)
        self.third_level = ConvBlock(in_channels=512, out_channels=256, maxpool=False)
        self.up3 = UP(in_channels=256, out_channels=128)
        self.second_level = ConvBlock(in_channels=256, out_channels=128, maxpool=False)
        self.up4 = UP(in_channels=128, out_channels=64)
        self.first_level = ConvBlock(in_channels=128, out_channels=64, maxpool=False)

        self.output_layer = nn.Conv3d(in_channels=64, out_channels=n_classes, kernel_size=1)

    def forward(self, output_features):
        [flv1, flv2, flv3, flv4, brdout] = output_features

        x = self.up1(brdout)
        gated_flv4 = self.attention1(x, flv4)
        x = self.cc(gated_flv4, x)
        x = self.fourth_level(x)

        x = self.up2(x)
        gated_flv3 = self.attention2(x, flv3)
        x = self.cc(gated_flv3, x)
        x = self.third_level(x)

        x = self.up3(x)
        gated_flv2 = self.attention3(x, flv2)
        x = self.cc(gated_flv2, x)
        x = self.second_level(x)

        x = self.up4(x)
        gated_flv1 = self.attention4(x, flv1)
        x = self.cc(gated_flv1, x)
        x = self.first_level(x)
        x = self.output_layer(x)

        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2):  # Changed default in_channels to 1
        super(UNet, self).__init__()
        self.contracting_path = ContractingPath(in_channels=in_channels)
        self.expanding_path = ExpandingPath(n_classes=n_classes)

    def forward(self, x):
        output_feature = self.contracting_path(x)
        output = self.expanding_path(output_feature)
        return output

if __name__ == "__main__":
    unet = UNet()
    # Modified input size for 3D (channels, depth, height, width)
    summary(unet, (1, 64, 128, 128))