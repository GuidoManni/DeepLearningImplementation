import torch
import torch.nn as nn
from torchsummary import summary
from UnetPlusPlus_utils import *


class ContractingPath(nn.Module):
    def __init__(self, in_channels=3):
        super(ContractingPath, self).__init__()

        # Encoder blocks following original U-Net architecture
        # Each level doubles the number of channels and halves spatial dimensions
        # These features will be shared across all nested U-Nets
        self.first_level = ConvBlock(in_channels=in_channels, out_channels=64, maxpool=True)  # X₁,₀
        self.second_level = ConvBlock(in_channels=64, out_channels=128, maxpool=True)  # X₂,₀
        self.third_level = ConvBlock(in_channels=128, out_channels=256, maxpool=True)  # X₃,₀
        self.fourth_level = ConvBlock(in_channels=256, out_channels=512, maxpool=True)  # X₄,₀

        # Bridge (bottom of the U)
        self.bridge = ConvBlock(in_channels=512, out_channels=1024, maxpool=True)  # X₅,₀

    def forward(self, x):
        # Forward pass through encoder, storing features for skip connections
        out1, flv1 = self.first_level(x)  # X₁,₀: First encoder block output
        out2, flv2 = self.second_level(out1)  # X₂,₀: Second encoder block output
        out3, flv3 = self.third_level(out2)  # X₃,₀: Third encoder block output
        out4, flv4 = self.fourth_level(out3)  # X₄,₀: Fourth encoder block output
        _, brdout = self.bridge(out4)  # X₅,₀: Bridge output

        return [flv1, flv2, flv3, flv4, brdout]


class ExpandingPath(nn.Module):
    def __init__(self, n_classes):
        super(ExpandingPath, self).__init__()

        self.cc = CropAndConc()  # For skip connections concatenation

        # Upsampling operations
        self.up1 = UP(in_channels=1024, out_channels=512)  # X₄,₁ upsampling
        self.up2 = UP(in_channels=512, out_channels=256)  # X₃,₁ upsampling
        self.up3 = UP(in_channels=256, out_channels=128)  # X₂,₁ upsampling
        self.up4 = UP(in_channels=128, out_channels=64)  # X₁,₁ upsampling

        # Decoder blocks for each level
        # Following paper's notation Xi,j where:
        # i: decoder level (1-4 from bottom to top)
        # j: block index within level (increases left to right)

        # Fourth level (closest to bridge)
        self.fourth_level = ConvBlock(in_channels=1024, out_channels=512, maxpool=False)

        # Third level blocks
        self.third_level_p1 = ConvBlock(in_channels=512, out_channels=256, maxpool=False) # path 1
        self.third_level_p2 = ConvBlock(in_channels=768, out_channels=256, maxpool=False)  # path 2

        # Second level blocks
        self.second_level_p1 = ConvBlock(in_channels=256, out_channels=128, maxpool=False) # path 1
        self.second_level_p2 = ConvBlock(in_channels=384, out_channels=128, maxpool=False) # path 2
        self.second_level_p3 = ConvBlock(in_channels=512, out_channels=128, maxpool=False) # path 3


        # First level blocks (closest to output)
        self.first_level_p1 = ConvBlock(in_channels=128, out_channels=64, maxpool=False) # path 1
        self.first_level_p2 = ConvBlock(in_channels=192, out_channels=64, maxpool=False) # path 2
        self.first_level_p3 = ConvBlock(in_channels=256, out_channels=64, maxpool=False) # path 3
        self.first_level_p4 = ConvBlock(in_channels=320, out_channels=64, maxpool=False) # path 4


        # Final 1x1 conv for class prediction
        # Separate output layers for deep supervision
        self.output_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1),
            nn.Sigmoid() if n_classes == 1 else nn.Identity()
        )
        self.output_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1),
            nn.Sigmoid() if n_classes == 1 else nn.Identity()
        )
        self.output_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1),
            nn.Sigmoid() if n_classes == 1 else nn.Identity()
        )
        self.output_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1),
            nn.Sigmoid() if n_classes == 1 else nn.Identity()
        )

    def forward(self, output_features):
        [x0_0, x1_0, x2_0, x3_0, x4_0] = output_features

        # Level 4 - Deepest decoder level
        x3_1 = self.up1(x4_0)  # Upsample bridge
        x3_1 = self.cc(x3_0, x3_1)  # Skip connection x3_0 -> x4_0
        x3_1 = self.fourth_level(x3_1)  # Conv operations

        # Level 3: 3 possible paths
        # Path 1:
        x2_1 = self.up2(x3_0)
        x2_1 = self.cc(x2_0, x2_1)  # skip connection x2_0 -> x2_1
        x2_1 = self.third_level_p1(x2_1)

        # Path 2
        x2_2 = self.up2(x3_1)
        x2_2 = self.cc(x2_0, x2_1, x2_2)  # skip connection x2_1 -> x2_2
        x2_2 = self.third_level_p2(x2_2)

        # Level 2: 3 possible paths
        # Path 1:
        x1_1 = self.up3(x2_0)
        x1_1 = self.cc(x1_0, x1_1)  # skip connection x1_0 -> x1_1
        x1_1 = self.second_level_p1(x1_1)

        # Path 2:
        x1_2 = self.up3(x2_1)
        x1_2 = self.cc(x1_0, x1_1, x1_2)
        x1_2 = self.second_level_p2(x1_2)

        # Path 3:
        x1_3 = self.up3(x2_2)
        x1_3 = self.cc(x1_0, x1_1, x1_2, x1_3)
        x1_3 = self.second_level_p3(x1_3)

        # Level 1: 4 possible paths

        # Path 1:
        x0_1 = self.up4(x1_0)
        x0_1 = self.cc(x0_0, x0_1)
        x0_1 = self.first_level_p1(x0_1)

        # Path 2:
        x0_2 = self.up4(x1_1)
        x0_2 = self.cc(x0_0, x0_1, x0_2)
        x0_2 = self.first_level_p2(x0_2)

        # Path 3:
        x0_3 = self.up4(x1_2)
        x0_3 = self.cc(x0_0, x0_1, x0_2, x0_3)
        x0_3 = self.first_level_p3(x0_3)

        # Path 4:
        x0_4 = self.up4(x1_3)
        x0_4 = self.cc(x0_0, x0_1, x0_2, x0_3, x0_4)
        x0_4 = self.first_level_p4(x0_4)

        # Output Path for Deep Supervision
        out1 = self.output_layer1(x0_1)
        out2 = self.output_layer2(x0_2)
        out3 = self.output_layer3(x0_3)
        out4 = self.output_layer4(x0_4)

        return out1, out2, out3, out4




class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, n_classes=2):
        super(UNetPlusPlus, self).__init__()
        self.contracting_path = ContractingPath(in_channels=in_channels)
        self.expanding_path = ExpandingPath(n_classes=n_classes)

    def forward(self, x):
        output_feature = self.contracting_path(x)
        output = self.expanding_path(output_feature)
        return output


if __name__ == "__main__":
    unet = UNetPlusPlus()
    summary(unet, (3, 572, 572))
