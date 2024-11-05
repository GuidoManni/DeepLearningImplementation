import torch
import torch.nn as nn
from torchsummary import summary
from unet_utils import *


class ContractingPath(nn.Module):
    def __init__(self, in_channels = 3):
        super(ContractingPath, self).__init__()

        # In the original U-Net implementation, there are 4 contracting levels (also called downsampling steps).
        # Starting from the input, each level reduces the spatial dimensions by half and doubles the channel dimensions.

        self.first_level = ConvBlock(in_channels=in_channels, out_channels=64, maxpool=True)
        self.second_level = ConvBlock(in_channels=64, out_channels=128, maxpool=True)
        self.third_level = ConvBlock(in_channels=128, out_channels=256, maxpool=True)
        self.fourth_level = ConvBlock(in_channels=256, out_channels=512, maxpool=True)

        # Then we have the bridge of the Unet
        self.bridge = ConvBlock(in_channels=512, out_channels=1024, maxpool=True)




    def forward(self, x):
        out1, flv1 = self.first_level(x)           # feature level 1 (flv1), out1 is flv1 after maxpooling
        out2, flv2 = self.second_level(out1)       # feature level 2 (flv2), out2 is flv2 after maxpooling
        out3, flv3 = self.third_level(out2)        # feature level 3 (flv3). out3 is flv3 after maxpooling
        out4, flv4 = self.fourth_level(out3)       # feature level 4 (flv4), out4 is flv4 after maxpooling
        _, brdout = self.bridge(out4)              # bridge output (brdout)

        return [flv1, flv2, flv3, flv4, brdout]


class ExpandingPath(nn.Module):
    def __init__(self, n_classes):
        super(ExpandingPath, self).__init__()

        # Symmetrically, there are 4 expanding levels (also called upsampling steps).
        # Starting from the output of the bridge, each level increases the spatial dimensions by two and half the channel dimensions.

        self.cc = CropAndConc()

        self.up1 = UP(in_channels=1024, out_channels=512)
        self.fourth_level = ConvBlock(in_channels=1024, out_channels=512, maxpool=False)
        self.up2 = UP(in_channels=512, out_channels=256)
        self.third_level = ConvBlock(in_channels=512, out_channels=256, maxpool=False)
        self.up3 = UP(in_channels=256, out_channels=128)
        self.second_level = ConvBlock(in_channels=256, out_channels=128, maxpool=False)
        self.up4 = UP(in_channels=128, out_channels=64)
        self.first_level = ConvBlock(in_channels=128, out_channels=64, maxpool=False)

        # output layer
        self.output_layer = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)



    def forward(self, output_features):
        [flv1, flv2, flv3, flv4, brdout] = output_features
        x = self.up1(brdout)
        x = self.cc(flv4, x)
        x = self.fourth_level(x)
        x = self.up2(x)
        x = self.cc(flv3, x)
        x = self.third_level(x)
        x = self.up3(x)
        x = self.cc(flv2, x)
        x = self.second_level(x)
        x = self.up4(x)
        x = self.cc(flv1, x)
        x = self.first_level(x)
        x = self.output_layer(x)

        return x




class UNet(nn.Module):
    def __init__(self, in_channels = 3, n_classes = 2):
        super(UNet, self).__init__()
        self.contracting_path = ContractingPath(in_channels=in_channels)
        self.expanding_path = ExpandingPath(n_classes=n_classes)

    def forward(self, x):
        output_feature = self.contracting_path(x)
        output = self.expanding_path(output_feature)
        return output



if __name__ == "__main__":
    unet = UNet()
    summary(unet, (3, 572, 572))
