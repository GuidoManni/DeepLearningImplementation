import torch
import torch.nn as nn



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        '''
        A separable convolution block is a combination of depthwise convolution and pointwise convolution. This means
        that the input tensor is convolved with a kernel of size (kernel_size, kernel_size) and then the output of this
        operation is convolved with a 1x1 kernel. This is done to reduce the number of parameters in the model.
        '''
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class XceptionEFlowBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(XceptionEFlowBlock, self).__init__()
        self.main_branch = nn.Sequential(
            SeparableConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            SeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.skip_connection = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x):
        x = self.main_branch(x) + self.skip_connection(x)
        return x


class XceptionMiddleFlowBlock(nn.Module):
    def __init__(self, in_channels):
        super(XceptionMiddleFlowBlock, self).__init__()
        self.main_branch = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            SeparableConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            SeparableConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        x = self.main_branch(x) + x
        return x



