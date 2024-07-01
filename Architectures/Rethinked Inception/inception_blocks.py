import torch
import torch.nn as nn


class ReductionBlock(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, is_second=False):
        super(ReductionBlock, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels1[0], kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels1[0], out_channels1[1], kernel_size=3, stride=2),
            nn.ReLU(inplace=True)
        )

        if not is_second:
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels2[0], kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels2[0], out_channels2[1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels2[1], out_channels2[2], kernel_size=3, stride=2),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels2[0], kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels2[0], out_channels2[1], kernel_size=(1, 7), stride=1, padding=(0, 3)),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels2[1], out_channels2[2], kernel_size=(7, 1), stride=1, padding=(3, 0)),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels2[2], out_channels2[3], kernel_size=3, stride=2),
                nn.ReLU(inplace=True)
            )

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Calculate the number of channels after concatenation
        total_channels = out_channels1[1] + out_channels2[-1] + in_channels

        # Add a 1x1 conv to adjust the number of output channels if necessary
        if is_second:
            self.adjust_channels = nn.Conv2d(total_channels, 1280, kernel_size=1, stride=1)
        else:
            self.adjust_channels = None

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        concat = torch.cat([branch1, branch2, branch3], 1)

        if self.adjust_channels:
            return self.adjust_channels(concat)
        else:
            return concat

class InceptionBlock5(nn.Module):
    '''
    This is the inception block displayed in Figure 5 of the paper.
    '''
    def __init__(self, in_channels, pool_features):
        super(InceptionBlock5, self).__init__()

        # this branch replace the 5x5 convolution with two 3x3 convolutions
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=48, kernel_size=1, stride=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=pool_features, kernel_size=1, stride=1),
            nn.BatchNorm2d(pool_features),
            nn.ReLU(),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], 1)


class InceptionBlock6(nn.Module):
    '''
    This is the inception block displayed in Figure 6 of the paper.
    '''
    def __init__(self, in_channels,
                 out_1x1=64,
                 red_7x7=64, out_7x7=96,
                 red_1x7_7x1=64, out_1x7_7x1=96,
                 out_pool=64):
        super(InceptionBlock6, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_7x7, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_7x7, red_7x7, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_7x7, out_7x7, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_1x7_7x1, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_1x7_7x1, red_1x7_7x1, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_1x7_7x1, red_1x7_7x1, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_1x7_7x1, red_1x7_7x1, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_1x7_7x1, out_1x7_7x1, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], 1)


class InceptionBlock7(nn.Module):
    '''
    This is the inception block displayed in Figure 7 of the paper,
    also known as InceptionE in some implementations.
    '''

    def __init__(self, in_channels):
        super(InceptionBlock7, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 320, kernel_size=1, stride=1),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True)
        )

        self.branch2_conv = nn.Sequential(
            nn.Conv2d(in_channels, 384, kernel_size=1, stride=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.branch2a = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.branch2b = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )

        self.branch3_conv = nn.Sequential(
            nn.Conv2d(in_channels, 448, kernel_size=1, stride=1),
            nn.BatchNorm2d(448),
            nn.ReLU(inplace=True),
            nn.Conv2d(448, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.branch3a = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.branch3b = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 192, kernel_size=1, stride=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)

        branch2 = self.branch2_conv(x)
        branch2 = torch.cat([self.branch2a(branch2), self.branch2b(branch2)], 1)

        branch3 = self.branch3_conv(x)
        branch3 = torch.cat([self.branch3a(branch3), self.branch3b(branch3)], 1)

        branch4 = self.branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], 1)