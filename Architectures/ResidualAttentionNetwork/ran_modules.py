import torch
import torch.nn as nn


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualUnit, self).__init__()
        self.sub_block1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, (4*in_channels), kernel_size=3, stride=stride, padding=1, bias=False),
        )
        self.sub_block2 = nn.Sequential(
            nn.BatchNorm2d(4*in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d((4*in_channels), out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

        # as a shortcut we are using the option B from the paper (projection shortcut)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        x = self.sub_block1(x)
        x = self.sub_block2(x)
        residual = self.shortcut(residual)
        x += residual
        return x



class ResidualAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_of_updown, t=2, r=1):
        super(ResidualAttentionModule, self).__init__()

        # Trunk branch
        residual_unit_trunck_list = []
        for _ in range(t):
            residual_unit_trunck_list.append(ResidualUnit(in_channels, out_channels))
        self.trunk_branch = nn.Sequential(*residual_unit_trunck_list)

        # Initial downsampling
        list_of_initial_downsampling = []
        for i in range(num_of_updown - 1):
            list_of_initial_downsampling.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            for _ in range(r):
                list_of_initial_downsampling.append(ResidualUnit(in_channels, out_channels))
        self.initial_downsampling = nn.Sequential(*list_of_initial_downsampling)

        # Ending downsampling
        list_of_ending_downsampling = []
        list_of_ending_downsampling.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for _ in range(2 * r):
            list_of_ending_downsampling.append(ResidualUnit(in_channels, out_channels))
        self.ending_downsampling = nn.Sequential(*list_of_ending_downsampling)

        # Upsampling
        list_of_upsampling = []
        list_of_upsampling.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        for _ in range(r):
            list_of_upsampling.append(ResidualUnit(in_channels, out_channels))

        for _ in range(num_of_updown - 1):
            list_of_upsampling.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            for _ in range(r):
                list_of_upsampling.append(ResidualUnit(in_channels, out_channels))
        self.upsampling = nn.Sequential(*list_of_upsampling)

        # Skip connection
        self.skip_connection = ResidualUnit(in_channels, out_channels)
        self.sigmoid = nn.Sigmoid()



    def forward(self, x):
        # Trunk branch
        trunk_output = self.trunk_branch(x)

        # Soft mask branch
        downsampled = self.initial_downsampling(x)
        downsampled = self.ending_downsampling(downsampled)
        upsampled = self.upsampling(downsampled)
        soft_mask = self.sigmoid(upsampled)
        print(soft_mask.shape)
        print(trunk_output.shape)

        # Skip connection
        skip_output = self.skip_connection(x)

        # Combining trunk and soft mask branches
        output = trunk_output * (1 + soft_mask) + skip_output
        return output
