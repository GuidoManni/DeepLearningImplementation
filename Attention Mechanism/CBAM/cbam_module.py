'''
The Convolutional Block Attention Module (CBAM) is a module that performs attention mechanism on the feature maps.
It consists of two sub-modules: Channel Attention Module and Spatial Attention Module.
- The Channel Attention Module performs attention mechanism on the channel dimension of the feature maps.
- The Spatial Attention Module performs attention mechanism on the spatial dimension of the feature maps.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction):
        super(ChannelAttentionModule, self).__init__()
        '''
        The channel attention module is comprised of:
        - Global Average Pooling & Max Pooling (in parallel)
        - MLP with one hidden layer (shared)
        - The output of the MLP is merged (element-wise addition) 
        - Then we apply the Sigmoid activation function
        '''

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_features=in_channels // reduction, out_features=in_channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)

        avg_pool = avg_pool.view(avg_pool.size(0), -1)
        max_pool = max_pool.view(max_pool.size(0), -1)

        avg_pool = self.mlp(avg_pool)
        max_pool = self.mlp(max_pool)

        channel_attention = self.sigmoid(avg_pool + max_pool).unsqueeze(2).unsqueeze(3)

        return channel_attention * x


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        '''
        The spatial attention module is comprised of:
        - average pooling & max pooling (in parallel)
        - concatenation of the two pooled features
        - 7x7 convolutional layer
        - Sigmoid activation function
        '''

        # Make sure kernel size is odd for same padding
        padding = kernel_size // 2

        # Single conv layer after concatenation
        self.conv = nn.Conv2d(
            in_channels=2,  # 2 because we concatenate avg_pool and max_pool
            out_channels=1,  # Output one attention map
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling along channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (B,1,H,W)

        # Max pooling along channel dimension
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (B,1,H,W)

        # Concatenate along the channel dimension
        concat = torch.cat([avg_pool, max_pool], dim=1)  # (B,2,H,W)

        # Generate attention map
        attention_map = self.conv(concat)  # (B,1,H,W)
        attention_map = self.sigmoid(attention_map)

        return attention_map * x


class CBAMModule(nn.Module):
    def __init__(self, in_channels, reduction):
        super(CBAMModule, self).__init__()
        '''
        The CBAM module is comprised of:
        - Channel Attention Module
        - Spatial Attention Module
        '''

        self.channel_attention = ChannelAttentionModule(in_channels, reduction)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        out = x + out
        return out