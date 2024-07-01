'''
Inception-v3 includes all the improvements of Inception-v2, plus the following additional enhancements:
- RMSProp optimizer
- BatchNorm in the auxiliary classifiers
- Label smoothing regularization
Since these changes are more about the training process than the architecture itself, we will directly use the Inception-v3.
'''

import torch
import torch.nn as nn
from torchsummary import summary

from inception_blocks import *

class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionV3, self).__init__()

        # Input layers
        self.input_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=80, out_channels=192, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Inception Blocks5 (from Figure 5 of the original paper)
        self.inception_block5_1 = InceptionBlock5(in_channels=192, pool_features=32)
        self.inception_block5_2 = InceptionBlock5(in_channels=256, pool_features=64)
        self.inception_block5_3 = InceptionBlock5(in_channels=288, pool_features=64)

        # Before the next inception block, we need to add a reduction block
        self.reduction_block1 = ReductionBlock(in_channels=288, out_channels1=[384, 384], out_channels2=[64, 96, 96], is_second=False)


        # Inception Blocks6 (from Figure 6 of the original paper)
        self.inception_block6_1 = InceptionBlock6(in_channels=768, out_1x1=192, red_7x7=160, out_7x7=192, red_1x7_7x1=160, out_1x7_7x1=192, out_pool=192)
        self.inception_block6_2 = InceptionBlock6(in_channels=768, out_1x1=192, red_7x7=160, out_7x7=192, red_1x7_7x1=160, out_1x7_7x1=192, out_pool=192)
        self.inception_block6_3 = InceptionBlock6(in_channels=768, out_1x1=192, red_7x7=160, out_7x7=192, red_1x7_7x1=160, out_1x7_7x1=192, out_pool=192)
        self.inception_block6_4 = InceptionBlock6(in_channels=768, out_1x1=192, red_7x7=160, out_7x7=192, red_1x7_7x1=160, out_1x7_7x1=192, out_pool=192)
        self.inception_block6_5 = InceptionBlock6(in_channels=768, out_1x1=192, red_7x7=160, out_7x7=192, red_1x7_7x1=160, out_1x7_7x1=192, out_pool=192)

        # Before the next inception block, we need to add a reduction block
        self.reduction_block2 = ReductionBlock(in_channels=768, out_channels1=[192, 320], out_channels2=[192, 192, 192, 192], is_second=True)

        # Inception Blocks7 (from Figure 7 of the original paper)
        self.inception_block7_1 = InceptionBlock7(in_channels=1280)
        self.inception_block7_2 = InceptionBlock7(in_channels=2048)

        # Output layers
        self.output_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )


    def forward(self, x):
        x = self.input_layers(x)
        x = self.inception_block5_1(x)
        x = self.inception_block5_2(x)
        x = self.inception_block5_3(x)
        x = self.reduction_block1(x)
        x = self.inception_block6_1(x)
        x = self.inception_block6_2(x)
        x = self.inception_block6_3(x)
        x = self.inception_block6_4(x)
        x = self.inception_block6_5(x)
        x = self.reduction_block2(x)
        x = self.inception_block7_1(x)
        x = self.inception_block7_2(x)
        x = self.output_layers(x)
        
        return x

if __name__ == '__main__':
    model = InceptionV3()
    summary(model, (3, 299, 299))