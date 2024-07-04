import torch
import torch.nn as nn

class FireModule(nn.Module):
    def __init__(self, in_channels, s1x1, e1x1, e3x3):
        '''
        :param in_channels: the input channel of the FireModule
        :param s1x1: the output channel of the squeeze layer
        :param e1x1: the output channel of the expand 1x1 layer
        :param e3x3: the output channel of the expand 3x3 layer
        '''
        super(FireModule, self).__init__()

        self.squeeze = nn.Conv2d(in_channels=in_channels, out_channels=s1x1, kernel_size=1)
        self.expand1x1 = nn.Conv2d(in_channels=s1x1, out_channels=e1x1, kernel_size=1)
        self.expand3x3 = nn.Conv2d(in_channels=s1x1, out_channels=e3x3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.squeeze(x)
        x = torch.cat([self.expand1x1(x), self.expand3x3(x)], 1)
        return x

