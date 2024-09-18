import torch
from ran_modules import *
import torch.nn as nn
from torchsummary import summary

class ResidualAttentionModule_92(nn.Module):
    def __init__(self, num_classes, in_channels, out_channels, num_of_updown, t=2, r=1):
        super(ResidualAttentionModule_92, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.residual_unit_1 = ResidualUnit(in_channels=64, out_channels=128)

        self.attention_module_1 = ResidualAttentionModule(in_channels=128, out_channels=128, num_of_updown=num_of_updown, t=t, r=r)

        self.residual_unit_2 = ResidualUnit(in_channels=128, out_channels=256, stride=2)

        self.attention_module_2 = ResidualAttentionModule(in_channels=256, out_channels=256, num_of_updown=num_of_updown, t=t, r=r)

        self.attention_module_3 = ResidualAttentionModule(in_channels=256, out_channels=256, num_of_updown=num_of_updown, t=t, r=r)

        self.residual_unit_3 = ResidualUnit(in_channels=256, out_channels=512, stride=2)

        self.attention_module_4 = ResidualAttentionModule(in_channels=512, out_channels=512, num_of_updown=num_of_updown, t=t, r=r)

        self.attention_module_5 = ResidualAttentionModule(in_channels=512, out_channels=512, num_of_updown=num_of_updown, t=t, r=r)

        self.residual_unit_4 = ResidualUnit(in_channels=512, out_channels=1024, stride=2)


        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )


    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_unit_1(x)
        x = self.attention_module_1(x)
        x = self.residual_unit_2(x)
        x = self.attention_module_2(x)
        x = self.attention_module_3(x)
        x = self.residual_unit_3(x)
        x = self.attention_module_4(x)
        x = self.attention_module_5(x)
        x = self.residual_unit_4(x)


        return x




if __name__ == "__main__":
    model = ResidualAttentionModule_92(num_classes=1000, in_channels=3, out_channels=64, num_of_updown=1, t=2, r=1)
    x = torch.randn(1, 3, 224, 224)
    summary(model, (3, 224, 224))