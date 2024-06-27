import torch
import torch.nn as nn
from torchsummary import summary

class VGG16Block1(nn.Module):
    def __init__(self, in_channels, out_conv):
        super(VGG16Block1, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels=out_conv, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = out_conv, out_channels=out_conv, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class VGG16Block2(nn.Module):
    def __init__(self, in_channels, out_conv):
        super(VGG16Block2, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_conv, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_conv, out_channels=out_conv, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_conv, out_channels=out_conv, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class VGG16(nn.Module):
    def __init__(self, n_classes):
        super(VGG16, self).__init__()

        # First Block
        self.block1 = VGG16Block1(in_channels=3, out_conv=64)

        # Second Block
        self.block2 = VGG16Block1(in_channels=64, out_conv=128)

        # Third Block
        self.block3 = VGG16Block2(in_channels=128, out_conv=256)

        # Fourth Block
        self.block4 = VGG16Block2(in_channels=256, out_conv=512)

        # Fifth Block
        self.block5 = VGG16Block2(in_channels=512, out_conv=512)

        # Output Block
        self.output_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=25088, out_features=4096),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=n_classes)
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.output_layers(x)

        return x

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VGG16(n_classes=1000).to(device)
    x = torch.randn(1, 3, 224, 224, device=device)
    summary(model, (3, 224, 224))
    print(model(x).shape)

