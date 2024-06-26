import torch
import torch.nn as nn
from torchsummary import summary

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, out_reduce_3x3, out_3x3, out_reduce_5x5, out_5x5, out_pool):
        super(InceptionModule, self).__init__()

        # 1x1 Convolution
        self.branch1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1, stride=1)

        # 1x1 Convolution -> 3x3 Convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_reduce_3x3, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(out_reduce_3x3, out_3x3, kernel_size=3, stride=1, padding=1)
        )

        # 1x1 Convolution -> 5x5 Convolution
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_reduce_5x5, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(out_reduce_5x5, out_5x5, kernel_size=5, stride=1, padding=2)
        )

        # 3x3 Max Pooling -> 1x1 Convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1, stride=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], 1)

class GoogLeNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GoogLeNet, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
        )

        # Second Convolutional Layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
        )

        # Inception Module 1
        self.inception1 = InceptionModule(in_channels=192, out_1x1=64, out_reduce_3x3=96, out_3x3=128, out_reduce_5x5=16, out_5x5=32, out_pool=32)

        # Inception Module 2
        self.inception2 = InceptionModule(in_channels=256, out_1x1=128, out_reduce_3x3=128, out_3x3=192, out_reduce_5x5=32, out_5x5=96, out_pool=64)

        # Max Pooling Layer
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception Module 3
        self.inception3 = InceptionModule(in_channels=480, out_1x1=192, out_reduce_3x3=96, out_3x3=208, out_reduce_5x5=16, out_5x5=48, out_pool=64)

        # Inception Module 4
        self.inception4 = InceptionModule(in_channels=512, out_1x1=160, out_reduce_3x3=112, out_3x3=224, out_reduce_5x5=24, out_5x5=64, out_pool=64)

        # Inception Module 5
        self.inception5 = InceptionModule(in_channels=512, out_1x1=128, out_reduce_3x3=128, out_3x3=256, out_reduce_5x5=24, out_5x5=64, out_pool=64)

        # Inception Module 6
        self.inception6 = InceptionModule(in_channels=512, out_1x1=112, out_reduce_3x3=144, out_3x3=288, out_reduce_5x5=32, out_5x5=64, out_pool=64)

        # Inception Module 7
        self.inception7 = InceptionModule(in_channels=528, out_1x1=256, out_reduce_3x3=160, out_3x3=320, out_reduce_5x5=32, out_5x5=128, out_pool=128)

        # Max Pooling Layer
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception Module 8
        self.inception8 = InceptionModule(in_channels=832, out_1x1=256, out_reduce_3x3=160, out_3x3=320, out_reduce_5x5=32, out_5x5=128, out_pool=128)

        # Inception Module 9
        self.inception9 = InceptionModule(in_channels=832, out_1x1=384, out_reduce_3x3=192, out_3x3=384, out_reduce_5x5=48, out_5x5=128, out_pool=128)

        # output layer
        self.output_layer = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, num_classes)
        )

        # auxilary classifier 1 -> connect to inception 4
        self.aux1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(512, 128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

        # auxilary classifier 2 -> connected to inception 6
        self.aux2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(528, 128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.max_pool(x)
        x = self.inception3(x)
        x4 = self.inception4(x)
        x = self.inception5(x4)
        x6 = self.inception6(x)
        x = self.inception7(x6)
        x = self.max_pool2(x)
        x = self.inception8(x)
        x = self.inception9(x)
        print(x.size())

        x = self.output_layer(x)

        aux1 = self.aux1(x4)
        aux2 = self.aux2(x6)

        return x, aux1, aux2


if __name__ == "__main__":
    model = GoogLeNet(in_channels=3, num_classes=1000)
    x = torch.randn(1, 3, 224, 224)
    y, aux1, aux2 = model(x)
    print(y.size())
    print(aux1.size())
    print(aux2.size())

    print(summary(model, (3, 224, 224)))

