import torch
import torch.nn as nn
from torchsummary import summary


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # Second Convolutional Layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # Third Convolutional Layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Fourth Convolutional Layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Fifth Convolutional Layer
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Dense Layer
        self.output = nn.Sequential(
            nn.Linear(128 * 2 * 2, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1000)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.reshape(x, (x.shape[0], 128 * 2 * 2)) # reshaping the tensor for the dense layer
        x = self.output(x)
        return x

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    model = AlexNet().to(device)
    x = torch.randn(1, 3, 224, 224, device=device)
    summary(model, (3, 224, 224))
    print(model(x).shape)





