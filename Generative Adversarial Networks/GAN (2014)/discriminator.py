import torch
import torch.nn as nn
from torchsummary import summary


class Discriminator(nn.Module):
    def __init__(self, input_dim=28):
        super(Discriminator, self).__init__()
        # Calculate input features (e.g., 28*28=784 for MNIST)
        in_features = input_dim * input_dim

        # Build the MLP architecture from original GAN paper:
        # 784 -> 240 -> 1
        self.mlp = nn.Sequential(
            # Flatten 2D image to 1D vector
            nn.Flatten(),  # (batch_size, 1, 28, 28) -> (batch_size, 784)

            # First layer transforms flattened image to hidden representation
            nn.Linear(in_features=in_features, out_features=240),
            nn.ReLU(),  # ReLU activation as per original paper

            # Output layer produces single scalar
            nn.Linear(in_features=240, out_features=1),
            nn.Sigmoid()  # Sigmoid to get probability in [0,1]
        )

    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        # output shape: (batch_size, 1)
        return self.mlp(x)

if __name__ == '__main__':
    disc = Discriminator(input_dim=28)
    summary(disc)
