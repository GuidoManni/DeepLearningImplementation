import torch
import torch.nn as nn
from torchsummary import summary


class Generator(nn.Module):
    def __init__(self, z_dim=100, output_dim=28):
        super(Generator, self).__init__()
        # Calculate total dimensions for output (e.g., 28*28=784 for MNIST)
        out_features = output_dim * output_dim
        self.output_dim = output_dim
        self.z_dim = z_dim

        # Build the MLP architecture from the original GAN paper:
        # z_dim -> 240 -> 784 (reshaped to 28x28)
        self.mlp = nn.Sequential(
            # First layer transforms from latent space to hidden layer
            nn.Linear(in_features=z_dim, out_features=240),
            nn.ReLU(),  # ReLU activation as per original paper

            # Output layer transforms to flattened image dimensions
            nn.Linear(in_features=240, out_features=out_features),
            nn.Sigmoid()  # Sigmoid ensures output is in [0,1] range for images
        )

    def sample_z(self, batch_size):
        # Sample from uniform distribution [-1, 1]
        # Shape: (batch_size, z_dim)
        z = 2 * torch.rand(batch_size, self.z_dim) - 1
        return z

    def to_image(self, out_flattened):
        # Reshape flat tensor to image format:
        # (batch_size, 784) -> (batch_size, 1, 28, 28)
        return out_flattened.view(out_flattened.size(0), 1, self.output_dim, self.output_dim)

    def forward(self, batch_size=None, z=None):
        # Allow both automatic sampling and manual z input
        if z is None:
            if batch_size is None:
                raise ValueError("Must provide either batch_size or z")
            # Sample random noise if z not provided
            z = self.sample_z(batch_size)

        # Generate flattened images
        out_flattened = self.mlp(z)

        # Reshape to proper image format
        out = self.to_image(out_flattened)
        return out



if __name__ == '__main__':
    gen = Generator(z_dim=100, output_dim=28)
    summary(gen)

