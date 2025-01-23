import torch
import torch.nn as nn
from torchsummary import summary

from transformer_module import TransformerEncoder
from transformer_utils import patchify, get_positional_embeddings


class VanillaViT(nn.Module):
    def __init__(self, input_dim=(3, 128, 128), n_patches=8, embed_dim=512, num_classes=1000):
        super(VanillaViT, self).__init__()
        self.channels, self.height, self.width = input_dim
        self.n_patches = n_patches
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        assert self.height % self.n_patches == 0, "Input height not entirely divisible by number of patches"
        assert self.width % self.n_patches == 0, "Input width not entirely divisible by number of patches"

        # The transformer has a linear mapper to embed the patches
        self.linear_mapper = nn.Linear(
            in_features=self.channels * (self.height // self.n_patches) * (self.width // self.n_patches),
            out_features=self.embed_dim
        )

        # The learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, 1, self.embed_dim))

        # Positional embeddings
        self.positional_embeddings = nn.Parameter(
            get_positional_embeddings(self.n_patches * self.n_patches + 1, self.embed_dim)
        )
        self.positional_embeddings.requires_grad = False

        # The transformer encoder
        self.transformer_encoder = TransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=8,
            ff_dim=2048,
            dropout=0.1
        )

        # The MLP head for classification
        self.mlp_head = nn.Linear(self.embed_dim, self.num_classes)



    def forward(self, x):
        # Split the image into patches
        patches = patchify(x, self.n_patches)

        # Embed the patches
        embeddings = self.linear_mapper(patches) # values (v)

        # Add the class token
        class_token = self.class_token.expand(x.size(0), -1, -1) # keys (k)
        embeddings = torch.cat([class_token, embeddings], dim = 1)

        # Add the positional embeddings
        embeddings += self.positional_embeddings # queries (q)
        # Run the transformer encoder
        embeddings = self.transformer_encoder(embeddings)

        class_representation = embeddings[:, 0, :] # Get the class token representation

        output = self.mlp_head(class_representation) # Classify the class token representation

        return output

if __name__ == '__main__':
    # Test the vanilla ViT
    model = VanillaViT(input_dim=(3, 128, 128), n_patches=8, embed_dim=512, num_classes=1000).to('cuda')
    x = torch.rand(16, 3, 128, 128).to('cuda') # 16 images of 3 channels with 128x128 resolution
    output = model(x)
    print(output.size())

    summary(model, (3, 128, 128))

