import torch
import torch.nn.functional as F
import numpy as np
def patchify(images, n_patches):
    '''
    Split the image into patches
    - images: input images
    - n_patches: number of patches to split the image into
    '''
    # Get the image size
    n, c, h, w = images.size()  # n = batch size, c = number of channels, h = height, w = width

    assert h == w, 'Input image must be square'

    # Calculate patch size
    patch_size = h // n_patches

    # Use unfold to create patches
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(n, c, n_patches, n_patches, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(n, n_patches * n_patches, -1)

    return patches


def get_positional_embeddings(sequence_length, d):
    '''
    Generate sinusoidal positional embeddings.
    - sequence_length: the length of the sequence (number of positions)
    - d: the dimension of the embeddings
    '''
    # Initialize a tensor to store the positional embeddings
    result = torch.ones(sequence_length, d)

    # Iterate over each position in the sequence
    for i in range(sequence_length):
        # Iterate over each dimension of the embedding
        for j in range(d):
            # Calculate the positional embedding value
            # Use sine for even indices and cosine for odd indices
            if j % 2 == 0:
                result[i][j] = np.sin(i / (10000 ** (j / d)))
            else:
                result[i][j] = np.cos(i / (10000 ** ((j - 1) / d)))

    return result