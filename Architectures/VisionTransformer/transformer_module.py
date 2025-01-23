'''
Pytorch implementation of Vision Transformer Encoder
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout = 0.1):
        '''
        Transformer Encoder
        - embed_dim: the dimension of the embeddings
        - num_heads: the number of attention heads
        - ff_dim: the dimension of the feedforward network
        - dropout: the dropout rate
        '''
        super(TransformerEncoder, self).__init__()

        # Layer normalization
        self.layer_norm_1 = nn.LayerNorm(normalized_shape = embed_dim)

        # Multi-head self-attention
        self.multi_head_self_attention = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, dropout = dropout)

        # Layer normalization
        self.layer_norm_2 = nn.LayerNorm(normalized_shape = embed_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_features = embed_dim, out_features = ff_dim),
            nn.ReLU(),
            nn.Linear(in_features = ff_dim, out_features = embed_dim)
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequences):
        '''
        Forward pass
        - sequences: the input sequences
        '''
        # Layer normalization
        sequences = self.layer_norm_1(sequences)

        # Multi-head self-attention
        attention_output, _ = self.multi_head_self_attention(sequences, sequences, sequences)

        # Dropout
        attention_output = self.dropout(attention_output)

        # Residual connection
        sequences = sequences + attention_output

        # Layer normalization
        sequences = self.layer_norm_2(sequences)

        # MLP
        mlp_output = self.mlp(sequences)

        # Dropout
        mlp_output = self.dropout(mlp_output)

        # Residual connection
        sequences = sequences + mlp_output

        return sequences

if __name__ == '__main__':
    # Test the transformer encoder
    encoder = TransformerEncoder(embed_dim = 512, num_heads = 8, ff_dim = 2048)
    x = torch.rand(16, 49, 512) # 16 sequences of 49 positions with 512 dimensions
    print(encoder(x).shape) # torch.Size([16, 49, 512])

