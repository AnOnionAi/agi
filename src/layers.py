import torch
import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000, learnable=True):
        super(PositionalEncoding, self).__init__()
        self.learnable = learnable

        # Initial sinusoidal positional encoding
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register as a buffer or a learnable parameter
        if self.learnable:
            self.pe = nn.Parameter(pe)  # Learnable positional embeddings
        else:
            self.register_buffer('pe', pe)  # Fixed sinusoidal embeddings

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class GPTTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout_rate):
        super(GPTTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

    def forward(self, x, mask=None):
        attention_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(self.dropout(attention_output) + x)  # Apply dropout after attention
        forward_output = self.feed_forward(x)
        out = self.norm2(self.dropout(forward_output) + x)  # Apply dropout after feed-forward network

        return out
