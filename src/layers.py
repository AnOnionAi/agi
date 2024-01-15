import torch
import math
import torch.nn as nn

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
