# layers.py
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

    def forward(self, x, attention_mask=None):
        # Use attention_mask instead of mask
        if attention_mask is not None:
            attention_mask = attention_mask[0]
            attention_mask = attention_mask.to(dtype=torch.bool)

        attention_output, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = self.norm1(self.dropout(attention_output) + x)
        forward_output = self.feed_forward(x)
        out = self.norm2(self.dropout(forward_output) + x)
        return out