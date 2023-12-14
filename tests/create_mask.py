import torch

def create_mask(mask, batch_size, heads, current_seq_length):
    mask = mask.unsqueeze(1)  # Now [batch_size, 1, seq_len]
    mask = mask.repeat(1, heads, 1)  # Now [batch_size, num_heads, seq_len]
    mask = mask.view(batch_size * heads, 1, current_seq_length)  # Now [batch_size*num_heads, 1, seq_len]
    mask = mask.repeat(1, current_seq_length, 1)  # Now [batch_size*num_heads, seq_len, seq_len]
    return mask

def test_create_mask():
    batch_size = 2
    heads = 4
    seq_length = 5
    mask = torch.ones(batch_size, seq_length)
    output_mask = create_mask(mask, batch_size, heads, seq_length)
    assert output_mask.shape == (batch_size * heads, seq_length, seq_length)

# Running the test
test_create_mask()
