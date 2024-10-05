import torch


def test_masks():
    batch_size = 2
    seq_len = 5
    embed_size = 768
    #model = GPTModel(...)  # Initialize with appropriate parameters
    model = None #GPTModel
    x = torch.randint(0, 50233, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, 1, seq_len)

    output = model(x, attention_mask=attention_mask)
    assert output.shape == (batch_size, seq_len, 50233), "Output shape mismatch"

test_masks()
