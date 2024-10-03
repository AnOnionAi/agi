# util.py
import random
import math
import torch
from dataset import TokenizedTextDataset  

# Initalize with sinusoidal positional encoding but still learnable 
def sinusoidal_positional_encoding(embed_size, max_len):
    pe = torch.zeros(max_len, embed_size)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def find_sequence_length():
    # Load the dataset
    train_dataset = TokenizedTextDataset('data/training_data.txt')

    # Ensure the sample size is not larger than the dataset size
    sample_size = min(10000, len(train_dataset))  # Adjust this value based on your needs

    # Sample a subset of the dataset
    samples = random.sample(list(train_dataset), sample_size)

    # Print out a few samples
    for i in range(5):
        print(samples[i])

    # Compute the maximum length of the samples
    max_len = max(max(tensor.shape[0] for tensor in sample) for sample in samples)
    print("Max Length", max_len)

#if __name__ == "__main__":
   # find_sequence_length()