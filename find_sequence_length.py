import random
from torch.utils.data import DataLoader
from src.dataset import TokenizedTextDataset  

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

if __name__ == "__main__":
    find_sequence_length()