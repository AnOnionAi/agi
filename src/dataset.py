from torch.utils.data import Dataset
import torch

class TokenizedTextDataset(Dataset):
    def __init__(self, file_path, sequence_length=50):
        # Load and process the data
        self.sequence_length = sequence_length
        with open(file_path, 'r') as file:
            lines = file.readlines()
        self.data = [list(map(int, line.strip().split())) for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the token sequence for the requested index
        sequence = self.data[idx]
        # Handle sequences shorter than the desired sequence length
        padded_sequence = sequence + [0] * (self.sequence_length - len(sequence))
        # Prepare input and target sequences
        input_sequence = padded_sequence[:-1]
        target_sequence = padded_sequence[1:]
        return torch.tensor(input_sequence), torch.tensor(target_sequence)
