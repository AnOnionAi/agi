from torch.utils.data import Dataset
import torch

class TokenizedTextDataset(Dataset):
    def __init__(self, file_path, sequence_length=50):
        self.sequence_length = sequence_length
        with open(file_path, 'r') as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        sequence = list(map(int, line.split()))

        # Padding or truncating the sequence to the desired length
        if len(sequence) < self.sequence_length:
            sequence += [0] * (self.sequence_length - len(sequence))  # Padding
        else:
            sequence = sequence[:self.sequence_length]  # Truncating

        input_sequence = torch.tensor(sequence[:-1], dtype=torch.long)
        target_sequence = torch.tensor(sequence[1:], dtype=torch.long)

        return input_sequence, target_sequence
