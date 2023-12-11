from torch.utils.data import Dataset
import torch

class TokenizedTextDataset(Dataset):
    def __init__(self, file_path, sequence_length=50):
        # Load and process the data
        self.sequence_length = sequence_length
        with open(file_path, 'r') as file:
            tokenized_text = file.readlines()
        self.data = [int(token.strip()) for token in tokenized_text]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Adjust to return a consistent sequence of tokens
        start_idx = idx * self.sequence_length
        end_idx = start_idx + self.sequence_length
        sequence = self.data[start_idx:end_idx]
        target_sequence = self.data[start_idx + 1:end_idx + 1]

        if len(sequence) < self.sequence_length:
            # Add padding if needed
            sequence.extend([0] * (self.sequence_length - len(sequence)))
            target_sequence.extend([0] * (self.sequence_length - len(target_sequence)))

        return torch.tensor(sequence), torch.tensor(target_sequence)
