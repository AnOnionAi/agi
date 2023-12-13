from torch.utils.data import Dataset
import torch

class TokenizedTextDataset(Dataset):
    def __init__(self, file_path, sequence_length=50, padding_token=0):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.padding_token = padding_token
        self.num_lines = self._get_num_lines()

    def _get_num_lines(self):
        with open(self.file_path, 'r') as file:
            return sum(1 for line in file)

    def __len__(self):
        return self.num_lines

    def __getitem__(self, idx):
        with open(self.file_path, 'r') as file:
            for i, line in enumerate(file):
                if i == idx:
                    sequence = list(map(int, line.strip().split()))
                    break

        # Padding or truncating the sequence
        if len(sequence) < self.sequence_length:
            sequence += [self.padding_token] * (self.sequence_length - len(sequence))
        else:
            sequence = sequence[:self.sequence_length]

        # Generate an attention mask for the sequence
        attention_mask = [1 if token != self.padding_token else 0 for token in sequence]

        input_sequence = torch.tensor(sequence[:-1], dtype=torch.long)
        target_sequence = torch.tensor(sequence[1:], dtype=torch.long)
        attention_mask = torch.tensor(attention_mask[:-1], dtype=torch.long)
        attention_mask = attention_mask.to(torch.bool)

        # Convert the 1D attention mask to a 2D attention mask
        attention_mask = attention_mask.unsqueeze(0).repeat(self.sequence_length-1, 1)

        return input_sequence, target_sequence, attention_mask
