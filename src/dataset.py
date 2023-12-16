from torch.utils.data import Dataset
import torch

class TokenizedTextDataset(Dataset):
    def __init__(self, file_path, sequence_length, padding_token=0, in_memory=True):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.padding_token = padding_token
        self.in_memory = in_memory
        self.data = []
        self.line_offsets = []

        if self.in_memory:
            self._load_dataset_into_memory()
        else:
            self._index_file_positions()

    def _load_dataset_into_memory(self):
        with open(self.file_path, 'r') as file:
            for line in file:
                self.data.append(line.strip())

    def _index_file_positions(self):
        with open(self.file_path, 'r') as file:
            offset = 0
            for line in file:
                self.line_offsets.append(offset)
                offset += len(line)

    def __len__(self):
        return len(self.data) if self.in_memory else len(self.line_offsets)

    def __getitem__(self, idx):
        if self.in_memory:
            line = self.data[idx]
        else:
            with open(self.file_path, 'r') as file:
                file.seek(self.line_offsets[idx])
                line = file.readline().strip()

        sequence = list(map(int, line.split()))

        # Padding or truncating the sequence
        if len(sequence) < self.sequence_length:
            sequence += [self.padding_token] * (self.sequence_length - len(sequence))
        else:
            sequence = sequence[:self.sequence_length]

        # Generate an attention mask for the sequence
        attention_mask = [1 if token != self.padding_token else 0 for token in sequence]

        input_sequence = torch.tensor(sequence[:-1], dtype=torch.long)
        target_sequence = torch.tensor(sequence[1:], dtype=torch.long)
        attention_mask = torch.tensor(attention_mask[:-1], dtype=torch.float)

        return input_sequence, target_sequence, attention_mask
