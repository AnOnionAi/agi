# dataset.py
import torch
import lightning as L
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import os
from gcs_utils import download_blob

# Collate function outside the dataset class
def collate_fn(batch):
    inputs, targets, masks = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)  # Pad attention masks

    return inputs_padded, targets_padded, masks_padded

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

class GPTDataModule(L.LightningDataModule):
    def __init__(self, train_blob_name, val_blob_name, bucket_name, batch_size=32, sequence_length=128, local_data_dir='temp_data'):
        super().__init__()
        self.train_blob_name = train_blob_name
        self.val_blob_name = val_blob_name
        self.bucket_name = bucket_name
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.local_data_dir = local_data_dir

    def setup(self, stage=None):
        """
        Downloads the training and validation data from GCS to a local directory.
        """
        if not os.path.exists(self.local_data_dir):
            os.makedirs(self.local_data_dir)

        # Download training data
        self.train_file = os.path.join(self.local_data_dir, os.path.basename(self.train_blob_name))
        download_blob(self.bucket_name, self.train_blob_name, self.train_file)

        # Download validation data
        self.val_file = os.path.join(self.local_data_dir, os.path.basename(self.val_blob_name))
        download_blob(self.bucket_name, self.val_blob_name, self.val_file)

        # Create dataset instances
        self.train_dataset = TokenizedTextDataset(self.train_file, self.sequence_length)
        self.val_dataset = TokenizedTextDataset(self.val_file, self.sequence_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)

    def teardown(self, stage=None):
        """
        Cleans up the local data directory after training/testing.
        """
        if os.path.exists(self.local_data_dir):
            import shutil
            shutil.rmtree(self.local_data_dir)
