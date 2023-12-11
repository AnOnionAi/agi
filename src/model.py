import torch
import lightning as L
import torch.nn as nn
import math

from torch.nn import functional as F
from layers import GPTTransformerBlock, PositionalEncoding
from dataset import TokenizedTextDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Collate function outside the dataset class
def collate_fn(batch):
    inputs, targets = zip(*batch)
    # Pad sequences to the maximum length of sequences in this batch
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded

class GPTModel(L.LightningModule):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout_rate, vocab_size, batch_size, trainable_pos_emb=False):
        super(GPTModel, self).__init__()
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.heads = heads
        self.forward_expansion = forward_expansion
        self.vocab_size = vocab_size
        train_dataset = TokenizedTextDataset('data/training_data.txt')
        self.max_len = max(len(sample) for sample in train_dataset)
        print("Max Length", self.max_len)
        self.batch_size = batch_size
        self.trainable_pos_emb = trainable_pos_emb

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.pos_embedding = self.init_pos_emb(self.trainable_pos_emb)

        self.layers = nn.ModuleList([
            GPTTransformerBlock(embed_size, heads, forward_expansion, dropout_rate)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def init_pos_emb(self, trainable):
        pos_emb = torch.zeros(1, self.max_len, self.embed_size)
        if trainable:
            pos_emb = nn.Parameter(pos_emb)
        return pos_emb

    def forward(self, x):
        x = self.embedding(x)

        # Get the current sequence length of the batch
        current_seq_length = x.size(1)

        # Generate positional embeddings for the current batch size
        pos_emb = self.pos_embedding_sinusoidal(current_seq_length)

        # Add positional embeddings to input embeddings
        x = x + pos_emb


        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

    def pos_embedding_sinusoidal(self, sequence_length):
        # sequence_length is already the maximum sequence length in the batch
        max_seq_length = sequence_length

        # positions is a tensor containing positions [max_seq_length].
        positions = torch.arange(max_seq_length, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_size, 2).float() * (-math.log(10000.0) / self.embed_size))
        div_term = div_term.to(self.device)
        pos_emb = torch.zeros((max_seq_length, self.embed_size), device=self.device)
        pos_emb[:, 0::2] = torch.sin(positions * div_term)
        
        # Add batch dimension with .unsqueeze
        pos_emb = pos_emb.unsqueeze(0)
        return pos_emb

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        # Reshape outputs to [batch_size * sequence_length, vocab_size]
        outputs = outputs.view(-1, self.vocab_size)
        # Flatten targets to [batch_size * sequence_length]
        targets = targets.view(-1)
        loss = F.cross_entropy(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        print(f"Validation Step - outputs shape: {outputs.shape}, targets shape: {targets.shape}")  # Debug
        outputs = outputs.view(-1, self.vocab_size)  # Flatten outputs
        targets = targets.view(-1)  # Flatten targets
        loss = F.cross_entropy(outputs, targets)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

 

    def train_dataloader(self):
        train_dataset = TokenizedTextDataset('data/training_data.txt')
        return DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        val_dataset = TokenizedTextDataset('data/validation_data.txt')
        return DataLoader(val_dataset, batch_size=self.batch_size, collate_fn=collate_fn, pin_memory=True)

    def configure_optimizers(self):
        # Create the AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.01)
        # Optionally, you can add a learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            },
    }
