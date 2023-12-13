import torch
import lightning as L
import torch.nn as nn
import math
import random
import torch.optim.lr_scheduler as lr_scheduler

from torch.cuda.amp import autocast
from torch.nn import functional as F
from layers import GPTTransformerBlock, PositionalEncoding
from dataset import TokenizedTextDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

global MAX_EPOCHS
MAX_EPOCHS = 1

# Collate function outside the dataset class
def collate_fn(batch):
    inputs, targets, masks = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)  # Pad attention masks

    return inputs_padded, targets_padded, masks_padded


class GPTModel(L.LightningModule):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout_rate, vocab_size, batch_size, max_length, trainable_pos_emb=False):
        super(GPTModel, self).__init__()
        self.save_hyperparameters() # Save the model's hyperparameters
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.heads = heads
        self.forward_expansion = forward_expansion
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.trainable_pos_emb = trainable_pos_emb

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 5000, embed_size))  # Add positional embeddings as a parameter

        self.layers = nn.ModuleList([
            GPTTransformerBlock(embed_size, heads, forward_expansion, dropout_rate)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def create_mask(self, mask, current_seq_length):
        if torch.isnan(mask).any() or torch.isinf(mask).any():
            print("NaN or inf value detected in mask at start of create_mask")
        # Expand mask for the number of heads and sequence length
        mask = mask.unsqueeze(1).unsqueeze(2)  # Now [batch_size, 1, 1, seq_len]
        mask = mask.repeat(1, self.heads, current_seq_length, 1)  # Now [batch_size, num_heads, seq_len, seq_len]

        # Reshape to [batch_size * num_heads, seq_len, seq_len]
        mask = mask.view(self.batch_size * self.heads, current_seq_length, current_seq_length)

        if torch.isnan(mask).any() or torch.isinf(mask).any():
            print("NaN or inf value detected in mask at end of create_mask")

        return mask

    def forward(self, x, mask=None):

        x = self.embedding(x)

        current_seq_length = x.size(1)

        # Add positional embeddings
        x = x + self.pos_embedding[:, :current_seq_length]

        # Transpose x to have shape (sequence_length, batch_size, embed_size)
        x = x.transpose(0, 1)

        # Adjust the mask for multi-head attention
        if mask is not None:
            mask = self.create_mask(mask, current_seq_length)

        for layer in self.layers:
            x = layer(x, mask=mask)  # Pass the mask to each layer
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("NaN or inf value detected after layer")

        x = self.output_layer(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN or inf value detected after output layer")

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

    def training_step(self, batch):
        inputs, targets, masks = batch 

        with autocast():  # Enable automatic mixed precision
            outputs = self(inputs, mask=masks)  # Pass the masks to the model
            outputs = outputs.view(-1, self.vocab_size)  # Flatten outputs
            targets = targets.view(-1)  # Flatten targets
            loss = F.cross_entropy(outputs, targets)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch):
        
        inputs, targets, masks = batch  # Unpack the attention masks along with inputs and targets
        with autocast():  # Enable automatic mixed precision
            outputs = self(inputs, mask=masks)  # Pass the masks to the model during forward pass
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.01)

        # Assuming train_dataset is defined and accessible
        train_dataset = TokenizedTextDataset('data/training_data.txt')
        num_batches_per_epoch = len(train_dataset) // self.batch_size
        if len(train_dataset) % self.batch_size != 0:
            num_batches_per_epoch += 1

        # Use the max_epochs variable you have defined
        num_epochs = MAX_EPOCHS
        total_steps = num_epochs * num_batches_per_epoch
        warmup_steps = int(0.1 * total_steps)  # Example: 10% of total steps for warmup

        scheduler = WarmupCosineLR(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # 'step' means the scheduler step is called after every batch
            },
        }


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr_scale = self.last_epoch / self.warmup_steps
        else:
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = 0.5 * (1.0 + torch.cos(torch.tensor(progress, device=self._get_device())))

        return [base_lr * lr_scale + self.min_lr for base_lr in self.base_lrs]

    def _get_device(self):
        return self.optimizer.param_groups[0]['params'][0].device
