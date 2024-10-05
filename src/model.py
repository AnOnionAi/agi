# model.py
import torch
import lightning as L
import torch.nn as nn

from torch.nn import functional as F
from layers import GPTTransformerBlock
from util import sinusoidal_positional_encoding

class GPTModel(L.LightningModule):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout_rate,
                 vocab_size, batch_size, sequence_length, max_epochs,
                 dataset_length):
        super(GPTModel, self).__init__()
        self.save_hyperparameters()  # Save the model's hyperparameters
        self.dataset_length = dataset_length  # Use the passed dataset_length

        self.embed_size = embed_size
        self.num_layers = num_layers
        self.heads = heads
        self.forward_expansion = forward_expansion
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_epochs = max_epochs

        # Example input array (adjust the shape according to your model's input)
        self.example_input_array = torch.zeros((1, sequence_length), dtype=torch.long)

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.pos_embbedings = nn.Parameter(sinusoidal_positional_encoding(embed_size, max_len=sequence_length))

        self.layers = nn.ModuleList([
            GPTTransformerBlock(embed_size, heads, forward_expansion, dropout_rate)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_size, vocab_size, bias=False)
        self.output_layer.weight = self.embedding.weight

    def forward(self, x, attention_mask=None):
        seq_len = x.size(1)
        device = x.device

        x = self.embedding(x)
        x = x + self.pos_embbedings[:, :seq_len, :]
        x = x.transpose(0, 1)  # Shape: [seq_len, batch_size, embed_size]

        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        # Create key padding mask [batch_size, seq_len]
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask.squeeze(1).to(device) == 0  # Convert to boolean mask

        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        x = self.output_layer(x)
        x = x.transpose(0, 1)  # Shape: [batch_size, seq_len, vocab_size]
        return x
    
    def create_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
        
    def masked_loss(self, outputs, targets, masks):
        # Flatten outputs and targets
        outputs_flat = outputs.reshape(-1, self.vocab_size)  # Use reshape instead of view
        targets_flat = targets.reshape(-1)

        # Use masks to filter out loss from padding tokens
        mask = masks.view(-1) == 1  # Flatten and convert to boolean mask
        outputs_masked = outputs_flat[mask]
        targets_masked = targets_flat[mask]

        # Calculate cross-entropy loss only on non-padded tokens
        return F.cross_entropy(outputs_masked, targets_masked)

    def training_step(self, batch):
        inputs, targets, masks = batch
        outputs = self(inputs, attention_mask=masks)
        loss = self.masked_loss(outputs, targets, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch):
        inputs, targets, masks = batch
        outputs = self(inputs, attention_mask=masks)
        loss = self.masked_loss(outputs, targets, masks)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)

        num_batches_per_epoch = self.dataset_length // self.batch_size
        if self.dataset_length % self.batch_size != 0:
            num_batches_per_epoch += 1

        total_steps = self.max_epochs * num_batches_per_epoch
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
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.000001, last_epoch=-1):
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


