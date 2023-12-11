import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from layers import GPTTransformerBlock, PositionalEncoding
import torch.nn as nn
import lightning

class GPTModel(lightning.LightningModule):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, vocab_size, train_dataset=None, val_dataset=None, batch_size=32):
        super(GPTModel, self).__init__()
        self.save_hyperparameters()  # This will save all the arguments of the __init__ method
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size)
        self.layers = nn.ModuleList([
            GPTTransformerBlock(embed_size, heads, forward_expansion)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_size, vocab_size)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        
    def forward(self, x):
        x = self.embed(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
    
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
        return DataLoader(train_dataset, batch_size=32, num_workers=8, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(val_dataset, batch_size=32, num_workers=8, persistent_workers=True)
    

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
