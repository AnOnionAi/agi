from torch.utils.data import Dataset
import torch

class TokenizedTextDataset(Dataset):
    def __init__(self, file_path, sequence_length, padding_token=0):
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
        attention_mask = torch.tensor(attention_mask[:-1], dtype=torch.float)

        return input_sequence, target_sequence, attention_mask
import tiktoken

def encode_file(file_path, output_file):
    # Initialize the tokenizer for GPT-4 model
    encoder = tiktoken.encoding_for_model("gpt-4")

    # Read the training data, assuming each line in the file is a separate sentence
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()

    # Encode each sentence into tokens and write to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            tokens = encoder.encode(sentence.strip())  # Strip whitespace and encode
            token_str = ' '.join(map(str, tokens))  # Join tokens into a string
            file.write(token_str + '\n')  # Write the token string to file

def split_data(file_path, train_file, val_file, val_ratio=0.1):
    # Read the tokens from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        tokens = [line.strip() for line in file]

    # Calculate the number of validation samples
    val_size = int(len(tokens) * val_ratio)

    # Split the tokens into training and validation sets
    train_tokens = tokens[:-val_size]
    val_tokens = tokens[-val_size:]

    # Write the training tokens to the training file
    with open(train_file, 'w', encoding='utf-8') as file:
        for token in train_tokens:
            file.write(f'{token}\n')

    # Write the validation tokens to the validation file
    with open(val_file, 'w', encoding='utf-8') as file:
        for token in val_tokens:
            file.write(f'{token}\n')


def find_vocab_size(file_path):
    max_token = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = [int(token) for token in line.strip().split()]
            max_token = max(max_token, max(tokens))
    return max_token + 1  # Assuming tokens start from 0


#vocab_size = find_vocab_size('data/training_data.txt')
#print("Vocabulary Size:", vocab_size)
import torch
import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class GPTTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout_rate):
        super(GPTTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

    def forward(self, x, mask=None):
        attention_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(self.dropout(attention_output) + x)  # Apply dropout after attention
        forward_output = self.feed_forward(x)
        out = self.norm2(self.dropout(forward_output) + x)  # Apply dropout after feed-forward network

        return out
# main.py
import argparse
import os
import torch

from torch.cuda.amp import GradScaler
from encode import encode_file
from model import GPTModel
from predict import predict_model

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

def train_model():
    # Initialize model
    model = GPTModel(
        embed_size=512, 
        num_layers=24, 
        heads=16, 
        forward_expansion=4, 
        dropout_rate=0.1,
        vocab_size=100232, # Adjust as needed
        batch_size=32,
        sequence_length=64, 
        max_epochs=1,
        trainable_pos_emb=True
    )

    # Initialize the TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="gpt", log_graph=True)

    trainer = Trainer(
        max_epochs=model.max_epochs,
        logger=logger,
        devices=1 if torch.cuda.is_available() else 1,
        accelerator="gpu" if torch.cuda.is_available() else 'auto',
        precision=16  # Add this line to enable 16-bit precision mixed precision (AMP)
    )

    # Train the model with AMP
    trainer.fit(model)
    logger.save()  # Save the TensorBoard logs

    # Optionally save the final model
    torch.save(model.state_dict(), 'final_model.pth')

def main(args):
    if args.command == 'encode':
        encode_file(args.input_file, args.output_file)
        print(f"Encoded Tokens written to {args.output_file}")
    elif args.command == 'train':
        train_model()
        print("Training Complete")
    elif args.command == 'predict':
        input_text = "Why Svelte Good?"
        predict_model(input_text)
    else:
        print("Invalid command")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGI Model Operations")
    subparsers = parser.add_subparsers(dest='command', help='select operation')

    # Subparser for encoding
    parser_encode = subparsers.add_parser('encode', help='Encode text data to tokens')
    parser_encode.add_argument('input_file', type=str, help='Input file path')
    parser_encode.add_argument('output_file', type=str, help='Output file path')

    # Subparser for training
    parser_train = subparsers.add_parser('train', help='Train the GPT model')

    # Add predict command
    predict_parser = subparsers.add_parser('predict', help='predict output for a given input text')

    # Parse the arguments and call the main function
    args = parser.parse_args()
    main(args)
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

# Collate function outside the dataset class
def collate_fn(batch):
    inputs, targets, masks = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)  # Pad attention masks

    return inputs_padded, targets_padded, masks_padded


class GPTModel(L.LightningModule):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout_rate, vocab_size, batch_size, sequence_length, max_epochs, trainable_pos_emb=False):
        super(GPTModel, self).__init__()
        self.save_hyperparameters() # Save the model's hyperparameters
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.heads = heads
        self.forward_expansion = forward_expansion
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_epochs = max_epochs
        self.trainable_pos_emb = trainable_pos_emb

        # Example input array (adjust the shape according to your model's input)
        self.example_input_array = torch.zeros((1, sequence_length), dtype=torch.long)

        with open('data/training_data.txt', 'r') as f:
            self.dataset_length = sum(1 for _ in f)

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 5000, embed_size))  # Add positional embeddings as a parameter

        self.layers = nn.ModuleList([
            GPTTransformerBlock(embed_size, heads, forward_expansion, dropout_rate)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def create_mask(self, mask, current_seq_length):

        # Expand mask for the number of heads and sequence length
        mask = mask.unsqueeze(1).unsqueeze(2)  # Now [batch_size, 1, 1, seq_len]
        mask = mask.repeat(1, self.heads, current_seq_length, 1)  # Now [batch_size, num_heads, seq_len, seq_len]

        # Reshape to [batch_size * num_heads, seq_len, seq_len]
        mask = mask.view(self.batch_size * self.heads, current_seq_length, current_seq_length)

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

    def training_step(self, batch):
        inputs, targets, masks = batch 

        outputs = self(inputs, mask=masks)  # Pass the masks to the model
        outputs = outputs.view(-1, self.vocab_size)  # Flatten outputs
        targets = targets.view(-1)  # Flatten targets
        loss = F.cross_entropy(outputs, targets)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch):
        inputs, targets, masks = batch  # Unpack the attention masks along with inputs and targets
        outputs = self(inputs, mask=masks)  # Pass the masks to the model during forward pass
        outputs = outputs.view(-1, self.vocab_size)  # Flatten outputs
        targets = targets.view(-1)  # Flatten targets
        loss = F.cross_entropy(outputs, targets)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def train_dataloader(self):
        train_dataset = TokenizedTextDataset('data/training_data.txt', self.sequence_length)
        return DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        val_dataset = TokenizedTextDataset('data/validation_data.txt', self.sequence_length)
        return DataLoader(val_dataset, batch_size=self.batch_size, collate_fn=collate_fn, pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.01)

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
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from model import GPTModel
import tiktoken
import argparse
import os
import yaml  

def get_latest_checkpoint(version):
    checkpoint_dir = f'tb_logs/my_model/version_{version}/checkpoints/'
    checkpoint_files = os.listdir(checkpoint_dir)
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
    return latest_checkpoint

def read_hparams(version):
    hparams_path = f'tb_logs/gpt/version_{version}/hparams.yaml'
    with open(hparams_path) as file:
        hparams = yaml.safe_load(file)
    return hparams

def predict_model(input_text, model_version=None):
    tokenizer = tiktoken.encoding_for_model("gpt-4")  # Ensure this matches your model's vocabulary

    # Use the latest version if no specific version is provided
    if model_version is None:
        versions = [d for d in os.listdir('tb_logs/gpt') if d.startswith('version_')]
        if versions:
            versions.sort(key=lambda v: int(v.split('_')[1]), reverse=True)
            model_version = versions[0].split('_')[1]
        else:
            raise Exception("No model versions found.")
    else:
        # Explicitly use version 12
        model_version = '12'  # Hardcoded to version 12 for testing

    print(f'Using model version {model_version}')
    hparams = read_hparams(model_version)
    model = GPTModel(
        embed_size=hparams['embed_size'],
        num_layers=hparams['num_layers'],
        heads=hparams['heads'],
        forward_expansion=hparams['forward_expansion'],
        dropout_rate=hparams['dropout_rate'],
        batch_size=hparams['batch_size'],
        vocab_size=hparams['vocab_size'],
        sequence_length=hparams['sequence_length'],
    )

    checkpoint_path = get_latest_checkpoint(model_version)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Generate text using the trained model
    generated_text = generate_text(input_text, tokenizer, model)
    return generated_text

def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    """ Filter a distribution of logits using nucleus (top-p) sampling """
    assert logits.dim() == 1  # batch size 1 for single word generation
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold (nucleus)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits

def generate_text(input_text, tokenizer, model, temperature=1.0, top_p=0.9):
    if not input_text.strip():
        raise ValueError("Input text is empty")

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text)

    if not input_ids:
        raise ValueError("Input text could not be tokenized")

    # Convert to tensor and add batch dimension
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    print(f"Encoded input ids: {input_ids}")  # Debug print

    # Generate text
    model.eval()
    with torch.no_grad():
        for i in range(model.sequence_length):
            outputs = model(input_ids)
            logits = outputs[:, -1, :] / temperature
            filtered_logits = top_p_filtering(logits.squeeze(), top_p=top_p)

            # Sample from the filtered distribution
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_token_id = Categorical(probabilities).sample()

            # Print the token being added
            print(f"Step {i}: Next token id: {next_token_id.item()}")  # Debug print

            # Stop generating if end-of-sequence token is produced
            if next_token_id.item() == tokenizer.eot_token:
                print("End of sequence token reached.")  # Debug print
                break

            # Add batch dimension to make it a 2D tensor
            next_token_id = next_token_id.unsqueeze(0).unsqueeze(0)  # Add two dimensions to make it a 2D tensor
            input_ids = torch.cat((input_ids, next_token_id), dim=-1)

    generated_text = tokenizer.decode(input_ids[0].tolist())
    print(f"Generated text: {generated_text}")  # Debug print
    return generated_text

