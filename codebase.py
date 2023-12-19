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
import os
import tiktoken

def encode_file(file_path, val_ratio=0.1):
    # Initialize the tokenizer for the GPT-2 model
    encoder = tiktoken.encoding_for_model("gpt2")

    # Define file paths for encoded data, training data, and validation data
    directory = os.path.dirname(file_path)
    encoded_file = os.path.join(directory, 'encoded_data.txt')
    train_file = os.path.join(directory, 'training_data.txt')
    val_file = os.path.join(directory, 'validation_data.txt')

    # Read and encode the data
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()

    with open(encoded_file, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            tokens = encoder.encode(sentence.strip())
            token_str = ' '.join(map(str, tokens))
            file.write(token_str + '\n')

    # Split the encoded data into training and validation datasets
    format_and_split_data(encoded_file, train_file, val_file, val_ratio)


def format_and_split_data(file_path, train_file, val_file, val_ratio=0.1):
    # Read the tokens from the file and join lines if they are not already joined
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        tokens = [' '.join(line.strip().split()) for line in lines]  # Join tokens on the same line

    # Calculate the number of validation samples
    val_size = int(len(tokens) * val_ratio)

    # Split the tokens into training and validation sets
    train_tokens = tokens[:-val_size]
    val_tokens = tokens[-val_size:]

    # Write the training tokens to the training file
    with open(train_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(train_tokens) + '\n')

    # Write the validation tokens to the validation file
    with open(val_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(val_tokens) + '\n')

def find_vocab_size(file_path):
    max_token = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = [int(token) for token in line.strip().split()]
            max_token = max(max_token, max(tokens))
    return max_token + 1  # Assuming tokens start from 0

# Example usage:
# encode_file('data/raw_data.txt', 'data/encoded_data.txt', 'data/training_data.txt', 'data/validation_data.txt')

vocab_size = find_vocab_size('data/training_data.txt')
print("Vocabulary Size:", vocab_size)
import torch.nn as nn

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


# Alternative Learnable Positional Embeddings
'''
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000, learnable=True):
        super(PositionalEncoding, self).__init__()
        self.learnable = learnable

        # Initial sinusoidal positional encoding
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register as a buffer or a learnable parameter
        if self.learnable:
            self.pe = nn.Parameter(pe)  # Learnable positional embeddings
        else:
            self.register_buffer('pe', pe)  # Fixed sinusoidal embeddings

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
'''# main.py
import argparse
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
        embed_size=1024, 
        num_layers=12, 
        heads=16, 
        forward_expansion=4, 
        dropout_rate=0.1,
        vocab_size=50233, # Adjust as needed
        batch_size=32,
        sequence_length=128, 
        max_epochs=1,
        training_file_path='data/training_data.txt',
        validation_file_path='data/validation_data.txt',
        trainable_pos_emb=False
    )

    print("Model Hyperparameters")
    print(model.hparams)  # Print the model's hyperparameters

    # Initialize the TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="gpt", log_graph=True)
    torch.set_float32_matmul_precision('medium')  # Enable mixed precision training

    trainer = Trainer(
        max_epochs=model.max_epochs,
        logger=logger,
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 1,
        accelerator="gpu" if torch.cuda.is_available() else 'auto',
        precision='16-mixed'  # Add this line to enable 16-bit precision mixed precision (AMP)
        #limit_train_batches=0.1,  # Limit the training data to 10% of the training set
        #limit_val_batches=0.1,  # Limit the validation data to 10% of the validation set
    )

    # Train the model with AMP
    trainer.fit(model)
    logger.save()  # Save the TensorBoard logs

    # Optionally save the final model
    torch.save(model.state_dict(), 'final_model.pth')

def main(args):
    if args.command == 'encode':
        input_file = args.input_file if args.input_file else 'data/raw_data.txt'
        encode_file(input_file)
        print(f"Encoded Tokens written")
    elif args.command == 'train':
        train_model()
        print("Training Complete")
    elif args.command == 'predict':
        input_text = "What is Svelte?"
        predict_model(input_text)
    else:
        print("Invalid command")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGI Model Operations")
    subparsers = parser.add_subparsers(dest='command', help='select operation')

    # Subparser for encoding
    parser_encode = subparsers.add_parser('encode', help='Encode text data to tokens')
    parser_encode.add_argument('input_file', type=str, nargs='?', default=None, help='Input file path')

    # Subparser for training
    parser_train = subparsers.add_parser('train', help='Train the GPT model')

    # Add predict command
    predict_parser = subparsers.add_parser('predict', help='predict output for a given input text')

    # Parse the arguments and call the main function
    args = parser.parse_args()
    main(args)import torch
import lightning as L
import torch.nn as nn
import math

from torch.nn import functional as F
from layers import GPTTransformerBlock
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
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout_rate, vocab_size, batch_size, sequence_length, max_epochs, training_file_path, validation_file_path, trainable_pos_emb=False,):
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
        self.training_file_path = training_file_path
        self.validation_file_path = validation_file_path

        # Example input array (adjust the shape according to your model's input)
        self.example_input_array = torch.zeros((1, sequence_length), dtype=torch.long)

        with open(training_file_path, 'r') as f:
            self.dataset_length = sum(1 for _ in f)

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        # self.pos_embedding = nn.Parameter(torch.zeros(1, 5000, embed_size))  # Comment out or remove
        self.pos_embedding = self.create_pos_embedding(sequence_length, embed_size)  # Call create_pos_embedding here

        self.layers = nn.ModuleList([
            GPTTransformerBlock(embed_size, heads, forward_expansion, dropout_rate)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_size, vocab_size)

    @staticmethod
    def create_pos_embedding(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return nn.Parameter(pe, requires_grad=False)

    def create_mask(self, mask, current_seq_length):
        batch_size = mask.size(0)  # Get the actual batch size
        # Expand mask for the number of heads and sequence length
        mask = mask.unsqueeze(1)  # Now [batch_size, 1, seq_len]
        mask = mask.repeat(1, self.heads, 1)  # Now [batch_size, num_heads, seq_len]
        mask = mask.view(batch_size * self.heads, 1, current_seq_length)  # Now [batch_size*num_heads, 1, seq_len]
        mask = mask.repeat(1, current_seq_length, 1)  # Now [batch_size*num_heads, seq_len, seq_len]
    
        return mask

    def forward(self, x, mask=None):

        x = self.embedding(x)
        current_seq_length = x.size(1)
        # Ensure positional embeddings have the correct shape
        pos_emb = self.pos_embedding[:, :current_seq_length, :]
        pos_emb = pos_emb.expand(x.size(0), -1, -1)  # Expand along batch size
        x = x + pos_emb

        # Transpose x to have shape (sequence_length, batch_size, embed_size)
        x = x.transpose(0, 1)

        # Adjust the mask for multi-head attention
        if mask is not None:
            mask = self.create_mask(mask, current_seq_length)

        for layer in self.layers:
            x = layer(x, mask=mask)  # Pass the mask to each layer

        x = self.output_layer(x)

        return x
        
    def masked_loss(self, outputs, targets, masks):
        # Flatten outputs and targets
        outputs_flat = outputs.view(-1, self.vocab_size)
        targets_flat = targets.view(-1)

        # Use masks to filter out loss from padding tokens
        mask = masks.view(-1) == 1  # Flatten and convert to boolean mask
        outputs_masked = outputs_flat[mask]
        targets_masked = targets_flat[mask]

        # Calculate cross-entropy loss only on non-padded tokens
        return F.cross_entropy(outputs_masked, targets_masked)

    def training_step(self, batch):
        inputs, targets, masks = batch
        outputs = self(inputs, mask=masks)
        loss = self.masked_loss(outputs, targets, masks)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch):
        inputs, targets, masks = batch
        outputs = self(inputs, mask=masks)
        loss = self.masked_loss(outputs, targets, masks)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def train_dataloader(self):
        train_dataset = TokenizedTextDataset(self.training_file_path, self.sequence_length)
        return DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        val_dataset = TokenizedTextDataset(self.validation_file_path, self.sequence_length)
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


import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from model import GPTModel
import tiktoken
import argparse
import os
import yaml  

def get_latest_checkpoint(version):
    checkpoint_dir = f'tb_logs/gpt/version_{version}/checkpoints/'
    checkpoint_files = os.listdir(checkpoint_dir)
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
    return latest_checkpoint

def read_hparams(version):
    hparams_path = f'tb_logs/gpt/version_{version}/hparams.yaml'
    with open(hparams_path) as file:
        hparams = yaml.safe_load(file)
    return hparams

def predict_model(input_text, model_version=None):
    tokenizer = tiktoken.encoding_for_model("gpt2")  # Ensure this matches your model's vocabulary

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
        max_epochs=hparams['max_epochs'],
        training_file_path=hparams['training_file_path'],
        validation_file_path=hparams['validation_file_path']
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
            logits = outputs[0, -1, :] / temperature  # Select the logits for the last word in the sequence
            filtered_logits = top_p_filtering(logits, top_p=top_p)

            # Sample from the filtered distribution
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_token_id = Categorical(probabilities).sample()

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

