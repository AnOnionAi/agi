# main.py
import argparse
import os
import torch

from encode import encode_file
from model import GPTModel
from dataset import TokenizedTextDataset

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

def train_model():
    # Initialize model
    model = GPTModel(
        embed_size=256, 
        num_layers=6, 
        heads=8, 
        forward_expansion=4, 
        vocab_size=100232,  # 50257 is size for GPT-2 and 100232 for GPT-4
        train_dataset=TokenizedTextDataset('data/training_data.txt'),
        val_dataset=TokenizedTextDataset('data/validation_data.txt'),
        batch_size=32
    )

    # Initialize the TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="my_model")

    # Inside your train_model function, before initializing the Trainer:
    tpu_cores = None
    if 'TPU_NAME' in os.environ:  # This environment variable is set if a TPU is available
        print("Using TPU")
        tpu_cores = 8  # Number of TPU cores

    # Initialize the Trainer with the logger
    trainer = Trainer(
        max_epochs=1,
        tpu_cores=tpu_cores,  # This will be None if no TPU is available, which is fine
        gpus=1 if torch.cuda.is_available() else None,
        logger=logger  # Add logger here
    )
    # Train the model
    trainer.fit(model)

def main(args):
    if args.command == 'encode':
        encode_file(args.input_file, args.output_file)
        print(f"Encoded Tokens written to {args.output_file}")
    elif args.command == 'train':
        train_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGI Model Operations")
    subparsers = parser.add_subparsers(dest='command', help='select operation')

    # Subparser for encoding
    parser_encode = subparsers.add_parser('encode', help='Encode text data to tokens')
    parser_encode.add_argument('input_file', type=str, help='Input file path')
    parser_encode.add_argument('output_file', type=str, help='Output file path')

    # Subparser for training
    parser_train = subparsers.add_parser('train', help='Train the GPT model')

    # Parse the arguments and call the main function
    args = parser.parse_args()
    main(args)
