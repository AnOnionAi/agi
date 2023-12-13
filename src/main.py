# main.py
import argparse
import os
import torch

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
        #vocab_size=100232,  # 50257 is size for GPT-2 and 100232 for GPT-4
        vocab_size=100232,
        batch_size=32,
        trainable_pos_emb=True
    )

    # Initialize the TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="my_model")

    # Initialize the Trainer with the logger
    trainer = Trainer(
        max_epochs=1,
        logger=logger,
        devices=1 if torch.cuda.is_available() else 1,
        accelerator="gpu" if torch.cuda.is_available() else 'auto',
    )
    # Train the model
    trainer.fit(model)

def main(args):
    if args.command == 'encode':
        encode_file(args.input_file, args.output_file)
        print(f"Encoded Tokens written to {args.output_file}")
    elif args.command == 'train':
        train_model()
        print("Training Complete")
    elif args.command == 'predict':
        input_text = "Why Svelte Good?"
        predict_model(input_text, 6)
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
