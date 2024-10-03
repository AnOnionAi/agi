# main.py
import os
import argparse
import torch
import wandb

from encode import encode_text_file, encode_jsonl_file, restructure_data
from model import GPTModel
from predict import predict_model
from dataset import GPTDataModule  # Import the updated GPTDataModule

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

# Assuming the key is in the project root directory
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "zeti-nube-dev-key.json"

def train_model(bucket_name, train_blob_name, val_blob_name):
    # Initialize model
    model = GPTModel(
        embed_size=768, 
        num_layers=12, 
        heads=16, 
        forward_expansion=4, 
        dropout_rate=0.1,
        vocab_size=50233,
        batch_size=32,
        sequence_length=1024, 
        max_epochs=10,
        training_file_path='',  # Not needed as data is handled by DataModule
        validation_file_path=''  # Not needed as data is handled by DataModule
    )

    print("Model Hyperparameters")
    print(model.hparams)  # Print the model's hyperparameters

    # Initialize your data module
    data_module = GPTDataModule(
        bucket_name=bucket_name,
        train_blob_name=train_blob_name,  # e.g., 'agi/svelte_docs/training_data.txt'
        val_blob_name=val_blob_name,      # e.g., 'agi/svelte_docs/validation_data.txt'
        batch_size=model.batch_size,
        sequence_length=model.sequence_length
    )

    # Initialize the TensorBoard logger
    tb_logger = TensorBoardLogger("tb_logs", name="gpt", log_graph=True)
    # Initialize the WandB logger and name your WandB project
    wandb_logger = WandbLogger(project='gpt')

    # Add your batch size to the WandB config
    wandb_logger.experiment.config["batch_size"] = model.batch_size
    torch.set_float32_matmul_precision('medium')  # Enable mixed precision training

    trainer = Trainer(
        max_epochs=model.max_epochs,
        logger=[tb_logger, wandb_logger],
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        accelerator="gpu" if torch.cuda.is_available() else 'auto',
        precision='16-mixed'  # Enable 16-bit precision mixed precision (AMP)
        #limit_train_batches=0.1,  # Uncomment to limit training data to 10%
        #limit_val_batches=0.1,    # Uncomment to limit validation data to 10%
    )

    # Train the model with AMP
    trainer.fit(model, datamodule=data_module)
    tb_logger.save()  # Save the TensorBoard logs
    wandb.finish()     # Finish the W&B run

    # Optionally save the final model
    torch.save(model.state_dict(), 'final_model.pth')

def main(args):
    if args.command == 'encode':
        input_file = args.input_file if args.input_file else 'data/svelte_docs/raw_data.txt'
        encode_text_file(input_file)
        print(f"Encoded Tokens written")
    elif args.command == 'restructure':
        input_file = args.input_file if args.input_file else 'data/svelte_docs/raw_data.txt'
        restructure_data(input_file, 'data/svelte_docs/structured_data.txt', 128)
        print(f"Structured data written")
    elif args.command == 'encode_json':
        input_file = args.input_file if args.input_file else 'data/web_text/small-117M.valid.jsonl'
        encode_jsonl_file(input_file)
        print(f"Encoded JSONL Tokens written")
    elif args.command == 'train':
        bucket_name = 'zdresearch'  # Your bucket name
        train_blob = 'agi/svelte_docs/training_data.txt'
        val_blob = 'agi/svelte_docs/validation_data.txt'
        train_model(bucket_name, train_blob, val_blob)
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
    predict_parser = subparsers.add_parser('predict', help='Predict output for a given input text')

    # Subparser for restructuring
    parser_restructure = subparsers.add_parser('restructure', help='Restructure raw text data into structured format')
    parser_restructure.add_argument('input_file', type=str, nargs='?', default=None, help='Input file path for restructuring')

    # Subparser for encoding JSONL
    parser_encode_json = subparsers.add_parser('encode_json', help='Encode JSONL file')
    parser_encode_json.add_argument('input_file', type=str, nargs='?', default=None, help='Input JSONL file path for encoding')

    # Parse the arguments and call the main function
    args = parser.parse_args()
    main(args)
