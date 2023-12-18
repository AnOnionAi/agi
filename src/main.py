# main.py
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
        num_layers=24, 
        heads=16, 
        forward_expansion=4, 
        dropout_rate=0.1,
        vocab_size=50233, # Adjust as needed
        batch_size=32,
        sequence_length=64, 
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
    main(args)