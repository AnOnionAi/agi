# main.py - Currently in GPU Configuration
import os
import argparse
import torch
import wandb
import datetime 
import yaml 

from encode import encode_text_file, encode_jsonl_file, restructure_data
from model import GPTModel
from predict import predict_model
from dataset import GPTDataModule  # Import the updated GPTDataModule
from gcs_utils import upload_blob
from callbacks import GCSCheckpointCallback, GCSTensorBoardLoggerCallback
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

# Assuming the key is in the project root directory
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "zeti-nube-dev-key.json"
os.environ['WANDB_MODE'] = 'offline'
# Initalize WandB Project For Loggin
wandb.init(project='agi', mode="offline")

if torch.cuda.is_available():
    print("CUDA is available!")
    print(torch.cuda.get_device_name(0), "baby!") 

def train_model(bucket_name, train_blob_name, val_blob_name):
    # Generate a unique timestamped directory name
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = f'agi/experiments/{timestamp}/'

    # Initialize your data module
    data_module = GPTDataModule(
        bucket_name=bucket_name,
        train_blob_name=train_blob_name,
        val_blob_name=val_blob_name,
        batch_size=32,
        sequence_length=1024
    )
    data_module.setup()

    # Calculate dataset length
    dataset_length = len(data_module.train_dataset)

    # Initialize model with hpparams such as dataset_length
    model = GPTModel(
        embed_size=1024,           # Reduced from 4096
        num_layers=16,             # Reduced from 36
        heads=8,                  # Reduced from 32
        forward_expansion=2,      # Reduced from 8
        dropout_rate=0.1,
        vocab_size=50233,         # Optional: Reduced from 50233
        batch_size=32,             # Reduced from 32
        sequence_length=1024,      # Reduced from 1024
        max_epochs=10,             # Reduced from 10
        dataset_length=dataset_length
)

    # Define local directories with the timestamp
    local_checkpoint_dir = f'checkpoints/{timestamp}/'
    local_tb_log_dir = f'tb_logs/{timestamp}/'

    # Ensure the local directories exist
    os.makedirs(local_checkpoint_dir, exist_ok=True)
    os.makedirs(local_tb_log_dir, exist_ok=True)

    # **Save hyperparameters to a YAML file**
    hparams_file = os.path.join(local_checkpoint_dir, 'hparams.yaml')
    with open(hparams_file, 'w') as f:
        yaml.dump(model.hparams, f)

    # **Upload hparams file to GCS**
    gcs_hparams_path = f'{experiment_dir}checkpoints/hparams.yaml'
    upload_blob(bucket_name, hparams_file, gcs_hparams_path)
    print(f"Hyperparameters saved to gs://{bucket_name}/{gcs_hparams_path}")

    # Initialize the loggers with the new local directory
    tb_logger = TensorBoardLogger(local_tb_log_dir, name="gpt", log_graph=True)
    wandb_logger = WandbLogger(project='gpt')
    wandb_logger.experiment.config["batch_size"] = model.batch_size

    # Define the GCS paths using the experiment directory
    gcs_checkpoint_path = f'{experiment_dir}checkpoints/'
    gcs_tb_log_dir = f'{experiment_dir}tb_logs/'

    # Define the custom checkpoint callback
    checkpoint_callback = GCSCheckpointCallback(
        bucket_name=bucket_name,
        gcs_ckpt_path=gcs_checkpoint_path,
        dirpath=local_checkpoint_dir,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    # Define the custom TensorBoard logger callback
    tb_logger_callback = GCSTensorBoardLoggerCallback(
        bucket_name=bucket_name,
        local_tb_log_dir=local_tb_log_dir,
        gcs_tb_log_dir=gcs_tb_log_dir,
        upload_interval=300  # Adjust the interval as needed
    )

    # Define a GPU monitoring callback (optional)
    class GPUStatsCallback(Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
             gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
             print(f"GPU Memory Allocated: {gpu_mem:.2f} GB")

    # Initialize the Trainer with the custom callbacks
    trainer = Trainer(
        max_epochs=model.max_epochs,
        logger=[tb_logger, wandb_logger],
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        accelerator="gpu" if torch.cuda.is_available() else 'auto',
        precision='16-mixed',
        strategy="deepspeed_stage_2",
        callbacks=[checkpoint_callback, tb_logger_callback, GPUStatsCallback()],
        log_every_n_steps=50,          # Log every 50 steps
        enable_progress_bar=True
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)
    tb_logger.save()
    wandb.finish()

    # Save and upload the final model (optional)
    local_model_path = 'final_model.pth'
    torch.save(model.state_dict(), local_model_path)
    gcs_model_path = f'{experiment_dir}final_model.pth'  # Use the experiment directory
    upload_blob(bucket_name, local_model_path, gcs_model_path)
    print(f"Model saved to gs://{bucket_name}/{gcs_model_path}")

    # Optionally, clean up local files
    # os.remove(local_model_path)
    # shutil.rmtree(local_checkpoint_dir)
    # shutil.rmtree(local_tb_log_dir)

def main(args):
    bucket_name = 'zdresearch'  # Define your bucket name
    experiments_folder = 'agi/experiments/'
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
        predict_model(input_text, bucket_name, experiments_folder)

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
