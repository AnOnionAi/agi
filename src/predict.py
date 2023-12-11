import torch
import torch.nn.functional as F
from model import GPTModel
import tiktoken

import argparse
import os

def generate_text(input_text, tokenizer, model, max_length=50, temperature=1.0, top_k=50):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text)

    # Convert to tensor and add batch dimension
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    # Generate text
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs[:, -1, :] / temperature
            probabilities = F.softmax(logits, dim=-1)

            # Apply top-k sampling
            top_k_probabilities, top_k_indices = torch.topk(probabilities, k=top_k)
            next_token_id = torch.multinomial(top_k_probabilities, num_samples=1)
            next_token_id = top_k_indices.gather(-1, next_token_id)

            # Stop generating if end-of-sequence token is produced
            if next_token_id.item() == tokenizer.eot_token:
                break

            input_ids = torch.cat((input_ids, next_token_id), dim=-1)

    generated_text = tokenizer.decode(input_ids[0].tolist())
    return generated_text

def get_latest_checkpoint(version):
    # Get the directory where the checkpoints are saved
    checkpoint_dir = f'tb_logs/my_model/version_{version}/checkpoints/'

    # Get a list of all checkpoint files
    checkpoint_files = os.listdir(checkpoint_dir)

    # Sort the checkpoint files by their modification time
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))

    # Get the path of the latest checkpoint file
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])

    return latest_checkpoint

def predict_model(input_text, model_version):

    # Initialize the tokenizer
    tokenizer = tiktoken.encoding_for_model("gpt-4")  # Ensure this matches your model's vocabulary

    # Load the trained model with the correct parameters used for training
    model = GPTModel(
        embed_size=256,
        num_layers=8,  # Adjusted to match the number of layers in the checkpoint
        heads=8,
        forward_expansion=4,
        dropout_rate=0.1,
        batch_size=32,
        vocab_size=100232
    )
    # If a version is specified, use it to create the checkpoint path
    if model_version is not None:
        # Use the function to get the latest checkpoint
        checkpoint_path = get_latest_checkpoint(model_version)
    else:
        # If no version is specified, find the latest version
        versions = [d for d in os.listdir('tb_logs/my_model') if d.startswith('version_')]
        versions.sort(key=lambda v: int(v.split('_')[1]), reverse=True)
        latest_version = versions[0].split('_')[1]
        checkpoint_path = f'tb_logs/my_model/version_{latest_version}/checkpoints/epoch=0-step=357.ckpt'

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Generate text using the trained model
    return generate_text(input_text, tokenizer, model, max_length=50, temperature=1.0, top_k=50)