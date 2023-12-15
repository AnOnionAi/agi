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

