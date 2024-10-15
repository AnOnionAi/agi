# predict.py
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import tiktoken
from gcs_utils import download_blob
from model import GPTModel
import yaml
from gcs_utils import download_blob, upload_blob, list_blobs
import logging
from google.cloud import exceptions as gcs_exceptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_latest_checkpoint(bucket_name, experiments_folder):
    checkpoint_dir = f"{experiments_folder}/checkpoints/"
    blobs = list_blobs(bucket_name, prefix=checkpoint_dir)
    checkpoint_files = [blob.name for blob in blobs if blob.name.endswith('.pth')]

    if not checkpoint_files:
        raise Exception("No checkpoint files found in the specified directory.")

    # Assuming checkpoints are named with step numbers or timestamps
    latest_checkpoint = sorted(checkpoint_files, reverse=True)[0]
    local_checkpoint = "latest_model.pth"
    download_blob(bucket_name, latest_checkpoint, local_checkpoint)
    return local_checkpoint

def read_hparams(bucket_name, experiments_folder, model_version=None):
    try: 
        if model_version:
            hparams_blob = f"{experiments_folder}/{model_version}/checkpoints/hparams.yaml"
        else:
            # Fetch all hparams.yaml files under experiments_folder
            versions = [d.name for d in list_blobs(bucket_name, prefix=f"{experiments_folder}") if 'hparams.yaml' in d.name]
            if not versions:
                raise Exception("No hparams.yaml files found.")
            # Sort the versions to find the latest one
            hparams_blob = sorted(versions, reverse=True)[0]
        
        local_hparams = "hparams.yaml"
        download_blob(bucket_name, hparams_blob, local_hparams)
        with open(local_hparams, 'r') as file:
            hparams = yaml.safe_load(file)
        return hparams
    except gcs_exceptions.NotFound:
        logger.error(f"Bucket or blob not found: {bucket_name}/{hparams_blob}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in read_hparams: {e}")
        raise

def load_model(bucket_name, experiments_folder, model_version=None):
    # Load hyperparameters
    hparams = read_hparams(bucket_name, experiments_folder, model_version)
    
    # Initialize the model
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
    
    # Load the latest checkpoint
    checkpoint_path = get_latest_checkpoint(bucket_name, experiments_folder)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model, hparams

def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    """ Filter a distribution of logits using nucleus (top-p) sampling """
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

def generate_text(input_text, tokenizer, model, sequence_length=128, temperature=1.0, top_p=0.9):
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
    with torch.no_grad():
        for _ in range(sequence_length):
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

def predict_model(input_text, bucket_name, experiments_folder, model_version=None):
    tokenizer = tiktoken.encoding_for_model("gpt2")  # Ensure this matches your model's vocabulary

    # Load model
    model, hparams = load_model(bucket_name, experiments_folder, model_version)
    sequence_length = hparams.get('sequence_length', 128)

    # Generate text using the trained model
    generated_text = generate_text(input_text, tokenizer, model, sequence_length=sequence_length)
    
    # Optionally, upload the generated text to GCS
    output_blob = f"{experiments_folder}/experiments_generated_text.txt"
    with open("generated_text.txt", "w", encoding='utf-8') as f:
        f.write(generated_text)
    upload_blob(bucket_name, "generated_text.txt", output_blob)
    
    return generated_text
