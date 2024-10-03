# encode.py
import os
import json
import tiktoken
from gcs_utils import download_blob, upload_blob, list_blobs

def restructure_data(all_text, context_window):
    """
    Restructure data into chunks of a specified context window.
    """
    chunks = [all_text[i:i+context_window] for i in range(0, len(all_text), context_window)]
    structured_data = []
    for idx, chunk in enumerate(chunks, start=1):
        json_obj = {
            "id": idx,
            "text": chunk.strip(),
            "length": len(chunk.split()),  # Number of words in the chunk
            "ended": idx == len(chunks)  # True if it's the last chunk
        }
        structured_data.append(json_obj)
    return structured_data

def encode_data(bucket_name, source_blob_name, destination_folder, context_window=1024, val_ratio=0.1):
    """
    Downloads data from GCS, encodes it, and uploads the processed data back to GCS.
    """
    local_raw_file = "raw_data.txt"
    download_blob(bucket_name, source_blob_name, local_raw_file)
    
    # Read and restructure data
    with open(local_raw_file, 'r', encoding='utf-8') as file:
        all_text = file.read()
    structured_data = restructure_data(all_text, context_window)
    
    # Write structured data to a temporary local file
    structured_file = "structured_data.jsonl"
    with open(structured_file, 'w', encoding='utf-8') as file:
        for item in structured_data:
            file.write(json.dumps(item) + '\n')
    
    # Encode the structured data
    encoder = tiktoken.encoding_for_model("gpt2")
    encoded_file = "encoded_data.txt"
    with open(structured_file, 'r', encoding='utf-8') as infile, open(encoded_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            tokens = encoder.encode(data['text'])
            token_str = ' '.join(map(str, tokens))
            outfile.write(token_str + '\n')
    
    # Split into training and validation sets
    train_file = "training_data.txt"
    val_file = "validation_data.txt"
    with open(encoded_file, 'r', encoding='utf-8') as infile:
        tokens = infile.readlines()
    val_size = int(len(tokens) * val_ratio)
    train_tokens = tokens[:-val_size]
    val_tokens = tokens[-val_size:]
    with open(train_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(train_tokens)
    with open(val_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(val_tokens)
    
    # Upload processed data back to GCS
    upload_blob(bucket_name, train_file, f"{destination_folder}/training_data.txt")
    upload_blob(bucket_name, val_file, f"{destination_folder}/validation_data.txt")
    print("Encoding and uploading complete.")

def encode_jsonl_file(input_file):
    # Initialize the tokenizer for GPT-2
    encoder = tiktoken.encoding_for_model("gpt2")

    # Construct output file name by replacing the extension with '.encoded.txt'
    base_name = os.path.splitext(input_file)[0]
    output_file = base_name + '.encoded.txt'

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Parse the JSON line
            data = json.loads(line)

            # Encode the text
            encoded_text = encoder.encode(data['text'])

            # Convert encoded tokens to a string and write to the output file
            outfile.write(' '.join(map(str, encoded_text)) + '\n')

    print(f"Encoded file saved as: {output_file}")

def encode_text_file(file_path, val_ratio=0.1, context_window=1024):
    structured_file = os.path.join(os.path.dirname(file_path), 'structured_data.txt')
    restructure_data(file_path, structured_file, context_window)

    encoder = tiktoken.encoding_for_model("gpt2")
    directory = os.path.dirname(structured_file)
    encoded_file = os.path.join(directory, 'encoded_data.txt')
    train_file = os.path.join(directory, 'training_data.txt')
    val_file = os.path.join(directory, 'validation_data.txt')

    with open(structured_file, 'r', encoding='utf-8') as file:
        sentences = file.readlines()

    with open(encoded_file, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            tokens = encoder.encode(sentence.strip())
            token_str = ' '.join(map(str, tokens))
            file.write(token_str + '\n')

    format_and_split_data(encoded_file, train_file, val_file, val_ratio)

def format_and_split_data(file_path, train_file, val_file, val_ratio=0.1):
    with open(file_path, 'r', encoding='utf-8') as file:
        tokens = [' '.join(line.strip().split()) for line in file.readlines()]

    val_size = int(len(tokens) * val_ratio)
    train_tokens = tokens[:-val_size]
    val_tokens = tokens[-val_size:]

    with open(train_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(train_tokens) + '\n')
    with open(val_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(val_tokens) + '\n')

def find_vocab_size(file_path):
    max_token = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = [int(token) for token in line.strip().split()]
            max_token = max(max_token, max(tokens))
    return max_token + 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process and encode data from GCS.')
    parser.add_argument('--bucket', type=str, required=True, help='Name of the GCS bucket.')
    parser.add_argument('--source_blob', type=str, required=True, help='GCS path to the source raw data file.')
    parser.add_argument('--destination_folder', type=str, default='agi/experiments/', help='GCS folder to upload processed data.')
    parser.add_argument('--context_window', type=int, default=1024, help='Context window size for data restructuring.')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation data ratio.')
    
    args = parser.parse_args()
    encode_data(args.bucket, args.source_blob, args.destination_folder, args.context_window, args.val_ratio)
