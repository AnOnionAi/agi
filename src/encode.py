import os
import json
import tiktoken

def restructure_data(input_file, output_file, context_window):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    all_text = "".join(lines)
    chunks = [all_text[i:i+context_window] for i in range(0, len(all_text), context_window)]

    with open(output_file, 'w', encoding='utf-8') as file:
        for idx, chunk in enumerate(chunks, start=1):  # Start the indexing from 1
            json_obj = {
                "id": idx,
                "text": chunk.strip(),
                "length": len(chunk.split()),  # Number of words in the chunk
                "ended": idx == len(chunks)  # True if it's the last chunk
            }
            file.write(json.dumps(json_obj) + '\n')

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

# Example usage:
# encode_file('data/raw_data.txt', val_ratio=0.1, context_window=1024)
#vocab_size = find_vocab_size('data/svelte_docs/training_data.txt')
#print("Vocabulary Size:", vocab_size)