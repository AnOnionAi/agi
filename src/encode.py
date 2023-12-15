import os
import tiktoken

def encode_file(file_path, val_ratio=0.1):
    # Initialize the tokenizer for the GPT-2 model
    encoder = tiktoken.encoding_for_model("gpt2")

    # Define file paths for encoded data, training data, and validation data
    directory = os.path.dirname(file_path)
    encoded_file = os.path.join(directory, 'encoded_data.txt')
    train_file = os.path.join(directory, 'training_data.txt')
    val_file = os.path.join(directory, 'validation_data.txt')

    # Read and encode the data
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()

    with open(encoded_file, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            tokens = encoder.encode(sentence.strip())
            token_str = ' '.join(map(str, tokens))
            file.write(token_str + '\n')

    # Split the encoded data into training and validation datasets
    format_and_split_data(encoded_file, train_file, val_file, val_ratio)


def format_and_split_data(file_path, train_file, val_file, val_ratio=0.1):
    # Read the tokens from the file and join lines if they are not already joined
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        tokens = [' '.join(line.strip().split()) for line in lines]  # Join tokens on the same line

    # Calculate the number of validation samples
    val_size = int(len(tokens) * val_ratio)

    # Split the tokens into training and validation sets
    train_tokens = tokens[:-val_size]
    val_tokens = tokens[-val_size:]

    # Write the training tokens to the training file
    with open(train_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(train_tokens) + '\n')

    # Write the validation tokens to the validation file
    with open(val_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(val_tokens) + '\n')

def find_vocab_size(file_path):
    max_token = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = [int(token) for token in line.strip().split()]
            max_token = max(max_token, max(tokens))
    return max_token + 1  # Assuming tokens start from 0

# Example usage:
# encode_file('data/raw_data.txt', 'data/encoded_data.txt', 'data/training_data.txt', 'data/validation_data.txt')

vocab_size = find_vocab_size('data/training_data.txt')
print("Vocabulary Size:", vocab_size)
