import tiktoken

def encode_file(file_path, output_file):
    # Initialize the tokenizer for GPT-4 model
    encoder = tiktoken.encoding_for_model("gpt-4")

    # Read the training data
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Encode the text into tokens
    tokens = encoder.encode(text)

    # Write the tokens to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        for token in tokens:
            file.write(f'{token}\n')

    return tokens


def split_data(file_path, train_file, val_file, val_ratio=0.1):
    # Read the tokens from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        tokens = [line.strip() for line in file]

    # Calculate the number of validation samples
    val_size = int(len(tokens) * val_ratio)

    # Split the tokens into training and validation sets
    train_tokens = tokens[:-val_size]
    val_tokens = tokens[-val_size:]

    # Write the training tokens to the training file
    with open(train_file, 'w', encoding='utf-8') as file:
        for token in train_tokens:
            file.write(f'{token}\n')

    # Write the validation tokens to the validation file
    with open(val_file, 'w', encoding='utf-8') as file:
        for token in val_tokens:
            file.write(f'{token}\n')


def find_vocab_size(file_path):
    with open(file_path, 'r') as file:
        tokens = [int(line.strip()) for line in file]
    return max(tokens) + 1  # Assuming tokens start from 0

vocab_size = find_vocab_size('data/training_data.txt')
print("Vocabulary Size:", vocab_size)
