import tiktoken

def encode_file(file_path, output_file):
    # Initialize the tokenizer for GPT-4 model
    encoder = tiktoken.encoding_for_model("gpt-4")

    # Read the training data, assuming each line in the file is a separate sentence
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()

    # Encode each sentence into tokens and write to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            tokens = encoder.encode(sentence.strip())  # Strip whitespace and encode
            token_str = ' '.join(map(str, tokens))  # Join tokens into a string
            file.write(token_str + '\n')  # Write the token string to file



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
    max_token = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = [int(token) for token in line.strip().split()]
            max_token = max(max_token, max(tokens))
    return max_token + 1  # Assuming tokens start from 0


vocab_size = find_vocab_size('data/training_data.txt')
print("Vocabulary Size:", vocab_size)
