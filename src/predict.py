import torch
import torch.nn.functional as F  # Import the functional module as F
from model import GPTModel
import tiktoken  # Ensure tiktoken is a module you have defined or installed

def generate_text(input_text, tokenizer, model, max_length=50, temperature=1.0, top_k=50):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text)

    # Convert to tensor and add batch dimension
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    
    # Generate text
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            # Apply temperature scaling
            logits = outputs[:, -1, :] / temperature
            # Convert logits to probabilities
            probabilities = F.softmax(logits, dim=-1)

             # Apply top-k sampling
            top_k_probabilities, top_k_indices = torch.topk(probabilities, k=top_k)
            next_token_id = torch.multinomial(top_k_probabilities, num_samples=1)
            next_token_id = top_k_indices.gather(-1, next_token_id)
            input_ids = torch.cat((input_ids, next_token_id), dim=-1)
            
            # Optionally, stop if end-of-sequence token is generated
            if next_token_id == tokenizer.eot_token:
                break

    generated_text = tokenizer.decode(input_ids[0].tolist())
    return generated_text

# Initialize the tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-4")  # Replace with your tokenizer's actual initialization method

# Load the trained model with the correct parameters used for training
model = GPTModel(
    embed_size=256,
    num_layers=6,
    heads=8,
    forward_expansion=4,
    vocab_size=100232,  # Replace with the actual vocab_size used during training
)

# Load the trained model's weights
checkpoint_path = 'tb_logs/my_model/version_13/checkpoints/epoch=0-step=5054.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Generate text using the trained model
input_text = "Why Svelte Good?"
generated_text = generate_text(input_text, tokenizer, model, max_length=50, temperature=1.0, top_k=50)
print(generated_text)
