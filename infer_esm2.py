import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import time

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model names to compare
model_paths = [
    "./esm2_t6_8M_UR50D",
    "./esm2_t12_35M_UR50D",
    "./esm2_t30_150M_UR50D",
    "./esm2_t33_650M_UR50D",
    "./esm2_t36_3B_UR50D"
]

# Prepare tensor for inputs
def prepare_inputs(sequence, tokenizer):
    tokens = tokenizer.tokenize(sequence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    inputs = torch.tensor([token_ids]).to(device)
    return inputs, token_ids

# Calculate pseudo-perplexity
def calculate_pseudo_perplexity(model, tokenizer, inputs, token_ids):
    attention_mask = torch.ones_like(inputs)
    log_probs = []

    model.eval()  # Set the model to evaluation mode
    start_time = time.time()  # Start timing

    with torch.no_grad():  # Disable gradient computation
        for i in range(len(token_ids)):
            if token_ids[i] == tokenizer.cls_token_id or token_ids[i] == tokenizer.sep_token_id:
                continue

            # Create a copy of the inputs and mask the i-th token
            masked_input = inputs.clone()
            masked_input[0, i] = tokenizer.mask_token_id

            # Perform a forward pass
            outputs = model(masked_input)
            predictions = outputs.logits

            # Get the log probability of the original token at position i
            predicted_prob = torch.nn.functional.softmax(predictions[0, i], dim=-1)
            original_token_id = token_ids[i]
            log_prob = torch.log(predicted_prob[original_token_id])
            log_probs.append(log_prob.item())

    end_time = time.time()  # End timing

    # Calculate the pseudo-perplexity
    pseudo_perplexity = np.exp(-np.mean(log_probs))
    inference_time = end_time - start_time  # Convert time to seconds
    return pseudo_perplexity, inference_time

# # problem1: Main loop to run the models and compare results
# # Define the sequence you want to infer
# sequence = "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGGILKKKGHHEAEVKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGFQG"  # Replace with your sequence

# results = []
# for model_path in model_paths:
#     # Load the tokenizer and model
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
    
#     # Prepare inputs
#     inputs, token_ids = prepare_inputs(sequence, tokenizer)
    
#     # Calculate pseudo-perplexity and inference time
#     pseudo_perplexity, inference_time = calculate_pseudo_perplexity(model, tokenizer, inputs, token_ids)
    
#     # Store the results
#     results.append((model_path, len(token_ids), inference_time, pseudo_perplexity))
#     print(f"Model: {model_path}, Sequence Length: {len(token_ids)}, Inference Time: {inference_time:.4f} seconds, Pseudo-Perplexity: {pseudo_perplexity:.4f}")

# # Print results
# for result in results:
#     model_name, seq_length, inf_time, pseudo_perplexity = result
#     # print(f"Model: {model_name}, Sequence Length: {seq_length}, Inference Time: {inf_time:.4f} seconds, Pseudo-Perplexity: {pseudo_perplexity:.4f}")

# problem2 : run inference on different sequence lengths
from fasta_dataset import FastaDataset
# Load the tokenizer and fasta dataset
fasta_file = "uniref50.fasta"
model_path = model_paths[3]
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
fasta_dataset = FastaDataset(fasta_file, tokenizer, limit=1000)
results = []
for idx in range(len(fasta_dataset)):
    sequence_data = fasta_dataset[idx]
    sequence = tokenizer.decode(sequence_data['input_ids'], skip_special_tokens=True)
    
    # Prepare inputs
    inputs, token_ids = prepare_inputs(sequence, tokenizer)
    
    # Calculate pseudo-perplexity and inference time
    pseudo_perplexity, inference_time = calculate_pseudo_perplexity(model, tokenizer, inputs, token_ids)
    
    # Store the results
    results.append((model_path, len(token_ids), inference_time, pseudo_perplexity))
    print(f"Model: {model_path}, Sequence Length: {len(token_ids)}, Inference Time: {inference_time:.4f} seconds, Pseudo-Perplexity: {pseudo_perplexity:.4f}")

