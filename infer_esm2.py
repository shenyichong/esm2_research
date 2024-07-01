import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import time
import random
from collections import defaultdict
import matplotlib.pyplot as plt


# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

# Calculate traditional perplexity
def calculate_perplexity(model, tokenizer, inputs, token_ids):
    seq_length = len(token_ids)
    num_masked_tokens = max(1, int(0.15 * seq_length))  # 15% of the sequence length
    mask_indices = random.sample(range(seq_length), num_masked_tokens)
    
    masked_input = inputs.clone()
    original_token_ids = masked_input[0, mask_indices].tolist()
    
    for idx in mask_indices:
        rand_val = random.random()
        if rand_val < 0.8:
            masked_input[0, idx] = tokenizer.mask_token_id
        elif rand_val < 0.9:
            masked_input[0, idx] = random.choice(list(tokenizer.get_vocab().values()))
        # else: keep the original token

    model.eval()  # Set the model to evaluation mode
    start_time = time.time()  # Start timing
    with torch.no_grad():  # Disable gradient computation
        outputs = model(masked_input)
        predictions = outputs.logits

    log_probs = []
    for original_token_id, idx in zip(original_token_ids, mask_indices):
        predicted_prob = torch.nn.functional.softmax(predictions[0, idx], dim=-1)
        log_prob = torch.log(predicted_prob[original_token_id])
        log_probs.append(log_prob.item())
    
    perplexity = np.exp(-np.mean(log_probs))
    end_time = time.time()  # End timing
    inference_time = end_time - start_time  # Convert time to seconds
    return perplexity, inference_time

# # -----------------------------------------------------------
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

# ------------------------------------------------------
# problem2 : run inference on different sequence lengths
from fasta_dataset import FastaDataset
# Load the tokenizer and fasta dataset
model_path = model_paths[3]
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)

# fasta_file = "uniref50.fasta"
# length_limit = 728
# fasta_dataset = FastaDataset(fasta_file, tokenizer, length_limit=length_limit, limit=1000)

fasta_file = "selected_sequences.fasta"
max_length = 20000
length_limit = 20000
log_file = "perplexities_by_length.txt"
fasta_dataset = FastaDataset(fasta_file, tokenizer, max_length, length_limit)

results = []
with open(log_file, 'w') as f:
    for idx in range(len(fasta_dataset)):
        sequence_data = fasta_dataset[idx]
        sequence = tokenizer.decode(sequence_data['input_ids'], skip_special_tokens=True)
        
        # Prepare inputs
        inputs, token_ids = prepare_inputs(sequence, tokenizer)
        
        # Calculate pseudo-perplexity and inference time
        # pseudo_perplexity, inference_time = calculate_pseudo_perplexity(model, tokenizer, inputs, token_ids)
        default_perplexity, inference_time = calculate_perplexity(model, tokenizer, inputs, token_ids)
        
        # Store the results
        results.append((model_path, len(token_ids), inference_time, default_perplexity))
        print(f"Model: {model_path}, Sequence Length: {len(token_ids)}, Inference Time: {inference_time:.4f} seconds, Default-Perplexity: {default_perplexity:.4f}")
        
        # write the results into a log file 
        f.write(f"Model: {model_path}, Sequence Length: {len(token_ids)}, Inference Time: {inference_time:.4f} seconds, Default-Perplexity: {default_perplexity:.4f}\n")

# plot a distribution of sequence lengths using different bins:
# bins: {"0-100","100-200","200-500","500-1000","1000-2000","2000-5000","5000-10000"}
# for each bin calculate the average of pseudo_perplexity with different sequence lengths within that bin
bins = [(0, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10000)]
bin_labels = ["0-100", "100-200", "200-500", "500-1000", "1000-2000", "2000-5000", "5000-10000"]

# Dictionary to hold perplexities for each bin
bin_perplexities = {bin_label: [] for bin_label in bin_labels}

# Dictionary to hold perplexities for each sequence length
length_perplexities = defaultdict(list)

for result in results: 
    model_name, seq_length, inf_time, perplexity = result
    for i,bin in enumerate(bins):
        if bin[0] < seq_length and bin[1] >= seq_length:
            bin_perplexities[bin_labels[i]].append(perplexity)
            break
    length_perplexities[seq_length].append(perplexity)
    
print("length_perplexities keys:", length_perplexities.keys())
for key in length_perplexities.keys():
    print("length of different key in bin:",len(length_perplexities[key]))


# Calculate average perplexity for each bin
average_perplexities = []
for bin_label in bin_labels:
    if bin_perplexities[bin_label]:
        avg_perplexity = np.mean(bin_perplexities[bin_label])
    else:
        avg_perplexity = 0
    average_perplexities.append(avg_perplexity)

# Calculate average perplexity for each sequence length
lengths = sorted(length_perplexities.keys())
average_perplexities_by_length = [np.mean(length_perplexities[length]) for length in lengths]

# Plotting the bin-based results
plt.figure(figsize=(10, 6))
plt.bar(bin_labels, average_perplexities, color='skyblue')
plt.xlabel('Sequence Length Bins')
plt.ylabel('Average Perplexity')
plt.title('Average Perplexity by Sequence Length Bins')
plt.xticks(rotation=45)
output_file = "avg_perplexity_by_seq_length_bins.png"
plt.savefig(output_file)
print(f"Plot saved to {output_file}")

# Plotting the length-based results
plt.figure(figsize=(12, 6))
plt.plot(lengths, average_perplexities_by_length, marker='o', linestyle='-', color='skyblue')
plt.xlabel('Sequence Length')
plt.ylabel('Average Perplexity')
plt.title('Average Perplexity by Sequence Length')
plt.ylim(top=100)  # Set the upper bound of y-axis to 100
plt.grid(True)
output_file = "avg_perplexity_by_seq_length.png"
plt.savefig(output_file)
print(f"Plot saved to {output_file}")