import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np

# Load the tokenizer and model
model_name = "./esm2_t33_650M_UR50D"  # Adjust the model name as needed 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Define the sequence you want to infer
sequence = "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGGILKKKGHHEAEVKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGFQG"  # Replace with your sequence
tokens = tokenizer.tokenize(sequence)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Prepare tensor for inputs
inputs = torch.tensor([token_ids])
attention_mask = torch.ones_like(inputs)

# Calculate pseudo-perplexity
log_probs = []

model.eval()  # Set the model to evaluation mode
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

# Calculate the pseudo-perplexity
pseudo_perplexity = np.exp(-np.mean(log_probs))
print(f"Pseudo-Perplexity of the sequence: {pseudo_perplexity}")