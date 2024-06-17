from transformers import EsmForMaskedLM, Trainer, TrainingArguments,EsmTokenizer
from fasta_dataset import FastaDataset

# Initialize the tokenizer
tokenizer = EsmTokenizer.from_pretrained("./esm2_t33_650M_UR50D")

# load the dataset 
fasta_file = "uniref50.fasta"
# fasta_dataset = FastaDataset(fasta_file, tokenizer)
fasta_dataset = FastaDataset(fasta_file, tokenizer, limit=100000)  # Limit to 100,000 sequences


# Define training arguments
training_args = TrainingArguments(
    output_dir="./esm2_output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=1000,
    save_total_limit=2,
    fp16=True,  # Use mixed precision training
    gradient_accumulation_steps=1,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500
)

# Initialize the model
model = EsmForMaskedLM.from_pretrained("./esm2_t33_650M_UR50D")

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=fasta_dataset,
)

# Fine-tune the model
trainer.train()