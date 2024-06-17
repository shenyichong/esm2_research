from torch.utils.data import Dataset
from Bio import SeqIO

class FastaDataset(Dataset):
    def __init__(self, fasta_file, tokenizer, max_length=1024, limit=None):
        self.fasta_file = fasta_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequence_start_lines = self._get_sequence_start_lines(limit)

    def _get_sequence_start_lines(self, limit=None):
        start_lines = []
        with open(self.fasta_file) as handle:
            line_num = 0
            for line in handle:
                if line.startswith(">"):
                    start_lines.append(line_num)
                line_num += 1
                if limit and len(start_lines) >= limit:
                    break
        return start_lines

    def __len__(self):
        return len(self.sequence_start_lines)

    def __getitem__(self, idx):
        start_line = self.sequence_start_lines[idx]
        with open(self.fasta_file) as handle:
            for _ in range(start_line):
                next(handle)  # Skip to the start line
            record = next(SeqIO.parse(handle, "fasta"))
            sequence = str(record.seq)
            tokenized = self.tokenizer(sequence, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            return {
                'input_ids': tokenized['input_ids'][0],
                'attention_mask': tokenized['attention_mask'][0],
                'labels': tokenized['input_ids'][0]
            }

