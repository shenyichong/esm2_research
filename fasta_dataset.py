from torch.utils.data import Dataset
from Bio import SeqIO

class FastaDataset(Dataset):
    def __init__(self, fasta_file, tokenizer, max_length=1024, length_limit=1024, limit=None):
        self.fasta_file = fasta_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.length_limit = length_limit # fileter sequence by length with length_limit or less
        self.sequence_start_lines = self._get_sequence_start_lines(limit)

    def _get_sequence_start_lines(self, limit=None):
        start_lines = []
        with open(self.fasta_file) as handle:
            line_num = 0
            while True:
                line = handle.readline()
                if not line:
                    break
                if line.startswith(">"):
                    current_pos = handle.tell()
                    record = next(SeqIO.parse(handle, "fasta"))
                    sequence_length = len(record.seq)
                    if sequence_length <= self.length_limit:
                        start_lines.append(line_num)
                    handle.seek(current_pos)  # Go back to the start of the record
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

