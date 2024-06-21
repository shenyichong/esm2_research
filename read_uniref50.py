import argparse
from collections import Counter
from Bio import SeqIO
import matplotlib.pyplot as plt

def get_sequence_length_distribution(fasta_file, limit=None):
    length_counter = Counter()
    num_sequences = 0
    lengths = []

    with open(fasta_file) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequence_length = len(record.seq)
            if sequence_length <= 5000:  # Only consider sequences with length <= 5000
                length_counter[sequence_length] += 1
            lengths.append(sequence_length)
            num_sequences += 1
            if limit and num_sequences >= limit:
                break

    return length_counter, lengths

def plot_length_distribution(length_counter, output_file):
    lengths = list(length_counter.keys())
    counts = list(length_counter.values())

    plt.figure(figsize=(10, 6))
    plt.bar(lengths, counts, width=1.0, edgecolor='black')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Sequence Length Distribution (Length ≤ 5000)')
    plt.xlim(0, 5000)  # Limit x-axis to 5000
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

def write_sequence_lengths_to_file(lengths, output_length_file):
    with open(output_length_file, 'w') as f:
        for length in lengths:
            f.write(f"{length}\n")
    print(f"Sequence lengths written to {output_length_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and plot sequence length distribution from a UniRef50 FASTA file")
    parser.add_argument("fasta_file", type=str, help="Path to the UniRef50 FASTA file")
    parser.add_argument("--limit", type=int, default=None, help="Limit on the number of sequences to process")
    parser.add_argument("--output_file", type=str, default="length_distribution.png", help="Output file to save the plot")
    parser.add_argument("--output_length_file", type=str, default="sequence_lengths.txt", help="Output file to save sequence lengths")
    args = parser.parse_args()

    length_counter, lengths = get_sequence_length_distribution(args.fasta_file, args.limit)
    plot_length_distribution(length_counter, args.output_file)
    write_sequence_lengths_to_file(lengths, args.output_length_file)
