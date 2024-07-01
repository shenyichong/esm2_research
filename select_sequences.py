import argparse
from Bio import SeqIO

def select_sequences(fasta_file, bins, limit_per_bin):
    bin_limits = {
        "0-100": (0, 100),
        "100-200": (100, 200),
        "200-500": (200, 500),
        "500-1000": (500, 1000),
        "1000-2000": (1000, 2000),
        "2000-5000": (2000, 5000),
        "5000-10000": (5000, 10000)
    }

    bin_sequences = {key: [] for key in bin_limits}

    with open(fasta_file) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequence_length = len(record.seq)
            for bin_name, (lower, upper) in bin_limits.items():
                if lower < sequence_length <= upper and len(bin_sequences[bin_name]) < limit_per_bin:
                    bin_sequences[bin_name].append(record)
                    break  # Stop checking bins once the sequence is added to one bin

    return bin_sequences

def write_sequences_to_fasta(bin_sequences, output_fasta_prefix):
    for bin_name, sequences in bin_sequences.items():
        output_file = f"{output_fasta_prefix}_{bin_name}.fasta"
        with open(output_file, "w") as output_handle:
            SeqIO.write(sequences, output_handle, "fasta")
        print(f"Written {len(sequences)} sequences to {output_file}")

def write_sequences_to_one_fasta(bin_sequences, output_fasta_prefix):
    output_file = f"{output_fasta_prefix}.fasta"
    for bin_name, sequences in bin_sequences.items():
        with open(output_file, "a") as output_handle:
            SeqIO.write(sequences, output_handle, "fasta")
            print(f"Written {len(sequences)} sequences to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select sequences of different lengths from a UniRef50 FASTA file into bins and write to new FASTA files")
    parser.add_argument("fasta_file", type=str, help="Path to the UniRef50 FASTA file")
    parser.add_argument("--limit_per_bin", type=int, default=200, help="Limit on the number of sequences per bin")
    parser.add_argument("--output_fasta_prefix", type=str, default="selected_sequences", help="Prefix for output FASTA files")
    args = parser.parse_args()

    bin_sequences = select_sequences(args.fasta_file, bins=[
        "0-100", "100-200", "200-500", "500-1000", "1000-2000", "2000-5000", "5000-10000"], limit_per_bin=args.limit_per_bin)
    
    write_sequences_to_fasta(bin_sequences, args.output_fasta_prefix)
    # write_sequences_to_one_fasta(bin_sequences, args.output_fasta_prefix)
