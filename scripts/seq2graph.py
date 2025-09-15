import os
import numpy
import pandas

# Extracting the data from the datasets
data_balanced = pandas.read_csv("data/processed/balanced_dataset.csv")
data_imbalanced = pandas.read_csv("data/dataset_original.csv")

# Obtaining the list of values across rows
seq_imbalanced = data_imbalanced.drop(["hash","malware"],axis=1).values
seq_balanced = data_balanced.values

# Number of API Calls
NUM_API_CALLS = 307

# Convert sequences per row to an adjacency graph
def sequences_to_graphs(sequences):
    """Converts API call sequences to adjacency matrix-based graphs"""
    n_sequences = len(sequences)
    adj_matrices = numpy.zeros((n_sequences, NUM_API_CALLS, NUM_API_CALLS), dtype=numpy.int8)
    
    for i in range(n_sequences):
        sequence = sequences[i]
        for j in range(len(sequence) - 1):
            current_api, next_api = sequence[j], sequence[j+1]
            if 0 <= current_api < NUM_API_CALLS and 0 <= next_api < NUM_API_CALLS:
                adj_matrices[i, current_api, next_api] = 1
    return adj_matrices

# Create graphs and save them
print(f"Converting all sequences to adjacency-matrix based graphs...")
numpy.save("data/graphs/balanced_graphs",sequences_to_graphs(seq_balanced))
numpy.save("data/graphs/imbalanced_graphs",sequences_to_graphs(seq_imbalanced))
print(f"All operations completed.")