"""
features.py

A script that creates the node matrix of features (X). 
"""

import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

try:
    # Loads the balanced sequences from the processed directory
    data_balanced = pd.read_csv("data/processed/balanced_dataset.csv")
    
    # Isolate the feature columns (API sequences). 
    seq_balanced = data_balanced.drop("malware", axis=1, errors='ignore').values
    
    print(f"Loaded balanced sequences (seq_balanced) shape: {seq_balanced.shape}")
    
except FileNotFoundError:
    print("FATAL ERROR: Required file 'data/processed/balanced_dataset.csv' not found. Cannot proceed with Part 1, Point 3.")
    exit()

# --- Part 1, Point 3: One-Hot Encoding ---
print("\nSTEP 3: Creating Node Feature Matrix X via One-Hot Encoding...")

# 1. Prepare data for encoder fitting
all_api_ids = seq_balanced.flatten().reshape(-1, 1)
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(all_api_ids)

# Determine the API vocabulary size (Feature Dimension)
num_api_types = len(encoder.categories_[0])
logging.info(f"API Vocabulary Size (Feature Dimension): {num_api_types}")

# 2. Apply one-hot encoding to the balanced sequences
X_OHE_list = []
for sequence in seq_balanced:
    sequence_reshaped = sequence.reshape(-1, 1)
    ohe_sequence = encoder.transform(sequence_reshaped)
    X_OHE_list.append(ohe_sequence)

# 3. Stack the list of OHE sequences into the final 3D tensor
X_OHE = np.stack(X_OHE_list)

# The resulting X_OHE is the Node Feature Matrix X
print(f"Shape of Node Feature Matrix X (X_OHE): {X_OHE.shape}")
print(f"Format: (Num_Samples, Sequence_Length, API_Feature_Dimension)")

# --- FIX: Ensure the output directory exists ---
output_dir = "data/features"
if not os.path.exists(output_dir):
    # This creates the directory 'data/features' if it does not exist
    os.makedirs(output_dir, exist_ok=True) 
    print(f"Created directory: {output_dir}")

# Save the Node Feature Matrix X
np.save(os.path.join(output_dir, "node_features_X"), X_OHE)
print(f"Node Feature Matrix X successfully saved to '{output_dir}/node_features_X.npy'.")