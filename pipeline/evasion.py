import os
import random
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.predictor import (
    predict_sgan,
    predict_dgcnn,
    predict_cnn,
    predict_dgcnn_sgan,
    predict_gat,
    predict_gat_sgan,
    predict_lstm
)

from .constants import STALLING_APIS, BENIGN_APIS

MODELS_CONFIG = {
    "SGAN": predict_sgan,
    "DGCNN": predict_dgcnn,
    "CNN": predict_cnn,
    "HYBRID-DGCNN": predict_dgcnn_sgan,
    "GAT": predict_gat,
    "HYBRID-GAT": predict_gat_sgan,
    "LSTM": predict_lstm
}


def safe_predict(func, features, seed=10, name="DEFAULT", silent=True):
    """Safely executes a prediction function, returning (None, None) on failure."""
    try:
        return func(features, seed)
    except Exception as e:
        if not silent:
            print(f"{name} failed: {e}")
        return None, None

def apply_stalling_padding(sequence, evasion_rate=0.3):
    """Injects useless or stall-inducing APIs randomly throughout the sequence."""
    seq = list(sequence)
    num_inserts = int(len(seq) * evasion_rate)
    
    for _ in range(num_inserts):
        insert_pos = random.randint(0, len(seq))
        seq.insert(insert_pos, random.choice(STALLING_APIS))
        
    return seq[:len(sequence)]


def apply_benign_mimicry(sequence, evasion_rate=0.3):
    """Replaces the trailing execution window with benign API footprints."""
    seq = np.array(sequence, copy=True)
    num_inserts = int(len(seq) * evasion_rate)
    
    if num_inserts > 0 and len(BENIGN_APIS) > 0:
        seq[-num_inserts:] = np.random.choice(BENIGN_APIS, size=num_inserts)
        
    return seq.tolist()


def apply_call_reordering(sequence, perturb_rate=0.2):
    """Swaps adjacent API call clusters to break signature continuity."""
    seq = list(sequence)
    num_swaps = int(len(seq) * perturb_rate)
    
    for _ in range(num_swaps):
        if len(seq) < 2:
            break
        idx = random.randint(0, len(seq) - 2)
        seq[idx], seq[idx+1] = seq[idx+1], seq[idx]
        
    return seq


def evaluate_evasion(X_val, y_val, evasion_rate=0.3, baseline=False):
    """Evaluates multiple deep learning engines against adversarial perturbation methods."""
    for seed in [10,20,30,40,50]:
        if not baseline:
            print(f"\n{'='*75}")
            print(f" EVASION EVALUATION - MANIPULATION RATE: {evasion_rate*100:.0f}% ")
            print(f"{'='*75}")
        else:
            print(f"\n{'='*75}\n BASELINE RUN (No Evasion Applied)\n{'='*75}")
        
        print(f"Evaluating models trained on seed={seed}...")
        
        malware_indices = np.where(y_val == 1)[0]
        X_malware = X_val[malware_indices]
        total_samples = len(X_malware)
        
        if total_samples == 0:
            print("No malware samples found in evaluation split.")
            return

        if baseline:
            techniques = {"Baseline (No Evasion)": lambda x: list(x)}
        else:
            techniques = {
                "Stalling Padding": lambda x: apply_stalling_padding(x, evasion_rate),
                "Benign Mimicry": lambda x: apply_benign_mimicry(x, evasion_rate),
                "Call Reordering": lambda x: apply_call_reordering(x, evasion_rate)
            }
        
        results_summary = []

        for tech_name, tech_func in techniques.items():
            print(f"Processing Technique: {tech_name}...")
            
            detected = {m: 0 for m in MODELS_CONFIG.keys()}
            total_time = {m: 0.0 for m in MODELS_CONFIG.keys()}
            
            for sequence in tqdm(X_malware):
                evaded_seq = tech_func(sequence)
                
                for model_name, model_func in MODELS_CONFIG.items():
                    t_exec, prob = safe_predict(model_func, evaded_seq, seed=seed, name=model_name)
                    
                    if t_exec is not None:
                        total_time[model_name] += t_exec
                    if prob is not None and prob > 0.5:
                        detected[model_name] += 1
                        
            for m in MODELS_CONFIG.keys():
                det_rate = (detected[m] / total_samples) * 100 if total_samples > 0 else 0
                results_summary.append({
                    "Technique": tech_name,
                    "Model": m,
                    "TPR (%)": f"{det_rate:.2f}%",
                    "Detected / Total": f"{detected[m]}/{total_samples}",
                    "Evasions Succeeded": total_samples - detected[m],
                    "Total Time (s)": f"{total_time[m]:.4f}"
                })

        df_results = pd.DataFrame(results_summary)
        print("\n" + df_results.to_string(index=False))

def load_data(filepath="dataset.csv"):
    """Loads and returns stratified train-test subsets from the filesystem."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing required execution file: {filepath}")
        
    df = pd.read_csv(filepath)
    if "hash" in df.columns:
        df = df.drop(columns=["hash"])

    X = df.drop(columns=['malware']).values.astype(np.float32)
    y = df['malware'].values.astype(np.int64)

    return train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )


if __name__ == "__main__":
    try:
        _, X_val, _, y_val = load_data("dataset.csv")
        
        # evaluate_evasion(X_val, y_val, baseline=True)
        
        for rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
            evaluate_evasion(X_val, y_val, evasion_rate=rate)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error interrupted runtime execution: {e}")