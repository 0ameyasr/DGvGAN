import os
import random
import time
import glob
import argparse  # Added for regime selection
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

from .constants import API_TO_ID, STALLING_APIS, BENIGN_APIS, DEPENDENCY_CHAINS

# --- Setup Directory for Checkpoints ---
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEPENDENCY_CHAINS_IDS = [
    {API_TO_ID[api] for api in chain if api in API_TO_ID} 
    for chain in DEPENDENCY_CHAINS
]

CRITICAL_APIS = set().union(*DEPENDENCY_CHAINS_IDS)

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

def get_safe_windows(sequence):
    window = []
    windows = []
    start = None

    for i, api in enumerate(sequence):
        if api in CRITICAL_APIS:
            if start is not None:
                windows.append((start, i))
                start = None
        else:
            if start is None:
                start = i

    if start is not None:
        windows.append((start, len(sequence)))

    return windows

def same_chain(api1, api2):
    for chain in DEPENDENCY_CHAINS_IDS:
        if api1 in chain and api2 in chain:
            return True
    return False

def apply_stalling_padding(sequence, evasion_rate=0.3):
    """
    Simulates sandbox timeout evasion. Prepends stalling APIs to shift the real behavior out of the 
    observation window, then truncates to the original size.
    """
    seq = list(sequence)
    orig_len = len(seq)
    num_stalls = int(orig_len * evasion_rate)
    
    if num_stalls == 0 or not STALLING_APIS:
        return seq
        
    stalling_prefix = [random.choice(STALLING_APIS) for _ in range(num_stalls)]
    extended_seq = stalling_prefix + seq
    return extended_seq[:orig_len]


def apply_benign_mimicry(sequence, evasion_rate=0.3):
    """Replace some with benign calls."""
    seq = list(sequence)
    safe_windows = get_safe_windows(seq)
    
    safe_indices = []
    for start, end in safe_windows:
        safe_indices.extend(range(start, end))
        
    if not safe_indices or not BENIGN_APIS:
        return seq
        
    num_replacements = int(len(seq) * evasion_rate)
    num_replacements = min(num_replacements, len(safe_indices)) 
    indices_to_replace = random.sample(safe_indices, num_replacements)
    
    for idx in indices_to_replace:
        seq[idx] = random.choice(BENIGN_APIS)
        
    return seq

def apply_call_reordering(sequence, perturb_rate=0.2, max_attempt_factor=10):
    """
    Swaps independent sequential calls. Uses DEPENDENCY_CHAINS 
    to ensure that mutually dependent APIs are never reordered.
    """
    seq = list(sequence)
    windows = get_safe_windows(seq)
    safe_positions = []

    for start, end in windows:
        if end - start > 1:
            safe_positions.extend(range(start, end - 1))

    if not safe_positions:
        return seq

    target_swaps = int(len(safe_positions) * perturb_rate)
    successful_swaps = 0
    max_attempts = target_swaps * max_attempt_factor
    attempts = 0

    while successful_swaps < target_swaps and attempts < max_attempts:
        idx = random.choice(safe_positions)
        
        if idx + 1 >= len(seq):
            attempts += 1
            continue

        a = seq[idx]
        b = seq[idx + 1]

        seq[idx], seq[idx + 1] = seq[idx + 1], seq[idx]
        successful_swaps += 1
        attempts += 1
        
    return seq

def get_completed_runs():
    """Scans the checkpoint directory to determine which tasks are already completed."""
    completed = set()
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "run_*.csv"))
    for f in checkpoint_files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                seed = int(df.iloc[0]["Seed"])
                tech = str(df.iloc[0]["Technique"])
                rate_val = df.iloc[0]["Rate"]
                rate = float(rate_val) if rate_val != "N/A" else "N/A"
                completed.add((seed, tech, rate))
        except Exception:
            continue  
    return completed

def evaluate_evasion(X_val, y_val, target_techniques=None, baseline=False, start_regime=None):
    """
    Evaluates multiple deep learning engines with an active checkpoint fallback strategy.
    Saves individual progress incrementally to disk.
    
    start_regime: dict containing {"seed": int, "technique": str, "rate": float} or None
    """
    seeds = [10, 20, 30, 40, 50]
    malware_indices = np.where(y_val == 1)[0]
    X_malware = X_val[malware_indices]
    total_samples = len(X_malware)
    
    if total_samples == 0:
        print("No malware samples found in evaluation split.")
        return []

    completed_runs = get_completed_runs()
    if completed_runs:
        print(f"[+] Found {len(completed_runs)} already completed execution steps. Skipping those runs.")

    # Regime tracking: if no start regime is provided, we start instantly active.
    is_active = True
    if start_regime and not baseline:
        is_active = False
        print(f"[!] Target Regime requested. Skipping everything until: Seed={start_regime['seed']}, Tech={start_regime['technique']}, Rate={start_regime['rate']}")

    for seed in seeds:
        if baseline:
            techniques_runs = [("Baseline (No Evasion)", "N/A", lambda x: list(x))]
        else:
            techniques_runs = []
            for tech_name, rates in target_techniques.items():
                for rate in rates:
                    r_formatted = round(float(rate), 2)
                    if tech_name == "Stalling Padding":
                        techniques_runs.append((tech_name, r_formatted, lambda x, r=r_formatted: apply_stalling_padding(x, r)))
                    elif tech_name == "Benign Mimicry":
                        techniques_runs.append((tech_name, r_formatted, lambda x, r=r_formatted: apply_benign_mimicry(x, r)))
                    elif tech_name == "Call Reordering":
                        techniques_runs.append((tech_name, r_formatted, lambda x, r=r_formatted: apply_call_reordering(x, r)))

        for tech_name, rate, tech_func in techniques_runs:
            # Check if we hit the target start regime threshold
            if not is_active and start_regime:
                if (seed == start_regime["seed"] and 
                    tech_name.lower() == start_regime["technique"].lower() and 
                    abs(float(rate) - float(start_regime["rate"])) < 1e-4):
                    is_active = True
                    print(f"[+] Target regime reached! Activating execution matrix from this point forward.")
                else:
                    continue  # Keep skipping

            # Standard checkpoint check
            if (seed, tech_name, rate) in completed_runs:
                print(f" -> Skipping: Seed {seed} | {tech_name} | Rate: {rate} (Already Done)")
                continue
                
            print(f" -> Running: Seed {seed} | {tech_name} | Rate: {rate if isinstance(rate, str) else f'{rate:.2f}'}")
            
            detected = {m: 0 for m in MODELS_CONFIG.keys()}
            total_time = {m: 0.0 for m in MODELS_CONFIG.keys()}
            
            for sequence in tqdm(X_malware, desc=f"{tech_name} ({rate})", leave=False):
                evaded_seq = tech_func(sequence)
                
                for model_name, model_func in MODELS_CONFIG.items():
                    t_exec, prob = safe_predict(model_func, evaded_seq, seed=seed, name=model_name)
                    
                    if t_exec is not None:
                        total_time[model_name] += t_exec
                    if prob is not None and prob > 0.5:
                        detected[model_name] += 1
            
            run_records = []
            for m in MODELS_CONFIG.keys():
                det_rate = (detected[m] / total_samples) * 100 if total_samples > 0 else 0
                run_records.append({
                    "Seed": seed,
                    "Technique": tech_name,
                    "Rate": rate,
                    "Model": m,
                    "TPR (%)": f"{det_rate:.2f}%",
                    "Detected / Total": f"{detected[m]}/{total_samples}",
                    "Evasions Succeeded": total_samples - detected[m],
                    "Total Time (s)": f"{total_time[m]:.4f}"
                })
            
            safe_tech_name = tech_name.replace(" ", "_").replace("(", "").replace(")", "")
            checkpoint_filename = f"run_s{seed}_{safe_tech_name}_r{rate}.csv"
            checkpoint_filepath = os.path.join(CHECKPOINT_DIR, checkpoint_filename)
            
            pd.DataFrame(run_records).to_csv(checkpoint_filepath, index=False)

    all_summary_records = []
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "run_*.csv"))
    for f in checkpoint_files:
        try:
            df = pd.read_csv(f)
            all_summary_records.extend(df.to_dict(orient="records"))
        except Exception:
            continue

    return all_summary_records

def load_data(filepath="dataset.csv"):
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
    # Setup parsing configuration for specific regime starts
    parser = argparse.ArgumentParser(description="Evaluate evasion frameworks with custom starting regimes.")
    parser.add_argument("--start_seed", type=int, default=50, help="Seed to start evaluation from (e.g. 20)")
    parser.add_argument("--start_tech", type=str, default='Stalling Padding', help="Technique to start from (e.g. 'Stalling Padding')")
    parser.add_argument("--start_rate", type=float, default=0.3, help="Rate parameter to start from (e.g. 0.7)")
    args = parser.parse_args()

    regime = None
    if args.start_seed is not None and args.start_tech is not None and args.start_rate is not None:
        regime = {
            "seed": args.start_seed,
            "technique": args.start_tech,
            "rate": args.start_rate
        }

    try:
        _, X_val, _, y_val = load_data("dataset.csv")
        all_summary_records = []
        
        specific_rates = {
            "Stalling Padding": [0.3, 0.5, 0.7, 0.9],
            "Benign Mimicry": [0.1, 0.2, 0.3, 0.4],
            "Call Reordering": [0.05, 0.10, 0.15, 0.20]
        }
        
        print(f"\n{'='*75}\n EVASION EVALUATION WITH TARGETED EXPERIMENT RATES\n{'='*75}")
        evasion_records = evaluate_evasion(X_val, y_val, target_techniques=specific_rates, baseline=False, start_regime=regime)
        all_summary_records.extend(evasion_records)
        
        if all_summary_records:
            df_summary = pd.DataFrame(all_summary_records)
            if "Seed" in df_summary.columns:
                df_summary = df_summary.sort_values(by=["Seed", "Technique", "Rate", "Model"])

            output_csv_path = "evasion_evaluation_summary.csv"
            df_summary.to_csv(output_csv_path, index=False)
            
            print(f"\n{'='*75}\n REWARDEE DATA SUMMARY REPORT\n{'='*75}")
            print(df_summary.to_string(index=False))
            print(f"\n[+] Successfully exported evaluation results across all combinations to: '{output_csv_path}'")
        else:
            print("\n[!] No records to process.")
        
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error interrupted runtime execution: {e}")