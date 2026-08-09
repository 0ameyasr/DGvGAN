import os
import random
import glob
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from .constants import API_TO_ID, STALLING_APIS, BENIGN_APIS, DEPENDENCY_CHAINS

DEPENDENCY_CHAINS_IDS = [
    {API_TO_ID[api] for api in chain if api in API_TO_ID} for chain in DEPENDENCY_CHAINS
]

CRITICAL_APIS = set().union(*DEPENDENCY_CHAINS_IDS)


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


def apply_stalling_padding(sequence, evasion_rate=0.3):
    seq = list(sequence)
    orig_len = len(seq)
    num_stalls = int(orig_len * evasion_rate)

    if num_stalls == 0 or not STALLING_APIS:
        return seq

    stalling_prefix = [random.choice(STALLING_APIS) for _ in range(num_stalls)]
    extended_seq = stalling_prefix + seq
    return extended_seq[:orig_len]


def apply_benign_mimicry(sequence, evasion_rate=0.3):
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


def evaluate_perturbation_impact(X_val, y_val, target_techniques):
    """
    Applies perturbation techniques to malware samples and counts how many
    sequences remained completely unaffected by the processing.
    """
    seeds = [10, 20, 30, 40, 50]
    malware_indices = np.where(y_val == 1)[0]
    X_malware = X_val[malware_indices]
    total_samples = len(X_malware)

    if total_samples == 0:
        print("No malware samples found in evaluation split.")
        return

    all_records = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        techniques_runs = []
        for tech_name, rates in target_techniques.items():
            for rate in rates:
                r_formatted = round(float(rate), 2)
                if tech_name == "Stalling Padding":
                    techniques_runs.append(
                        (
                            tech_name,
                            r_formatted,
                            lambda x, r=r_formatted: apply_stalling_padding(x, r),
                        )
                    )
                elif tech_name == "Benign Mimicry":
                    techniques_runs.append(
                        (
                            tech_name,
                            r_formatted,
                            lambda x, r=r_formatted: apply_benign_mimicry(x, r),
                        )
                    )
                elif tech_name == "Call Reordering":
                    techniques_runs.append(
                        (
                            tech_name,
                            r_formatted,
                            lambda x, r=r_formatted: apply_call_reordering(x, r),
                        )
                    )

        for tech_name, rate, tech_func in techniques_runs:
            unaffected_count = 0

            for sequence in tqdm(
                X_malware,
                desc=f"Processing: {tech_name} ({rate}) Seed {seed}",
                leave=False,
            ):
                orig_seq = list(sequence)
                perturbed_seq = tech_func(orig_seq)
                if orig_seq == perturbed_seq:
                    unaffected_count += 1

            unaffected_pct = (unaffected_count / total_samples) * 100

            all_records.append(
                {
                    "Seed": seed,
                    "Technique": tech_name,
                    "Rate": rate,
                    "Total Samples": total_samples,
                    "Unaffected Samples": unaffected_count,
                    "Unaffected (%)": f"{unaffected_pct:.2f}%",
                }
            )

    df_summary = pd.DataFrame(all_records)
    df_summary = df_summary.sort_values(by=["Seed", "Technique", "Rate"])

    print(
        f"\n{'='*75}\n ADVERSARIAL PERTURBATION IMPACT REPORT (UNAFFECTED SEQUENCES)\n{'='*75}"
    )
    print(df_summary.to_string(index=False))

    output_csv_path = "perturbation_impact_summary.csv"
    df_summary.to_csv(output_csv_path, index=False)
    print(f"\n[+] Results successfully exported to: '{output_csv_path}'")


def load_data(filepath="dataset.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing required execution file: {filepath}")

    df = pd.read_csv(filepath)
    if "hash" in df.columns:
        df = df.drop(columns=["hash"])

    X = df.drop(columns=["malware"]).values.astype(np.float32)
    y = df["malware"].values.astype(np.int64)

    return train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)


if __name__ == "__main__":
    try:
        _, X_val, _, y_val = load_data("dataset.csv")

        specific_rates = {
            "Stalling Padding": [0.3, 0.5, 0.7, 0.9],
            "Benign Mimicry": [0.1, 0.2, 0.3, 0.4],
            "Call Reordering": [0.05, 0.10, 0.15, 0.20],
        }

        evaluate_perturbation_impact(X_val, y_val, target_techniques=specific_rates)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error interrupted runtime execution: {e}")
