import os
import pandas as pd
import numpy as np
from feature_engineering.extractor import extract_features_from_report
from pipeline.predictor import (
    predict_sgan,
    predict_dgcnn,
    predict_cnn,
    predict_dgcnn_sgan,
    predict_gat_sgan,
    predict_gat,
    predict_lstm
)
from models.ensemble import ensemble

REPORTS_DIR = "sandbox/reports/processed"

MODELS_CONFIG = {
    "sgan": {"func": predict_sgan, "name": "SGAN"},
    "dgcnn": {"func": predict_dgcnn, "name": "DGCNN"},
    "cnn": {"func": predict_cnn, "name": "CNN"},
    "hybrid": {"func": predict_dgcnn_sgan, "name": "HYBRID-DGCNN"},
    "gat": {"func": predict_gat, "name": "GAT"},
    "hybrid_gat": {"func": predict_gat_sgan, "name": "HYBRID-GAT"},
    "lstm": {"func": predict_lstm, "name": "LSTM"},
}

evaluated = False

def safe_predict(func, features, seed=10, name="DEFAULT", silent=True):
    try:
        return func(features, seed)
    except Exception as e:
        if not silent:
            print(f"{name} failed: {e}")
        return None, None


def process_file(file, seed, silent=True):
    path = os.path.join(REPORTS_DIR, file)
    try:
        features = extract_features_from_report(path)
        # print(features)
    except Exception as e:
        print(f"\nError extracting features from {file}: {e}")
        return None

    results = {"sample": file}
    preds_to_ensemble = []

    for key, cfg in MODELS_CONFIG.items():
        tp, p = safe_predict(cfg["func"], features, seed, cfg["name"])
        
        results[key] = round(p, 3) if p is not None else None
        results[f"t_{key}"] = round(tp, 9) if tp is not None else None
        
        if p is not None and not np.isnan(p):
            preds_to_ensemble.append(p)

    if not preds_to_ensemble:
        if not silent:
            print(f"All models failed for: {file}")
        return None

    try:
        final = ensemble(preds_to_ensemble)
    except Exception:
        final = sum(preds_to_ensemble) / len(preds_to_ensemble)

    results["final"] = round(final, 3)
    results["malware"] = "MALWARE" if final > 0.5 else "BENIGN"
    
    return results


def load_training_hashes():
    """Loads and returns training dataset hashes to identify leaked data upfront."""
    try:
        df = pd.read_csv("dataset.csv")
        return set(df['hash'].astype(str))
    except Exception as e:
        print(f"Warning: Could not load dataset.csv for data leakage checks: {e}")
        return set()


def run():
    global evaluated
    if not os.path.exists(REPORTS_DIR):
        print("Reports directory not found!")
        return

    all_files = [
        f for f in os.listdir(REPORTS_DIR)
        if os.path.isfile(os.path.join(REPORTS_DIR, f)) and f.lower().endswith(".json")
    ]

    if not all_files:
        print("No JSON files found!")
        return

    # Load compromised hashes
    training_hashes = load_training_hashes()
    
    # FIXED: Upfront tracking filtering to completely drop compromised validation instances
    files = []
    for f in all_files:
        md5_hash = f[:-5] # Drops '.json'
        if md5_hash in training_hashes:
            print(f"Skipping Leakage/Compromised Sample: {f}")
        else:
            files.append(f)

    if not files:
        print("All target files were marked compromised/filtered out!")
        return

    target_cols = list(MODELS_CONFIG.keys()) + ["final"]
    summary_metrics = {col: {} for col in target_cols}
    seeds = [10, 20, 30, 40, 50]

    for seed in seeds:
        print(f"\n--- Running Seed = {seed} ---")
        results_list = []

        for file in files:
            res = process_file(file, seed)
            if res:
                results_list.append(res)
        
        if not results_list:
            print(f"No successful outputs generated for seed {seed}.")
            continue

        rdf = pd.DataFrame(results_list)
        
        if not evaluated:
            rdf['sample'].to_csv("evaluated.csv", index=False)
            evaluated = True
        
        print(rdf)
        
        print("\nOOD generalization summary:")
        for col in target_cols:
            valid_preds = rdf[col].dropna()
            if len(valid_preds) > 0:
                rate = len(valid_preds[valid_preds >= 0.50]) / len(rdf)
                print(f"{col} --> {rate:.2f}")
                summary_metrics[col][seed] = rate
            else:
                print(f"{col} --> N/A (All models failed)")
                summary_metrics[col][seed] = np.nan

    # --- FINAL SUMMARY SECTION ---
    print("\n" + "="*50)
    print("FINAL SUMMARY PER SEED AND MODEL")
    print("="*50)
    
    summary_df = pd.DataFrame(summary_metrics).T
    summary_df.columns = [f"Seed {s}" for s in seeds]
    print(summary_df.to_string())
    
    print("\n" + "-"*50)
    print("AGGREGATE STATISTICS (Mean ± Std)")
    print("-"*50)
    
    for col in target_cols:
        rates = [summary_metrics[col][s] for s in seeds if not np.isnan(summary_metrics[col][s])]
        if rates:
            mean_val = np.mean(rates)
            std_val = np.std(rates)
            print(f"{col:<12} --> {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"{col:<12} --> N/A (No valid runs)")
    print("="*50)


if __name__ == "__main__":
    run()