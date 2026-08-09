import os
import json
import pandas as pd
import numpy as np
from pipeline.predictor import (
    predict_sgan,
    predict_dgcnn,
    predict_cnn,
    predict_dgcnn_sgan,
    predict_gat_sgan,
    predict_gat,
    predict_lstm,
    predict_transformer,
    predict_transformer_sgan,
    predict_lstm_sgan,
)

REPORTS_DIR = "reports/processed"

MODELS_CONFIG = {
    "sgan": {"func": predict_sgan, "name": "CNN-SGAN"},
    "dgcnn": {"func": predict_dgcnn, "name": "DGCNN"},
    "cnn": {"func": predict_cnn, "name": "CNN"},
    "hybrid": {"func": predict_dgcnn_sgan, "name": "DGCNN-SGAN"},
    "gat": {"func": predict_gat, "name": "GAT"},
    "hybrid_gat": {"func": predict_gat_sgan, "name": "GAT-SGAN"},
    "lstm": {"func": predict_lstm, "name": "LSTM"},
    "lstm_sgan": {"func": predict_lstm_sgan, "name": "LSTM-SGAN"},
    "trans": {"func": predict_transformer, "name": "TRANSFORMER"},
    "tsgan": {"func": predict_transformer_sgan, "name": "TRANSFORMER-SGAN"},
}

evaluated = False


def ensemble(preds):
    return sum(preds) / len(preds)


def extract_features_from_report(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    api_calls = data.get("api", [])

    if not isinstance(api_calls, list):
        api_calls = []

    return api_calls


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
    try:
        df = pd.read_csv("dataset.csv")
        return set(df["hash"].astype(str))
    except Exception as e:
        print(f"Warning: Could not load dataset.csv for data leakage checks: {e}")
        return set()


def get_benign_hashes():
    try:
        with open("benign.txt", "r") as b:
            hashes = b.read().split("\n")
        return set([hash for hash in hashes if hash])
    except Exception as e:
        print(f"Warning: Could not load benign.txt: {e}")
        return set()


def evaluate_subset(files, subset_name="TARGET", is_benign=False):
    global evaluated
    if not files:
        print(f"\nNo target files to process for {subset_name} subset!")
        return

    print(f"\n==================================================")
    print(f"STARTING EVALUATION FOR SUBSET: {subset_name} ({len(files)} files)")
    print(
        f"Metric target rule: {'Predictions < 0.50 (Benign Target)' if is_benign else 'Predictions >= 0.50 (Malware Target)'}"
    )
    print(f"==================================================")

    target_cols = list(MODELS_CONFIG.keys()) + ["final"]
    summary_metrics = {col: {} for col in target_cols}
    seeds = [
        10,
    ]  # 20, 30, 40, 50]

    for seed in seeds:
        print(f"\n[ Running Seed = {seed} ]")
        results_list = []

        for file in files:
            res = process_file(file, seed)
            if res:
                results_list.append(res)

        if not results_list:
            print(f"No successful outputs generated for seed {seed}.")
            continue

        rdf = pd.DataFrame(results_list)
        rdf = rdf.head(500)

        if not evaluated:
            rdf["sample"].to_csv("evaluated.csv", index=False)
            evaluated = True

        if is_benign:
            eval_benign = [h[:-5] for h in rdf["sample"].tolist()]
            with open("benign_evals.txt", "w") as f:
                f.write("\n".join(eval_benign))
            print(rdf)

        if not is_benign:
            eval_malware = [h[:-5] for h in rdf["sample"].tolist()]
            with open("malware_evals.txt", "w") as f:
                f.write("\n".join(eval_malware))
            print(rdf)

        print("\nOOD generalization summary:")
        for col in target_cols:
            valid_preds = rdf[col].dropna()
            if len(valid_preds) > 0:
                if is_benign:
                    correct_preds = valid_preds[valid_preds < 0.50]
                else:
                    correct_preds = valid_preds[valid_preds >= 0.50]

                rate = len(correct_preds) / len(rdf)
                print(f"{col} --> {rate:.2f}")
                summary_metrics[col][seed] = rate
            else:
                print(f"{col} --> N/A (All models failed)")
                summary_metrics[col][seed] = np.nan

    print("\n" + "=" * 50)
    print(f"FINAL SUMMARY PER SEED AND MODEL ({subset_name})")
    print("=" * 50)

    summary_df = pd.DataFrame(summary_metrics).T
    summary_df.columns = [f"Seed {s}" for s in seeds]
    print(summary_df.to_string())

    print("\n" + "-" * 50)
    print(f"AGGREGATE STATISTICS (Mean +/- Std) : {subset_name}")
    print("-" * 50)

    for col in target_cols:
        rates = [
            summary_metrics[col][s]
            for s in seeds
            if not np.isnan(summary_metrics[col][s])
        ]
        if rates:
            mean_val = np.mean(rates)
            std_val = np.std(rates)
            print(f"{col:<12} --> {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"{col:<12} --> N/A (No valid runs)")
    print("=" * 50)


def run(mode="separate"):
    """
    Supported modes:
      - "separate" : Runs Malware (>=0.5) and Benign (<0.5) subsets independently.
      - "malware"  : Runs ONLY the malware subset (>=0.5).
      - "benign"   : Runs ONLY the benign subset (<0.5).
      - "all"      : Runs all valid files combined together (Defaults to >=0.5).
    """
    if not os.path.exists(REPORTS_DIR):
        print("Reports directory not found!")
        return

    all_files = [
        f
        for f in os.listdir(REPORTS_DIR)
        if os.path.isfile(os.path.join(REPORTS_DIR, f)) and f.lower().endswith(".json")
    ]

    if not all_files:
        print("No JSON files found!")
        return

    training_hashes = load_training_hashes()
    benign_hashes = get_benign_hashes()

    malware_files = []
    benign_files = []

    for f in all_files:
        md5_hash = f[:-5]

        if md5_hash in training_hashes:
            print(f"Skipping Leakage/Compromised MALWARE Sample: {f}")
            continue

        if md5_hash in benign_hashes:
            benign_files.append(f)
        else:
            malware_files.append(f)

    if mode == "separate":
        evaluate_subset(malware_files, subset_name="MALWARE ONLY", is_benign=False)
        evaluate_subset(benign_files, subset_name="BENIGN ONLY", is_benign=True)

    elif mode == "malware":
        evaluate_subset(malware_files, subset_name="MALWARE ONLY", is_benign=False)

    elif mode == "benign":
        evaluate_subset(benign_files, subset_name="BENIGN ONLY", is_benign=True)

    elif mode == "all":
        combined_files = malware_files + benign_files
        evaluate_subset(
            combined_files, subset_name="ALL SAMPLES COMBINED", is_benign=False
        )

    else:
        print(
            f"Invalid mode '{mode}' selected. Choose from: 'separate', 'malware', 'benign', 'all'"
        )


if __name__ == "__main__":
    run(mode="separate")
