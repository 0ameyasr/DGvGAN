import os
import traceback
import numpy
import pandas 
import random
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from scripts.predictor import (
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

def set_seed(seed=42):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    
OOD_CSV_PATH = "data/ood.csv"
DATASET_CSV_PATH = "data/dataset.csv"
SEEDS = [10, 20, 30, 40, 50]

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

def load_training_hashes(path=DATASET_CSV_PATH):
    try:
        df = pandas.read_csv(path)
        return set(df["hash"].astype(str))
    except Exception as e:
        print(f"Could not load {path} for data leakage check: {e}")
        return set()

def safe_predict(func, features, seed=10, name="DEFAULT", log_first_error=False):
    try:
        sanitized_features = [int(x) for x in features]
        tp, p = func(sanitized_features, seed)
        if p is None or numpy.isnan(p):
            if log_first_error:
                print(f"{name} returned NaN or None with raw output: ({tp}, {p})")
            return None, numpy.nan
        p_clamped = float(numpy.clip(p, 0.0, 1.0))
        return tp, p_clamped
    except Exception as e:
        if log_first_error:
            print(f"\nException caught in {name}: {e}")
            traceback.print_exc()
        return None, numpy.nan


def compute_metrics(y_true, y_scores, threshold=0.5, fallback_score=0.5):
    y_true = numpy.array(y_true, dtype=int)
    y_scores = numpy.array(y_scores, dtype=float)

    total_count = len(y_true)
    valid_mask = ~numpy.isnan(y_scores)
    valid_count = int(valid_mask.sum())

    if total_count == 0 or len(numpy.unique(y_true)) < 2:
        return {
            "AUC-ROC": numpy.nan,
            "TPR": numpy.nan,
            "TNR": numpy.nan,
            "valid_count": valid_count,
            "total_count": total_count,
        }

    y_scores_imputed = numpy.where(valid_mask, y_scores, fallback_score)

    try:
        auc = roc_auc_score(y_true, y_scores_imputed)
    except Exception:
        auc = numpy.nan

    y_pred = (y_scores_imputed >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else numpy.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else numpy.nan
    return {
        "AUC-ROC": auc,
        "TPR": tpr,
        "TNR": tnr,
        "valid_count": valid_count,
        "total_count": total_count,
    }


def run_multi_seed_evaluation(seeds=SEEDS, check_leakage=True, min_ensemble_quorum_ratio=0.5):
    set_seed()
    print(f"Loading OOD samples from {OOD_CSV_PATH}")
    df = pandas.read_csv(OOD_CSV_PATH)

    # Data leakage check
    if check_leakage and "hash" in df.columns:
        train_hashes = load_training_hashes()
        if train_hashes:
            init_len = len(df)
            df = df[~df["hash"].astype(str).isin(train_hashes)].reset_index(drop=True)
            dropped = init_len - len(df)
            if dropped > 0:
                print(f"Filtered out {dropped} training leakage samples...")

    print(f"Total test instances: {len(df)}")
    feature_cols = [f"t_{i}" for i in range(100)]
    df[feature_cols] = df[feature_cols].astype(int)
    y_true = df["malware"].astype(int).values
    eval_targets = list(MODELS_CONFIG.keys()) + ["final"]
    total_base_models = len(MODELS_CONFIG)
    min_quorum = int(numpy.ceil(total_base_models * min_ensemble_quorum_ratio))

    all_seed_results = {s: {} for s in seeds}
    logged_errors = {key: False for key in eval_targets}
    for s_idx, seed in enumerate(seeds):
        print(f"STARTING SEED {seed} ({s_idx + 1}/{len(seeds)})...")
        seed_predictions = {key: [] for key in eval_targets}
        for _, row in df.iterrows():
            features = row[feature_cols].tolist()
            sample_preds = []
            for key, cfg in MODELS_CONFIG.items():
                should_log = not logged_errors[key]
                tp, p = safe_predict(
                    cfg["func"],
                    features,
                    seed=seed,
                    name=cfg["name"],
                    log_first_error=should_log,
                )
                if should_log and (p is None or numpy.isnan(p)):
                    logged_errors[key] = True

                seed_predictions[key].append(p)
                if not numpy.isnan(p):
                    sample_preds.append(p)
                    
            if len(sample_preds) >= min_quorum:
                seed_predictions["final"].append(float(numpy.mean(sample_preds)))
            else:
                seed_predictions["final"].append(numpy.nan)

        print(f"\nMetrics for seed {seed}:")
        seed_table = []
        for key in eval_targets:
            metrics = compute_metrics(y_true, seed_predictions[key])
            all_seed_results[seed][key] = metrics

            name = MODELS_CONFIG[key]["name"] if key != "final" else "ENSEMBLE"
            seed_table.append(
                {
                    "Model": name,
                    "AUC-ROC": f"{metrics['AUC-ROC']:.4f}" if not numpy.isnan(metrics["AUC-ROC"]) else "NaN",
                    "TPR": f"{metrics['TPR']:.4f}" if not numpy.isnan(metrics["TPR"]) else "NaN",
                    "TNR": f"{metrics['TNR']:.4f}" if not numpy.isnan(metrics["TNR"]) else "NaN",
                    "Valid Preds": f"{metrics['valid_count']}/{metrics['total_count']}",
                }
            )
        print(pandas.DataFrame(seed_table).to_string(index=False))

    # Aggregate cross-seed statistics
    print("Final summary across seeds (mean ± std with ddof=1):")
    summary_records = []
    for key in eval_targets:
        name = MODELS_CONFIG[key]["name"] if key != "final" else "ENSEMBLE"
        aucs = [all_seed_results[s][key]["AUC-ROC"] for s in seeds if not numpy.isnan(all_seed_results[s][key]["AUC-ROC"])]
        tprs = [all_seed_results[s][key]["TPR"] for s in seeds if not numpy.isnan(all_seed_results[s][key]["TPR"])]
        tnrs = [all_seed_results[s][key]["TNR"] for s in seeds if not numpy.isnan(all_seed_results[s][key]["TNR"])]
        
        def format_mean_std(vals):
            if len(vals) == 0:
                return "N/A"
            if len(vals) == 1:
                return f"{vals[0]:.4f} ± 0.0000"
            return f"{numpy.mean(vals):.4f} ± {numpy.std(vals, ddof=1):.4f}"
        summary_records.append(
            {
                "Model": name,
                "AUC-ROC (mean ± std)": format_mean_std(aucs),
                "TPR (mean ± std)": format_mean_std(tprs),
                "TNR (mean ± std)": format_mean_std(tnrs),
            }
        )

    final_df = pandas.DataFrame(summary_records)
    print(final_df.to_string(index=False))

if __name__ == "__main__":
    run_multi_seed_evaluation(seeds=SEEDS, check_leakage=True)