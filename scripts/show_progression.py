#!/usr/bin/env python
"""
Show the progression of model improvements:
73.68% (TF-IDF + Naive Bayes) → 89.47% (TF-IDF + LR) → 94.74% (LR+)
"""
import sys
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Add src directory to path for importing feats_extra
root = Path(__file__).parent
sys.path.insert(0, str(root / "src"))

def load_split_ids(p):
    return [x.strip() for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]

def evaluate_model(pipe, df_test, model_name):
    """Evaluate a model on test set and return metrics."""
    X_test = df_test["text"].tolist()
    y_test = df_test["label"].tolist()
    
    y_pred = pipe.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    # Confusion matrix
    y_test_binary = [1 if t == "good" else 0 for t in y_test]
    y_pred_binary = [1 if p == "good" else 0 for p in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_binary, labels=[0, 1]).ravel()
    
    return {
        "model": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "correct": tp + tn,
        "total": len(y_test)
    }

def main():
    print("=" * 80)
    print("RAP BATTLE JUDGE - MODEL PROGRESSION")
    print("=" * 80)
    print()
    
    # Load test set
    root = Path(__file__).parent
    df = pd.read_csv(root / "data/raw/dataset.csv")
    test_ids = load_split_ids(root / "data/processed/test.txt")
    df_test = df[df["id"].astype(str).isin(test_ids)].copy()
    
    print(f"Test set size: {len(df_test)} songs")
    print(f"  - Good songs: {sum(df_test['label'] == 'good')}")
    print(f"  - Bad songs: {sum(df_test['label'] == 'bad')}")
    print()
    
    results = []
    
    # Model 1: TF-IDF + Naive Bayes
    print("Loading Model 1: TF-IDF + Naive Bayes...")
    nb_model_path = root / "models" / "nb_v1" / "song_nb_v1_20251026.joblib"
    if not nb_model_path.exists():
        nb_model_path = root / "models" / "nb_v1" / "song_nb_v1_20251023.joblib"
    
    if nb_model_path.exists():
        pipe_nb = joblib.load(nb_model_path)
        metrics_nb = evaluate_model(pipe_nb, df_test, "TF-IDF + Naive Bayes")
        results.append(metrics_nb)
        print(f"[OK] Loaded: {nb_model_path.name}")
    else:
        print(f"[X] Model not found: {nb_model_path}")
        print("  Run: python src/train_baselines.py --in data/raw/dataset.csv")
        return
    
    print()
    
    # Model 2: TF-IDF + Logistic Regression
    print("Loading Model 2: TF-IDF + Logistic Regression...")
    lr_model_path = root / "models" / "lr_v1" / "song_lr_v1_20251026.joblib"
    if not lr_model_path.exists():
        lr_model_path = root / "models" / "lr_v1" / "song_lr_v1_20251023.joblib"
    
    if lr_model_path.exists():
        pipe_lr = joblib.load(lr_model_path)
        metrics_lr = evaluate_model(pipe_lr, df_test, "TF-IDF + Logistic Regression")
        results.append(metrics_lr)
        print(f"[OK] Loaded: {lr_model_path.name}")
    else:
        print(f"[X] Model not found: {lr_model_path}")
        print("  Run: python src/train_baselines.py --in data/raw/dataset.csv")
        return
    
    print()
    
    # Model 3: LR+ (Word + Char + Stats)
    print("Loading Model 3: LR+ (Word + Char + Statistical Features)...")
    lr_plus_model_path = root / "models" / "lr_v2_plus" / "song_lr_v2_plus_20251026.joblib"
    if not lr_plus_model_path.exists():
        lr_plus_model_path = root / "models" / "lr_v2_plus" / "song_lr_v2_plus.joblib"
    
    if lr_plus_model_path.exists():
        pipe_lr_plus = joblib.load(lr_plus_model_path)
        metrics_lr_plus = evaluate_model(pipe_lr_plus, df_test, "LR+ (Word+Char+Stats)")
        results.append(metrics_lr_plus)
        print(f"[OK] Loaded: {lr_plus_model_path.name}")
    else:
        print(f"[X] Model not found: {lr_plus_model_path}")
        print("  Run: python src/train_lr_plus.py --csv data/raw/dataset.csv")
        return
    
    print()
    print("=" * 80)
    print("PROGRESSION RESULTS (Test Set)")
    print("=" * 80)
    print()
    
    # Display results in a table
    print(f"{'Model':<40} {'Accuracy':<12} {'Macro-F1':<12} {'Correct':<10} {'Total':<8}")
    print("-" * 80)
    
    for i, m in enumerate(results, 1):
        acc_pct = m["accuracy"] * 100
        f1_pct = m["macro_f1"] * 100
        print(f"{i}. {m['model']:<38} {acc_pct:>6.2f}%     {f1_pct:>6.2f}%     {m['correct']:>3}/{m['total']:<5}")
    
    print()
    print("=" * 80)
    print("DETAILED BREAKDOWN")
    print("=" * 80)
    print()
    
    for i, m in enumerate(results, 1):
        acc_pct = m["accuracy"] * 100
        f1_pct = m["macro_f1"] * 100
        improvement = ""
        if i > 1:
            prev_acc = results[i-2]["accuracy"] * 100
            improvement = f" (+{acc_pct - prev_acc:.2f}%)"
        
        print(f"{i}. {m['model']}")
        print(f"   Accuracy: {acc_pct:.2f}%{improvement}")
        print(f"   Macro-F1: {f1_pct:.2f}%")
        print(f"   Confusion Matrix: TN={m['tn']}, FP={m['fp']}, FN={m['fn']}, TP={m['tp']}")
        print(f"   Correct: {m['correct']}/{m['total']} songs")
        print()
    
    print("=" * 80)
    print("PROGRESSION SUMMARY")
    print("=" * 80)
    print()
    print("Model Evolution:")
    print(f"  1. Baseline (TF-IDF + NB):     {results[0]['accuracy']*100:.2f}%")
    print(f"  2. Baseline (TF-IDF + LR):      {results[1]['accuracy']*100:.2f}%  (+{results[1]['accuracy']*100 - results[0]['accuracy']*100:.2f}%)")
    print(f"  3. Advanced (LR+ Features):    {results[2]['accuracy']*100:.2f}%  (+{results[2]['accuracy']*100 - results[1]['accuracy']*100:.2f}%)")
    print()
    print(f"Total Improvement: {results[2]['accuracy']*100 - results[0]['accuracy']*100:.2f}%")
    print(f"  ({results[0]['accuracy']*100:.2f}% → {results[2]['accuracy']*100:.2f}%)")
    print()

if __name__ == "__main__":
    main()

