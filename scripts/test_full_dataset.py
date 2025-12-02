#!/usr/bin/env python
"""Test model on entire dataset to check accuracy and overfitting."""
import argparse
import joblib
import pandas as pd
from pathlib import Path
import sys
import re
import warnings

# Ignore scikit-learn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

def load_split_ids(p: Path):
    """Load song IDs from split file."""
    if not p.exists():
        return set()
    return set(x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip())

def main():
    ap = argparse.ArgumentParser(
        description="Test model on entire dataset to check accuracy and overfitting"
    )
    ap.add_argument("--model", required=True, help="Path to trained model (.joblib)")
    ap.add_argument("--dataset", default="data/raw/dataset.csv", help="Dataset CSV file")
    ap.add_argument("--th_good", type=float, default=0.5, help="Threshold for 'good' classification")
    ap.add_argument("--out", default="reports/full_dataset_test.csv", help="Output CSV file")
    args = ap.parse_args()
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    
    if "v2_plus" in str(model_path):
        src_dir = Path(__file__).parent / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
    
    print(f"Loading model: {model_path}", file=sys.stderr)
    pipe = joblib.load(model_path)
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(df)} songs", file=sys.stderr)
    
    train_ids = load_split_ids(Path("data/processed/train.txt"))
    valid_ids = load_split_ids(Path("data/processed/valid.txt"))
    test_ids = load_split_ids(Path("data/processed/test.txt"))
    
    print(f"Loaded splits: train={len(train_ids)}, valid={len(valid_ids)}, test={len(test_ids)}", file=sys.stderr)
    results = []
    for idx, row in df.iterrows():
        song_id = str(row["id"])
        true_label = row["label"]
        text = row["text"]
        
        text_processed = re.sub(r"[\r\n\t]+", " ", text).strip()
        try:
            proba = pipe.predict_proba([text_processed])[0]
            clf = pipe.named_steps.get("clf", None)
            classes = list(getattr(clf, "classes_", [])) if clf else ["bad", "good"]
            
            if "good" in classes and "bad" in classes:
                i_good = classes.index("good")
                i_bad = classes.index("bad")
            else:
                i_bad, i_good = 0, 1
            
            p_good = float(proba[i_good])
            p_bad = float(proba[i_bad])
            pred = "good" if p_good >= args.th_good else "bad"
            correct = (pred == true_label)
            
            split = "unknown"
            if song_id in train_ids:
                split = "train"
            elif song_id in valid_ids:
                split = "valid"
            elif song_id in test_ids:
                split = "test"
            
            results.append({
                "id": song_id,
                "true_label": true_label,
                "pred": pred,
                "p_good": p_good,
                "p_bad": p_bad,
                "correct": correct,
                "split": split
            })
            
        except Exception as e:
            print(f"Error predicting song {song_id}: {e}", file=sys.stderr)
            continue
    
    df_results = pd.DataFrame(results)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path}", file=sys.stderr)
    
    print("\n" + "=" * 70, file=sys.stderr)
    print("FULL DATASET EVALUATION", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Total songs tested: {len(df_results)}", file=sys.stderr)
    
    if len(df_results) > 0:
        overall_correct = df_results["correct"].sum()
        overall_acc = overall_correct / len(df_results) * 100
        print(f"\nOVERALL ACCURACY: {overall_correct}/{len(df_results)} ({overall_acc:.2f}%)", file=sys.stderr)
        
        print(f"\nBy Label:", file=sys.stderr)
        for label in ["good", "bad"]:
            label_data = df_results[df_results["true_label"] == label]
            if len(label_data) > 0:
                label_correct = label_data["correct"].sum()
                label_acc = label_correct / len(label_data) * 100
                print(f"  {label.upper()}: {label_correct}/{len(label_data)} ({label_acc:.2f}%)", file=sys.stderr)
        
        print(f"\nBy Split (Overfitting Check):", file=sys.stderr)
        split_results = {}
        for split in ["train", "valid", "test"]:
            split_data = df_results[df_results["split"] == split]
            if len(split_data) > 0:
                split_correct = split_data["correct"].sum()
                split_acc = split_correct / len(split_data) * 100
                split_results[split] = split_acc
                print(f"  {split.upper()}: {split_correct}/{len(split_data)} ({split_acc:.2f}%)", file=sys.stderr)
        
        print(f"\nOverfitting Analysis:", file=sys.stderr)
        if "train" in split_results and "test" in split_results:
            train_acc = split_results["train"]
            test_acc = split_results["test"]
            diff = train_acc - test_acc
            
            if diff > 10:
                print(f"  ⚠️  WARNING: Potential overfitting detected!", file=sys.stderr)
                print(f"     Train accuracy ({train_acc:.2f}%) is {diff:.2f}% higher than test ({test_acc:.2f}%)", file=sys.stderr)
            elif diff > 5:
                print(f"  ⚠️  Minor overfitting: Train ({train_acc:.2f}%) > Test ({test_acc:.2f}%) by {diff:.2f}%", file=sys.stderr)
            else:
                print(f"  ✅ No overfitting: Train ({train_acc:.2f}%) and Test ({test_acc:.2f}%) are similar (diff: {diff:.2f}%)", file=sys.stderr)
        
        if "train" in split_results and "valid" in split_results:
            train_acc = split_results["train"]
            valid_acc = split_results["valid"]
            diff = train_acc - valid_acc
            
            if diff > 10:
                print(f"  ⚠️  WARNING: Train accuracy ({train_acc:.2f}%) is {diff:.2f}% higher than validation ({valid_acc:.2f}%)", file=sys.stderr)
            elif diff > 5:
                print(f"  ⚠️  Minor gap: Train ({train_acc:.2f}%) > Valid ({valid_acc:.2f}%) by {diff:.2f}%", file=sys.stderr)
            else:
                print(f"  ✅ Train ({train_acc:.2f}%) and Valid ({valid_acc:.2f}%) are similar (diff: {diff:.2f}%)", file=sys.stderr)
        
        print(f"\nPrediction Distribution:", file=sys.stderr)
        pred_counts = df_results["pred"].value_counts()
        for pred, count in pred_counts.items():
            pct = count / len(df_results) * 100
            print(f"  {pred.upper()}: {count} ({pct:.1f}%)", file=sys.stderr)
        
        avg_p_good = df_results["p_good"].mean()
        print(f"\nAverage p_good: {avg_p_good:.3f}", file=sys.stderr)
        
        errors = df_results[df_results["correct"] == False]
        if len(errors) > 0:
            print(f"\nErrors ({len(errors)} total):", file=sys.stderr)
            for _, err in errors.head(10).iterrows():
                print(f"  {err['id']:40s} | true={err['true_label']:4s} pred={err['pred']:4s} | p_good={err['p_good']:.3f} | split={err['split']}", file=sys.stderr)
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors", file=sys.stderr)
        else:
            print(f"\n✅ No errors - perfect accuracy!", file=sys.stderr)
    
    print("=" * 70, file=sys.stderr)

if __name__ == "__main__":
    main()

