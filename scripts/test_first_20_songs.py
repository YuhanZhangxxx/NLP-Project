#!/usr/bin/env python
"""Test first N songs from dataset to check for overfitting."""
import argparse
import joblib
import pandas as pd
from pathlib import Path
import sys
import re

def main():
    ap = argparse.ArgumentParser(
        description="Test first 20 songs from dataset to check for overfitting"
    )
    ap.add_argument("--model", required=True, help="Path to trained model (.joblib)")
    ap.add_argument("--dataset", default="data/raw/dataset.csv", help="Dataset CSV file")
    ap.add_argument("--num_songs", type=int, default=20, help="Number of songs to test (default: 20)")
    ap.add_argument("--th_good", type=float, default=0.5, help="Threshold for 'good' classification")
    ap.add_argument("--out", default="reports/first_20_songs_test.csv", help="Output CSV file")
    ap.add_argument("--check-splits", action="store_true", help="Check which split each song belongs to")
    args = ap.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    
    # Always add src/ and skill_detection/ to path for v2_plus models
    src_dir = Path(__file__).parent.parent / "src"
    skill_dir = Path(__file__).parent.parent / "skill_detection"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(skill_dir) not in sys.path:
        sys.path.insert(0, str(skill_dir))
    
    print(f"Loading model: {model_path}", file=sys.stderr)
    pipe = joblib.load(model_path)
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(df)} songs", file=sys.stderr)
    
    df_test = df.head(args.num_songs).copy()
    print(f"Testing first {len(df_test)} songs from dataset", file=sys.stderr)
    
    split_info = {}
    if args.check_splits:
        for split_name in ["train", "valid", "test"]:
            split_file = Path(f"data/processed/{split_name}.txt")
            if split_file.exists():
                split_ids = set(split_file.read_text().splitlines())
                split_info[split_name] = split_ids
                print(f"Loaded {split_name} split: {len(split_ids)} songs", file=sys.stderr)
    
    results = []
    for idx, row in df_test.iterrows():
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
            if args.check_splits:
                for split_name, split_ids in split_info.items():
                    if song_id in split_ids:
                        split = split_name
                        break
            
            results.append({
                "id": song_id,
                "true_label": true_label,
                "pred": pred,
                "p_good": p_good,
                "p_bad": p_bad,
                "correct": correct,
                "split": split,
                "text_preview": text[:100] + "..." if len(text) > 100 else text
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
    print("SUMMARY", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Total songs tested: {len(df_results)}", file=sys.stderr)
    
    if len(df_results) > 0:
        correct_count = df_results["correct"].sum()
        accuracy = correct_count / len(df_results) * 100
        print(f"Correct predictions: {correct_count}/{len(df_results)} ({accuracy:.1f}%)", file=sys.stderr)
        
        print(f"\nBy true label:", file=sys.stderr)
        for label in ["good", "bad"]:
            label_data = df_results[df_results["true_label"] == label]
            if len(label_data) > 0:
                label_correct = label_data["correct"].sum()
                label_acc = label_correct / len(label_data) * 100
                print(f"  {label.upper()}: {label_correct}/{len(label_data)} correct ({label_acc:.1f}%)", file=sys.stderr)
        
        if args.check_splits and "split" in df_results.columns:
            print(f"\nBy split:", file=sys.stderr)
            for split in ["train", "valid", "test"]:
                split_data = df_results[df_results["split"] == split]
                if len(split_data) > 0:
                    split_correct = split_data["correct"].sum()
                    split_acc = split_correct / len(split_data) * 100
                    print(f"  {split.upper()}: {split_correct}/{len(split_data)} correct ({split_acc:.1f}%)", file=sys.stderr)
        
        print(f"\nPrediction distribution:", file=sys.stderr)
        pred_counts = df_results["pred"].value_counts()
        for pred, count in pred_counts.items():
            print(f"  {pred.upper()}: {count}", file=sys.stderr)
        
        avg_p_good = df_results["p_good"].mean()
        print(f"\nAverage p_good: {avg_p_good:.3f}", file=sys.stderr)
        
        errors = df_results[df_results["correct"] == False]
        if len(errors) > 0:
            print(f"\nErrors ({len(errors)}):", file=sys.stderr)
            for _, err in errors.iterrows():
                print(f"  {err['id']:30s} | true={err['true_label']:4s} pred={err['pred']:4s} | p_good={err['p_good']:.3f}", file=sys.stderr)
    
    print("=" * 70, file=sys.stderr)

if __name__ == "__main__":
    main()

