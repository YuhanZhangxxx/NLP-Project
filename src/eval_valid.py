import argparse, sys, json
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def load_split_ids(p: Path):
    # read ids line by line (supports plain id text)
    with p.open("r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    return set(ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="valid", choices=["valid","test","train"])
    ap.add_argument("--th", type=float, default=None, help="single threshold; if not provided, do scan")
    ap.add_argument("--th_start", type=float, default=0.40)
    ap.add_argument("--th_end", type=float, default=0.80)
    ap.add_argument("--th_step", type=float, default=0.05)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root)
    ds = pd.read_csv(root/"data/raw/dataset.csv")
    split_file = root/f"data/processed/{args.split}.txt"
    if not split_file.exists():
        print(f"Split file not found: {split_file}", file=sys.stderr)
        sys.exit(2)
    keep = load_split_ids(split_file)
    df = ds[ds["id"].astype(str).isin(keep)].copy()
    if df.empty:
        print(f"No rows for split={args.split}.", file=sys.stderr)
        sys.exit(3)

    pipe = load(args.model)
    classes = list(pipe.classes_)
    if "good" not in classes or "bad" not in classes:
        print(f"Classes in model: {classes}. Need 'good' and 'bad'.", file=sys.stderr)
        sys.exit(4)
    good_idx = classes.index("good")

    proba = pipe.predict_proba(df["text"].tolist())
    p_good = proba[:, good_idx]
    y_true = (df["label"] == "good").astype(int).values

    def eval_at(th):
        y_pred = (p_good >= th).astype(int)
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        return {"th":th, "accuracy":acc, "macro_f1":f1, "tn":tn, "fp":fp, "fn":fn, "tp":tp}

    out = []
    if args.th is not None:
        out.append(eval_at(args.th))
    else:
        th = args.th_start
        while th <= args.th_end + 1e-9:
            out.append(eval_at(round(th, 4)))
            th += args.th_step

    res = pd.DataFrame(out).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    reports = root/"reports"
    reports.mkdir(parents=True, exist_ok=True)
    out_csv = reports/f"{args.split}_threshold_scan.csv"
    res.to_csv(out_csv, index=False)
    best = res.iloc[0].to_dict()
    print(f"[{args.split}] best macro-F1 @ th={best['th']:.2f}  "
          f"F1={best['macro_f1']:.4f}  acc={best['accuracy']:.4f}  "
          f"cm=[tn={int(best['tn'])}, fp={int(best['fp'])}, fn={int(best['fn'])}, tp={int(best['tp'])}]")
    if args.th is not None:
        print(res.head(1).to_string(index=False))
        print("(single threshold result written)")
    else:
        print(f"Full scan written → {out_csv}")

if __name__ == "__main__":
    main()
