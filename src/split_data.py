#!/usr/bin/env python
import argparse, pandas as pd, numpy as np, pathlib, json, re

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_csv", required=True, help="Path to raw CSV with id,text,label")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--valid", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    inp = pathlib.Path(args.input_csv)
    df = pd.read_csv(inp)
    assert {"id","text","label"}.issubset(df.columns), "CSV must contain id,text,label"

    # Minimal clean: strip control chars from text
    df["text"] = df["text"].astype(str).str.replace(r"[\r\n\t]+", " ", regex=True).str.strip()

    # Shuffle
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    df = df.iloc[idx].reset_index(drop=True)

    n = len(df)
    n_train = int(n * args.train)
    n_valid = int(n * args.valid)
    n_test  = n - n_train - n_valid

    train_ids = df.iloc[:n_train]["id"].tolist()
    valid_ids = df.iloc[n_train:n_train+n_valid]["id"].tolist()
    test_ids  = df.iloc[n_train+n_valid:]["id"].tolist()

    out_dir = pathlib.Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir/"train.txt").write_text("\n".join(train_ids), encoding="utf-8")
    (out_dir/"valid.txt").write_text("\n".join(valid_ids), encoding="utf-8")
    (out_dir/"test.txt").write_text("\n".join(test_ids), encoding="utf-8")

    print(f"Wrote splits: train={len(train_ids)}, valid={len(valid_ids)}, test={len(test_ids)} → data/processed/*.txt")

if __name__ == "__main__":
    main()
