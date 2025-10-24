#!/usr/bin/env python
import argparse, pathlib, pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Merge transcripts (id,text,label?) with a separate labels.csv (id,label)")
    ap.add_argument("--trans", required=True, help="CSV with columns id,text,label (label can be empty)")
    ap.add_argument("--labels", required=True, help="CSV with columns id,label")
    ap.add_argument("--out", default="data/raw/dataset.csv", help="Output merged dataset path")
    args = ap.parse_args()

    trans = pathlib.Path(args.trans)
    labels = pathlib.Path(args.labels)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df_t = pd.read_csv(trans)
    df_l = pd.read_csv(labels)

    assert {"id","text"}.issubset(df_t.columns), "transcripts must have id,text"
    assert {"id","label"}.issubset(df_l.columns), "labels must have id,label"

    df = df_t[["id","text"]].merge(df_l[["id","label"]], on="id", how="left")
    missing = df["label"].isna().sum()
    if missing:
        print(f"Warning: {missing} rows have no label after merge. Fill your labels.csv properly.", flush=True)
    df = df.fillna({"label": ""})

    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote merged dataset to {out} (rows={len(df)})")

if __name__ == "__main__":
    main()
