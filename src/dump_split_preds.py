import argparse
from pathlib import Path
import pandas as pd
from joblib import load

def read_ids(p): return [x.strip() for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="valid", choices=["train","valid","test"])
    ap.add_argument("--th_good", type=float, default=0.45)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    df = pd.read_csv(root/"data/raw/dataset.csv")
    keep = set(read_ids(root/f"data/processed/{args.split}.txt"))
    df = df[df["id"].astype(str).isin(keep)].copy()

    pipe = load(args.model)
    proba = pipe.predict_proba(df["text"].tolist())
    classes = list(pipe.classes_)
    gi = classes.index("good"); bi = classes.index("bad")
    df["p_good"] = proba[:, gi]
    df["p_bad"]  = proba[:, bi]
    df["pred"]   = (df["p_good"] >= args.th_good).map({True:"good", False:"bad"})
    df["correct"]= (df["pred"] == df["label"])

    out = args.out or root/f"reports/{args.split}_preds_th{args.th_good:.2f}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df[["id","label","p_good","p_bad","pred","correct","text"]].to_csv(out, index=False)
    print(f"Wrote → {out}")
