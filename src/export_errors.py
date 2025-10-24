#!/usr/bin/env python
import argparse, pathlib, csv, json
import pandas as pd
import joblib
from sklearn.metrics import classification_report

def snippet(s, n=120):
    s = str(s).replace("\n"," ").strip()
    return (s[:n] + "…") if len(s) > n else s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_csv", required=True, help="raw CSV with id,text,label")
    ap.add_argument("--model", default=None, help="optional specific model, else use best from models/manifest.json")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    man_path = pathlib.Path("models/manifest.json")
    assert man_path.exists(), "Run training first to create models/manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    if args.model:
        model_path = args.model
    else:
        best = manifest["best_on_valid"]
        model_path = manifest["models"][best]["path"]

    pipe = joblib.load(model_path)

    # Load test ids
    test_ids = set(pathlib.Path("data/processed/test.txt").read_text(encoding="utf-8").splitlines())
    dtest = df[df["id"].isin(test_ids)].copy()

    preds = pipe.predict(dtest["text"])
    dtest["pred"] = preds

    fp = dtest[(dtest["label"]=="bad") & (dtest["pred"]=="good")]
    fn = dtest[(dtest["label"]=="good") & (dtest["pred"]=="bad")]

    out = pathlib.Path("reports/errors_fp_fn.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type","id","gold","pred","snippet"])
        for _,r in fp.iterrows():
            w.writerow(["FP", r["id"], r["label"], r["pred"], snippet(r["text"])])
        for _,r in fn.iterrows():
            w.writerow(["FN", r["id"], r["label"], r["pred"], snippet(r["text"])])    

    print(f"Exported {len(fp)} FP and {len(fn)} FN to {out}")

if __name__ == "__main__":
    main()
