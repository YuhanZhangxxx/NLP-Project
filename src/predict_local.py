#!/usr/bin/env python
import argparse, pathlib, csv, joblib
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to a saved .joblib pipeline")
    ap.add_argument("--text", action="append", help="Text snippet to classify (can be repeated)")
    args = ap.parse_args()

    pipe = joblib.load(args.model)

    texts = args.text or [
        "He choked two rounds; filler and stale angles.",
        "Layered multis with clean structure and sustained energy."
    ]
    preds = pipe.predict(texts)

    proba = None
    try:
        proba = pipe.predict_proba(texts)
    except Exception:
        proba = None

    out = pathlib.Path("reports/pred_samples.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if proba is not None:
            # Assuming class order is ['bad','good']
            w.writerow(["text","pred","score_bad","score_good"])
            for t, p, pr in zip(texts, preds, proba):
                w.writerow([t, p, float(pr[0]), float(pr[1])])
        else:
            w.writerow(["text","pred"])
            for t, p in zip(texts, preds):
                w.writerow([t, p])

    print(f"Wrote predictions to {out}")
    for i, t in enumerate(texts, 1):
        if proba is not None:
            print(f"{i}. {preds[i-1]}  (bad={proba[i-1][0]:.3f}, good={proba[i-1][1]:.3f})  :: {t}")
        else:
            print(f"{i}. {preds[i-1]}  :: {t}")

if __name__ == "__main__":
    main()
