import argparse, re
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--th_good", type=float, default=None, help="optional: for error boundary analysis; doesn't affect scoring")
ap.add_argument("--k", type=int, default=5, help="show Top-K misclassified examples")
args = ap.parse_args()

p = Path(args.csv)
df = pd.read_csv(p)

# basic validation
need = {"id","label","pred","p_good","p_bad","correct"}
missing = need - set(df.columns)
if missing:
    raise SystemExit(f"CSV missing columns: {missing}")

y_true = (df["label"]=="good").astype(int).values
y_pred = (df["pred"] =="good").astype(int).values

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, labels=[0,1], zero_division=0)

print(f"File: {p}")
print(f"accuracy={acc:.4f}  macro-F1={macro_f1:.4f}")
print(f"cm: tn={tn}, fp={fp}, fn={fn}, tp={tp}")
print("Per-class (label=bad,good):")
print(f"  precision: {prec}")
print(f"  recall   : {rec}")
print(f"  f1       : {f1}")
print(f"  support  : {supp}")

# export misclassifications
errs = df[df["label"]!=df["pred"]].copy()
out_err = p.with_name(p.stem+"_errors.csv")
errs.to_csv(out_err, index=False, encoding="utf-8")
print(f"Wrote errors → {out_err}")

# Top-K misclassification examples (most confident & closest to threshold)
if args.th_good is None:
    # try to infer threshold from filename (e.g. *_th0.45.csv)
    m = re.search(r"th(\d+(?:\.\d+)?)", p.stem)
    th = float(m.group(1)) if m else 0.5
else:
    th = args.th_good
errs["margin"] = (errs["p_good"] - th).abs()

conf_fp = errs[(errs["label"]=="bad") & (errs["pred"]=="good")].sort_values("p_good", ascending=False).head(args.k)
conf_fn = errs[(errs["label"]=="good") & (errs["pred"]=="bad")].sort_values("p_good", ascending=True).head(args.k)
near_th = errs.sort_values("margin", ascending=True).head(args.k)

def brief(dfv):
    return dfv[["id","label","pred","p_good","p_bad"]]

print("\nTop confident false positives (pred=good,label=bad):")
print(brief(conf_fp).to_string(index=False))
print("\nTop confident false negatives (pred=bad,label=good):")
print(brief(conf_fn).to_string(index=False))
print("\nTop near-threshold mistakes:")
print(brief(near_th).to_string(index=False))
