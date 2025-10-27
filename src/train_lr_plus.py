import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from joblib import dump
from scipy import sparse
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from feats_extra import text_stats

def load_ids(p: Path):
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def evaluate(pipe, df, name):
    X = df["text"].tolist(); y = df["label"].tolist()
    yp = pipe.predict(X)
    acc = accuracy_score(y, yp)
    f1  = f1_score(y, yp, average="macro", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(
        [0 if t=="bad" else 1 for t in y],
        [0 if p=="bad" else 1 for p in yp],
        labels=[0,1]).ravel()
    print(f"{name}: acc={acc:.4f}  macro-F1={f1:.4f}  cm=[tn={tn}, fp={fp}, fn={fn}, tp={tp}]")
    return {"split":name,"accuracy":acc,"macro_f1":f1,"tn":tn,"fp":fp,"fn":fn,"tp":tp}

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/dataset.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--word_min_df", type=int, default=2)
    ap.add_argument("--word_max_df", type=float, default=0.95)
    ap.add_argument("--word_ngram", nargs=2, type=int, default=[1,2])
    ap.add_argument("--char_ngram", nargs=2, type=int, default=[3,5])
    ap.add_argument("--sublinear_tf", action="store_true")
    ap.add_argument("--keep_apostrophe", action="store_true")
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    token_pattern = r"(?u)\b\w+\b"
    if args.keep_apostrophe:
        token_pattern = r"(?u)\b[\w']+\b"

    word_vec = TfidfVectorizer(
        ngram_range=tuple(args.word_ngram),
        min_df=args.word_min_df,
        max_df=args.word_max_df,
        token_pattern=token_pattern,
        sublinear_tf=args.sublinear_tf,
        lowercase=True
    )
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=tuple(args.char_ngram),
        lowercase=True,
        sublinear_tf=args.sublinear_tf
    )
    stats = Pipeline([
        ("stats", FunctionTransformer(text_stats, validate=False)),
        ("scale", MaxAbsScaler())
    ])

    feats = FeatureUnion([
        ("word", word_vec),
        ("char", char_vec),
        ("stats", stats)
    ])

    clf = LogisticRegression(
        solver="liblinear",
        C=args.C,
        max_iter=500,
        random_state=args.seed
    )

    pipe = Pipeline([("feats", feats), ("clf", clf)])

    root = Path(".")
    df = pd.read_csv(args.csv)
    tr = df[df["id"].astype(str).isin(load_ids(root/"data/processed/train.txt"))].copy()
    va = df[df["id"].astype(str).isin(load_ids(root/"data/processed/valid.txt"))].copy()
    te = df[df["id"].astype(str).isin(load_ids(root/"data/processed/test.txt"))].copy()

    pipe.fit(tr["text"].tolist(), tr["label"].tolist())

    evaluate(pipe, va, "valid")
    evaluate(pipe, te, "test")

    Path("models").mkdir(parents=True, exist_ok=True)
    outm = Path("models")/"song_lr_v2_plus.joblib"
    dump(pipe, outm)
    print(f"Saved → {outm}")

    Path("reports").mkdir(parents=True, exist_ok=True)
    cfg = {
      "word_vec":{"ngram":args.word_ngram,"min_df":args.word_min_df,"max_df":args.word_max_df,
                  "sublinear_tf":args.sublinear_tf,"keep_apostrophe":args.keep_apostrophe},
      "char_vec":{"ngram":args.char_ngram,"sublinear_tf":args.sublinear_tf},
      "stats":["n_chars","n_lines","avg_line","n_tok","uniq_tok_ratio","repeat_line_ratio","punct"],
      "clf":{"C":args.C,"solver":"liblinear","max_iter":500}
    }
    (Path("reports")/"run_config_plus.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
