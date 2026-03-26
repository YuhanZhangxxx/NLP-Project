import argparse, json
from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def load_split_ids(p):
    return [x.strip() for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="csv", default="data/raw/dataset.csv")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--min_df", type=int, default=3)
ap.add_argument("--max_df", type=float, default=0.9)
ap.add_argument("--ngram", nargs=2, type=int, default=[1,2])
ap.add_argument("--stop_words", default="english", choices=["english","none"])
ap.add_argument("--keep_apostrophe", action="store_true")
ap.add_argument("--binary", action="store_true")
ap.add_argument("--sublinear_tf", action="store_true")
args = ap.parse_args()

root = Path(".")
df = pd.read_csv(args.csv)

train_ids = load_split_ids(root/"data/processed/train.txt")
valid_ids = load_split_ids(root/"data/processed/valid.txt")
test_ids  = load_split_ids(root/"data/processed/test.txt")

split = {
  "train": df[df["id"].astype(str).isin(train_ids)].copy(),
  "valid": df[df["id"].astype(str).isin(valid_ids)].copy(),
  "test" : df[df["id"].astype(str).isin(test_ids)].copy(),
}

token_pattern = r"(?u)\b\w+\b"
if args.keep_apostrophe:
    token_pattern = r"(?u)\b[\w']+\b"

stop = None if args.stop_words=="none" else "english"

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=tuple(args.ngram),
        min_df=args.min_df,
        max_df=args.max_df,
        stop_words=stop,
        token_pattern=token_pattern,
        binary=args.binary,
        sublinear_tf=args.sublinear_tf,
    )),
    ("clf", LogisticRegression(max_iter=200, random_state=args.seed))
])

Xtr, ytr = split["train"]["text"].tolist(), split["train"]["label"].tolist()
pipe.fit(Xtr, ytr)

def eval_on(name):
    X = split[name]["text"].tolist()
    y = split[name]["label"].tolist()
    yp = pipe.predict(X)
    acc = accuracy_score(y, yp)
    f1  = f1_score(y, yp, average="macro", zero_division=0)
    tn, fp, fn, tp = confusion_matrix([1 if t=="good" else 0 for t in y],
                                      [1 if p=="good" else 0 for p in yp], labels=[0,1]).ravel()
    return {"split":name, "accuracy":acc, "macro_f1":f1, "tn":tn,"fp":fp,"fn":fn,"tp":tp}

res = [eval_on("valid"), eval_on("test")]
for r in res:
    print(f"{r['split']}: acc={r['accuracy']:.4f}  macro-F1={r['macro_f1']:.4f}  cm=[tn={r['tn']}, fp={r['fp']}, fn={r['fn']}, tp={r['tp']}]")

models = Path("models")
(models / "lr_v1").mkdir(exist_ok=True, parents=True)
outp = models / "lr_v1" / "song_lr_v1_tweaked.joblib"
dump(pipe, outp)
print(f"Saved tweaked model → {outp}")

# log run config
rc = {
  "tfidf": {
    "ngram": args.ngram,
    "min_df": args.min_df,
    "max_df": args.max_df,
    "stop_words": args.stop_words,
    "token_pattern": token_pattern,
    "binary": args.binary,
    "sublinear_tf": args.sublinear_tf
  },
  "clf": {"C":1.0, "max_iter":200, "random_state":args.seed}
}
Path("reports").mkdir(exist_ok=True, parents=True)
(Path("reports")/"run_config_tweaked.json").write_text(json.dumps(rc, indent=2), encoding="utf-8")
