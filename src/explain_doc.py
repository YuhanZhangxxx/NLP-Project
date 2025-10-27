import argparse
from pathlib import Path
import numpy as np, pandas as pd
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB

def find_step_by_type(pipe, typ):
    for name, step in pipe.named_steps.items():
        if isinstance(step, typ):
            return name, step
    return None, None

def good_coef_from_clf(clf, classes):
    # return weight vector pointing toward 'good'
    if isinstance(clf, LogisticRegression):
        w = clf.coef_[0]            # binary classification: shape=(1, n_features)
        cls = list(classes)
        if len(cls) == 2:
            # scikit's w corresponds to classes_[1]; flip if classes_[1] is not 'good'
            return w if cls[-1] == 'good' else -w
        return w
    elif isinstance(clf, (MultinomialNB, ComplementNB)):
        cls = list(classes)
        gi = cls.index('good')
        bi = 1 - gi
        # approximate log P(w|good) - log P(w|bad) as "weight toward good"
        return clf.feature_log_prob_[gi] - clf.feature_log_prob_[bi]
    else:
        raise TypeError(f"Unsupported classifier type: {type(clf)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.root)
    df = pd.read_csv(root/"data/raw/dataset.csv")
    row = df[df["id"].astype(str)==args.id]
    if row.empty:
        raise SystemExit(f"id not found in dataset: {args.id}")
    row = row.iloc[0]
    text, label = row["text"], row["label"]

    pipe = load(args.model)

    # auto-locate Vectorizer and Classifier
    _, vec = find_step_by_type(pipe, TfidfVectorizer)
    if vec is None:
        raise SystemExit(f"Could not find TfidfVectorizer in steps: {list(pipe.named_steps.keys())}")

    clf = None
    for typ in (LogisticRegression, MultinomialNB, ComplementNB):
        _, clf = find_step_by_type(pipe, typ)
        if clf is not None:
            break
    if clf is None:
        raise SystemExit(f"Could not find supported classifier in steps: {list(pipe.named_steps.keys())}")

    X = vec.transform([text])
    classes = getattr(clf, "classes_", ["bad","good"])
    # probability (if available)
    if hasattr(pipe, "predict_proba"):
        p = pipe.predict_proba([text])[0]
        try:
            gi = list(classes).index('good')
        except ValueError:
            gi = -1
        p_good = float(p[gi])
    else:
        # use decision function when no prob (for debug print only)
        p_good = float(getattr(clf, "decision_function")(X)[0])

    w_good = good_coef_from_clf(clf, classes)

    x = X.toarray().ravel()
    idx = np.flatnonzero(x)
    contrib = { i: x[i]*w_good[i] for i in idx }
    top = args.top
    top_good = sorted(idx, key=lambda i: contrib[i], reverse=True)[:top]
    top_bad  = sorted(idx, key=lambda i: contrib[i])[:top]
    feat = vec.get_feature_names_out()

    print(f"ID={args.id}  true_label={label}  p_good={p_good:.3f}")
    print("\nTop pushes toward GOOD:")
    for i in top_good:
        print(f"  {feat[i]:<24} +{contrib[i]:.5f}")
    print("\nTop pushes toward BAD:")
    for i in top_bad:
        print(f"  {feat[i]:<24} {contrib[i]:.5f}")

if __name__ == "__main__":
    main()
