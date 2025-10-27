import argparse
from pathlib import Path
import numpy as np
from joblib import load
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

STAT_NAMES = ["n_chars","n_lines","avg_line","n_tok","uniq_tok_ratio","repeat_line_ratio","punct"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--file",  required=True)
    ap.add_argument("--th", type=float, default=0.50)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    text = Path(args.file).read_text(encoding="utf-8", errors="ignore")
    pipe = load(args.model)
    feats: FeatureUnion = pipe.named_steps["feats"]
    clf = pipe.named_steps["clf"]

    classes = list(getattr(clf, "classes_", ["bad","good"]))
    coef = clf.coef_[0]
    # make weights point toward 'good'
    if classes[-1] != "good":
        coef = -coef

    # compute offset for each feature block
    blocks = []
    offset = 0
    for name, tr in feats.transformer_list:
        Xb = tr.transform([text])  # (1, dim)
        dim = Xb.shape[1]
        sl = slice(offset, offset+dim)
        blocks.append((name, tr, Xb, sl))
        offset += dim

    # probability & prediction
    p_good = float(pipe.predict_proba([text])[0][classes.index("good")])
    pred = "good" if p_good >= args.th else "bad"
    print(f"FILE={args.file}\nmodel={Path(args.model).name}\nthreshold={args.th}")
    print(f"p_good={p_good:.3f}  => PRED={pred}\n")

    # WORD contributions
    for name, tr, Xb, sl in blocks:
        if isinstance(tr, TfidfVectorizer) and tr.analyzer != "char":
            feat = tr.get_feature_names_out()
            x = Xb.toarray().ravel()
            w = coef[sl]
            idx = np.flatnonzero(x)
            contrib = {i: x[i]*w[i] for i in idx}
            top_pos = sorted(idx, key=lambda i: contrib[i], reverse=True)[:args.top]
            top_neg = sorted(idx, key=lambda i: contrib[i])[:args.top]
            print("WORD n-grams pushing GOOD:")
            for i in top_pos:
                print(f"  {feat[i]:<24} +{contrib[i]:.5f}")
            print("\nWORD n-grams pushing BAD:")
            for i in top_neg:
                print(f"  {feat[i]:<24} {contrib[i]:.5f}")
            print()
            break

    # CHAR contributions
    for name, tr, Xb, sl in blocks:
        if isinstance(tr, TfidfVectorizer) and tr.analyzer == "char":
            feat = tr.get_feature_names_out()
            x = Xb.toarray().ravel()
            w = coef[sl]
            idx = np.flatnonzero(x)
            contrib = {i: x[i]*w[i] for i in idx}
            top_pos = sorted(idx, key=lambda i: contrib[i], reverse=True)[:args.top]
            top_neg = sorted(idx, key=lambda i: contrib[i])[:args.top]
            print("CHAR n-grams pushing GOOD:")
            for i in top_pos:
                print(f"  {feat[i]!r:<24} +{contrib[i]:.5f}")
            print("\nCHAR n-grams pushing BAD:")
            for i in top_neg:
                print(f"  {feat[i]!r:<24} {contrib[i]:.5f}")
            print()
            break

    # text stats contributions (list by name)
    for name, tr, Xb, sl in blocks:
        # stats is a Pipeline(stats -> scale), just take the values
        if hasattr(tr, "transform") and not isinstance(tr, TfidfVectorizer):
            xb = Xb.toarray().ravel()
            w = coef[sl]
            contrib = xb * w
            print("TEXT-STATS contributions:")
            for i, (val, c) in enumerate(zip(xb, contrib)):
                nm = STAT_NAMES[i] if i < len(STAT_NAMES) else f"stat_{i}"
                print(f"  {nm:<18} value={val:.4f}  contrib={c:+.5f}")
            print()
            break

if __name__ == "__main__":
    main()
