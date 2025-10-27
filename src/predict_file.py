#!/usr/bin/env python
import argparse, joblib
from pathlib import Path

def read_text(p: Path):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return p.read_text(encoding=enc, errors="ignore")
        except Exception:
            pass
    return p.read_text(errors="ignore")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--th_good", type=float, default=0.5, help="Probability threshold for class 'good'")
    args = ap.parse_args()

    txt = read_text(Path(args.file))
    pipe = joblib.load(args.model)

    try:
        proba = pipe.predict_proba([txt])[0]
        # get indices of 'good' and 'bad' in probability vector (based on actual model classes_)
        clf = getattr(pipe, "named_steps", {}).get("clf", None)
        classes = list(getattr(clf, "classes_", []))
        if classes and ("good" in classes and "bad" in classes):
            i_good = classes.index("good")
            i_bad  = classes.index("bad")
        else:
            # fallback: assume order [bad, good]
            i_bad, i_good = 0, 1

        p_good = float(proba[i_good])
        p_bad  = float(proba[i_bad])
        pred   = "good" if p_good >= args.th_good else "bad"
        print(f"{pred} (bad={p_bad:.3f}, good={p_good:.3f}, th_good={args.th_good})")
    except Exception:
        pred = pipe.predict([txt])[0]
        print(pred)

if __name__ == "__main__":
    main()
