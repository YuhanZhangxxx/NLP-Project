#!/usr/bin/env python
import argparse, pathlib, csv, re
import pandas as pd

def read_text(path, encoding):
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        return f.read()

def clean_text(s: str) -> str:
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    ap = argparse.ArgumentParser(description="Ingest TXT files from folder structure into a CSV dataset.")
    ap.add_argument("--root", required=True, help="Root folder containing subfolders 'good' and 'bad' (case-insensitive).")
    ap.add_argument("--out", default="data/raw/dataset.csv", help="Output CSV path.")
    ap.add_argument("--encoding", default="utf-8", help="Encoding for reading .txt files.")
    ap.add_argument("--id_style", choices=["auto","fname"], default="auto", help="ID generation: 'auto' uses G0001/B0001; 'fname' uses file stem.")
    ap.add_argument("--min_chars", type=int, default=1, help="Skip files shorter than this character count after cleaning.")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    # find 'good' and 'bad' dirs (case-insensitive)
    def find_dir(name):
        for cand in [name, name.capitalize(), name.upper(), name.title()]:
            p = root / cand
            if p.exists() and p.is_dir():
                return p
        return None

    good_dir = find_dir("good")
    bad_dir = find_dir("bad")
    if good_dir is None or bad_dir is None:
        raise SystemExit("Could not find both 'good' and 'bad' folders under the root (case-insensitive).")

    rows = []
    counters = {"good":0, "bad":0}

    def walk_collect(lbl, base):
        nonlocal rows
        for path in base.rglob("*.txt"):
            txt = clean_text(read_text(path, args.encoding))
            if len(txt) < args.min_chars:
                continue
            if args.id_style == "fname":
                fid = f"{lbl[0].upper()}_{path.stem}"
            else:
                counters[lbl] += 1
                fid = f"{lbl[0].upper()}{counters[lbl]:04d}"
            rows.append({"id": fid, "text": txt, "label": lbl})

    walk_collect("good", good_dir)
    walk_collect("bad", bad_dir)

    if not rows:
        raise SystemExit("No .txt files found under 'good' or 'bad'.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows, columns=["id","text","label"])
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    print("Label counts:", df["label"].value_counts().to_dict())

if __name__ == "__main__":
    main()
