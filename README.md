# Rap Judge — Phase 1 (Baseline)

A simple **text‑only** judge that tags a full song/battle as **good** or **bad** using **TF‑IDF** with two baselines:
- **Multinomial Naive Bayes**
- **Logistic Regression**

You get fixed splits, metrics, confusion matrices, and ready‑to‑load **.joblib** pipelines.

---

## Repo layout
```
rap-judge/
├─ data/
│  ├─ raw/               # dataset.csv lives here (id,text,label)
│  ├─ raw_txt/           # drop .txt into good/ bad/ (and *dirty/ variants if you keep raw crawls)
│  └─ processed/         # train.txt / valid.txt / test.txt (reproducible splits)
├─ models/               # exported models + manifest.json
├─ reports/              # metrics.csv / confusion_matrix_*.png / run_config.json / etc.
├─ nb/                   # optional notebooks
├─ src/
│  ├─ ingest_txt.py           # TXT folders → dataset.csv
│  ├─ split_data.py           # fixed ratio splits only (train/valid/test)
│  ├─ train_baselines.py      # train + eval (NB/LR)
│  ├─ train_lr_tweaks.py      # LR tweaks (stopwords/ngram etc.)
│  ├─ train_lr_plus.py        # LR+ (word+char+extra stats)
│  ├─ predict_file.py         # predict from a file (won’t echo/save lyrics)
│  ├─ predict_local.py        # quick test from inline text (logs to CSV)
│  ├─ dump_split_preds.py     # batch-predict entire split files
│  ├─ explain_file_v2.py      # explain LR/NB with top n-grams
│  ├─ explain_doc.py          # explain a single doc
│  ├─ export_errors.py        # dump FP/FN for manual review
│  ├─ eval_valid.py           # threshold sweep on valid set
│  ├─ score_preds_csv.py      # score prediction CSVs
│  ├─ feats_extra.py          # extra text stats features
│  ├─ merge_labels.py         # merge labels from multiple sources
│  ├─ batch_whisper_lyrics.py # Whisper batch ASR
│  ├─ fw_batch_lyrics.py      # Faster-Whisper batch ASR
│  └─ transcribe_whisper.py   # single-file ASR helper
├─ clean_lyrics.py        # root-level cleaner (NOT under src/)
├─ requirements.txt
└─ README.md
```

---

## Setup (Windows PowerShell)
```powershell
cd "C:\Users\<you>\Desktop\rap-judge"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

# (optional) console UTF-8 to avoid mojibake
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
chcp 65001 > $null
```

---

## Add data (pick one)
### A) TXT folders (easy for labeling)
```
# put your files like this

data/raw_txt/good/*.txt
data/raw_txt/bad/*.txt

# turn into CSV
python .\src\ingest_txt.py --root data\raw_txt --out data\raw\dataset.csv --id_style fname
```
This creates **data/raw/dataset.csv** with columns: `id,text,label`.

### B) Already have a CSV?
Drop `dataset.csv` in **data/raw/** with columns:
```
id,text,label  # label must be "good" or "bad"
```

---

## Clean lyrics (idempotent)
`clean_lyrics.py` is at the **repo root**. Usage is simple: pass a **file or a folder**; it writes `*.clean.txt` **next to originals** and **won’t double‑clean**.

```powershell
# Single file → writes track1.clean.txt next to track1.txt
python .\clean_lyrics.py "data\raw_txt\good\track1.txt"

# Folder (recurses) → for each *.txt, writes a *.clean.txt sibling
python .\clean_lyrics.py "data\raw_txt"
```
What it removes: section headers like `[Chorus]`, ad/shoutout lines, "You might also like" blocks, ticket spam lines, etc.

> Tip: You can **ingest the cleaned files only** by pointing `ingest_txt.py` at a directory that contains the `*.clean.txt` files (or move them into a `raw_txt_clean/` folder first).

---

## Make fixed splits (reproducible)
```powershell
python .\src\split_data.py --in data\raw\dataset.csv --train 0.8 --valid 0.1 --test 0.1 --seed 42
# outputs: data/processed/train.txt, valid.txt, test.txt
```


## Train + evaluate (baselines)
```powershell
python .\src\train_baselines.py --in data\raw\dataset.csv --seed 42
```
**Check outputs:**
- `reports/metrics.csv` — accuracy + macro‑F1 on valid/test for NB/LR
- `reports/confusion_matrix_nb.png`, `reports/confusion_matrix_lr.png`
- `reports/run_config.json` — TF‑IDF + model params + seed + split files
- Models in **models/** — `song_nb_v1_YYYYMMDD.joblib`, `song_lr_v1_YYYYMMDD.joblib`, `manifest.json`

**Defaults (tweak via CLI):**
- TF‑IDF: `ngram=(1,2)`, `min_df=2`, `max_df=0.9`
- NB: `alpha=1.0`
- LR: `C=1.0`, `max_iter=200`

---

## Predict (no lyrics printed/saved)
Use when you don’t want lyrics echoed anywhere.
```powershell
# grab the newest LR model
$model = (Get-ChildItem .\models\song_lr_v1_*.joblib | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName

# predict a single song file (only shows result)
python .\src\predict_file.py --model "$model" --file "C:\path\to\song.txt"

# stricter bar for calling "good" (default is 0.50)
python .\src\predict_file.py --model "$model" --file "C:\path\to\song.txt" --th_good 0.60
```

### Quick local probe (logs text for debugging)
```powershell
python .\src\predict_local.py --model .\models\song_lr_v1_*.joblib --text "Some lyric here"
# writes to reports/pred_samples.csv
```

### Batch‑predict a folder (without printing lyrics)
```powershell
Get-ChildItem -Path "C:\path\to\lyrics" -Filter *.txt -Recurse | ForEach-Object {
  python .\src\predict_file.py --model "$model" --file $_.FullName --th_good 0.60
}
```

---

## Explain a prediction (why LR/NB decided so)
Feature‑level view of n‑grams that push **GOOD** vs **BAD**.
```powershell
python .\src\explain_file_v2.py --model "$model" --file "C:\path\to\song.txt" --th 0.50 --top 12
```
Outputs top contributing n‑grams for each side plus the final label.

---

## Error analysis (find FP/FN quickly)
```powershell
python .\src\export_errors.py --in data\raw\dataset.csv
# open reports/errors_fp_fn.csv (bad→good = FP, good→bad = FN)
```
Skim for patterns: repetition, filler, crawler noise, very short/long texts, etc.

---

## (Optional) Transcribe audio → lyrics (Whisper)
For building your corpus from battle videos or MP3s.
```powershell
python .\src\batch_whisper_lyrics.py `
  --input  "C:\path\to\audio_or_video_folder" `
  --output "data\lyrics" `
  --model  large-v3 `
  --device cuda `
  --language en
```
Then clean (`clean_lyrics.py`) and ingest (`ingest_txt.py`).

---

## How we judge “done” for Phase 1
- Fixed splits saved in `data/processed/*.txt`
- Metrics for both baselines in `reports/metrics.csv`
- Picked a “best” baseline (justify with numbers + a few error examples)
- Exported pipelines under `models/` (+ `manifest.json`)
- Run record: `reports/run_config.json`
- (If used) `reports/pred_samples.csv`, `reports/errors_fp_fn.csv`
- This README in the repo

---

## Quick cheat sheet
```powershell
# activate
cd "C:\Users\<you>\Desktop\rap-judge"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# TXT -> CSV
python .\src\ingest_txt.py --root data\raw_txt --out data\raw\dataset.csv --id_style fname

# split
python .\src\split_data.py --in data\raw\dataset.csv --seed 42

# train
python .\src\train_baselines.py --in data\raw\dataset.csv --seed 42

# predict (file; no lyrics in output)
$model = (Get-ChildItem .\models\song_lr_v1_*.joblib | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
python .\src\predict_file.py --model "$model" --file "C:\path\to\song.txt" --th_good 0.60

# explain
python .\src\explain_file_v2.py --model "$model" --file "C:\path\to\song.txt" --th 0.50 --top 12

# export mistakes
python .\src\export_errors.py --in data\raw\dataset.csv
```

---

## Troubleshooting (Windows)
- **“No suitable Python runtime found”** → Install Python 3.11, then run `py -3.11 -m venv .venv` again.
- **Unicode boxes/garbage** → enable UTF‑8 (see Setup) and ensure your `.txt` files are UTF‑8.
- **GPU Whisper stalls** → try `--device cpu` or smaller models; close other GPU apps.
- **Very imbalanced data** → add more from the minority class or use stricter `--th_good`.

---

## What’s new in v2
- Fixed **clean_lyrics.py** usage and location (root-level, `python .\clean_lyrics.py <file_or_folder>`; auto-writes `*.clean.txt`).
- Removed non-existent **K-fold** / **Bootstrap** CLI flags from `split_data.py` (only fixed ratios are supported).
- Expanded **Repo layout** to include utility scripts actually present in this repo.
- Kept baseline training/prediction instructions the same.

---

## Appendix: .gitignore (recommended)
Create a `.gitignore` at repo root with the following defaults:

```
# Python
__pycache__/
*.py[cod]
*.pyd
*.pyo
*.so
*.dylib

# Virtual environments
.venv/
venv/
env/
.env/
.python-version

# OS / IDE
.DS_Store
Thumbs.db
desktop.ini
.vscode/
.idea/
*.code-workspace

# Notebooks
.ipynb_checkpoints/

# Logs / temp
logs/
tmp/
*.log

# Models & artifacts
models/**
!models/.gitkeep

# Reports (plots, large CSVs)
reports/**
!reports/.gitkeep

# Raw text & audio (keep only small samples under a separate sample dir if needed)
data/raw_txt/**
data/lyrics/**
*.mp3
*.wav
*.flac
*.m4a
*.ogg
*.opus
*.mp4
*.mov
*.srt
*.vtt

# Whisper caches (if any)
whisper_cache/
```
**Keep** `data/processed/*.txt` in version control (reproducible splits) and `data/raw/dataset.csv` if it’s small and sanitized.

