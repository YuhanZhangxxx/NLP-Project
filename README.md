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
│  ├─ train_lr_plus.py        # LR+ (word+char+extra stats) ⭐ Recommended
│  ├─ analyze_song.py         # 🎤 Detailed analysis with interactive mode ⭐ NEW
│  ├─ predict_file.py         # simple prediction from file
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
├─ scripts/              # Testing and utility scripts
│  ├─ test_first_20_songs.py  # Test first N songs for overfitting check
│  ├─ test_full_dataset.py    # Test entire dataset for overfitting
│  └─ show_progression.py     # (if used) Model progression visualization
├─ judge.py              # ⭐ Quick judge (auto-selects model)
├─ clean_lyrics.py       # Root-level cleaner (NOT under src/)
├─ requirements.txt
└─ README.md
```

---

## Setup

### Mac/Linux
```bash
cd /path/to/CS486-NLP-Project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell
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

## Add Data

### A) TXT Folders (Recommended)
Organize your lyrics into folders:
```
data/raw_txt/good/*.txt
data/raw_txt/bad/*.txt
```

**Step 1: Clean lyrics** (removes ads, transcription artifacts)
```bash
# Clean all files in raw_txt directory
python clean_lyrics.py data/raw_txt
# Creates *.clean.txt files next to originals
```

**Step 2: Convert to CSV**
```bash
# Mac/Linux
python src/ingest_txt.py --root data/raw_txt --out data/raw/dataset.csv --id_style fname

# Windows
python src\ingest_txt.py --root data\raw_txt --out data\raw\dataset.csv --id_style fname
```

This creates **data/raw/dataset.csv** with columns: `id,text,label`.

### B) Already have a CSV?
Drop `dataset.csv` in **data/raw/** with columns:
```
id,text,label  # label must be "good" or "bad"
```

---

## Clean Lyrics

`clean_lyrics.py` is at the **repo root**. It automatically removes ads, transcription artifacts, and section markers.

```bash
# Single file → writes track1.clean.txt next to track1.txt
python clean_lyrics.py data/raw_txt/good/track1.txt

# Folder (recurses) → for each *.txt, writes a *.clean.txt sibling
python clean_lyrics.py data/raw_txt
```

**What it removes:**
- Section headers: `[Chorus]`, `[Verse 1]`, `[Intro]`, etc.
- Transcription artifacts: "We'll be right back", "Do not censor", "Thank you for watching", etc.
- Ad lines: "Get tickets...", "See tickets near...", etc.
- "You might also like" recommendation blocks
- All-caps short shouts (like "DESERT STORM!!")

> **Tip**: The cleaning function is also built into `analyze_song.py` and `judge.py`, so lyrics are automatically cleaned when you use those tools.

---

## Make Fixed Splits (Reproducible)

```bash
# Mac/Linux
python src/split_data.py --in data/raw/dataset.csv --train 0.8 --valid 0.1 --test 0.1 --seed 42

# Windows
python src\split_data.py --in data\raw\dataset.csv --train 0.8 --valid 0.1 --test 0.1 --seed 42
```

**Outputs:** `data/processed/train.txt`, `valid.txt`, `test.txt` (lists of song IDs)


## Train Models

### Baseline Models (NB + LR v1)
```bash
python src/train_baselines.py --in data/raw/dataset.csv --seed 42
```

**Outputs:**
- `reports/metrics.csv` — accuracy + macro‑F1 on valid/test for NB/LR
- `reports/confusion_matrix_nb.png`, `reports/confusion_matrix_lr.png`
- `reports/run_config.json` — TF‑IDF + model params + seed + split files
- Models: `models/song_nb_v1_YYYYMMDD.joblib`, `models/song_lr_v1_YYYYMMDD.joblib`

**Defaults:**
- TF‑IDF: `ngram=(1,2)`, `min_df=2`, `max_df=0.9`
- NB: `alpha=1.0`
- LR: `C=1.0`, `max_iter=200`

### LR+ Model (Recommended - Better Performance) ⭐
```bash
python src/train_lr_plus.py --csv data/raw/dataset.csv --seed 42
```

**Features:**
- Word n-grams (1-2)
- Character n-grams (3-5)
- Extra statistical features (length, vocabulary diversity, repetition, etc.)

**Outputs:**
- Model: `models/song_lr_v2_plus_YYYYMMDD.joblib` or `models/song_lr_v2_plus.joblib`
- Config: `reports/run_config_plus.json`

---

## Predict & Judge

### 🎤 Interactive Mode (Recommended - NEW!)
**Best for pasting lyrics directly with special characters** (quotes, brackets, etc.)

```bash
# Interactive mode - paste lyrics directly
python src/analyze_song.py --model models/song_lr_v2_plus_20251026.joblib --interactive
# Then paste your lyrics, press Ctrl+D (Mac/Linux) or Ctrl+Z+Enter (Windows) when done
# Or type 'END' on a separate line to finish
```

**Features:**
- ✅ Handles any special characters (no shell parsing issues)
- ✅ Automatic lyrics cleaning (removes ads, transcription artifacts)
- ✅ Detailed analysis with quality score, feature contributions, statistics
- ✅ Optional line-by-line analysis with `--lines` flag

### Quick Judge (Simple Text Input)
**For simple lyrics without special characters:**

```bash
# Simple judge (auto-selects latest model)
python judge.py "Your rap lyrics here"

# With custom model
python judge.py --model models/song_lr_v2_plus_20251026.joblib "Your lyrics"
```

### Detailed Analysis (File or Text)
**Full analysis with quality score and explanations:**

```bash
# From file
python src/analyze_song.py --model models/song_lr_v2_plus_20251026.joblib --file lyrics.txt

# Direct text input
python src/analyze_song.py --model models/song_lr_v2_plus_20251026.joblib --text "Your lyrics here"

# With line-by-line analysis
python src/analyze_song.py --model models/song_lr_v2_plus_20251026.joblib --file lyrics.txt --lines

# Adjust threshold
python src/analyze_song.py --model models/song_lr_v2_plus_20251026.joblib --file lyrics.txt --th_good 0.60
```

### Simple Prediction (No Lyrics Printed)
**Use when you only want the result, no detailed analysis:**

```bash
# Predict from file (simple output)
python src/predict_file.py --model models/song_lr_v2_plus_20251026.joblib --file lyrics.txt

# With custom threshold
python src/predict_file.py --model models/song_lr_v2_plus_20251026.joblib --file lyrics.txt --th_good 0.60
```

### Quick Test (Logs to CSV)
**For debugging - saves predictions to CSV:**

```bash
python src/predict_local.py --model models/song_lr_v2_plus_20251026.joblib --text "Some lyric here"
# writes to reports/pred_samples.csv
```

### Batch Predict Folder
```bash
# Linux/Mac
find /path/to/lyrics -name "*.txt" -exec python src/predict_file.py --model models/song_lr_v2_plus_20251026.joblib --file {} \;

# Windows PowerShell
Get-ChildItem -Path "C:\path\to\lyrics" -Filter *.txt -Recurse | ForEach-Object {
  python src\predict_file.py --model models\song_lr_v2_plus_20251026.joblib --file $_.FullName --th_good 0.60
}
```

---

## Explain Predictions

Feature‑level view of n‑grams that push **GOOD** vs **BAD**.

```bash
python src/explain_file_v2.py --model models/song_lr_v2_plus_20251026.joblib --file lyrics.txt --th 0.50 --top 12
```

**Outputs:**
- Top contributing n-grams for "good" classification
- Top contributing n-grams for "bad" classification
- Final label and probabilities

**Note:** `analyze_song.py` also provides feature contributions as part of its detailed analysis.

---

## Error Analysis & Evaluation

### Export Errors (FP/FN)
Find false positives and false negatives for manual review:

```bash
python src/export_errors.py --in data/raw/dataset.csv
# Output: reports/errors_fp_fn.csv
# - bad→good = False Positive (FP)
# - good→bad = False Negative (FN)
```

Look for patterns: repetition, filler, crawler noise, very short/long texts, etc.

### Evaluate on Validation Set
Test different thresholds on validation set:

```bash
python src/eval_valid.py --model models/song_lr_v2_plus_20251026.joblib --split valid
# Output: reports/valid_threshold_scan.csv
```

### Check for Overfitting

**Test first N songs from dataset:**
```bash
python scripts/test_first_20_songs.py --model models/song_lr_v2_plus_20251026.joblib --num_songs 40 --check-splits
# Tests first 40 songs, shows accuracy by split (train/valid/test)
# Output: reports/first_20_songs_test.csv
```

**Test entire dataset:**
```bash
python scripts/test_full_dataset.py --model models/song_lr_v2_plus_20251026.joblib --dataset data/raw/dataset.csv
# Tests all songs in dataset, analyzes overfitting by split
# Output: reports/full_dataset_test.csv
```

---

## (Optional) Transcribe Audio → Lyrics

### End-to-End: Transcribe + Judge (Recommended)

**Single command to transcribe audio and judge quality:**

```bash
# Basic usage
python scripts/transcribe_and_judge.py --audio audio/song.mp3

# With detailed analysis
python scripts/transcribe_and_judge.py --audio audio/song.mp3 --detailed

# Use large-v3 for best quality
python scripts/transcribe_and_judge.py --audio audio/song.mp3 --model large-v3 --detailed

# Keep transcript files
python scripts/transcribe_and_judge.py --audio audio/song.mp3 --detailed --keep-transcript
```

**What it does:**
1. Transcribes audio with Whisper
2. Cleans lyrics (removes ads, artifacts)
3. Judges quality with the model
4. Optionally keeps transcript files

### Batch Transcription (For Building Corpus)

For building your corpus from battle videos or MP3s using Whisper.

```bash
python src/batch_whisper_lyrics.py \
  --input  "/path/to/audio_or_video_folder" \
  --output "data/transcribed_lyrics" \
  --model  medium \
  --device cpu \
  --language en
```

**Model options:**
- `medium` - ⭐ **Default** - Good balance of speed and quality
- `large-v3` - Best quality but slower (requires more memory)

> **Note**: `base` and smaller models produce poor transcription quality and are not recommended.

**Then:**
1. Clean: `python clean_lyrics.py data/transcribed_lyrics`
2. Ingest: `python src/ingest_txt.py --root data/transcribed_lyrics --out data/raw/dataset.csv --id_style fname`

> **Note**: After transcription, you can move cleaned files to `data/raw_txt/good/` or `data/raw_txt/bad/` for organization.

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

## Quick Cheat Sheet

### Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# or: .venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Pipeline
```bash
# Clean lyrics (removes ads, artifacts)
python clean_lyrics.py data/raw_txt

# TXT folders -> CSV
python src/ingest_txt.py --root data/raw_txt --out data/raw/dataset.csv --id_style fname

# Create fixed splits
python src/split_data.py --in data/raw/dataset.csv --train 0.8 --valid 0.1 --test 0.1 --seed 42
```

### Training
```bash
# Train baselines (NB + LR v1)
python src/train_baselines.py --in data/raw/dataset.csv --seed 42

# Train LR+ (v2_plus - recommended, better performance)
python src/train_lr_plus.py --csv data/raw/dataset.csv --seed 42
```

### Judging/Prediction (Choose based on your needs)

**🎤 Interactive Mode (Best for pasting lyrics):**
```bash
python src/analyze_song.py --model models/song_lr_v2_plus_20251026.joblib --interactive
```

**⚡ Quick Judge (Simple text):**
```bash
python judge.py "Your lyrics here"
```

**📊 Detailed Analysis (File or text with full report):**
```bash
python src/analyze_song.py --model models/song_lr_v2_plus_20251026.joblib --file lyrics.txt --lines
```

**🔍 Simple Prediction (Just result):**
```bash
python src/predict_file.py --model models/song_lr_v2_plus_20251026.joblib --file lyrics.txt
```

### Analysis & Evaluation
```bash
# Explain prediction (feature contributions)
python src/explain_file_v2.py --model models/song_lr_v2_plus_20251026.joblib --file lyrics.txt --th 0.50 --top 12

# Export errors (FP/FN analysis)
python src/export_errors.py --in data/raw/dataset.csv

# Evaluate on validation set
python src/eval_valid.py --model models/song_lr_v2_plus_20251026.joblib --split valid

# Test first N songs for overfitting check
python scripts/test_first_20_songs.py --model models/song_lr_v2_plus_20251026.joblib --num_songs 40 --check-splits
```

---

## Model Selection

**Available Models:**
- `song_lr_v1_*.joblib` - Baseline Logistic Regression (TF-IDF only)
- `song_lr_v2_plus_*.joblib` - ⭐ **Recommended** - Enhanced LR with word+char n-grams + stats
- `song_nb_v1_*.joblib` - Baseline Multinomial Naive Bayes

**Which to use?**
- **For best performance**: Use `song_lr_v2_plus_*.joblib` (v2_plus models)
- **For simple predictions**: Any model works, but v2_plus is more accurate
- **Auto-selection**: `judge.py` automatically selects the latest v2_plus model

**Note:** v2_plus models require the `feats_extra.py` module. Scripts automatically handle this.

---

## Troubleshooting

### General Issues
- **"No suitable Python runtime found"** → Install Python 3.11+, then recreate venv
- **Unicode boxes/garbage** → Ensure `.txt` files are UTF‑8 encoded
- **ModuleNotFoundError: feats_extra** → Scripts should auto-handle this, but if it persists, ensure `src/` is in Python path
- **Very imbalanced data** → Add more from minority class or use stricter `--th_good` threshold

### Windows-Specific
- **PowerShell execution policy** → Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
- **Path separators** → Use backslashes `\` in PowerShell, forward slashes `/` in bash

### Model Issues
- **GPU Whisper stalls** → Try `--device cpu` or smaller models; close other GPU apps
- **Version warnings** → scikit-learn version mismatches are usually harmless (model was trained with older version)

---

## What's New

### Latest Updates
- 🎤 **Interactive Mode**: `analyze_song.py --interactive` - paste lyrics directly, handles any special characters
- ⚡ **Quick Judge**: `judge.py` - simple command-line judge with auto model selection
- 📊 **Enhanced Analysis**: `analyze_song.py` - detailed quality scores, feature contributions, line-by-line analysis
- 🧹 **Improved Cleaning**: Better removal of transcription artifacts and ads
- ✅ **Overfitting Check**: `scripts/test_first_20_songs.py` - test model performance on dataset songs

### Previous Updates (v2)
- Fixed **clean_lyrics.py** usage and location (root-level, auto-writes `*.clean.txt`)
- Removed non-existent **K-fold** / **Bootstrap** CLI flags from `split_data.py`
- Expanded **Repo layout** to include utility scripts actually present in this repo

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
data/transcribed_lyrics/**
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

