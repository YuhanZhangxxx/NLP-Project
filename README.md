# FCPBRL Battle Rap Hybrid Scorer

A hybrid battle rap skill detection scorer using **GPT-4o-mini + rule engine** to analyze battle rap transcripts and return a structured JSON score based on the FCPBRL basketball-scoring taxonomy.

---

## How It Works

```
Text Input
  ↓
Rule Engine (DD / TVL / CHK only)
  +
Windowed LLM Scoring (GPT-4o-mini)
  · 10-line windows, 3-line overlap
  · Per-window position context (opener / middle / closer)
  ↓
Merge → Deduplicate → Dynamic Cap Scaling
  ↓
JSON Score Output
```

**Rule engine** handles 3 trusted detections only:
- `DD` — Deliberate Duplicate (rhetorical repeat, similarity ≥ 93%)
- `TVL` — Travel/Stumble (filler sounds, stutters)
- `CHK` — Choke (forgetting lines, restart)

All other 30+ techniques are LLM-only.

**Dynamic cap scaling** prevents under-scoring on long rounds:
```python
scale = min(total_lines / 45.0, 3.0)
# STRUCTURAL skills (FCS, OGR, 4QP, SD, HCS, SPM, PM, BKW) → never scale
# CONTENT skills (3PT, CO, HS, MR, LU, FL, etc.) → ceil(base * scale)
```

**LLM response cache** stored at `skill_detection/.llm_cache.json` (gitignored). Key = md5(model + prompts). Auto-invalidates on prompt/model changes.

**Auto-split** — if input is a wall-of-text (e.g. raw Deepgram output with no line breaks), the scorer automatically splits it into per-bar lines before scoring.

---

## Repo Layout (beta branch)

```
CS486-NLP-Project/
├── skill_detection/
│   ├── hybrid_scorer_gpt.py   ← main scorer (entry point)
│   ├── rap_techniques.py      ← rule-based technique detectors
│   ├── skill_engine.py        ← rule engine + score_round_json()
│   └── .llm_cache.json        ← LLM response cache (gitignored)
├── data/
│   └── battles/
│       ├── ralph_vs_rico/
│       │   ├── ralph_round1.txt   (150 lines)
│       │   └── rico_round1.txt    (78 lines)
│       └── award_vs_jakkboy/
│           ├── award_round1-3.txt
│           ├── jakkboy_round1-3.txt
│           └── jakkboy_round1_deepgram.txt  ← raw Deepgram test
├── .env                       ← API keys (gitignored, create manually)
├── requirements.txt           ← locked versions
└── README.md
```

---

## Setup (New Machine)

### 1. Clone and enter the repo

```bash
git clone <repo-url>
cd CS486-NLP-Project
git checkout beta
```

### 2. Create virtual environment

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Locked versions:
```
openai==2.30.0
python-dotenv==1.0.1
nltk==3.9.2
numpy==1.26.4
scipy==1.12.0
```

### 4. Create `.env` file

Create `.env` in the **project root** (not inside `skill_detection/`):

```
OPENAI_API_KEY=sk-...your-key-here...
```

The scorer auto-loads this file on startup.

### 5. Download NLTK data (first run only)

The scorer handles this automatically on first run. If it fails manually:

```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
```

---

## Usage

### From file

```bash
python skill_detection/hybrid_scorer_gpt.py --file data/battles/ralph_vs_rico/ralph_round1.txt
```

### With options

```bash
python skill_detection/hybrid_scorer_gpt.py \
  --file data/battles/ralph_vs_rico/ralph_round1.txt \
  --battler "Ralph" \
  --round 1 \
  --opponent-file data/battles/ralph_vs_rico/rico_round1.txt  # enables STL detection
```

### Inline text

```bash
python skill_detection/hybrid_scorer_gpt.py "Your bars here..."
```

### From stdin (Node.js / backend integration)

```bash
echo "bar bar bar" | python skill_detection/hybrid_scorer_gpt.py
```

### Change model

```bash
# More accurate, higher cost
python skill_detection/hybrid_scorer_gpt.py --file lyrics.txt --model gpt-4o

# Default (cost-efficient)
python skill_detection/hybrid_scorer_gpt.py --file lyrics.txt --model gpt-4o-mini
```

### Save output to file

```bash
python skill_detection/hybrid_scorer_gpt.py --file lyrics.txt > result.json
```

---

## Output Format

```json
{
  "round_number": 1,
  "battler": "Ralph",
  "model": "gpt-4o-mini",
  "total_score": 50.8,
  "highlights": [
    {
      "skill_id": "SD",
      "skill_name": "Slam Dunk",
      "points": 4.25,
      "direction": "positive",
      "lines": ["...quoted bar..."],
      "reason": "...",
      "source": "llm"
    }
  ],
  "deductions": [...],
  "all_detections": [...],
  "windows_used": 21
}
```

---

## Node.js Backend Integration

The scorer reads from **stdin** when no `--file` or text argument is provided. Typical Express integration:

```typescript
// AIService.ts
score(text: string): Promise<object> {
  return new Promise((resolve, reject) => {
    const py = spawn('python', [
      'C:/path/to/CS486-NLP-Project/skill_detection/hybrid_scorer_gpt.py'
    ]);
    let out = '';
    py.stdin.write(text);
    py.stdin.end();
    py.stdout.on('data', d => out += d);
    py.on('close', code => {
      if (code !== 0) reject(new Error('Scorer failed'));
      else resolve(JSON.parse(out));
    });
  });
}
```

Pipeline: **Audio → Deepgram transcription → text → Python scorer (stdin) → JSON**

The scorer's auto-split handles raw Deepgram output (wall-of-text) automatically — no pre-processing needed.

---

## Skill Taxonomy (FCPBRL)

| Code | Name | Points | Notes |
|------|------|--------|-------|
| SD | Slam Dunk | +4.25 | Elite layered bar, 1 per 10-15 lines |
| FCS | Full-Court Shot | +5.00 | 3+ distinct domains in one bar |
| HCS | Half-Court Shot | +3.75 | Ambitious bar that mostly lands, 1 per round |
| FB | Fast Break | +3.00 | 2-3 punches back-to-back |
| 3PT | 3-Pointer | +2.85 | Pop culture / history / opponent's record |
| ES | Euro Step | +2.75 | Word chain A→B→C, 3+ pivot nodes |
| STL | Steal/Rebuttal | +2.50 | Uses opponent's exact words as weapon |
| CO | Crossover | +2.25 | Genuine direction change mid-bar |
| HS | Hook Shot | +2.00 | Unexpected metaphor as attack vehicle |
| OGR | Out-the-Gate | +2.00 | Strong opener (first 20% of round) |
| 4QP | Fourth-Quarter | +2.00 | Strong closer (last 20% of round) |
| SPM | Spin Move | +1.90 | Setup/reversal: "you're X… actually opposite" |
| PM | Post Move | +1.75 | Extended identity breakdown 3+ lines |
| ISO | Isolation | +1.50 | Sustained 4+ line attack on one topic |
| MR | Mid-Range | +1.35 | Well-crafted bar with clean punch, max 2/round |
| LU | Layup | +1.25 | Simple bar with actual double meaning, max 1/round |
| FL | Floater | +0.75 | Light wordplay with clear hook |
| DD | Double Dribble | -2.00 | Near-verbatim repetition (rule engine) |
| TVL | Travel/Stumble | -1.50 | Filler sounds, stutters (rule engine) |
| CHK | Choke/Turnover | -2.75 | Forgetting lines, restart (rule engine) |
| FA | Forced Angle | -1.35 | Clearly forced/awkward connection |
| DRG | Backcourt Viol. | -1.25 | Pure filler lines |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'skill_detection.funcs'`**
→ Make sure you're using the `beta` branch. Run `git checkout beta`.

**`ERROR: OPENAI_API_KEY environment variable not set`**
→ Create `.env` in the project root with `OPENAI_API_KEY=sk-...`

**Score is -1.5 and 0 highlights on Deepgram transcript**
→ Input is wall-of-text with no line breaks. The auto-split should handle this automatically. Check that you're on the latest beta commit.

**Rate limit errors (429)**
→ Scorer retries up to 5x with exponential backoff automatically. For long rounds, the 1.5s inter-window sleep prevents most rate limit issues. Check your OpenAI tier if it persists.

**Windows: `python3` not found**
→ Use `python` instead of `python3` on Windows.

**Windows: PowerShell execution policy**
→ Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating venv.
