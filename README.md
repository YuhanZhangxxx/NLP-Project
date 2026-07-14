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
│   ├── hybrid_scorer_gpt.py   ← single-round scorer (entry point)
│   ├── match_scorer.py        ← multi-round match scorer (stdin JSON → stdout JSON)
│   ├── score_from_db.py       ← DB-driven scorer: reads transcripts, saves verdict
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

## Model

The judge model is configurable. Resolution order is: `--model` (CLI), `payload.model`, `OPENAI_JUDGE_MODEL`, then `gpt-4o-mini`.

```bash
# Single performance
python skill_detection/hybrid_scorer_gpt.py --file data/battles/ralph_vs_rico/ralph_round1.txt --model gpt-4.1-mini

# Full match JSON
python skill_detection/match_scorer.py --file payload.json --model gpt-4.1-nano

# DB-backed scoring
python skill_detection/score_from_db.py --livestream-id <uuid> --model gpt-5-mini
```

For repeatable local testing, set `OPENAI_JUDGE_MODEL` in the environment. The scorer still uses the same prompt and rule engine; only the OpenAI judge model changes.

---

## Match Scorer (`match_scorer.py`)

Scores a full multi-round match. Reads a JSON payload from stdin, runs `hybrid_score()` for each rapper per round, determines per-round winners, and outputs a structured verdict.

```bash
cat payload.json | python skill_detection/match_scorer.py
python skill_detection/match_scorer.py --file payload.json  # debug
```

**Input:**
```json
{
  "match_id": "optional-uuid",
  "battler_a": "Rapper A",
  "battler_b": "Rapper B",
  "tie_threshold": 0.75,
  "rounds": [
    {"a": "A round 1 transcript", "b": "B round 1 transcript"},
    {"a": "A round 2 transcript", "b": "B round 2 transcript"}
  ]
}
```

**Output:**
```json
{
  "schema_version": 1,
  "status": "ok",
  "match_winner": "A",
  "summary": "Rapper A wins 2-1",
  "rounds": [
    {"round": 1, "winner": "A", "score_a": 18.0, "score_b": 12.5, "detail_a": {...}, "detail_b": {...}}
  ]
}
```

Rebuttal chain: B1 rebuts A1, A2 rebuts B1, B2 rebuts A2, etc. — enables STL detection.

---

## DB Scorer (`score_from_db.py`)

End-to-end scorer that reads transcripts from Neon DB, runs `match_scorer`, and optionally saves the verdict + W/L/BP back to the DB. Triggered automatically by the Node.js backend when all rounds of a match are submitted.

```bash
# Dry run (print verdict only)
python skill_detection/score_from_db.py --livestream-id <uuid>

# Save verdict to DB
python skill_detection/score_from_db.py --livestream-id <uuid> --save

# List all livestreams with transcripts
python skill_detection/score_from_db.py --list
```

Requires `DATABASE_URL` in `.env`. Rapper A/B order is determined by `livestreams.first_rapper` (explicit) or Round 1 earliest timestamp (fallback).

The scorer is **idempotent** — if a verdict already exists it skips the save and account updates.

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

**Automatic match scoring pipeline:**

```
Audio (WebM/Opus)
  → Deepgram nova-3 transcription
  → saveTranscript() [DB, idempotent upsert]
  → becameComplete? → spawnDbScorer(livestreamId)
      → score_from_db.py --livestream-id <uuid> --save
          → verdict + W/L/BP written to DB
```

The scorer's auto-split handles raw Deepgram output (wall-of-text) automatically — no pre-processing needed.

Backend scorer logs are written to `FCPBRL-Backend/logs/scorer-<uuid>.log` (stdout + stderr captured).

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

## Debugging Scorer in Production

### 1. Check if scorer was triggered

On the backend server, scorer logs are written to `FCPBRL-Backend/logs/scorer-<livestream-uuid>.log`.

```bash
# List all scorer logs
ls /home/ec2-user/FCPBRL-Backend/logs/

# If no log file exists for a match → scorer was never triggered
# (becameComplete never became true — match is likely incomplete)
```

### 2. Read the scorer log

```bash
cat /home/ec2-user/FCPBRL-Backend/logs/scorer-<uuid>.log
```

A successful run ends with:
```
[stderr]   Verdict saved: score1=X, score2=Y
[close] code=0 signal=null
```

A failed run will show a Python traceback in `[stderr]` lines and `[close] code=1`.

### 3. Check transcript completeness

If no log file exists, the match may be incomplete. Check via DB:

```bash
node -e "
const { Pool } = require('pg');
const pool = new Pool({ connectionString: process.env.DATABASE_URL });
pool.query(\`
  SELECT m.total_rounds, m.rapper1_bruf, m.rapper2_bruf,
         t.round, t.rapper, t.created
  FROM matches m
  JOIN transcripts t ON t.livestream_id = m.livestream_id
  WHERE m.livestream_id = '<uuid>'
  ORDER BY t.round, t.created
\`, (err, r) => { console.log(err || JSON.stringify(r.rows, null, 2)); pool.end(); });
"
```

Scorer triggers only when `actual == total_rounds * 2` (every rapper has submitted every round).

### 4. Check verdict in DB

```bash
node -e "
const { Pool } = require('pg');
const pool = new Pool({ connectionString: process.env.DATABASE_URL });
pool.query(\"SELECT score1, score2, verdict FROM matches WHERE livestream_id = '<uuid>'\",
  (err, r) => { console.log(err || JSON.stringify(r.rows[0], null, 2)); pool.end(); });
"
```

`verdict: null` means scorer either hasn't run or failed.

### 5. Manually trigger scorer

If the match is complete but verdict is still null (e.g. scorer crashed), run manually:

```bash
# On EC2
cd /home/ec2-user/NLP-Project
python3 skill_detection/score_from_db.py --livestream-id <uuid> --save

# Locally (Windows)
cd C:\Users\tuhai\Desktop\CS486-NLP-Project
.venv\Scripts\python.exe skill_detection/score_from_db.py --livestream-id <uuid> --save
```

The scorer is idempotent — safe to re-run. If verdict already exists it skips automatically.

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
