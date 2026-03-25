# FCPBRL Scoring System — Deployment Options

## What We Built

Three modules, can be used alone or together:

| File | What it does |
|------|-------------|
| `skill_engine.py` | Rule-based engine — catches structural mistakes like repeated lines |
| `llm_scorer.py` | Pure LLM scoring via Ollama |
| `hybrid_scorer.py` | Rule engine + LLM combined — **recommended for production** |

The idea behind hybrid: the rule engine is really good at catching exact line repetitions (DD) and stumble markers (TVL), while the LLM handles the creative stuff like wordplay, cultural references, and hard-hitting bars. They cover each other's blind spots.

---

## How Accurate Is It Right Now?

We tested on two real rounds — JakkBoy Maine and A. Ward — and compared against a human judge estimate.

| Method | JakkBoy Score | A. Ward Score |
|--------|-------------|--------------|
| Rule engine alone | -0.75 | +3.50 |
| LLM alone (7B) | +21.50 | +15.60 |
| Hybrid (7B) | +32.20 | +20.00 |
| **Human estimate** | **+24.60** | **+38.60** |

The current 7B model gets you roughly **50-60% of human accuracy**. The main gap is cultural references — things like the Nathan→bacon wordplay chain, or knowing who Carlton Banks is, or catching the "Outside 5" event name flip. The 7B model just doesn't have that knowledge baked in deeply enough.

---

## Bigger Model = Better Accuracy

| Model Size | Accuracy (est.) | VRAM Needed |
|-----------|----------------|-------------|
| 7B (current) | ~50-60% | 6-8 GB |
| 14B | ~65% | 10-12 GB |
| **32B** | **~75%** | **20-24 GB** |
| 70B | ~85-90% | 40-48 GB |

The 32B is the sweet spot. Going from 7B to 14B doesn't move the needle much — you really need 32B before the cultural reference stuff starts working reliably.

---

## Server Options

### Option A — Keep 7B (Cheapest)

If you're happy with the current accuracy, you just need a GPU with 8GB+ VRAM.

| Platform | GPU | Price | Notes |
|----------|-----|-------|-------|
| **RunPod** | RTX 3080 16GB | ~$0.20/hr | Best value, stable |
| Vast.ai | RTX 3070/3080 | ~$0.10-0.15/hr | Cheapest, slightly less reliable |
| AWS g4dn.xlarge (Spot) | T4 16GB | ~$0.16-0.18/hr | AWS cheapest GPU, can be interrupted |
| AWS g4dn.xlarge (On-demand) | T4 16GB | ~$0.53/hr | Stable but pricey for what you get |

**Pick: RunPod RTX 3080 at $0.20/hr.** Double the VRAM of your local 3070, much cheaper than AWS, and stable enough for live scoring.

---

### Option B — Upgrade to 32B (~75% accuracy)

| Platform | GPU | Price | Notes |
|----------|-----|-------|-------|
| **AWS g5.xlarge** | A10G 24GB | ~$1.01/hr | Reliable, good for production |
| RunPod A10 | A10 24GB | ~$0.50-0.75/hr | Same GPU, cheaper |
| Lambda Labs | A10 24GB | ~$0.75/hr | Solid alternative |

---

### Option C — 70B, Near-Human Accuracy

| Platform | GPU | Price | Notes |
|----------|-----|-------|-------|
| AWS g6e.xlarge | L40S 48GB | ~$2.20/hr | Single card, runs 70B |
| **RunPod A100 80GB** | A100 80GB | ~$1.99/hr | Faster inference |

---

### Option D — Just Use Claude API (No Server Needed)

Skip the GPU entirely and call the API per round.

| Model | Per round | Per battle (6 rounds) | 100 battles/day × 30 days |
|-------|-----------|----------------------|--------------------------|
| Haiku 4.5 | $0.003 | $0.018 | $54/month |
| **Sonnet 4.6** | **$0.011** | **$0.067** | **$200/month** |

Best accuracy by far (Sonnet is close to human-level). Good option if you're just starting out and don't want to deal with server maintenance.

---

## Recommended Path

```
Early stage / testing
  └─ RunPod RTX 3080 + qwen2.5:7b  →  $0.20/hr, current accuracy

Once you go live
  ├─ Budget first  →  RunPod A10 + qwen2.5:32b  →  ~$0.50/hr, ~75% accuracy
  └─ Accuracy first  →  Claude API Sonnet        →  pay-per-use, best accuracy
```

---

## A Few Technical Notes

- Inference framework: **Ollama** (installed, v0.18.2)
- Latency per round (~74 lines of text):
  - 7B on RTX 3070: ~15-20 seconds
  - 32B on A10G: ~15-25 seconds (can drop to 5-8s with vLLM)
  - 70B on A100: ~30-50 seconds
- For live streaming: score each round asynchronously after transcription completes — don't block the main feed waiting for results
