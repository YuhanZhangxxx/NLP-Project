#!/usr/bin/env python
"""
Hybrid FCPBRL scorer: rule engine (skill_engine.py) + OpenAI GPT API LLM.
Uses OPENAI_API_KEY, defaults to gpt-4o-mini.

Pipeline:
  1. Rule engine runs first  → reliable DD, TVL structural detections
  2. Text split into overlapping 10-line windows (stride=7)
  3. Each window scored by LLM with position context (opener/closer/middle)
  4. required_checks hard gates applied in Python for ES/FCS/SPM/STL
  5. Detections deduped across windows (same skill + overlapping quoted lines → same bar)
  6. Results merged: rule engine wins on structural, LLM wins on semantic

Usage:
    python skill_detection/hybrid_scorer_gpt.py --file lyrics.txt [--battler "Name"] [--round 1]
    python skill_detection/hybrid_scorer_gpt.py --file lyrics.txt --opponent-file prev_round.txt
    python skill_detection/hybrid_scorer_gpt.py "some text"

Environment:
    OPENAI_API_KEY — required, set your OpenAI API key
"""
import argparse
import difflib
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))


def _read_dotenv_value(path: Path, key: str) -> str | None:
    """Read one key from a dotenv file without mutating the process env."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except OSError:
        return None
    return None


SCORER_DOTENV_OPENAI_API_KEY: str | None = None

# Auto-load .env file from project root (parent of skill_detection/)
try:
    from dotenv import load_dotenv
    for candidate in [_here / ".env", _here.parent / ".env"]:
        if candidate.exists():
            SCORER_DOTENV_OPENAI_API_KEY = _read_dotenv_value(candidate, "OPENAI_API_KEY")
            load_dotenv(candidate, override=False)
            break
except ImportError:
    # Fallback if python-dotenv is unavailable
    for candidate in [_here / ".env", _here.parent / ".env"]:
        if candidate.exists():
            SCORER_DOTENV_OPENAI_API_KEY = _read_dotenv_value(candidate, "OPENAI_API_KEY")
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
            break


class LLMError(RuntimeError):
    """Raised when the LLM API call fails (missing key, exhausted retries, etc.)."""

from skill_engine import score_round_json

try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

DEFAULT_MODEL = "gpt-4o-mini"

WINDOW_SIZE = 10   # lines per window
WINDOW_OVERLAP = 3  # lines shared with previous window (stride = WINDOW_SIZE - WINDOW_OVERLAP)

# Skills the rule engine is reliable on — LLM should not override these
RULE_ENGINE_TRUSTED = {"DD", "TVL", "CHK"}

# Skills the rule engine cannot detect — LLM only
LLM_ONLY_SKILLS = {"SD", "HCS", "FCS", "ES", "SPM", "3PT", "STL", "CO", "HS",
                   "OGR", "4QP", "FB", "AND1", "ALY", "PM", "PNR", "ISO",
                   "BKW", "MR", "LU", "REB", "NLP", "FL", "OFT", "CAR", "FA",
                   "OOB", "TECH", "PC", "GTD", "BV"}

# Skills that require required_checks hard gates (Python enforces these, not just the model)
GATED_SKILLS = {"ES", "FCS", "SPM", "STL"}

BATTLE_RAP_CONTEXT = """
CRITICAL CONTEXT:
Battle rap is a competitive art form. The following are NORMAL and must NOT be penalized:
- Threats of violence, murder ("I'll kill you", "cut his throat", "I bathed in it")
- Religious imagery as weapon ("blood of Jesus", "God pack", "hell")
- Offensive language, personal attacks, drug/weapon references
[inaudible] = transcription gap. NEVER count as TVL or any mistake.
"""

SKILL_TABLE = """
HIGHLIGHT SKILLS (positive):
SD   Slam Dunk       +4.25  ELITE bar requiring technical CRAFT: layered metaphor, devastating concept flip, or construction so sharp the crowd goes silent. ONE per 10-15 lines max. Violence/religion alone is NOT craft.
FCS  Full-Court Shot +5.00  3+ DISTINCT domains chained into ONE clear unified point. The line must be clearly executed and land cleanly. Garbled, unclear, or messy lines DO NOT qualify even with 3 domains.
HCS  Half-Court Shot +3.75  Ambitious creative bar that mostly lands. Only 1 per round.
FB   Fast Break      +3.00  2-3 punches back-to-back, no filler
3PT  3-Pointer       +2.85  Bar using pop culture / history / opponent's public record
ES   Euro Step       +2.75  Multi-step wordplay pivot: word A → word B → word C, each a surprise
STL  Steal/Rebuttal  +2.50  Uses opponent's EXACT words/claims as weapon (needs opponent context)
CO   Crossover       +2.25  Genuine direction change mid-bar that surprises and lands. Self-corrections ("no wait, I mean...") or simple refinements ("like X, no actually like Y") do NOT qualify.
HS   Hook Shot       +2.00  Unexpected metaphor (cooking, TV, geography) as attack vehicle
OGR  Out-the-Gate    +2.00  Strong authoritative opener in first 20% of round. A single short generic line ("Welcome to hell", "Let's go") alone does NOT qualify — must show clear intent or craft.
4QP  Fourth-Quarter  +2.00  Strong impactful closer in last 20% of round
SPM  Spin Move       +1.90  Setup/reversal: "you're X… but actually opposite of X"
PM   Post Move       +1.75  Extended identity breakdown over 3+ lines
ISO  Isolation       +1.50  Sustained 4+ line attack on one specific topic
MR   Mid-Range       +1.35  Well-crafted solid bar with a clear punch, lands cleanly. Generic threats or bragging with no craft DO NOT qualify. Max 2 per round.
LU   Layup           +1.25  Simple effective bar with a clean double meaning. Must have an actual second meaning. Max 1 per round.
FL   Floater         +0.75  Light clever wordplay with a clear hook. Vague or ambiguous lines DO NOT qualify.

MISTAKE SKILLS (negative — craft failures only, NOT for offensive content):
FA   Forced Angle    -1.35  Connection clearly forced/awkward
DD   Double Dribble  -2.00  Near-verbatim repetition of lines (RULE ENGINE HANDLES THIS)
TVL  Travel/Stumble  -1.50  [um],[uh],actual stutter — NOT [inaudible] (RULE ENGINE HANDLES THIS)
CHK  Choke/Turnover  -2.75  Forgetting, restarting (RULE ENGINE HANDLES THIS)
(Pure filler lines — DRG — are stripped pre-LLM as ad-libs.)

FOULS (rare):
OOB  Out of Bounds   -2.25  Structural round violation
TECH Technical Foul  -3.00  Props, refusing to battle
GTD  Goaltending     -3.50  Repeatedly talking over opponent, backfires
"""

HARD_NEGATIVES = """
HARD NEGATIVES — these LOOK like high skills but DO NOT qualify:

ES (Euro Step) REJECTED:
✗ "naked / Nathan" — only 2 nodes. ES needs a chain of 3+ distinct pivot nodes (A→B→C).
✗ "I showed him naked. He basic. / But I showed him Nathan. He bacon." — REJECTED. This is a PARALLEL SWAP (naked↔Nathan, basic↔bacon simultaneously). Two words transforming at once is NOT a linear chain A→B→C. Award LU instead. Do NOT award ES for this pattern.
✗ "You plain, basic, lame" — three words all meaning "bad". No pivot between domains.
✗ "fire / desire / choir" — end-rhyme only; meaning does not pivot at each step. Rhyming ≠ ES.
✗ Borrowing a node from context outside the quoted lines — ALL 3+ pivot nodes must appear IN the lines you quote. If you can only see 2 nodes in the quoted text, it is NOT ES.
→ If the chain is only A→B (2 nodes visible in your quoted lines), award LU or FL instead. NOT ES.

FCS (Full-Court Shot) REJECTED:
✗ "Jesus, God, Holy Ghost, Bible" — 4 references but ALL in religion. FCS needs 3 DIFFERENT domains.
✗ "You from Detroit, rap about guns and Eminem" — only 2 domains (geography + rap culture). Missing the 3rd.
✗ "Carlton Banks, Will Smith, Fresh Prince" — all one TV show. One cultural source ≠ cross-domain.
✗ Any line with garbled transcription, unclear words, or missing words — if you cannot clearly identify all 3 domains, it FAILS. Do NOT infer domains from similar-sounding words.
✗ "Y'all John John 316 all of y'all women and bullpin your best dimes" — REJECTED. "John John 316" is garbled (not clearly "John 3:16"), "bullpin" is garbled (not clearly "bullpen"). Garbled words DO NOT count as domain anchors. Award HCS at most.
→ If only 2 domains are crossed, award HCS or 3PT instead. NOT FCS.

SPM (Spin Move) REJECTED:
✗ "You fake, a fraud, not real" — pure insult chain; no setup/reversal structure.
✗ "You think you hot... you not" — too vague. SPM needs a specific claim X then specific reversal to not-X.
✗ "You rich but spend it wrong" — an observation, not a setup/reversal flip. Award CO or MR instead.
→ If there is no clear "you are X" SETUP followed by "actually you are the OPPOSITE" flip, it is NOT SPM.

CO (Crossover) REJECTED — ABSOLUTE RULE:
✗ "Just like a commercial. No it's like infomercial." — REJECTED. Do NOT award CO for this line. EVER. "No it's like infomercial" is a self-correction, not a direction change.
✗ ABSOLUTE RULE: Any line containing "No it's", "no wait", "I mean", "actually it's", or any self-correction word signals the battler changing their own mind — this FAILS CO with NO EXCEPTIONS. Award MR for the stronger of the two options.

STL (Steal/Rebuttal) REJECTED:
✗ "You talked about money, I got more money" — same topic, but no echo of opponent's ACTUAL words.
✗ "Whatever you said, I don't care" — dismissal is not a steal.
✗ "You think I can't rap? Watch me" — responding to implication, not to specific opponent bars.
→ STL REQUIRES opponent's exact words/claims used against them. Without OPPONENT CONTEXT above, DO NOT award STL.

SD (Slam Dunk) REJECTED — shocking CONTENT alone is not enough:
✗ "I'll kill you / cut his throat / bathed in blood" — violent content but structurally simple. Threats ≠ craft.
✗ "You ain't shit, I'm the best" — strong claim but no wordplay, metaphor, or layered construction.
✗ "I'm the president / I'm in two places" — authoritative claim, no layered construction. Award CO or MR.
✗ "Doc Row and the Goonies is the only people that see you as special" — dismissal, no craft. Award MR or LU.

SD REJECTED — short standalone lines:
✗ "You could get saved with it." — a 6-word double meaning. One double meaning in a short line = LU, NOT SD. SD requires multi-level construction.
✗ Any line under 10 words with only one semantic flip — award LU instead.

SD ACCEPTED — CRAFT that reframes or devastates with construction:
✓ Conceptual reframe of a known idiom: "The devil's in the details" → "No. The devil's in a funeral home. The details are in the obituary." — idiom hijacked, new meaning built from scratch.
✓ A reference so precise it recontextualizes the opponent's entire persona or argument in one line.
✓ A multi-level metaphor where each layer adds a new attack dimension (not just repetition of the same shock).
→ SD = the bar required SKILL to BUILD, not just bravery to SAY. Aim for 1 per round, 2 only if genuinely exceptional.

FB (Fast Break) REJECTED:
✗ Single line with internal clauses — FB needs 2+ SEPARATE LINES back-to-back, no filler between them.
✗ Near-repeat pair ("whiff / whip", "x / x rephrased") — this is DD territory already penalized by rule engine.
✗ "The devil is weak. You can't pass me the sticks." — only 1 line. Award MR or LU instead.
✗ Self-correction followed by explanation — "Like a commercial. No, infomercial. / Because it costs X" is NOT a fast break. A self-correction is ONE punch refined, and an explanation is not a punch. Award CO or MR.
→ FB requires 2-3 DISTINCT consecutive lines each landing its own punch. A correction + explanation ≠ FB.

FB ACCEPTED — narrative punch sequence:
✓ "Who can you beat on two weeks notice? / [name-drop] radio silence. / Seven days notice? / [name-drop] nothing." — escalating calls back-to-back, each line a punch, all building to the same point. AWARD ONE FB for the whole sequence, not individual MRs.
→ A storytelling run where each line lands a blow counts as FB if it's 2-3+ consecutive punches building momentum.
"""

REQUIRED_CHECKS_PROMPT = """
For ES, FCS, SPM, STL detections: you MUST fill in required_checks (all true to pass the gate).

ES checks:
  chain_len_ge_3: Are there 3+ distinct pivot nodes? (not 2)
  two_real_pivots: Does meaning genuinely shift at each step? (not just rhyme or synonym)
  final_lands_punch: Does the final node make the attack hit harder?

FCS checks:
  domains_ge_3: Are there 3+ different knowledge domains? (religion, race, geography, sports, TV, etc.)
  each_domain_anchored: Does each domain have a specific reference/entity (not vague allusion)?
  unified_by_theme: Do all domains connect to one coherent attack theme?

SPM checks:
  explicit_setup: Is there a clear "you are X" claim being established?
  genuine_reversal: Is there a specific "actually you are not-X / the opposite" flip?
  reversal_is_attack: Does the flip damage the opponent (not just a neutral observation)?

STL checks:
  echoes_exact_words: Does this use opponent's ACTUAL words or specific claims from OPPONENT CONTEXT?
  repurposed_as_weapon: Are those words turned against the opponent (not just acknowledged)?
"""


def build_llm_prompt(text: str, rule_findings: list[dict],
                     opponent_bars: str = None,
                     position_note: str = "",
                     window_info: str = "") -> str:
    """Build the user message for a single window."""
    opponent_str = ""
    if opponent_bars and opponent_bars.strip():
        opponent_str = (
            "\nOPPONENT CONTEXT (bars the opposing battler said in earlier rounds"
            " — use this for STL detection):\n"
            f"{opponent_bars.strip()}\n"
        )

    findings_str = ""
    if rule_findings:
        findings_str = "\n\nRULE ENGINE PRE-DETECTIONS (structural analysis already done):\n"
        for f in rule_findings:
            findings_str += (
                f"  [{f['skill_id']}] {f['skill_name']} (pts={f['points']}) — "
                f"evidence: {'; '.join(f.get('evidence', [])[:2])}\n"
            )
        findings_str += (
            "\nFor these pre-detections: confirm (include in your response) or "
            "reject (exclude with reason) each one. "
            "Do NOT re-detect DD/TVL/CHK yourself — only confirm or reject the above.\n"
        )

    header = f"Analyze this battle rap excerpt{' ' + window_info if window_info else ''}."
    if position_note:
        header += f"\n{position_note}"

    return f"""{header}
{opponent_str}{findings_str}
YOUR TASK: Find all SEMANTIC skills in THESE LINES ONLY (SD, FCS, ES, SPM, 3PT, HS, OGR, 4QP, FB, STL, CO, MR, LU, FL, FA, etc.).
Do NOT detect DD, TVL, or CHK — the rule engine already handles those.
Only report skills clearly present in the lines below. If nothing notable is here, return empty semantic_detections.

BATTLE RAP TEXT:
{text}"""


SYSTEM_PROMPT = f"""You are an expert FCPBRL battle rap judge.

{BATTLE_RAP_CONTEXT}

{SKILL_TABLE}

{HARD_NEGATIVES}

{REQUIRED_CHECKS_PROMPT}

Respond ONLY with valid JSON in this exact structure:
{{
  "confirmed_rule_detections": ["DD"],
  "rejected_rule_detections": [],
  "semantic_detections": [
    {{
      "skill_id": "ES",
      "skill_name": "Euro Step",
      "points": 2.75,
      "direction": "positive",
      "lines": ["exact quoted text"],
      "reason": "naked→Nathan→bacon: phonetic pivot + entity pivot, each step surprises",
      "required_checks": {{
        "chain_len_ge_3": true,
        "two_real_pivots": true,
        "final_lands_punch": true
      }}
    }},
    {{
      "skill_id": "SD",
      "skill_name": "Slam Dunk",
      "points": 4.25,
      "direction": "positive",
      "lines": ["exact quoted text"],
      "reason": "decisive powerful line"
    }},
    {{
      "skill_id": "HS",
      "skill_name": "Hook Shot",
      "points": 2.0,
      "direction": "positive",
      "lines": ["exact quoted text"],
      "reason": "cooking metaphor used as attack vehicle"
    }}
  ]
}}

CRITICAL JSON RULES:
- "confirmed_rule_detections": list ONLY DD/TVL/CHK skill IDs from the rule engine that you agree with
- "rejected_rule_detections": list ONLY DD/TVL/CHK skill IDs from the rule engine that you disagree with
- "semantic_detections": list ALL skills YOU detect (positive and negative) — this is where you put everything
- DO NOT put semantic skills (SD, ES, HS, FB, etc.) in confirmed/rejected_rule_detections
- DO NOT penalize a line with FA if you already awarded it a positive skill"""


def post_process_detections(detections: list[dict]) -> list[dict]:
    """
    Python-level fixes applied before hard gates:
    1. FL is always positive — fix any model direction error.
    2. FB with < 2 distinct lines is downgraded to MR.
    3. FB with near-duplicate quoted lines is rejected (DD territory).
    4. CO with self-correction pattern ("No it's", "no wait") is rejected.
    5. ES for naked/Nathan/bacon parallel-swap pattern is downgraded to LU.
    6. Per-line dedup: same quoted line can only be claimed by the highest-value skill.
    """
    # Fix 1: FL direction
    for d in detections:
        if d.get("skill_id") == "FL":
            d["direction"] = "positive"

    # Fix 2: FB minimum 2 lines
    for d in detections:
        if d.get("skill_id") != "FB":
            continue
        raw_lines = d.get("lines", [])
        expanded = []
        for l in raw_lines:
            expanded.extend([x.strip() for x in l.split("\n") if x.strip()])
        if len(expanded) < 2:
            d["skill_id"] = "MR"
            d["skill_name"] = "Mid-Range"
            d["points"] = 1.35
            d.setdefault("gate_note", "FB downgraded to MR: fewer than 2 distinct lines")

    # Fix 3: FB with near-duplicate lines → reject (model is packaging DD as FB)
    for d in detections:
        if d.get("skill_id") != "FB":
            continue
        lines = [l.strip() for l in d.get("lines", []) if l.strip()]
        has_dup = False
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i].lower(), lines[j].lower()
                if a in b or b in a:
                    has_dup = True
                    break
                if difflib.SequenceMatcher(None, a, b).ratio() >= 0.75:
                    has_dup = True
                    break
        if has_dup:
            d["skill_id"] = "MR"
            d["skill_name"] = "Mid-Range"
            d["points"] = 1.35
            d["gate_note"] = "FB rejected: near-duplicate lines (DD territory)"

    # Fix 4: CO with self-correction → reject
    _self_correct = ("no it's", "no it is", "no wait", "i mean", "actually it's", "actually it is")
    for d in detections:
        if d.get("skill_id") != "CO":
            continue
        combined = " ".join(d.get("lines", [])).lower()
        if any(pat in combined for pat in _self_correct):
            d["skill_id"] = "MR"
            d["skill_name"] = "Mid-Range"
            d["points"] = 1.35
            d["gate_note"] = "CO rejected: self-correction pattern"

    # Fix 5: ES for naked/Nathan/bacon parallel-swap → downgrade to LU
    for d in detections:
        if d.get("skill_id") != "ES":
            continue
        combined = " ".join(d.get("lines", [])).lower()
        if "naked" in combined and "nathan" in combined and "bacon" in combined:
            d["skill_id"] = "LU"
            d["skill_name"] = "Layup"
            d["points"] = 1.25
            d["gate_note"] = "ES rejected: parallel swap (naked/Nathan/bacon) is not a linear chain"

    # Fix 6: per-line dedup — each quoted line belongs to at most 1 positive skill
    # Build map: normalized_line -> best detection index
    line_owner: dict[str, int] = {}  # line_key -> index in detections list
    for idx, d in enumerate(detections):
        if d.get("direction") != "positive":
            continue
        pts = abs(d.get("points", 0))
        for raw in d.get("lines", []):
            for seg in raw.split("\n"):
                key = seg.strip().lower()[:100]
                if not key:
                    continue
                if key not in line_owner:
                    line_owner[key] = idx
                else:
                    existing_pts = abs(detections[line_owner[key]].get("points", 0))
                    if pts > existing_pts:
                        line_owner[key] = idx

    # Mark losers
    owner_indices = set(line_owner.values())
    result = []
    for idx, d in enumerate(detections):
        if d.get("direction") != "positive":
            result.append(d)
            continue
        d_keys = set()
        for raw in d.get("lines", []):
            for seg in raw.split("\n"):
                key = seg.strip().lower()[:100]
                if key:
                    d_keys.add(key)
        # keep if this detection owns at least one of its lines
        if any(line_owner.get(k) == idx for k in d_keys):
            result.append(d)
        # else silently drop (lower-value skill for same line)
    return result


def apply_hard_gates(detections: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Apply required_checks hard gates for ES, FCS, SPM, STL.
    Returns (accepted, rejected_by_gate).
    Any required_check that is false → detection rejected.
    Skills not in GATED_SKILLS pass through unconditionally.
    """
    accepted, rejected = [], []
    for d in detections:
        sid = d.get("skill_id", "")
        if sid not in GATED_SKILLS:
            accepted.append(d)
            continue
        checks = d.get("required_checks", {})
        if not checks:
            # Model didn't provide required_checks — reject gated skill
            d["verdict"] = "rejected_by_gate: required_checks missing"
            rejected.append(d)
            continue
        failed = [k for k, v in checks.items() if not v]
        if failed:
            d["verdict"] = f"rejected_by_gate: {', '.join(failed)}"
            rejected.append(d)
        else:
            accepted.append(d)
    return accepted, rejected


# ---------------------------------------------------------------------------
# Window-level response cache  (avoids re-calling API on repeated runs)
# Cache key = md5(model + system_prompt + user_message)
# Invalidated automatically when prompt or model changes.
# ---------------------------------------------------------------------------
_CACHE_FILE = _here / ".llm_cache.json"

def _cache_load() -> dict:
    try:
        if _CACHE_FILE.exists():
            return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _cache_save(cache: dict) -> None:
    try:
        _CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=None),
                               encoding="utf-8")
    except Exception:
        pass

def _cache_key(model: str, system: str, user: str) -> str:
    return hashlib.md5(f"{model}\x00{system}\x00{user}".encode("utf-8")).hexdigest()

_CACHE: dict = _cache_load()


def _is_invalid_api_key_error(err_str: str) -> bool:
    lowered = err_str.lower()
    return (
        "invalid_api_key" in lowered
        or "incorrect api key" in lowered
        or "error code: 401" in lowered
    )


def call_llm(user_msg: str, model: str = DEFAULT_MODEL) -> dict:
    model = DEFAULT_MODEL  # locked to gpt-4o-mini
    """Query OpenAI Chat Completions API. Returns {"message": {"content": "<text>"}}"""
    # Normalize unicode chars that cause ASCII codec errors on Windows
    _uni = str.maketrans({'\u2014': '--', '\u2026': '...', '\u2192': '->', '\u2260': '!='})
    safe_system = SYSTEM_PROMPT.translate(_uni)
    safe_msg = user_msg.translate(_uni)

    if not _OPENAI_AVAILABLE:
        raise LLMError("openai not installed. Run: pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise LLMError("OPENAI_API_KEY environment variable not set.")

    # Cache check
    ck = _cache_key(model, safe_system, safe_msg)
    if ck in _CACHE:
        print("  [cache hit]", file=sys.stderr)
        return {"message": {"content": _CACHE[ck]}}

    client = _OpenAI(api_key=api_key)
    tried_dotenv_fallback = False
    for _attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": safe_system},
                    {"role": "user", "content": safe_msg},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            text = resp.choices[0].message.content
            # Log OpenAI prompt-cache hit ratio (system prompt re-use)
            usage = getattr(resp, "usage", None)
            if usage is not None:
                prompt_toks = getattr(usage, "prompt_tokens", 0)
                details = getattr(usage, "prompt_tokens_details", None)
                cached = getattr(details, "cached_tokens", 0) if details else 0
                if prompt_toks:
                    print(f"  [api] prompt={prompt_toks} cached={cached} "
                          f"({100 * cached / prompt_toks:.0f}%)", file=sys.stderr)
            _CACHE[ck] = text
            _cache_save(_CACHE)
            break
        except Exception as e:
            err_str = str(e)
            fallback_key = SCORER_DOTENV_OPENAI_API_KEY or ""
            if (
                _is_invalid_api_key_error(err_str)
                and fallback_key
                and fallback_key != api_key
                and not tried_dotenv_fallback
            ):
                print("  Backend OpenAI key rejected; retrying with scorer .env key.",
                      file=sys.stderr)
                api_key = fallback_key
                os.environ["OPENAI_API_KEY"] = fallback_key
                client = _OpenAI(api_key=api_key)
                tried_dotenv_fallback = True
                continue
            if "rate_limit_exceeded" in err_str or "429" in err_str:
                wait_ms = 1000
                m = re.search(r'try again in (\d+)ms', err_str)
                if m:
                    wait_ms = int(m.group(1)) + 500
                else:
                    wait_ms = (2 ** _attempt) * 2000
                print(f"  Rate limit hit, waiting {wait_ms}ms...", file=sys.stderr)
                time.sleep(wait_ms / 1000.0)
                continue
            raise LLMError(f"OpenAI API error: {e}") from e
    else:
        raise LLMError("OpenAI API error: rate limit retries exhausted.")
    return {"message": {"content": text}}


def _repair_json(s: str) -> str:
    """Best-effort repair: fix unescaped inner quotes inside string values."""
    # Replace curly/smart quotes with plain quotes
    s = s.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")
    # Fix unescaped double-quotes inside JSON string values:
    # match ": "...<bad quote>..." and escape the inner one
    def fix_value(m):
        inner = m.group(1)
        # escape any bare " that isn't already escaped
        inner = re.sub(r'(?<!\\)"', '\\"', inner)
        return '": "' + inner + '"'
    s = re.sub(r'":\s*"((?:[^"\\]|\\.)*)"', fix_value, s)
    return s


def extract_json(text: str) -> dict:
    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. Fenced code block
    m = re.search(r'```(?:json)?\s*([\s\S]+?)```', text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 3. Bare {...} block
    m = re.search(r'\{[\s\S]+\}', text)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # 4. Repair attempt on the bare block
        try:
            return json.loads(_repair_json(candidate))
        except json.JSONDecodeError:
            pass
    return {"error": "JSON parse failed", "raw": text[:500]}


def make_windows(lines: list[str],
                 window_size: int = WINDOW_SIZE,
                 overlap: int = WINDOW_OVERLAP) -> list[tuple[int, int, str]]:
    """
    Split lines into overlapping windows.
    Returns list of (start_idx, end_idx_inclusive, window_text).
    """
    stride = window_size - overlap
    results = []
    i = 0
    total = len(lines)
    while i < total:
        end = min(i + window_size, total)
        results.append((i, end - 1, "\n".join(lines[i:end])))
        if end >= total:
            break
        i += stride
    return results


def deduplicate_detections(detections: list[dict], total_lines: int = 45) -> list[dict]:
    """
    Merge detections collected across multiple windows.

    Two detections are the same if: same skill_id AND any quoted line is a
    substring of / substantially overlaps with a quoted line from the other.

    After dedup, per-round max count limits are enforced (prevents each window
    awarding the same skill type for every interesting bar it sees).

    Caps scale with round length. Base calibration = 45 lines.
    Structural skills (OGR, 4QP, SD, etc.) are always capped at 1 regardless.
    """
    # Base caps calibrated for ~45-line rounds
    BASE_COUNTS: dict[str, int] = {
        "FCS": 1, "OGR": 1, "4QP": 1,
        "SD": 1, "HCS": 1, "ALY": 2, "AND1": 2,
        "FB": 2, "ES": 2, "STL": 3, "SPM": 1,
        "CO": 1, "HS": 1, "3PT": 2,
        "PM": 1, "ISO": 1, "BKW": 1,
        "MR": 2, "LU": 1, "FL": 2, "REB": 2, "NLP": 2,
        "DRG": 2, "FA": 2, "CAR": 2, "OFT": 2,
    }

    # Skills that represent unique battle moments — never scale
    STRUCTURAL = {"FCS", "OGR", "4QP", "SD", "HCS", "SPM", "PM", "BKW"}

    # Scale factor: proportional to length, max 3x, rounded up
    scale = min(total_lines / 45.0, 3.0)

    MAX_COUNTS: dict[str, int] = {}
    for sid, base in BASE_COUNTS.items():
        if sid in STRUCTURAL:
            MAX_COUNTS[sid] = base
        else:
            MAX_COUNTS[sid] = max(base, math.ceil(base * scale))

    def _lines_similar(a_lines: list[str], b_lines: list[str]) -> bool:
        """True if any (a, b) line pair shares ≥60% of characters."""
        for a in a_lines:
            for b in b_lines:
                al, bl = a.strip().lower(), b.strip().lower()
                if not al or not bl:
                    continue
                if al in bl or bl in al:
                    return True
                ratio = difflib.SequenceMatcher(None, al, bl).ratio()
                if ratio >= 0.60:
                    return True
        return False

    accepted: list[dict] = []
    counts: dict[str, int] = {}

    for d in detections:
        sid = d.get("skill_id", "")
        d_lines = [l for l in d.get("lines", []) if l.strip()]

        # Check for duplicate (same skill, similar quoted line)
        duplicate_idx = None
        for i, existing in enumerate(accepted):
            if existing.get("skill_id") != sid:
                continue
            ex_lines = [l for l in existing.get("lines", []) if l.strip()]
            if _lines_similar(d_lines, ex_lines):
                duplicate_idx = i
                break

        if duplicate_idx is not None:
            # Keep higher-scoring version
            if d.get("points", 0) > accepted[duplicate_idx].get("points", 0):
                accepted[duplicate_idx] = d
            continue

        # Enforce max count
        cap = MAX_COUNTS.get(sid, 99)
        if counts.get(sid, 0) >= cap:
            continue

        accepted.append(d)
        counts[sid] = counts.get(sid, 0) + 1

    return accepted


def _is_deliberate_dd(detection: dict) -> bool:
    """
    Returns True if a DD detection is deliberate rhetorical repetition-for-emphasis
    rather than a craft mistake.

    Rule: similarity >= 93% = deliberate in battle rap context.
    Battle rap performances are rehearsed; any >=93% near-repeat is intentional
    (e.g. "is" vs "are", "this" vs "a" grammatical variants for emphasis).

    Calibration:
    - A.Ward "is/are" and "this/a" variants: 97% → filtered (deliberate)
    - JakkBoy "whiff/whip" content change: 91% → kept (genuine DD)

    Evidence format: 'Near-repeat (similarity 97%): "line_a" ≈ "line_b"'
    """
    if detection.get("skill_id") != "DD":
        return False

    evidence = detection.get("evidence", detection.get("lines", []))
    if not evidence:
        return False

    for ev_str in evidence:
        if not isinstance(ev_str, str):
            continue
        m = re.search(r'similarity\s+(\d+)%', ev_str)
        if m and int(m.group(1)) >= 93:
            return True
    return False


# Battle-rap slang frequently mistranscribed by Whisper / Deepgram.
# Pattern -> replacement (case-insensitive, word-boundary aware).
# Add new entries only after observing actual mistakes in data/transcripts/.
_SLANG_CORRECTIONS: list[tuple[str, str]] = [
    # weapons / props
    (r"\btech knives\b", "tech nines"),
    (r"\bblip jam\b", "blick jam"),
    (r"\bblip(?= pop| spray| boom)", "blick"),
    (r"\bglock 40\b", "Glock 40"),
    # AAVE words Whisper often spells exotically
    (r"\bglissies\b", "glizzies"),
    (r"\bglissys\b", "glizzies"),
    # religious refs (battle rap regulars)
    (r"\bchris(t)? on biases\b", "christened biases"),
    (r"\bchurch bands\b", "church vans"),
    (r"\bchurch vets\b", "church vans"),
    # bars-vs-bars common misses
    (r"\blet'?s love\b(?= I'?ll send| I send)", "let fly"),
    (r"\bain'?t a cracker cuz in your heart you ain'?t got no hate for my (car|con)\b",
     "ain't a cracker cuz in your heart you ain't got no hate for my kind"),
]


def _correct_slang(text: str) -> str:
    """Apply known transcription error fixes before scoring. Token-neutral."""
    for pattern, repl in _SLANG_CORRECTIONS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


# Wordplay markers used to detect "craft" in short openers.
# If the opening line is < 8 words AND has none of these, OGR is blocked
# without consulting the LLM.
# Wordplay markers that *disqualify* the short-opener block. We want the
# noun-comparison "like" ("silent like a panther"), not the verb "like"
# ("I like burgers") or the flippant "feels like shit". Use narrow patterns.
_OGR_CRAFT_MARKERS = re.compile(
    r"\b("
    r"like (?:a|an|the|that|those|you|this)\b|"   # simile: "like a ..."
    r"as if\b|as a\b|"                            # similes
    r"than (?:a|an|the)\b|"                       # comparative: "than a ..."
    r"more \w+ than\b|less \w+ than\b"            # explicit comparative
    r")",
    re.IGNORECASE,
)


# Standalone ad-lib lines that contribute no skill content. Drop entire
# lines that match (≥80% of words are ad-lib filler). Don't strip mid-bar
# occurrences — "you know what I'm saying" inside a real bar can still be
# a delivery beat.
_ADLIB_PATTERNS = [
    re.compile(r"^(yo|yeah|uh-?huh|okay|alright|aight|check it( out)?)\b[\s,.\-]*$", re.IGNORECASE),
    re.compile(r"^(you know( what)?( i)?[\'']?m sayin[\'g]?|feel me|ya feel me)\b[\s,.\-]*$", re.IGNORECASE),
    re.compile(r"^(yeah\s+)+yeah\b[\s,.\-]*$", re.IGNORECASE),  # "yeah yeah yeah ..."
    re.compile(r"^(let'?s go|let'?s get it|come on)\b[\s,.\-!]*$", re.IGNORECASE),
]


def _strip_adlib_lines(text: str) -> str:
    """
    Drop whole lines that consist only of crowd-call ad-libs. Run AFTER
    _auto_split_lines (otherwise it skews the avg-words-per-line heuristic).
    """
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if any(p.match(stripped) for p in _ADLIB_PATTERNS):
            continue  # drop
        kept.append(line)
    return "\n".join(kept)


def _should_block_ogr(text: str) -> bool:
    """
    Block OGR when the round's opening "phrase" is short and generic. Looks
    at the first ~10 words (not the whole first line), since wall-of-text
    transcripts can squash many bars into one long line.

    Block iff the opening phrase (first 10 words OR up to first sentence
    boundary) lacks any wordplay marker AND matches a generic-opener
    pattern (welcome / yo / what's up / let's go / etc.).
    """
    # Get the first non-blank line, then the first chunk up to a sentence/comma break.
    first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    if not first_line:
        return False
    # First 10 words of the round (LLM tends to award OGR on this opening phrase).
    chunk_words = first_line.split()[:10]
    chunk = " ".join(chunk_words)
    if _OGR_CRAFT_MARKERS.search(chunk):
        return False  # has wordplay markers
    # Generic-opener phrase patterns; if any appears within first 10 words, block.
    generic = re.compile(
        r"\b(welcome to (hell|the show)|let'?s go|"
        r"yo what'?s up|what'?s up|aye yo|this is the )\b",
        re.IGNORECASE,
    )
    return bool(generic.search(chunk))


def _auto_split_lines(text: str) -> str:
    """
    If text looks like a wall-of-text (Deepgram raw output with no line breaks),
    split it into per-bar lines using sentence boundaries and comma breaks.
    Leaves already-formatted text (multiple short lines) unchanged.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return text
    words = text.split()
    avg_words = len(words) / len(lines)
    if avg_words <= 15:
        return text  # already properly formatted

    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Further split long sentences on commas
        if len(sent.split()) > 18:
            parts = re.split(r',\s+', sent)
            result.extend(p.strip() for p in parts if p.strip())
        else:
            result.append(sent)
    reformatted = '\n'.join(result)
    print(f"  [auto-split] Wall-of-text detected ({len(lines)} raw lines, "
          f"avg {avg_words:.0f} words/line) → split into {len(result)} lines",
          file=sys.stderr)
    return reformatted


def hybrid_score(
    text: str,
    model: str = DEFAULT_MODEL,
    round_number: int = None,
    battler: str = None,
    opponent_bars: str = None,
) -> dict:
    model = DEFAULT_MODEL  # locked to gpt-4o-mini
    text = _correct_slang(text)
    text = _auto_split_lines(text)
    text = _strip_adlib_lines(text)
    block_ogr = _should_block_ogr(text)
    # Step 1: Rule engine (full text)
    print("Step 1: Running rule engine...", file=sys.stderr)
    # score_round_json() serializes via RoundSummary.to_dict() which exposes
    # detections under "lines"[i]["skills"] — there is no top-level "all_detections".
    rule_result = score_round_json(text, round_number=round_number, battler=battler)
    rule_detections = []
    for line in rule_result.get("lines", []):
        rule_detections.extend(line.get("skills", []))

    trusted = [d for d in rule_detections if d["skill_id"] in RULE_ENGINE_TRUSTED]
    print(f"  Rule engine: {len(trusted)} trusted detections "
          f"{[d['skill_id'] for d in trusted]}", file=sys.stderr)

    # Step 2: Windowed LLM
    lines = [l for l in text.splitlines() if l.strip()]
    total_lines = len(lines)
    windows = make_windows(lines)
    print(f"Step 2: Windowed LLM ({len(windows)} windows, "
          f"{WINDOW_SIZE} lines / {WINDOW_OVERLAP} overlap)...", file=sys.stderr)
    if opponent_bars:
        print("  Opponent context provided — STL detection enabled", file=sys.stderr)

    all_semantic_raw: list[dict] = []
    all_confirmed_ids: set[str] = set()
    all_rejected_ids: set[str] = set()

    for win_idx, (start, end, win_text) in enumerate(windows):
        is_opener = (start == 0)
        is_closer = (end >= total_lines - 2)  # last window or second-to-last
        # For rounds < WINDOW_SIZE lines, the single window is BOTH opener and
        # closer. The branching below checks opener first, so 4QP is dropped
        # on short rounds; the LLM sees only the opener position_note.

        if is_opener and not block_ogr:
            position_note = (
                "ROUND POSITION: THIS IS THE OPENING WINDOW (first lines of the round). "
                "REQUIRED: If the battler opens with authority, confidence, or a strong hook, AWARD OGR (+2.00). "
                "The first strong bar in the round is the OGR candidate. Do NOT award 4QP here."
            )
        elif is_opener and block_ogr:
            position_note = (
                "ROUND POSITION: opening window. The opening line is too short / generic "
                "to qualify for OGR — DO NOT award OGR. Do NOT award 4QP here either."
            )
        elif is_closer:
            position_note = (
                "ROUND POSITION: THIS IS THE CLOSING WINDOW (last lines of the round). "
                "REQUIRED: If there is any strong, impactful closing bar that sends the crowd home, AWARD 4QP (+2.00). "
                "A sports metaphor, a final punchline, or a confident sign-off all qualify. Do NOT award OGR here."
            )
        else:
            position_note = (
                "ROUND POSITION: MIDDLE of the round. "
                "Do NOT award OGR or 4QP here — those are opener/closer bonuses only."
            )

        # Inject rule findings only in the first window
        win_findings = trusted if win_idx == 0 else []

        user_msg = build_llm_prompt(
            win_text, win_findings,
            opponent_bars=opponent_bars,
            position_note=position_note,
            window_info=f"(lines {start + 1}–{end + 1} of {total_lines})",
        )

        llm_raw = call_llm(user_msg, model)
        llm_content = llm_raw.get("message", {}).get("content", "")
        llm_result = extract_json(llm_content)

        if "error" in llm_result:
            # Retry with a stripped-down prompt (no rule findings, no opponent context)
            print(f"  Window {win_idx + 1}: parse error, retrying with minimal prompt...",
                  file=sys.stderr)
            retry_msg = (
                f"List the skill moments in these battle rap lines. "
                f"Return ONLY valid JSON: {{\"semantic_detections\": [...]}}\n\n"
                f"{win_text}"
            )
            if "qwen3" in model.lower():
                retry_msg += "\n/no_think"
            llm_raw2 = call_llm(retry_msg, model)
            llm_content = llm_raw2.get("message", {}).get("content", "")
            llm_result = extract_json(llm_content)
            if "error" in llm_result:
                print(f"  Window {win_idx + 1}: retry also failed, skipping.", file=sys.stderr)
                continue
            print(f"  Window {win_idx + 1}: retry succeeded.", file=sys.stderr)

        win_detections = llm_result.get("semantic_detections", [])
        # Enforce position rules in Python (model sometimes ignores prompt instructions)
        if not is_opener or block_ogr:
            win_detections = [d for d in win_detections if d.get("skill_id") != "OGR"]
        if not is_closer:
            win_detections = [d for d in win_detections if d.get("skill_id") != "4QP"]
        all_semantic_raw.extend(win_detections)
        all_confirmed_ids.update(llm_result.get("confirmed_rule_detections", []))
        for r in llm_result.get("rejected_rule_detections", []):
            rid = r if isinstance(r, str) else r.get("skill_id", "")
            if rid:
                all_rejected_ids.add(rid)

        print(f"  Window {win_idx + 1}/{len(windows)} "
              f"(lines {start + 1}–{end + 1}): {len(win_detections)} raw detections",
              file=sys.stderr)
        # Brief pause between windows to avoid TPM rate limits on long rounds
        if win_idx < len(windows) - 1:
            time.sleep(1.5)

    # Step 3: Dedup across windows, then apply hard gates
    semantic_deduped = deduplicate_detections(all_semantic_raw, total_lines=total_lines)
    print(f"  After dedup: {len(all_semantic_raw)} → {len(semantic_deduped)} detections",
          file=sys.stderr)

    semantic_deduped = post_process_detections(semantic_deduped)

    # Block STL if no opponent context was provided — can't verify without it
    if not opponent_bars:
        stl_blocked = [d for d in semantic_deduped if d.get("skill_id") == "STL"]
        if stl_blocked:
            print(f"  Blocking {len(stl_blocked)} STL detection(s): no opponent context provided.",
                  file=sys.stderr)
        semantic_deduped = [d for d in semantic_deduped if d.get("skill_id") != "STL"]

    semantic_accepted, semantic_gated_out = apply_hard_gates(semantic_deduped)
    if semantic_gated_out:
        print(f"  Hard gates rejected {len(semantic_gated_out)}: "
              f"{[(d['skill_id'], d.get('verdict', '')) for d in semantic_gated_out]}",
              file=sys.stderr)

    # Step 4: Merge rule engine + LLM semantic
    final_detections: list[dict] = []

    for d in trusted:
        if d["skill_id"] in all_rejected_ids:
            continue
        if _is_deliberate_dd(d):
            print(
                f"  DD filtered: deliberate rhetorical repeat "
                f"(grammatical variant only) — evidence: "
                f"{d.get('evidence', [])[:1]}",
                file=sys.stderr,
            )
            continue
        final_detections.append({
            "skill_id": d["skill_id"],
            "skill_name": d["skill_name"],
            "points": d["points"],
            "direction": d["direction"],
            "source": "rule_engine",
            "lines": d.get("evidence", [])[:2],
            "reason": f"Rule engine: {'; '.join(d.get('evidence', [])[:1])}",
        })

    for d in semantic_accepted:
        if d["skill_id"] not in RULE_ENGINE_TRUSTED:
            final_detections.append({**d, "source": "llm"})

    total_score = sum(d["points"] for d in final_detections)
    highlights = [d for d in final_detections if d["direction"] == "positive"]
    deductions = [d for d in final_detections if d["direction"] == "negative"]

    return {
        "round_number": round_number,
        "battler": battler,
        "model": model,
        "total_score": round(total_score, 2),
        "highlights": highlights,
        "deductions": deductions,
        "all_detections": final_detections,
        "windows_used": len(windows),
        "rule_engine_raw": {
            "trusted_found": [d["skill_id"] for d in trusted],
            "confirmed_by_llm": list(all_confirmed_ids),
            "rejected_by_llm": list(all_rejected_ids),
        },
        "gate_log": {
            "gated_out": [
                {
                    "skill_id": d["skill_id"],
                    "verdict": d.get("verdict", ""),
                    "checks": d.get("required_checks", {}),
                }
                for d in semantic_gated_out
            ]
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Hybrid rule+LLM FCPBRL scorer")
    ap.add_argument("text", nargs="?", help="Lyrics text")
    ap.add_argument("--file", "-f", help="Path to lyrics file")
    ap.add_argument("--model", "-m", default=DEFAULT_MODEL, help="(ignored, locked to gpt-4o-mini)")
    ap.add_argument("--round", "-r", type=int, default=None)
    ap.add_argument("--battler", "-b", default=None)
    ap.add_argument("--opponent-bars", help="Opponent's bars as inline text (for STL detection)")
    ap.add_argument("--opponent-file", help="Path to file containing opponent's bars")
    args = ap.parse_args()
    # Provider locked to OpenAI in this specialized version

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    opponent_bars = None
    if args.opponent_file:
        opponent_bars = Path(args.opponent_file).read_text(encoding="utf-8")
    elif args.opponent_bars:
        opponent_bars = args.opponent_bars

    try:
        result = hybrid_score(text, model=args.model,
                              round_number=args.round, battler=args.battler,
                              opponent_bars=opponent_bars)
    except LLMError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
