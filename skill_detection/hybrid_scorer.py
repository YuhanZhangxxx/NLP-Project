#!/usr/bin/env python
"""
Hybrid FCPBRL scorer: rule engine (skill_engine.py) + local LLM (Ollama).

Pipeline:
  1. Rule engine runs first  → reliable DD, TVL structural detections
  2. Text split into overlapping 6-line windows (stride=4)
  3. Each window scored by LLM with position context (opener/closer/middle)
  4. required_checks hard gates applied in Python for ES/FCS/SPM/STL
  5. Detections deduped across windows (same skill + overlapping quoted lines → same bar)
  6. Results merged: rule engine wins on structural, LLM wins on semantic

Usage:
    python skill_detection/hybrid_scorer.py --file lyrics.txt [--battler "Name"] [--round 1]
    python skill_detection/hybrid_scorer.py --file lyrics.txt --opponent-file prev_round.txt
    python skill_detection/hybrid_scorer.py "some text"
"""
import argparse
import json
import sys
import re
import urllib.request
import urllib.error
from pathlib import Path

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from skill_engine import score_round_json

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b"

WINDOW_SIZE = 6    # lines per window
WINDOW_OVERLAP = 2  # lines shared with previous window (stride = WINDOW_SIZE - WINDOW_OVERLAP)

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
SD   Slam Dunk       +4.25  One decisive powerful line — violence/religion is NORMAL here
FCS  Full-Court Shot +5.00  3+ cross-domain references chained (gun+religion+race+culture)
HCS  Half-Court Shot +3.75  Ambitious creative bar, mostly lands
FB   Fast Break      +3.00  2-3 punches back-to-back, no filler
3PT  3-Pointer       +2.85  Bar using pop culture / history / opponent's public record
ES   Euro Step       +2.75  Multi-step wordplay pivot: word A → word B → word C, each a surprise
STL  Steal/Rebuttal  +2.50  Uses opponent's EXACT words/claims as weapon (needs opponent context)
CO   Crossover       +2.25  Rapid angle switch that surprises then lands
HS   Hook Shot       +2.00  Unexpected metaphor (cooking, TV, geography) as attack vehicle
OGR  Out-the-Gate    +2.00  Strong authoritative opener in first 20% of round
4QP  Fourth-Quarter  +2.00  Strong impactful closer in last 20% of round
SPM  Spin Move       +1.90  Setup/reversal: "you're X… but actually opposite of X"
PM   Post Move       +1.75  Extended identity breakdown over 3+ lines
ISO  Isolation       +1.50  Sustained 4+ line attack on one specific topic
MR   Mid-Range       +1.35  Well-crafted solid bar, lands cleanly
LU   Layup           +1.25  Simple effective bar, clean double meaning
FL   Floater         +0.75  Light clever wordplay

MISTAKE SKILLS (negative — craft failures only, NOT for offensive content):
DRG  Backcourt Viol  -1.25  Pure filler lines, adds nothing
FA   Forced Angle    -1.35  Connection clearly forced/awkward
DD   Double Dribble  -2.00  Near-verbatim repetition of lines (RULE ENGINE HANDLES THIS)
TVL  Travel/Stumble  -1.50  [um],[uh],actual stutter — NOT [inaudible] (RULE ENGINE HANDLES THIS)
CHK  Choke/Turnover  -2.75  Forgetting, restarting (RULE ENGINE HANDLES THIS)

FOULS (rare):
OOB  Out of Bounds   -2.25  Structural round violation
TECH Technical Foul  -3.00  Props, refusing to battle
GTD  Goaltending     -3.50  Repeatedly talking over opponent, backfires
"""

HARD_NEGATIVES = """
HARD NEGATIVES — these LOOK like high skills but DO NOT qualify:

ES (Euro Step) REJECTED:
✗ "naked / Nathan" — only 2 nodes. ES needs a chain of 3+ distinct pivot nodes (A→B→C).
✗ "You plain, basic, lame" — three words all meaning "bad". No pivot between domains.
✗ "fire / desire / choir" — end-rhyme only; meaning does not pivot at each step. Rhyming ≠ ES.
→ If the chain is only A→B (2 nodes), award LU or FL instead. NOT ES.

FCS (Full-Court Shot) REJECTED:
✗ "Jesus, God, Holy Ghost, Bible" — 4 references but ALL in religion. FCS needs 3 DIFFERENT domains.
✗ "You from Detroit, rap about guns and Eminem" — only 2 domains (geography + rap culture). Missing the 3rd.
✗ "Carlton Banks, Will Smith, Fresh Prince" — all one TV show. One cultural source ≠ cross-domain.
→ If only 2 domains are crossed, award HCS or 3PT instead. NOT FCS.

SPM (Spin Move) REJECTED:
✗ "You fake, a fraud, not real" — pure insult chain; no setup/reversal structure.
✗ "You think you hot... you not" — too vague. SPM needs a specific claim X then specific reversal to not-X.
✗ "You rich but spend it wrong" — an observation, not a setup/reversal flip. Award CO or MR instead.
→ If there is no clear "you are X" SETUP followed by "actually you are the OPPOSITE" flip, it is NOT SPM.

STL (Steal/Rebuttal) REJECTED:
✗ "You talked about money, I got more money" — same topic, but no echo of opponent's ACTUAL words.
✗ "Whatever you said, I don't care" — dismissal is not a steal.
✗ "You think I can't rap? Watch me" — responding to implication, not to specific opponent bars.
→ STL REQUIRES opponent's exact words/claims used against them. Without OPPONENT CONTEXT above, DO NOT award STL.
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
YOUR TASK: Find all SEMANTIC skills in THESE LINES ONLY (SD, FCS, ES, SPM, 3PT, HS, OGR, 4QP, FB, STL, CO, MR, LU, FL, DRG, FA, etc.).
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
- DO NOT penalize a line with DRG or FA if you already awarded it a positive skill"""


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
            # Model didn't provide checks — accept but flag
            d.setdefault("gate_note", "no required_checks provided — accepted without gate")
            accepted.append(d)
            continue
        failed = [k for k, v in checks.items() if not v]
        if failed:
            d["verdict"] = f"rejected_by_gate: {', '.join(failed)}"
            rejected.append(d)
        else:
            accepted.append(d)
    return accepted, rejected


def query_ollama(user_msg: str, model: str = DEFAULT_MODEL) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 4096},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"Ollama error: {e}", file=sys.stderr)
        sys.exit(1)


def extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'```(?:json)?\s*([\s\S]+?)```', text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r'\{[\s\S]+\}', text)
    if m:
        try:
            return json.loads(m.group(0))
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


def deduplicate_detections(detections: list[dict]) -> list[dict]:
    """
    Merge detections collected across multiple windows.

    Two detections are the same if: same skill_id AND any quoted line is a
    substring of / substantially overlaps with a quoted line from the other.

    After dedup, per-round max count limits are enforced (prevents each window
    awarding the same skill type for every interesting bar it sees).
    """
    import difflib

    # Max times each skill can fire in one round
    MAX_COUNTS: dict[str, int] = {
        "FCS": 1, "OGR": 1, "4QP": 1,
        "SD": 2, "HCS": 2, "ALY": 2, "AND1": 2,
        "FB": 2, "ES": 2, "STL": 3, "SPM": 2,
        "CO": 2, "HS": 2, "3PT": 2,
        "PM": 1, "ISO": 1, "BKW": 1,
        "MR": 4, "LU": 4, "FL": 5, "REB": 2, "NLP": 2,
        "DRG": 2, "FA": 2, "CAR": 2, "OFT": 2,
    }

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


def hybrid_score(
    text: str,
    model: str = DEFAULT_MODEL,
    round_number: int = None,
    battler: str = None,
    opponent_bars: str = None,
) -> dict:
    # Step 1: Rule engine (full text)
    print("Step 1: Running rule engine...", file=sys.stderr)
    rule_result = score_round_json(text, round_number=round_number, battler=battler)
    rule_detections = rule_result.get("all_detections") or []
    if not rule_detections:
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

        if is_opener:
            position_note = (
                "ROUND POSITION: OPENING of the round. "
                "OGR (Out-the-Gate) MAY apply if the opening bars are strong. "
                "Do NOT award 4QP here."
            )
        elif is_closer:
            position_note = (
                "ROUND POSITION: CLOSING of the round. "
                "4QP (Fourth-Quarter Push) MAY apply if the closing bars are strong. "
                "Do NOT award OGR here."
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

        llm_raw = query_ollama(user_msg, model)
        llm_content = llm_raw.get("message", {}).get("content", "")
        llm_result = extract_json(llm_content)

        if "error" in llm_result:
            print(f"  Window {win_idx + 1}: parse error — {llm_result.get('error')}", file=sys.stderr)
            continue

        win_detections = llm_result.get("semantic_detections", [])
        all_semantic_raw.extend(win_detections)
        all_confirmed_ids.update(llm_result.get("confirmed_rule_detections", []))
        for r in llm_result.get("rejected_rule_detections", []):
            rid = r if isinstance(r, str) else r.get("skill_id", "")
            if rid:
                all_rejected_ids.add(rid)

        print(f"  Window {win_idx + 1}/{len(windows)} "
              f"(lines {start + 1}–{end + 1}): {len(win_detections)} raw detections",
              file=sys.stderr)

    # Step 3: Dedup across windows, then apply hard gates
    semantic_deduped = deduplicate_detections(all_semantic_raw)
    print(f"  After dedup: {len(all_semantic_raw)} → {len(semantic_deduped)} detections",
          file=sys.stderr)

    semantic_accepted, semantic_gated_out = apply_hard_gates(semantic_deduped)
    if semantic_gated_out:
        print(f"  Hard gates rejected {len(semantic_gated_out)}: "
              f"{[(d['skill_id'], d.get('verdict', '')) for d in semantic_gated_out]}",
              file=sys.stderr)

    # Step 4: Merge rule engine + LLM semantic
    final_detections: list[dict] = []

    for d in trusted:
        if d["skill_id"] not in all_rejected_ids:
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
    ap.add_argument("--model", "-m", default=DEFAULT_MODEL)
    ap.add_argument("--round", "-r", type=int, default=None)
    ap.add_argument("--battler", "-b", default=None)
    ap.add_argument("--opponent-bars", help="Opponent's bars as inline text (for STL detection)")
    ap.add_argument("--opponent-file", help="Path to file containing opponent's bars")
    args = ap.parse_args()

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

    result = hybrid_score(text, model=args.model,
                          round_number=args.round, battler=args.battler,
                          opponent_bars=opponent_bars)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
