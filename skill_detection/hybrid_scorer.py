#!/usr/bin/env python
"""
Hybrid FCPBRL scorer: rule engine (skill_engine.py) + local LLM (Ollama).

Pipeline:
  1. Rule engine runs first  → reliable DD, TVL structural detections
  2. Rule findings injected into LLM prompt as "pre-detected facts"
  3. LLM confirms/rejects rule findings AND adds semantic skills (SD, ES, FCS, SPM, 3PT…)
  4. Results merged: rule engine wins on structural, LLM wins on semantic

Usage:
    python skill_detection/hybrid_scorer.py --file lyrics.txt [--battler "Name"] [--round 1]
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

# Skills the rule engine is reliable on — LLM should not override these
RULE_ENGINE_TRUSTED = {"DD", "TVL", "CHK"}

# Skills the rule engine cannot detect — LLM only
LLM_ONLY_SKILLS = {"SD", "HCS", "FCS", "ES", "SPM", "3PT", "STL", "CO", "HS",
                   "OGR", "4QP", "FB", "AND1", "ALY", "PM", "PNR", "ISO",
                   "BKW", "MR", "LU", "REB", "NLP", "FL", "OFT", "CAR", "FA",
                   "OOB", "TECH", "PC", "GTD", "BV"}

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
STL  Steal/Rebuttal  +2.50  Uses opponent's own words/claims as weapon
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


def build_llm_prompt(text: str, rule_findings: list[dict]) -> str:
    """Build the user message including rule engine pre-detections."""
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

    return f"""Analyze this battle rap round and identify every skill moment.

{findings_str}
YOUR TASK: Find all SEMANTIC skills (SD, FCS, ES, SPM, 3PT, HS, OGR, 4QP, FB, STL, CO, MR, LU, FL, DRG, FA, OOB, GTD, etc.).
Do NOT detect DD, TVL, or CHK — the rule engine already handles those.

BATTLE RAP TEXT:
{text}"""


SYSTEM_PROMPT = f"""You are an expert FCPBRL battle rap judge.

{BATTLE_RAP_CONTEXT}

{SKILL_TABLE}

Respond ONLY with valid JSON:
{{
  "confirmed_rule_detections": ["DD", "TVL"],
  "rejected_rule_detections": [{{"skill_id": "X", "reason": "why rejected"}}],
  "semantic_detections": [
    {{
      "skill_id": "<ID>",
      "skill_name": "<name>",
      "points": <float>,
      "direction": "positive" | "negative",
      "lines": ["<exact quoted text>"],
      "reason": "<1 sentence>"
    }}
  ]
}}"""


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


def hybrid_score(
    text: str,
    model: str = DEFAULT_MODEL,
    round_number: int = None,
    battler: str = None,
) -> dict:
    # Step 1: Rule engine
    print("Step 1: Running rule engine...", file=sys.stderr)
    rule_result = score_round_json(text, round_number=round_number, battler=battler)
    rule_detections = rule_result.get("all_detections") or []
    # Flatten from lines if needed
    if not rule_detections:
        for line in rule_result.get("lines", []):
            rule_detections.extend(line.get("skills", []))

    trusted = [d for d in rule_detections if d["skill_id"] in RULE_ENGINE_TRUSTED]
    print(f"  Rule engine found {len(trusted)} trusted detections: "
          f"{[d['skill_id'] for d in trusted]}", file=sys.stderr)

    # Step 2: LLM
    print("Step 2: Querying LLM...", file=sys.stderr)
    user_msg = build_llm_prompt(text, trusted)
    llm_raw = query_ollama(user_msg, model)
    llm_content = llm_raw.get("message", {}).get("content", "")
    llm_result = extract_json(llm_content)

    if "error" in llm_result:
        print(f"  LLM parse error: {llm_result}", file=sys.stderr)

    # Step 3: Merge
    confirmed_ids = set(llm_result.get("confirmed_rule_detections", []))
    rejected_ids = {r["skill_id"] for r in llm_result.get("rejected_rule_detections", [])}
    semantic = llm_result.get("semantic_detections", [])

    final_detections = []

    # Include trusted rule detections that LLM confirmed (or didn't explicitly reject)
    for d in trusted:
        if d["skill_id"] not in rejected_ids:
            final_detections.append({
                "skill_id": d["skill_id"],
                "skill_name": d["skill_name"],
                "points": d["points"],
                "direction": d["direction"],
                "source": "rule_engine",
                "lines": d.get("evidence", [])[:2],
                "reason": f"Rule engine: {'; '.join(d.get('evidence', [])[:1])}",
            })

    # Add LLM semantic detections (skip if skill_id overlaps with trusted)
    for d in semantic:
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
        "summary": llm_result.get("summary", ""),
        "rule_engine_raw": {
            "trusted_found": [d["skill_id"] for d in trusted],
            "confirmed_by_llm": list(confirmed_ids),
            "rejected_by_llm": list(rejected_ids),
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Hybrid rule+LLM FCPBRL scorer")
    ap.add_argument("text", nargs="?", help="Lyrics text")
    ap.add_argument("--file", "-f", help="Path to lyrics file")
    ap.add_argument("--model", "-m", default=DEFAULT_MODEL)
    ap.add_argument("--round", "-r", type=int, default=None)
    ap.add_argument("--battler", "-b", default=None)
    args = ap.parse_args()

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    result = hybrid_score(text, model=args.model,
                          round_number=args.round, battler=args.battler)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
