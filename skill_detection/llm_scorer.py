#!/usr/bin/env python
"""
LLM-based FCPBRL skill scorer using local Ollama.

Usage:
    python skill_detection/llm_scorer.py --file lyrics.txt [--model qwen2.5:7b] [--battler "Name"] [--round 1]
    python skill_detection/llm_scorer.py "some battle rap text"
"""
import argparse
import json
import sys
import re
import urllib.request
import urllib.error
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b"

BATTLE_RAP_CONTEXT = """
CRITICAL CONTEXT — READ FIRST:
Battle rap is a competitive art form where performers deliver pre-written verses attacking their opponent.
The following are ALL completely NORMAL in battle rap and must NEVER be marked as fouls or mistakes:
- Threats of violence, murder, death ("I'll kill you", "I'll shoot you", "I'll cut your throat")
- Religious imagery used as weapons ("blood of Jesus", "bathing in it", "God pack", "hell")
- Extremely offensive language, insults, personal attacks
- References to drugs, weapons, crime
- Provocative or disturbing metaphors
These are the CRAFT of battle rap. Penalize them ONLY if they are clumsy/unfunny/forced, not merely offensive.

[inaudible] in the text = a word the transcriber could not hear. It is NOT a stumble or mistake by the battler. IGNORE all [inaudible] markers when evaluating TVL/CHK.

POSITIVE SKILL EXAMPLES (what to look for):
- SD (Slam Dunk): "Don't preach to me about the blood of Jesus when I cut his throat and I bathed in it" — decisive, powerful, lands hard. THIS IS SD, not OFT.
- ES (Euro Step): "I showed him naked / He basic / But I showed him Nathan / He bacon" — leads one direction (naked=vulnerable) then pivots (Nathan→bacon=gun). Multi-word chain that misdirects.
- FCS (Full-Court Shot): A sequence connecting 3+ unrelated domains (e.g. gun → religious metaphor → racial history → pop culture reference) all in one cohesive attack.
- SPM (Spin Move): "You ain't a cracker cause in your heart... But then again, you IS a cracker cause in your brain..." — sets up a compliment then reverses it into an attack.
- 3PT (3-Pointer): "I left Fresh Prince cause Carlton found God with speed" — uses knowledge of a TV show character's arc as a punchline. Shows external research/cultural knowledge.
- HS (Hook Shot): "my uncle chefing in Hell's Kitchen for your soul until it's tender" — uses an unexpected cooking/location metaphor to attack.
- LU (Layup): "I send you and yours to heaven / You could get saved with it" — simple, clean double-meaning threat. Works without being complex.
- DD (Double Dribble): Same lines repeated word-for-word or near-verbatim later in the round.
- DRG (Backcourt Violation): "Into hell it is. Into hell with this." — pure filler that just repeats without adding content.
"""

SKILL_TABLE = """
HIGHLIGHT SKILLS (positive points):
FCS  Full-Court Shot      +5.00  3+ cross-domain references chained into one punch (gun+religion+race+culture). Extremely rare.
SD   Slam Dunk            +4.25  One decisive powerful line that clearly "wins" the moment. Violence/religion imagery is NORMAL here.
HCS  Half-Court Shot      +3.75  Ambitious creative bar, high-risk, mostly lands
ALY  Alley-Oop            +3.50  Clear set-up line followed by a payoff line that wouldn't work without the setup
AND1 And-1                +3.25  Punch lands cleanly despite interruption or crowd noise
FB   Fast Break           +3.00  2-3 strong punches back-to-back with no filler
3PT  3-Pointer            +2.85  Bar showing external research: pop culture, history, opponent's public record
ES   Euro Step            +2.75  Multi-step wordplay pivot: word A → word B → word C, each step a surprise
STL  Steal / Rebuttal     +2.50  Uses opponent's own words/claims as the weapon against them
CO   Crossover            +2.25  Rapid angle switch mid-bar that surprises then lands
HS   Hook Shot            +2.00  Unusual unexpected metaphor (cooking, sports, TV, geography) used as attack vehicle
OGR  Out-the-Gate Run     +2.00  Strong authoritative opener in first 20% of round
4QP  Fourth-Quarter Push  +2.00  Strong impactful closer in last 20% of round
SPM  Spin Move            +1.90  Setup/reversal: "you're X... but actually you're the opposite of X"
PM   Post Move            +1.75  Extended identity breakdown over 3+ lines on one topic
PNR  Pick & Roll          +1.65  Battler sets up their own angle, then knocks it down
ISO  Isolation            +1.50  Sustained 4+ line attack on one specific topic
BKW  Breakaway            +1.40  Clean uninterrupted flow for 8+ lines, no stumbles
MR   Mid-Range            +1.35  Well-crafted solid bar, lands cleanly
LU   Layup               +1.25  Simple effective bar, works without complexity
REB  Rebound             +1.15  Good recovery bar after a weak moment
NLP  No-Look Pass        +0.90  Devastating line delivered casually, as an aside
FL   Floater             +0.75  Soft clever wordplay that lands lightly

MISTAKE SKILLS (negative points — only for craft failures, NOT for offensive content):
OFT  Offensive Foul      -0.50  Offensive content with ZERO lyrical value AND no punchline (shock for shock's sake only)
CAR  Carry               -1.00  Setup that runs too long (4+ lines) without payoff
DRG  Backcourt Violation -1.25  Pure filler lines — repetitive phrases that add nothing
FA   Charge/Forced Angle -1.35  Connection forced that clearly doesn't fit (awkward, reaches too far)
TVL  Travel/Stumble      -1.50  [um], [uh], actual stuttering, or losing the line mid-delivery. NOT [inaudible].
DD   Double Dribble      -2.00  Near-verbatim repetition of lines already used in this round
CHK  Turnover/Choke      -2.75  Forgetting lines, restarting, completely losing thread

FOUL SKILLS (negative — rare, only for rule violations):
OOB  Out of Bounds       -2.25  Structural round violation
TECH Technical Foul      -3.00  Props, refusing to battle, mic drop violation
PC   Defensive Foul      -3.00  Crossing an explicitly off-limits topic agreed before battle
GTD  Goaltending         -3.50  Repeatedly talking through opponent in a way that hurts the round
BV   Boundary Violation  -4.00  Doxxing, real threats with specific details, explicit rule violations
"""

SYSTEM_PROMPT = f"""You are an expert FCPBRL battle rap judge analyzing a single battler's round.

{BATTLE_RAP_CONTEXT}

SCORING TAXONOMY:
{SKILL_TABLE}

RULES:
- Violence, threats, religion, and offensive language are NORMAL battle rap — do NOT penalize them unless they are purely gratuitous with zero craft
- [inaudible] = transcription gap, NEVER counts as TVL
- Award positive skills generously when the craft is genuinely there
- Only award FCS/SD for truly exceptional moments (1-2 per round max)
- DD requires near-verbatim repetition of whole lines, not just similar themes

Respond ONLY with valid JSON in this exact format:
{{
  "total_score": <float>,
  "detections": [
    {{
      "skill_id": "<ID>",
      "skill_name": "<name>",
      "points": <float>,
      "direction": "positive" | "negative",
      "lines": ["<exact quoted text>"],
      "reason": "<1 sentence explanation>"
    }}
  ],
  "summary": "<2-3 sentence overall assessment>"
}}
"""


def query_ollama(text: str, model: str = DEFAULT_MODEL) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this battle rap round:\n\n{text}"}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 4096,
        }
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result
    except urllib.error.URLError as e:
        print(f"Error connecting to Ollama at {OLLAMA_URL}: {e}", file=sys.stderr)
        print("Is Ollama running? Try: ollama serve", file=sys.stderr)
        sys.exit(1)


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from ```json ... ``` block
    m = re.search(r'```(?:json)?\s*([\s\S]+?)```', text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding first { ... } block
    m = re.search(r'\{[\s\S]+\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"error": "Could not parse JSON", "raw": text}


def score_with_llm(
    text: str,
    model: str = DEFAULT_MODEL,
    round_number: int = None,
    battler: str = None,
) -> dict:
    print(f"Querying {model}...", file=sys.stderr)
    result = query_ollama(text, model)
    raw_content = result.get("message", {}).get("content", "")
    parsed = extract_json(raw_content)

    # Attach metadata
    parsed["model"] = model
    if round_number is not None:
        parsed["round_number"] = round_number
    if battler:
        parsed["battler"] = battler

    return parsed


def main():
    ap = argparse.ArgumentParser(description="Score a battle rap round with local LLM")
    ap.add_argument("text", nargs="?", help="Lyrics text (or use --file)")
    ap.add_argument("--file", "-f", help="Path to lyrics file")
    ap.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    ap.add_argument("--round", "-r", type=int, default=None, help="Round number")
    ap.add_argument("--battler", "-b", default=None, help="Battler name")
    ap.add_argument("--list-models", action="store_true", help="List available Ollama models and exit")
    args = ap.parse_args()

    if args.list_models:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            for m in data.get("models", []):
                size_gb = m["size"] / 1024**3
                print(f"  {m['name']:40s}  {size_gb:.1f} GB")
        return

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    result = score_with_llm(text, model=args.model, round_number=args.round, battler=args.battler)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
