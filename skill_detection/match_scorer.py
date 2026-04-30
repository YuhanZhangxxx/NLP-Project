#!/usr/bin/env python
"""
Multi-round battle rap match scorer.

Reads stdin JSON, scores each performance via hybrid_score(), determines
per-round winners + match verdict, writes stdout JSON.

Input shape:
    {
      "match_id": "<optional, echoed back>",
      "battler_a": "Jack Boy Main",
      "battler_b": "A Ward",
      "tie_threshold": 0.75,           // optional, default 0.75
      "model": "gpt-4o-mini",          // optional, defaults to scorer default
      "rounds": [
        {"a": "A1 transcript", "b": "B1 transcript"},
        {"a": "A2 transcript", "b": "B2 transcript"},
        {"a": "A3 transcript", "b": "B3 transcript"}
      ]
    }

Rebuttal (opponent_bars) chain:
    A1 -> None         (opener, no rebut possible)
    B1 -> A1
    A2 -> B1           (carries from previous pair)
    B2 -> A2
    A3 -> B2
    B3 -> A3

Output shape (schema_version 1):
    {
      "schema_version": 1,
      "status": "ok",
      "match_id": "<echoed>",
      "battler_a": "...",
      "battler_b": "...",
      "model": "gpt-4o-mini",
      "tie_threshold": 0.75,
      "rounds": [
        {
          "round": 1, "winner": "A"|"B"|"tie",
          "score_a": 4.1, "score_b": 2.5,
          "detail_a": {...}, "detail_b": {...}
        }, ...
      ],
      "match_winner": "A"|"B"|"tie",
      "summary": "A wins 2-1",
      "error": null
    }

On failure:
    {
      "schema_version": 1, "status": "error", "error": "<message>",
      "match_id": "<echoed if available>", "rounds": [],
      "match_winner": null, "summary": null
    }

Usage:
    cat payload.json | python skill_detection/match_scorer.py
    python skill_detection/match_scorer.py --file payload.json     # debug
"""
import argparse
import json
import sys
from pathlib import Path

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from hybrid_scorer_gpt import DEFAULT_MODEL, LLMError, hybrid_score

SCHEMA_VERSION = 1
DEFAULT_TIE_THRESHOLD = 0.75


def _round_winner(score_a: float, score_b: float, tie_threshold: float) -> str:
    diff = score_a - score_b
    # <= so that exactly equal scores tie even when tie_threshold=0.
    if abs(diff) <= tie_threshold:
        return "tie"
    return "A" if diff > 0 else "B"


def _verdict_summary(a_wins: float, b_wins: float, battler_a: str, battler_b: str) -> tuple[str, str]:
    """Return (match_winner, human-readable summary)."""
    if a_wins > b_wins:
        return "A", f"{battler_a} wins {a_wins:g}-{b_wins:g}"
    if b_wins > a_wins:
        return "B", f"{battler_b} wins {b_wins:g}-{a_wins:g}"
    return "tie", f"Tie {a_wins:g}-{b_wins:g}"


def score_match(payload: dict) -> dict:
    """
    Score an N-round match. Pure function: no I/O.

    This function RAISES on bad input (ValueError, LLMError, etc.).
    The schema_version=1 error envelope is only guaranteed on the CLI
    path (main()). Library callers must catch exceptions themselves.
    """
    rounds_input = payload.get("rounds") or []
    if not rounds_input:
        raise ValueError("payload.rounds must be a non-empty list")

    battler_a = payload.get("battler_a") or "A"
    battler_b = payload.get("battler_b") or "B"
    tie_threshold = float(payload.get("tie_threshold") or DEFAULT_TIE_THRESHOLD)
    model = payload.get("model") or DEFAULT_MODEL

    round_results: list[dict] = []
    a_wins = 0.0
    b_wins = 0.0
    prev_b_text: str | None = None  # what A rebuts in rounds 2+

    for i, pair in enumerate(rounds_input):
        round_num = i + 1
        a_text = pair.get("a") or ""
        b_text = pair.get("b") or ""
        if not a_text.strip() or not b_text.strip():
            raise ValueError(f"round {round_num}: both 'a' and 'b' transcripts are required")

        # A goes first; rebuts the IMMEDIATELY PRECEDING B (None for round 1).
        detail_a = hybrid_score(
            a_text, model=model, round_number=round_num,
            battler=battler_a, opponent_bars=prev_b_text,
        )
        # B follows; rebuts A's current round.
        detail_b = hybrid_score(
            b_text, model=model, round_number=round_num,
            battler=battler_b, opponent_bars=a_text,
        )

        score_a = float(detail_a.get("total_score", 0.0))
        score_b = float(detail_b.get("total_score", 0.0))
        winner = _round_winner(score_a, score_b, tie_threshold)
        if winner == "A":
            a_wins += 1
        elif winner == "B":
            b_wins += 1
        else:
            a_wins += 0.5
            b_wins += 0.5

        round_results.append({
            "round": round_num,
            "winner": winner,
            "score_a": round(score_a, 2),
            "score_b": round(score_b, 2),
            "detail_a": detail_a,
            "detail_b": detail_b,
        })
        prev_b_text = b_text

    match_winner, summary = _verdict_summary(a_wins, b_wins, battler_a, battler_b)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "match_id": payload.get("match_id"),
        "battler_a": battler_a,
        "battler_b": battler_b,
        "model": model,
        "tie_threshold": tie_threshold,
        "rounds": round_results,
        "match_winner": match_winner,
        "summary": summary,
        "error": None,
    }


def _error_envelope(msg: str, match_id=None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "error": msg,
        "match_id": match_id,
        "rounds": [],
        "match_winner": None,
        "summary": None,
    }


def main():
    ap = argparse.ArgumentParser(description="Multi-round match scorer (stdin JSON in, stdout JSON out)")
    ap.add_argument("--file", "-f", help="Read payload from JSON file (debug); default is stdin")
    args = ap.parse_args()

    raw = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    match_id = None
    try:
        payload = json.loads(raw)
        match_id = payload.get("match_id") if isinstance(payload, dict) else None
        result = score_match(payload)
    except json.JSONDecodeError as e:
        result = _error_envelope(f"invalid JSON input: {e}", match_id)
    except (ValueError, KeyError) as e:
        result = _error_envelope(f"invalid payload: {e}", match_id)
    except LLMError as e:
        result = _error_envelope(f"LLM call failed: {e}", match_id)
    except Exception as e:
        result = _error_envelope(f"unexpected error: {type(e).__name__}: {e}", match_id)

    print(json.dumps(result, indent=2, ensure_ascii=True))
    sys.exit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
