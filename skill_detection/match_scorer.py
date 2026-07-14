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
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from hybrid_scorer_gpt import DEFAULT_MODEL, LLMError, get_usage, hybrid_score, reset_usage

SCHEMA_VERSION = 1
DEFAULT_TIE_THRESHOLD = 0.75
ProgressCallback = Callable[[dict], None]

# Estimated OpenAI judge prices in USD per 1M tokens:
# (input_uncached, input_cached, output). Keep this as a default estimate only;
# override with env OPENAI_PRICE_<MODEL>=input,cached,output if prices change.
MODEL_PRICE_USD_PER_1M: dict[str, tuple[float, float, float]] = {
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.6-luna": (0.50, 0.05, 3.00),
    "gpt-5.6-terra": (1.25, 0.125, 7.50),
    "gpt-5.6-sol": (2.50, 0.25, 15.00),
}


def _price_env_key(model: str) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").upper()
    return f"OPENAI_PRICE_{key}"


def _price_for_model(model: str) -> tuple[tuple[float, float, float] | None, str]:
    env_key = _price_env_key(model)
    raw = os.environ.get(env_key)
    if raw:
        try:
            parts = [float(x.strip()) for x in raw.split(",")]
            if len(parts) == 3:
                return (parts[0], parts[1], parts[2]), f"env:{env_key}"
        except ValueError:
            pass
    if model in MODEL_PRICE_USD_PER_1M:
        return MODEL_PRICE_USD_PER_1M[model], "default_estimate"
    return None, "unknown_model"


def _estimate_cost(model: str, usage: dict) -> dict:
    price, source = _price_for_model(model)
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int(usage.get("cached_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    uncached = max(prompt - cached, 0)
    if price is None:
        return {
            "currency": "USD",
            "estimated": True,
            "available": False,
            "model": model,
            "price_source": source,
            "total_usd": None,
        }
    input_rate, cached_rate, output_rate = price
    input_usd = uncached / 1_000_000 * input_rate
    cached_usd = cached / 1_000_000 * cached_rate
    output_usd = completion / 1_000_000 * output_rate
    total = input_usd + cached_usd + output_usd
    return {
        "currency": "USD",
        "estimated": True,
        "available": True,
        "model": model,
        "price_source": source,
        "rates_per_1m_tokens": {
            "input_uncached": input_rate,
            "input_cached": cached_rate,
            "output": output_rate,
        },
        "billable_tokens": {
            "input_uncached": uncached,
            "input_cached": cached,
            "output": completion,
        },
        "input_uncached_usd": round(input_usd, 6),
        "input_cached_usd": round(cached_usd, 6),
        "output_usd": round(output_usd, 6),
        "total_usd": round(total, 6),
    }


def _text_stats(text: str) -> dict:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return {
        "chars": len(text),
        "non_whitespace_chars": len(re.sub(r"\s+", "", text)),
        "words": len(re.findall(r"\b[\w']+\b", text)),
        "lines": len(lines),
    }


def _input_stats(rounds_input: list[dict], battler_a: str, battler_b: str) -> dict:
    by_round = []
    totals = {
        "chars": 0,
        "non_whitespace_chars": 0,
        "words": 0,
        "lines": 0,
        "rounds": len(rounds_input),
        "performances": len(rounds_input) * 2,
    }
    for idx, pair in enumerate(rounds_input, start=1):
        a = _text_stats(str(pair.get("a") or ""))
        b = _text_stats(str(pair.get("b") or ""))
        by_round.append({"round": idx, "a": a, "b": b})
        for stats in (a, b):
            for key in ("chars", "non_whitespace_chars", "words", "lines"):
                totals[key] += stats[key]
    return {
        "battler_a": battler_a,
        "battler_b": battler_b,
        "totals": totals,
        "rounds": by_round,
    }


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


def score_match(payload: dict, progress_callback: ProgressCallback | None = None) -> dict:
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
    model = str(payload.get("model") or os.environ.get("OPENAI_JUDGE_MODEL") or DEFAULT_MODEL)
    total_performances = len(rounds_input) * 2
    reset_usage()
    started = time.perf_counter()
    input_stats = _input_stats(rounds_input, battler_a, battler_b)

    round_results: list[dict] = []
    a_wins = 0.0
    b_wins = 0.0
    prev_b_text: str | None = None  # what A rebuts in rounds 2+
    completed_performances = 0

    def emit_progress(step: int, message: str) -> None:
        if not progress_callback:
            return
        percent = 15 + int((step / max(total_performances, 1)) * 70)
        progress_callback({
            "phase": "scoring",
            "step": step,
            "total": total_performances,
            "percent": percent,
            "message": message,
        })

    emit_progress(0, "Preparing AI judge")

    for i, pair in enumerate(rounds_input):
        round_num = i + 1
        a_text = pair.get("a") or ""
        b_text = pair.get("b") or ""
        if not a_text.strip() or not b_text.strip():
            raise ValueError(f"round {round_num}: both 'a' and 'b' transcripts are required")

        # A goes first; rebuts the IMMEDIATELY PRECEDING B (None for round 1).
        emit_progress(completed_performances, f"Scoring round {round_num}: {battler_a}")
        detail_a = hybrid_score(
            a_text, model=model, round_number=round_num,
            battler=battler_a, opponent_bars=prev_b_text,
        )
        completed_performances += 1
        emit_progress(completed_performances, f"Finished round {round_num}: {battler_a}")

        # B follows; rebuts A's current round.
        emit_progress(completed_performances, f"Scoring round {round_num}: {battler_b}")
        detail_b = hybrid_score(
            b_text, model=model, round_number=round_num,
            battler=battler_b, opponent_bars=a_text,
        )
        completed_performances += 1
        emit_progress(completed_performances, f"Finished round {round_num}: {battler_b}")

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
    elapsed_seconds = round(time.perf_counter() - started, 3)
    usage = get_usage()
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
        "usage": usage,
        "elapsed_seconds": elapsed_seconds,
        "input_stats": input_stats,
        "cost_estimate": _estimate_cost(model, usage),
        "error": None,
    }


def _error_envelope(msg: str, match_id=None) -> dict:
    usage = get_usage()
    model = os.environ.get("OPENAI_JUDGE_MODEL") or DEFAULT_MODEL
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "error": msg,
        "match_id": match_id,
        "rounds": [],
        "match_winner": None,
        "summary": None,
        "usage": usage,
        "cost_estimate": _estimate_cost(model, usage),
    }


def main():
    ap = argparse.ArgumentParser(description="Multi-round match scorer (stdin JSON in, stdout JSON out)")
    ap.add_argument("--file", "-f", help="Read payload from JSON file (debug); default is stdin")
    ap.add_argument("--model", "-m", help="Override the payload/environment judge model")
    args = ap.parse_args()

    raw = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    match_id = None
    try:
        payload = json.loads(raw)
        if args.model:
            payload["model"] = args.model
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
