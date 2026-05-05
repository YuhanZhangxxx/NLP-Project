#!/usr/bin/env python
"""
End-to-end match scorer that reads transcripts from Neon DB,
runs match_scorer.score_match(), and optionally saves the verdict back.

Determines rapper A (goes first) by who submitted Round 1 first
(earliest `created` timestamp).

Usage:
    python skill_detection/score_from_db.py --livestream-id <uuid>
    python skill_detection/score_from_db.py --livestream-id <uuid> --save
    python skill_detection/score_from_db.py --list    # show livestreams with transcripts

Environment:
    DATABASE_URL in .env (loaded automatically)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

# Load .env
for candidate in [_here / ".env", _here.parent / ".env"]:
    if candidate.exists():
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
        break

try:
    import psycopg2
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary", file=sys.stderr)
    sys.exit(1)

from match_scorer import score_match, SCHEMA_VERSION

# Neon DB connection string (from .env or environment)
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set. Add it to .env or export it.", file=sys.stderr)
    sys.exit(1)


from contextlib import contextmanager

@contextmanager
def get_connection(autocommit=False):
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = autocommit
    try:
        yield conn
    finally:
        conn.close()


def list_livestreams_with_transcripts():
    """Show livestreams that have transcripts, with row counts."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT t.livestream_id, l.title, COUNT(*) as rounds,
                   MIN(t.created) as first, MAX(t.created) as last
            FROM transcripts t
            JOIN livestreams l ON l.id = t.livestream_id
            GROUP BY t.livestream_id, l.title
            ORDER BY MAX(t.created) DESC;
        """)
        rows = cur.fetchall()
    if not rows:
        print("No livestreams with transcripts found.")
        return
    print(f"{'livestream_id':<40} {'title':<25} {'rows':>5}  {'first':>20}  {'last':>20}")
    print("-" * 115)
    for r in rows:
        print(f"{str(r[0]):<40} {str(r[1]):<25} {r[2]:>5}  {str(r[3])[:19]:>20}  {str(r[4])[:19]:>20}")


def fetch_transcripts(livestream_id: str) -> list[dict]:
    """Fetch one latest transcript per round/rapper for a livestream."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT ON (round, rapper)
                   round, rapper, text, created
            FROM transcripts
            WHERE livestream_id = %s
            ORDER BY round, rapper, created DESC;
        """, (livestream_id,))
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return sorted(rows, key=lambda r: (r["round"], r["created"]))


def resolve_rapper_name(bruf_id: str) -> str:
    """Look up display name from brufids/accounts. Falls back to bruf_id."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT COALESCE(
                NULLIF(a.stage_name, ''),
                NULLIF(a.username, ''),
                NULLIF(b.owner, ''),
                b.id
            )
            FROM brufids b
            LEFT JOIN accounts a ON a.id::text = b.owner
            WHERE b.id = %s;
        """, (bruf_id,))
        row = cur.fetchone()
        return row[0] if row and row[0] else bruf_id


def _get_first_rapper(livestream_id: str) -> str | None:
    """Check livestreams.first_rapper for explicit ordering."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT first_rapper FROM livestreams WHERE id = %s;", (livestream_id,))
        row = cur.fetchone()
        return row[0] if row and row[0] else None


def build_payload(livestream_id: str, rows: list[dict]) -> dict:
    """
    Build the scorer JSON payload from DB rows.
    Rapper A priority: livestreams.first_rapper (explicit) → Round 1
    earliest `created` timestamp (fallback).
    """
    if not rows:
        raise ValueError(f"No transcripts found for livestream {livestream_id}")

    round1 = [r for r in rows if r["round"] == 1]
    if len(round1) < 2:
        raise ValueError(f"Round 1 incomplete: only {len(round1)} row(s)")

    # Priority 1: explicit first_rapper field
    explicit_first = _get_first_rapper(livestream_id)
    if explicit_first and explicit_first in {r["rapper"] for r in round1}:
        rapper_a_id = explicit_first
        rapper_b_id = next(r["rapper"] for r in round1 if r["rapper"] != explicit_first)
        print("  Order source: livestreams.first_rapper (explicit)", file=sys.stderr)
    else:
        # Priority 2: fallback to created timestamp
        rapper_a_id = round1[0]["rapper"]
        rapper_b_id = round1[1]["rapper"]
        print("  Order source: Round 1 INSERT order (fallback)", file=sys.stderr)

    # Group by round
    rounds_grouped: dict[int, dict[str, str]] = {}
    for r in rows:
        rn = r["round"]
        rounds_grouped.setdefault(rn, {})
        if r["rapper"] == rapper_a_id:
            rounds_grouped[rn]["a"] = r["text"]
        else:
            rounds_grouped[rn]["b"] = r["text"]

    # Build ordered rounds list
    round_nums = sorted(rounds_grouped.keys())
    rounds_list = []
    for rn in round_nums:
        pair = rounds_grouped[rn]
        if "a" not in pair or "b" not in pair:
            raise ValueError(f"Round {rn} incomplete: missing {'A' if 'a' not in pair else 'B'}")
        rounds_list.append(pair)

    payload = {
        "match_id": livestream_id,
        "battler_a": resolve_rapper_name(rapper_a_id),
        "battler_b": resolve_rapper_name(rapper_b_id),
        "rounds": rounds_list,
    }
    return payload, rapper_a_id, rapper_b_id


def _update_account_record(cur, bruf_id: str, win: bool, round_score: int) -> None:
    """Increment wins or losses and add bp for the account owning bruf_id."""
    if not bruf_id:
        return
    if win:
        cur.execute("""
            UPDATE accounts
            SET wins = COALESCE(wins, 0) + 1,
                bp   = COALESCE(bp,   0) + %s
            WHERE id::text = (SELECT owner FROM brufids WHERE id = %s)
               OR username  = (SELECT owner FROM brufids WHERE id = %s);
        """, (round_score, bruf_id, bruf_id))
    else:
        cur.execute("""
            UPDATE accounts
            SET losses = COALESCE(losses, 0) + 1,
                bp     = COALESCE(bp,     0) + %s
            WHERE id::text = (SELECT owner FROM brufids WHERE id = %s)
               OR username  = (SELECT owner FROM brufids WHERE id = %s);
        """, (round_score, bruf_id, bruf_id))
    rows = cur.rowcount
    print(f"  account update ({'win' if win else 'loss'}) bruf={bruf_id} bp+={round_score} rows={rows}", file=sys.stderr)


def save_verdict(livestream_id: str, result: dict, rapper_a_id: str | None = None, rapper_b_id: str | None = None):
    """Write match verdict back to the matches table and update account W/L/BP."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT verdict
            FROM matches
            WHERE livestream_id = %s
            FOR UPDATE;
        """, (livestream_id,))
        match_rows = cur.fetchall()
        if len(match_rows) == 0:
            conn.rollback()
            print(f"  ERROR: no match row found for livestream_id={livestream_id}. "
                  f"Create a match with this livestream_id first.", file=sys.stderr)
            return
        if len(match_rows) > 1:
            conn.rollback()
            print(f"  WARNING: {len(match_rows)} match rows found for livestream_id={livestream_id}; "
                  f"expected 1. Skipping save to avoid duplicate account updates.", file=sys.stderr)
            return
        if match_rows[0][0]:
            conn.rollback()
            print("  Verdict already exists; skipping save and account record updates.", file=sys.stderr)
            return

        # Compute wins from rounds data
        a_wins = 0.0
        b_wins = 0.0
        for r in result.get("rounds", []):
            w = r.get("winner", "")
            if w == "A":
                a_wins += 1
            elif w == "B":
                b_wins += 1
            else:
                a_wins += 0.5
                b_wins += 0.5

        # score1/score2 in matches table are INTEGERs — store round wins.
        # Winner gets ceil, loser gets floor so total = number of rounds.
        # e.g. 2.5-0.5 → 3-0 (total=3 rounds, winner clear).
        if a_wins >= b_wins:
            score1 = int(math.ceil(a_wins))
            score2 = int(math.floor(b_wins))
        else:
            score1 = int(math.floor(a_wins))
            score2 = int(math.ceil(b_wins))

        cur.execute("""
            UPDATE matches
            SET score1 = %s, score2 = %s, verdict = %s
            WHERE livestream_id = %s;
        """, (score1, score2, json.dumps(result, ensure_ascii=True), livestream_id))

        updated = cur.rowcount

        # Update account W/L/BP
        if updated == 1 and rapper_a_id and rapper_b_id:
            a_won = a_wins >= b_wins
            _update_account_record(cur, rapper_a_id, win=a_won,     round_score=score1 if a_won else score2)
            _update_account_record(cur, rapper_b_id, win=not a_won, round_score=score2 if a_won else score1)
        conn.commit()

    if updated == 1:
        print(f"  Verdict saved: score1={score1}, score2={score2}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Score a match from DB transcripts")
    ap.add_argument("--livestream-id", "-l", help="UUID of the livestream")
    ap.add_argument("--save", action="store_true", help="Save verdict back to matches table")
    ap.add_argument("--list", action="store_true", help="List livestreams with transcripts")
    ap.add_argument("--tie-threshold", type=float, default=0.75)
    args = ap.parse_args()

    if args.list:
        list_livestreams_with_transcripts()
        return

    if not args.livestream_id:
        print("ERROR: --livestream-id required (or use --list to browse)", file=sys.stderr)
        sys.exit(1)

    # Step 1: Fetch transcripts
    print(f"Fetching transcripts for {args.livestream_id}...", file=sys.stderr)
    rows = fetch_transcripts(args.livestream_id)
    print(f"  Found {len(rows)} rows", file=sys.stderr)

    # Step 2: Build payload
    try:
        payload, rapper_a_id, rapper_b_id = build_payload(args.livestream_id, rows)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    payload["tie_threshold"] = args.tie_threshold
    print(f"  Battler A: {payload['battler_a']} (bruf={rapper_a_id})", file=sys.stderr)
    print(f"  Battler B: {payload['battler_b']} (bruf={rapper_b_id})", file=sys.stderr)
    print(f"  Rounds: {len(payload['rounds'])}", file=sys.stderr)

    # Step 3: Score
    print("Scoring...", file=sys.stderr)
    try:
        result = score_match(payload)
    except Exception as e:
        print(f"ERROR: scorer failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 4: Output
    print(json.dumps(result, indent=2, ensure_ascii=True))

    # Step 5: Save (optional)
    if args.save and result.get("status") == "ok":
        save_verdict(args.livestream_id, result, rapper_a_id, rapper_b_id)

    sys.exit(0 if result.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()
