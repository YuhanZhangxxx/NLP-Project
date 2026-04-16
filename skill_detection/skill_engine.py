"""
FCPBRL Skill Detection Engine
==============================
Multi-label battle rap skill and action detector.
Maps text to the FCPBRL basketball-scoring taxonomy.

Public API
----------
    detect_skills(text, threshold=0.4)  -> SkillScanResult
    scan_round(text)                    -> dict  (JSON-serializable)

Backward Compat
---------------
    detect_rap_techniques(texts)        -> scipy.sparse.csr_matrix (n, 4)
    (drop-in replacement for the same function in rap_techniques.py)

Skill Taxonomy (35 skills)
--------------------------
    Highlights (23): FCS, SD, HCS, ALY, AND1, FB, 3PT, ES, STL, CO, HS,
                     OGR, 4QP, SPM, PM, PNR, ISO, BKW, MR, LU, REB, NLP, FL
    Mistakes   ( 7): OFT, CAR, DRG, FA, TVL, DD, CHK
    Fouls      ( 5): OOB, TECH, PC, GTD, BV

Skills requiring live observation (crowd/timing/physical) are intentionally
excluded — they cannot be reliably inferred from transcript text alone.
"""

from __future__ import annotations

import re
import sys
import json
import dataclasses
from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy import sparse

# ---------------------------------------------------------------------------
# Resolve skill_detection/ on sys.path so we can import rap_techniques
# ---------------------------------------------------------------------------
_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from rap_techniques import (
    detect_full_court_shot,
    detect_slam_dunk,
    detect_half_court_shot,
    detect_alley_oop,
    normalize_text,
    NLTK_AVAILABLE,
)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SkillDetection:
    """A single detected skill in a piece of text."""
    skill_id:   str           # Short code, e.g. "SD", "STL"
    skill_name: str           # Full name, e.g. "Slam Dunk"
    confidence: float         # 0.0–1.0 raw detector score
    points:     float         # Official FCPBRL point value (±)
    direction:  str           # "positive" | "negative"
    category:   str           # "highlight" | "mistake" | "foul"
    evidence:   List[str]     # Matched text snippets (up to 3)


@dataclass
class SkillScanResult:
    """Complete multi-label scan result for a piece of text."""
    detections:  List[SkillDetection]
    total_score: float                      # sum of points for all detected skills
    highlights:  List[SkillDetection]       # direction == "positive"
    mistakes:    List[SkillDetection]       # direction == "negative"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def __str__(self) -> str:
        lines = [f"Total Score: {self.total_score:+.2f}  "
                 f"({len(self.highlights)} highlights, {len(self.mistakes)} deductions)"]
        if self.highlights:
            lines.append("\nHIGHLIGHTS:")
            for d in self.highlights:
                lines.append(f"  [{d.skill_id}] {d.skill_name:25s} +{d.points:.2f}pts  "
                              f"conf={d.confidence:.0%}")
                for ev in d.evidence[:1]:
                    lines.append(f"      -> {ev[:100]}")
        if self.mistakes:
            lines.append("\nDEDUCTIONS:")
            for d in self.mistakes:
                lines.append(f"  [{d.skill_id}] {d.skill_name:25s} {d.points:.2f}pts  "
                              f"conf={d.confidence:.0%}")
                for ev in d.evidence[:1]:
                    lines.append(f"      -> {ev[:100]}")
        return "\n".join(lines)


# ============================================================================
# Skill Registry
# ============================================================================
# Tuple layout: (key, id, name, points, category, threshold)
# First 4 entries MUST stay in this order (legacy sparse-matrix compat):
#   col0 = full_court_shot, col1 = slam_dunk,
#   col2 = half_court_shot, col3 = alley_oop

SKILL_REGISTRY = [
    # ---- Highlights (positive) — first 4 FIXED for legacy sparse-matrix compat ----
    ("full_court_shot",     "FCS",  "Full-Court Shot",       +5.00, "highlight", 0.45),
    ("slam_dunk",           "SD",   "Slam Dunk",             +4.25, "highlight", 0.35),
    ("half_court_shot",     "HCS",  "Half-Court Shot",       +3.75, "highlight", 0.35),
    ("alley_oop",           "ALY",  "Alley-Oop",             +3.50, "highlight", 0.35),
    # remaining highlights, sorted by points desc
    ("and_1",               "AND1", "And-1",                 +3.25, "highlight", 0.40),
    ("fast_break",          "FB",   "Fast Break",            +3.00, "highlight", 0.35),
    ("three_pointer",       "3PT",  "3-Pointer",             +2.85, "highlight", 0.35),
    ("euro_step",           "ES",   "Euro Step",             +2.75, "highlight", 0.35),
    ("steal",               "STL",  "Steal / Rebuttal",      +2.50, "highlight", 0.40),
    ("crossover",           "CO",   "Crossover",             +2.25, "highlight", 0.35),
    ("hook_shot",           "HS",   "Hook Shot",             +2.00, "highlight", 0.35),
    ("out_the_gate",        "OGR",  "Out-the-Gate Run",      +2.00, "highlight", 0.40),
    ("fourth_quarter",      "4QP",  "Fourth-Quarter Push",   +2.00, "highlight", 0.40),
    ("spin_move",           "SPM",  "Spin Move",             +1.90, "highlight", 0.40),
    ("post_move",           "PM",   "Post Move",             +1.75, "highlight", 0.40),
    ("pick_and_roll",       "PNR",  "Pick & Roll",           +1.65, "highlight", 0.40),
    ("isolation",           "ISO",  "Isolation",             +1.50, "highlight", 0.40),
    ("breakaway",           "BKW",  "Breakaway",             +1.40, "highlight", 0.40),
    ("midrange",            "MR",   "Mid-Range",             +1.35, "highlight", 0.40),
    ("layup",               "LU",   "Layup",                 +1.25, "highlight", 0.40),
    ("rebound",             "REB",  "Rebound",               +1.15, "highlight", 0.40),
    ("no_look_pass",        "NLP",  "No-Look Pass",          +0.90, "highlight", 0.40),
    ("floater",             "FL",   "Floater",               +0.75, "highlight", 0.35),
    # ---- Mistakes (negative) ----
    ("offensive_foul",      "OFT",  "Offensive Foul",        -0.50, "mistake",   0.55),
    ("carry",               "CAR",  "Carry",                 -1.00, "mistake",   0.55),
    ("backcourt_violation", "DRG",  "Backcourt Violation",   -1.25, "mistake",   0.50),
    ("forced_angle",        "FA",   "Charge / Forced Angle", -1.35, "mistake",   0.50),
    ("travel",              "TVL",  "Travel / Stumble",      -1.50, "mistake",   0.50),
    ("double_dribble",      "DD",   "Double Dribble",        -2.00, "mistake",   0.50),
    ("choke",               "CHK",  "Turnover / Choke",      -2.75, "mistake",   0.50),
    # ---- Fouls (negative) ----
    ("out_of_bounds",       "OOB",  "Out of Bounds",         -2.25, "foul",      0.60),
    ("technical_foul",      "TECH", "Technical Foul",        -3.00, "foul",      0.65),
    ("physical_contact",    "PC",   "Defensive Foul",        -3.00, "foul",      0.70),
    ("goaltending",         "GTD",  "Goaltending",           -3.50, "foul",      0.60),
    ("boundary_violation",  "BV",   "Boundary Violation",    -4.00, "foul",      0.70),
]

# Lookup dict: key -> (id, name, points, category, threshold)
_SKILL_META = {key: (sid, name, pts, cat, thr)
               for key, sid, name, pts, cat, thr in SKILL_REGISTRY}


# ============================================================================
# Helpers
# ============================================================================

def _norm(text: str) -> str:
    """Normalise for matching (lower, collapse whitespace, smart quotes)."""
    s = text.lower()
    s = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", s)
    s = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def _lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _snip(text: str, m: re.Match, pad: int = 35) -> str:
    """Extract a context snippet around a regex match (newlines flattened)."""
    start = max(0, m.start() - pad)
    end   = min(len(text), m.end() + pad)
    snippet = text[start:end].strip()
    return re.sub(r'\s*[\r\n]+\s*', ' | ', snippet)


def _score_patterns(s: str, patterns: List[str],
                    per_hit: float, cap: float) -> Tuple[float, List[re.Match]]:
    """Count pattern hits, return (capped score, list of matches)."""
    hits, matches = 0, []
    for p in patterns:
        m = re.search(p, s, re.IGNORECASE)
        if m:
            hits += 1
            matches.append(m)
    return min(hits * per_hit, cap), matches


def _score_keywords(s: str, keywords, per_hit: float, cap: float) -> float:
    count = sum(1 for kw in keywords if kw in s)
    return min(count * per_hit, cap)


# ============================================================================
# Positive Skill Detectors  (4 legacy delegated below; 19 new here)
# ============================================================================

def _detect_steal(text: str) -> Tuple[float, List[str]]:
    """Steal / Rebuttal: turns opponent's own words/setup against them."""
    s = _norm(text)
    score, evidence = 0.0, []

    echo_patterns = [
        r"\byou\s+said\b",
        r"\byou\s+came\s+(?:in\s+here|out|up)\s+(?:talking|saying|claiming|bragging)\b",
        r"\bsame\s+(?:\w+\s+){1,4}you\s+(?:used|said|tried|came\s+with)\b",
        r"\byou(?:r)?\s+own\s+(?:words?|bars?|angle|setup|lines?)\b",
        r"\b(?:using|flip(?:ped)?|turned)\s+(?:your|his|her)\s+own\s+(?:bar|line|setup|angle|words?)\b",
        r"\bthat\s+(?:same|very)\s+(?:bar|line|angle|setup)\s+you\b",
        r"\byou\s+handed\s+(?:me|us)\s+(?:the\s+)?(?:ammo|material|gun|weapon)\b",
    ]
    sc, ms = _score_patterns(s, echo_patterns, 0.25, 0.65)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m))

    rebuttal_patterns = [
        r"\b(?:let\s+me\s+address|wait\s+(?:wait|hold|no)|hold\s+on|clap\s+back|rebuttal)\b",
        r"\b(?:you\s+(?:lied|made\s+that\s+up|forgot|didn'?t\s+know))\b",
        r"\bflip\s+(?:the\s+)?script\b",
        r"\bthat'?s?\s+(?:ironic|funny)\s+because\b",
        r"\becho\b",
    ]
    sc2, ms2 = _score_patterns(s, rebuttal_patterns, 0.15, 0.35)
    score += sc2
    for m in ms2[:1]:
        if len(evidence) < 3:
            evidence.append(_snip(text, m))

    return min(score, 1.0), evidence[:3]


def _detect_three_pointer(text: str) -> Tuple[float, List[str]]:
    """3-Pointer: calculated long-range attack using opponent's history/career."""
    s = _norm(text)
    score, evidence = 0.0, []

    research_patterns = [
        r"\b(?:back\s+in|way\s+back|remember\s+when|back\s+when)\b",
        r"\byour\s+(?:history|past|record|career|track\s+record|reputation|background)\b",
        r"\b(?:public\s+record|on\s+(?:record|camera|tape|video|film))\b",
        r"\b(?:looked\s+(?:you|it)\s+up|researched|investigated|dug\s+(?:up|into))\b",
        r"\bbefore\s+(?:this|tonight|today|the\s+battle)\b",
        r"\byears?\s+ago\b",
    ]
    sc, ms = _score_patterns(s, research_patterns, 0.18, 0.55)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m))

    depth_patterns = [
        r"\b(?:calculated|premeditated|planned|studied|prepared|researched)\b",
        r"\b(?:three\s+pointer|long\s+(?:range|shot)|deep\s+(?:ball|shot))\b",
        r"\b(?:from\s+(?:deep|distance|far|afar|way\s+out))\b",
    ]
    sc2, ms2 = _score_patterns(s, depth_patterns, 0.2, 0.4)
    score += sc2
    for m in ms2[:1]:
        if len(evidence) < 3:
            evidence.append(_snip(text, m))

    return min(score, 1.0), evidence[:3]


def _detect_fast_break(text: str) -> Tuple[float, List[str]]:
    """Fast Break: multiple consecutive punches in quick succession."""
    s = _norm(text)
    lines = _lines(text)
    score, evidence = 0.0, []

    combo_patterns = [
        r"\bcombo\b",
        r"\bone,?\s+two,?\s+(?:three|punch)\b",
        r"\bpunch\s+after\s+punch\b|\bbar\s+after\s+bar\b|\bhit\s+after\s+hit\b",
        r"\b(?:run\s+of\s+(?:bars?|punches?)|opening\s+run|closing\s+(?:run|push))\b",
        r"\bkept\s+(?:coming|hitting|punching)\b",
        r"\b(?:wave\s+after\s+wave|volley|barrage)\b",
    ]
    sc, ms = _score_patterns(s, combo_patterns, 0.25, 0.55)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m))

    # Punch-density window: 2+ impact words in any 3-line window
    impact_re = re.compile(
        r'\b(?:dead|bodied|murdered|killed|done|over|finished|gone|erased|destroyed)\b'
    )
    if len(lines) >= 3:
        for i in range(len(lines) - 2):
            window = " ".join(lines[i:i+3])
            hits = len(impact_re.findall(_norm(window)))
            if hits >= 2:
                score += 0.25
                evidence.append(" | ".join(lines[i:i+3])[:120])
                break

    return min(score, 1.0), evidence[:3]


def _detect_euro_step(text: str) -> Tuple[float, List[str]]:
    """Euro Step: sets expectation then pivots mid-bar to deliver unexpected punch."""
    s = _norm(text)
    score, evidence = 0.0, []

    setup_patterns = [
        r"\b(?:everyone|they|people)\s+(?:thinks?|thought|expected|assumed)\b",
        r"\b(?:normally|usually|typically|traditionally)\b",
        r"\b(?:looks?\s+like|seems?\s+like|appears?\s+to\s+be)\b",
        r"\b(?:you'?d?\s+(?:think|expect|assume))\b",
        r"\b(?:they\s+(?:said|expected|predicted))\b",
    ]
    pivot_patterns = [
        r"\b(?:but\s+(?:actually|wait|no|nah|hold\s+on|then|here))\b",
        r"\b(?:plot\s+twist|surprise|wrong|nah|however)\b",
        r"\b(?:except|until|suddenly)\b",
    ]

    has_setup = any(re.search(p, s, re.IGNORECASE) for p in setup_patterns)
    has_pivot = any(re.search(p, s, re.IGNORECASE) for p in pivot_patterns)

    if has_setup and has_pivot:
        score += 0.50
        for p in setup_patterns:
            m = re.search(p, s, re.IGNORECASE)
            if m:
                evidence.append(_snip(text, m, pad=50))
                break
    elif has_pivot:
        score += 0.20

    misdirect_patterns = [
        r"\b(?:misdirect(?:ed)?|pivot(?:ed)?|redirect(?:ed)?|angle\s+shift)\b",
        r"\b(?:fake\s+out?|feint(?:ed)?|faked)\b",
        r"\b(?:didn'?t\s+(?:see\s+it|expect)\s+coming)\b",
    ]
    sc2, ms2 = _score_patterns(s, misdirect_patterns, 0.2, 0.35)
    score += sc2
    for m in ms2[:1]:
        if len(evidence) < 3:
            evidence.append(_snip(text, m))

    return min(score, 1.0), evidence[:3]


def _detect_hook_shot(text: str) -> Tuple[float, List[str]]:
    """Hook Shot: curved indirect approach — hits from an unexpected angle."""
    s = _norm(text)
    score, evidence = 0.0, []

    indirect_patterns = [
        r"\b(?:came\s+from\s+(?:the\s+)?(?:side|angle|left|right|back|blind\s+side))\b",
        r"\b(?:went\s+around|didn'?t\s+come\s+(?:straight|directly\s+at))\b",
        r"\b(?:curved|arc(?:ed)?|hook)\b",
        r"\b(?:indirect|roundabout|circumvent|bypass)\b",
    ]
    sc, ms = _score_patterns(s, indirect_patterns, 0.2, 0.50)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m))

    # Using someone/something else to reach the target
    indirect_attack = re.search(
        r"\b(?:through|via|by\s+way\s+of|using|bringing\s+up)\b"
        r".{5,80}"
        r"\b(?:about\s+you|talking\s+to\s+you|really\s+about\s+you)\b",
        s, re.IGNORECASE
    )
    if indirect_attack:
        score += 0.30
        evidence.append(_snip(text, indirect_attack))

    return min(score, 1.0), evidence[:3]


def _detect_floater(text: str) -> Tuple[float, List[str]]:
    """Floater: short direct shot that connects cleanly."""
    s = _norm(text)
    score, evidence = 0.0, []

    subtle_patterns = [
        r"\b(?:quietly|softly|subtly|smoothly|low[\s-]key)\b",
        r"\b(?:slipped?\s+(?:in|by|through|past)|sneaked?\s+(?:in|by|through|past))\b",
        r"\b(?:under\s+(?:their|his|your)\s+(?:radar|guard|nose))\b",
        r"\b(?:barely|just\s+barely)\s+\w+\b",
        r"\b(?:floated?|glided?|drifted?)\b",
        r"\b(?:whispered?|understated)\b",
    ]
    sc, ms = _score_patterns(s, subtle_patterns, 0.2, 0.55)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m))

    indirect_ref = [
        r"\b(?:not\s+(?:naming\s+names|pointing\s+fingers|saying\s+(?:who|names)))\b",
        r"\b(?:hypothetically|theoretically|imagine\s+if)\b",
        r"\b(?:speaking\s+of|which\s+reminds\s+me)\b",
    ]
    sc2, ms2 = _score_patterns(s, indirect_ref, 0.15, 0.30)
    score += sc2
    for m in ms2[:1]:
        if len(evidence) < 3:
            evidence.append(_snip(text, m))

    return min(score, 1.0), evidence[:3]


def _detect_crossover(text: str) -> Tuple[float, List[str]]:
    """Crossover: fake one angle, switch direction at the last second."""
    s = _norm(text)
    score, evidence = 0.0, []

    shift_patterns = [
        r"\b(?:switch(?:ed)?\s+(?:up|it\s+up)|flip(?:ped)?\s+(?:the\s+)?(?:script|style|delivery))\b",
        r"\b(?:change\s+of\s+(?:pace|tone|register|style|gear|tempo))\b",
        r"\b(?:sped?\s+(?:it\s+)?up|slowed?\s+(?:it\s+)?down)\b",
        r"\b(?:tempo\s+(?:change|shift|drop))\b",
        r"\b(?:code[\s-]switch(?:ed)?|different\s+energy|switched\s+lanes?)\b",
        r"\b(?:went\s+from\s+\w+\s+to\s+\w+)\b",
    ]
    sc, ms = _score_patterns(s, shift_patterns, 0.25, 0.70)
    score += sc
    for m in ms[:3]:
        evidence.append(_snip(text, m))

    return min(score, 1.0), evidence[:3]


def _detect_rebound(text: str) -> Tuple[float, List[str]]:
    """Rebound: strong recovery after stumble — turns a weak moment into strength."""
    s = _norm(text)
    score, evidence = 0.0, []

    recovery_patterns = [
        r"\b(?:recovered?|bounced?\s+back|came\s+back|got\s+back)\b",
        r"\b(?:still\s+came|still\s+hit|still\s+landed|still\s+won)\b",
        r"\b(?:recovery|comeback|come\s+back|rebound)\b",
        r"\b(?:even\s+after\s+(?:that|the|stumbling|choking|forgetting))\b",
        r"\b(?:came\s+back\s+(?:harder|stronger|better))\b",
    ]
    sc, ms = _score_patterns(s, recovery_patterns, 0.22, 0.55)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m))

    # Stumble-then-recover sequence within the same text
    stumble_recover = re.search(
        r"\b(?:stumbled?|choked?|forgot|messed\s+up|lost\s+(?:my\s+)?(?:place|flow))\b"
        r".{0,120}"
        r"\b(?:but|still|then|yet|however)\b.{0,60}"
        r"\b(?:came\s+back|recovered?|bounced?|landed|hit|won)\b",
        s, re.IGNORECASE | re.DOTALL
    )
    if stumble_recover:
        score += 0.35
        evidence.append(_snip(text, stumble_recover, pad=10))

    return min(score, 1.0), evidence[:3]


def _detect_midrange(text: str) -> Tuple[float, List[str]]:
    """Mid-Range: strong, consistent punchline — reliable heavy execution."""
    s = _norm(text)
    lines = _lines(text)
    score, evidence = 0.0, []

    solid_patterns = [
        r"\breceipts?\b|\bproof\b|\bevidence\b|\bdocumented\b|\bverified\b",
        r"\byou\s+(?:said|did|were|never|always|used\s+to)\b",
        r"\bcheck\s+(?:the\s+)?(?:record|tape|footage|history)\b",
        r"\b(?:exposed?|proven?|shown|clear|obvious|facts?|check)\b",
        r"\b(?:truth|real|genuine|authentic|credible?)\b",
    ]
    sc, ms = _score_patterns(s, solid_patterns, 0.12, 0.45)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m))

    # Density of non-filler lines
    filler_words = {'yeah', 'ok', 'uh', 'um', 'right', 'now', 'so', 'and', 'the', 'a'}
    non_filler = [
        ln for ln in lines
        if len(ln.split()) >= 6 and
        len(set(ln.lower().split()) - filler_words) >= 4
    ]
    if len(non_filler) >= 6:
        score += 0.25
    elif len(non_filler) >= 3:
        score += 0.12

    return min(score, 1.0), evidence[:3]


def _detect_layup(text: str) -> Tuple[float, List[str]]:
    """Layup: narrative jab — light diss via storytelling, clean close-range hit."""
    s = _norm(text)
    score, evidence = 0.0, []

    direct_patterns = [
        r"\byou\s+(?:can'?t|couldn'?t|don'?t)\b",
        r"\byou\s+(?:ain'?t|aren'?t|weren'?t|never\s+(?:were|was|have\s+been))\b",
        r"\b(?:no\s+cap|no\s+lie|facts?\s+only|just\s+facts)\b",
        r"\b(?:bottom\s+line|point\s+blank|plain\s+and\s+simple|straight\s+up)\b",
        r"\b(?:simple|basic|plain|real\s+talk|straight)\b",
    ]
    sc, ms = _score_patterns(s, direct_patterns, 0.12, 0.45)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m))

    # Basic me-vs-you contrast
    me_vs_you = re.search(
        r"\b(?:i|my)\s+\w+\s+(?:is|are|was|were)\s+(?:better|stronger|real|greater)\s+"
        r"(?:than|compared\s+to)\s+(?:you|your|his|her)\b",
        s, re.IGNORECASE
    )
    if me_vs_you:
        score += 0.22
        evidence.append(_snip(text, me_vs_you))

    return min(score, 1.0), evidence[:3]


def _detect_alley_oop(text: str) -> Tuple[float, List[str]]:
    """Alley-Oop: explicit coordinated setup → payoff, or tagged-in structure."""
    s = _norm(text)
    score, evidence = 0.0, []

    patterns = [
        r"\bset(?:ting)?\s+(?:him|them|you|it)\s+up\b",
        r"\bpass(?:ed|ing)?\s+(?:him|them|it|me)\s+(?:the|a)\s+\w+",
        r"\btag(?:ged|ging)?\s+(?:in|out)\b",
        r"\b(?:my\s+)?(?:partner|teammate|co[\s-]writer)\s+(?:said|told|wrote|set)",
        r"\bthat(?:\'s|\s+was)\s+the\s+set[\s-]?up\b",
        r"\b(?:alley[\s-]?oop|assist(?:ed)?)\b",
        r"\bhe\s+(?:set|gave|passed)\s+(?:me|it)\s+(?:up|over)\b",
    ]
    sc, ms = _score_patterns(s, patterns, 0.40, 0.80)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m))

    return min(score, 1.0), evidence[:3]


def _detect_and_1(text: str) -> Tuple[float, List[str]]:
    """And-1: punch lands cleanly even while being interrupted or talked through."""
    s = _norm(text)
    score, evidence = 0.0, []

    interference_patterns = [
        r"\[(?:talks?\s+over|speaks?\s+over|interrupts?|yelling|crowd\s+interference)\]",
        r"\b(?:you\s+(?:yelling|talking|screaming)\s+(?:through|during|in)\s+my\s+(?:round|setup|bar))\b",
        r"\b(?:punch(?:es?)?\s+(?:through|past)\s+(?:the\s+)?(?:noise|resistance|interference))\b",
        r"\b(?:even\s+(?:while|with|through)\s+(?:you|them|the\s+crowd)\s+(?:talking|yelling|interrupting))\b",
        r"\b(?:bar\s+(?:still\s+)?(?:landed|hit|scored)\s+(?:through|despite))\b",
        r"\b(?:punch\s+(?:still\s+)?connects?\s+(?:even\s+)?(?:through|while))\b",
        r"\b(?:interrupting\s+only\s+(?:made|proves?|shows?))\b",
        r"\b(?:talking\s+through\s+my\s+setup\s+won'?t)\b",
    ]
    sc, ms = _score_patterns(s, interference_patterns, 0.35, 0.85)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m, pad=40))

    return min(score, 1.0), evidence[:3]


def _detect_spin_move(text: str) -> Tuple[float, List[str]]:
    """Spin Move: takes opponent's own angle and reverses it against them."""
    s = _norm(text)
    score, evidence = 0.0, []

    # The key pattern: "you said X — [irony/flip] because [Y that negates X]"
    flip_patterns = [
        r"\byou\s+(?:said|call|claim)\s+.{5,60}(?:funny|ironic|crazy|wild|but)\b",
        r"\byou\s+(?:said|call)\s+(?:i|me)\s+.{3,40}(?:but|yet|funny\s+how)\b",
        r"\b(?:ironic|funny|crazy|wild)\s+(?:how|that|because|since)\s+you\b",
        r"\b(?:the\s+same\s+one\s+who|but\s+you'?re?\s+the\s+one\s+who)\b",
        r"\b(?:you\s+brag|you\s+claim)\s+.{5,60}(?:but|yet)\b",
        r"\byour\s+own\s+(?:logic|argument|angle|point)\s+(?:backfires?|works?\s+against|proves?)\b",
        r"\b(?:using\s+your\s+(?:own\s+)?angle|turning\s+(?:it|that|your\s+point)\s+(?:back|around))\b",
    ]
    sc, ms = _score_patterns(s, flip_patterns, 0.30, 0.75)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m, pad=50))

    reversal_words = [
        r"\b(?:spin\s+move|angle\s+flip|reverse(?:d)?|backfired?)\b",
        r"\b(?:irony|hypocrite|contradict|pot\s+calling\s+the\s+kettle)\b",
    ]
    sc2, ms2 = _score_patterns(s, reversal_words, 0.20, 0.35)
    score += sc2
    for m in ms2[:1]:
        if len(evidence) < 3:
            evidence.append(_snip(text, m))

    return min(score, 1.0), evidence[:3]


def _detect_post_move(text: str) -> Tuple[float, List[str]]:
    """Post Move: direct close-range breakdown of opponent's identity."""
    s = _norm(text)
    score, evidence = 0.0, []

    breakdown_patterns = [
        r"\b(?:let'?s?\s+talk\s+about\s+(?:you|who\s+you\s+(?:really\s+)?are))\b",
        r"\b(?:forget\s+(?:your\s+)?(?:clique|crew|bars?|stories?|persona|reputation))\b",
        r"\b(?:who\s+you\s+(?:really|actually)\s+are)\b",
        r"\b(?:peel\s+back\s+(?:the\s+)?(?:layers?|mask|lies?|truth))\b",
        r"\b(?:i'?m\s+(?:not\s+)?(?:attacking|going\s+after)\s+(?:your\s+bars?|the\s+person))\b",
        r"\b(?:the\s+(?:real|true|actual)\s+(?:you|version\s+of\s+you))\b",
        r"\b(?:behind\s+(?:the\s+)?(?:persona|mask|image|bars?|front))\b",
        r"\b(?:stripped?\s+(?:down|away)|expose\s+(?:who\s+you|the\s+real))\b",
    ]
    sc, ms = _score_patterns(s, breakdown_patterns, 0.28, 0.75)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m, pad=50))

    return min(score, 1.0), evidence[:3]


def _detect_pick_and_roll(text: str) -> Tuple[float, List[str]]:
    """Pick & Roll: self-generated setup feeding into own punchline."""
    s = _norm(text)
    score, evidence = 0.0, []

    self_setup_patterns = [
        r"\b(?:i\s+(?:set\s+(?:myself|that)\s+up|gave\s+myself|threw\s+(?:it|the\s+setup)\s+to\s+myself))\b",
        r"\b(?:self[\s-]alley[\s-]oop|passed\s+(?:it\s+)?to\s+myself)\b",
        r"\b(?:i\s+(?:built|loaded|stacked|set)\s+(?:the\s+)?(?:setup|scheme|trap)\s+(?:for\s+myself|myself))\b",
        r"\b(?:nobody\s+(?:assist(?:ing|ed)?|help(?:ing|ed)?)\s+(?:my|this)\s+(?:career|round))\b",
        r"\b(?:i\s+(?:pitched|rolled|screened)\s+(?:it|that)\s+(?:soft|to\s+myself|then\s+slam))\b",
    ]
    sc, ms = _score_patterns(s, self_setup_patterns, 0.35, 0.80)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m, pad=40))

    return min(score, 1.0), evidence[:3]


def _detect_isolation(text: str) -> Tuple[float, List[str]]:
    """Isolation: extended solo attack focused on a single flaw or topic."""
    s = _norm(text)
    lines = _lines(text)
    score, evidence = 0.0, []

    iso_patterns = [
        r"\b(?:for\s+(?:eight|8|sixteen|16|the\s+whole|all)\s+(?:bars?|round|lines?))\b",
        r"\b(?:(?:the\s+)?whole\s+round\s+(?:i'?m\s+on|on)\s+(?:just\s+)?you)\b",
        r"\b(?:stay\s+(?:right\s+)?there|don'?t\s+(?:look|move|run))\b",
        r"\b(?:i'?m\s+cooking\s+(?:just|only)\s+you)\b",
        r"\b(?:one\s+(?:angle|target|focus|topic)\s+(?:all\s+round|the\s+whole|start\s+to\s+finish))\b",
        r"\b(?:iso(?:lation)?|singled?\s+out|one[\s-]on[\s-]one|personal\s+foul)\b",
        r"\b(?:can'?t\s+escape\s+(?:when\s+i|the\s+iso))\b",
    ]
    sc, ms = _score_patterns(s, iso_patterns, 0.30, 0.70)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m, pad=40))

    # Topic repetition: same word appearing 3+ times across 4+ lines suggests focused attack
    if len(lines) >= 4:
        words = re.findall(r'\b[a-z]{4,}\b', s)
        word_counts = Counter(words)
        top_word, top_cnt = word_counts.most_common(1)[0] if word_counts else ("", 0)
        if top_cnt >= 4 and top_word not in {'that', 'this', 'with', 'your', 'just', 'like',
                                              'have', 'been', 'what', 'when', 'they', 'then',
                                              'every', 'about', 'from', 'make'}:
            score += 0.20
            evidence.append(f'Focused angle: "{top_word}" repeated {top_cnt}x')

    return min(score, 1.0), evidence[:3]


def _detect_breakaway(text: str) -> Tuple[float, List[str]]:
    """Breakaway: extended clean uninterrupted run with no stumbles."""
    s = _norm(text)
    lines = _lines(text)
    score, evidence = 0.0, []

    flow_patterns = [
        r"\b(?:kept\s+(?:going|flowing|cooking|running)|didn'?t\s+stop)\b",
        r"\b(?:clean\s+(?:run|streak|stretch|flow)|smooth\s+(?:run|flow))\b",
        r"\b(?:no\s+(?:breaks?|stops?|stumbles?|pauses?))\b",
        r"\b(?:rolling|momentum|streak|straight\s+through)\b",
        r"\b(?:eight\s+bars?\s+(?:in|straight|clean)|(?:eight|8)\s+clean\s+bars?)\b",
        r"\b(?:uninterrupted|non[\s-]stop)\b",
    ]
    sc, ms = _score_patterns(s, flow_patterns, 0.25, 0.55)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m, pad=40))

    # Positive signal: 8+ lines with no stumble markers
    stumble_re = re.compile(
        r"(?:\[(?:um+|uh+|er+|erm+)\]|\bum+\b|\buh+\b)|\.{3,}"
        r"|\[(?:pause|stumble|hesitation)\]",
        re.IGNORECASE
    )
    if len(lines) >= 8 and not stumble_re.search(s):
        score += 0.30
        evidence.append(f"{len(lines)} lines with no stumble markers detected")

    return min(score, 1.0), evidence[:3]


def _detect_no_look_pass(text: str) -> Tuple[float, List[str]]:
    """No-Look Pass: punch disguised behind casual, offhand delivery."""
    s = _norm(text)
    score, evidence = 0.0, []

    casual_patterns = [
        r"\b(?:wasn'?t\s+even\s+(?:aiming|trying|targeting)\s+(?:at\s+)?you)\b",
        r"\b(?:(?:off|out)\s+the\s+(?:top|cuff|dome|head))\b",
        r"\b(?:accidentally|by\s+accident|didn'?t\s+mean\s+to)\b",
        r"\b(?:i\s+wasn'?t\s+even\s+(?:looking|aiming|focused\s+on\s+you))\b",
        r"\b(?:(?:said|dropped)\s+it\s+(?:casually|offhand|in\s+passing|without\s+trying))\b",
        r"\b(?:wrote\s+this\s+(?:part\s+)?for\s+nobody)\b",
        r"\b(?:slipped?\s+(?:that|it)\s+in\s+(?:casually|quietly|without))\b",
        r"\b(?:no[\s-]look|sleeper|hidden\s+punch)\b",
    ]
    sc, ms = _score_patterns(s, casual_patterns, 0.30, 0.80)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m, pad=45))

    return min(score, 1.0), evidence[:3]


def _detect_out_the_gate(text: str) -> Tuple[float, List[str]]:
    """Out-the-Gate Run: strong powerful opening sequence."""
    s = _norm(text)
    lines = _lines(text)
    score, evidence = 0.0, []

    opening_patterns = [
        r"\b(?:out\s+(?:the|of\s+the)\s+gate)\b",
        r"\b(?:from\s+(?:the\s+)?(?:jump|start|top|gate|beginning|first\s+bar))\b",
        r"\b(?:opened?\s+(?:the\s+)?(?:round|battle|set))\b",
        r"\b(?:first\s+(?:bar|line|few\s+bars?)\s+(?:in|hit|landed))\b",
        r"\b(?:so\s+much\s+pressure\s+in\s+the\s+first)\b",
        r"\b(?:started?\s+(?:so\s+)?(?:strong|hard|cold|hot|fast))\b",
    ]
    sc, ms = _score_patterns(s, opening_patterns, 0.35, 0.70)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m, pad=45))

    # Position-based: if first 25% of lines have impact words
    if len(lines) >= 4:
        opening_lines = lines[:max(2, len(lines) // 4)]
        opening_text = " ".join(opening_lines).lower()
        impact_count = len(re.findall(
            r'\b(?:dead|bodied|killed|destroyed|pressure|shook|hit|landed|punched)\b',
            opening_text
        ))
        if impact_count >= 2:
            score += 0.25
            evidence.append("Strong impact density in opening lines")
        elif impact_count == 1:
            score += 0.10

    return min(score, 1.0), evidence[:3]


def _detect_fourth_quarter(text: str) -> Tuple[float, List[str]]:
    """Fourth-Quarter Push: strong powerful closing sequence."""
    s = _norm(text)
    lines = _lines(text)
    score, evidence = 0.0, []

    closing_patterns = [
        r"\b(?:closed?\s+(?:the\s+)?(?:round|battle)\s+(?:like|with|so))\b",
        r"\b(?:ended?\s+(?:the\s+)?round\s+(?:so|with|like))\b",
        r"\b(?:(?:last|final|closing)\s+(?:bar|bars?|shot|punch|push))\b",
        r"\b(?:fourth[\s-]quarter|buzzer[\s-]beater)\b",
        r"\b(?:finished\s+(?:so\s+)?(?:cold|hot|hard|strong))\b",
        r"\b(?:outro|closing\s+(?:run|push|stretch))\b",
        r"\b(?:curtain\s+closer|lights\s+out|signed\s+off)\b",
    ]
    sc, ms = _score_patterns(s, closing_patterns, 0.35, 0.70)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m, pad=45))

    # Position-based: if last 25% of lines have impact words
    if len(lines) >= 4:
        closing_lines = lines[-(max(2, len(lines) // 4)):]
        closing_text = " ".join(closing_lines).lower()
        impact_count = len(re.findall(
            r'\b(?:dead|bodied|killed|destroyed|pressure|shook|hit|landed|punched|done|finished)\b',
            closing_text
        ))
        if impact_count >= 2:
            score += 0.25
            evidence.append("Strong impact density in closing lines")
        elif impact_count == 1:
            score += 0.10

    return min(score, 1.0), evidence[:3]


# ============================================================================
# Negative Skill Detectors
# ============================================================================

def _detect_offensive_foul(text: str) -> Tuple[float, List[str]]:
    """Offensive Foul: distasteful aggression with no lyrical value."""
    s = _norm(text)
    score, evidence = 0.0, []

    patterns = [
        r"\[(?:offensive\s+foul|distasteful|aggression|barking)\]",
        r"\b(?:i'?ll\s+(?:spit\s+on|slap|fight|hurt)\s+(?:your|you|his))\b",
        r"\b(?:get\s+in\s+(?:your|his)\s+face|stepped?\s+to\s+(?:him|you))\b",
        r"\b(?:bark(?:ing|ed)?\s+(?:at|in)\s+(?:his|your|their)\s+face)\b",
        r"\b(?:say\s+something\s+else\s+and\s+i'?ll)\b",
        r"\b(?:disrespect(?:ful)?\s+(?:with\s+)?no\s+(?:bar|punchline|purpose))\b",
    ]
    sc, ms = _score_patterns(s, patterns, 0.35, 0.80)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m, pad=25))

    return min(score, 1.0), evidence[:3]


def _detect_carry(text: str) -> Tuple[float, List[str]]:
    """Carry: 4+ bar setup with no punch or anticlimactic payoff."""
    s = _norm(text)
    lines = _lines(text)
    score, evidence = 0.0, []

    # Multiple setup phrases in sequence
    setup_re = re.compile(
        r"\b(?:i\s+(?:did|checked|found|studied|analyzed|researched|mapped|"
        r"investigated|prepared|planned|built|loaded|stacked|set\s+up|constructed))\b",
        re.IGNORECASE
    )
    setup_hits = list(setup_re.finditer(s))

    if len(setup_hits) >= 3:
        score += 0.40
        for m in setup_hits[:2]:
            evidence.append(_snip(text, m))
    elif len(setup_hits) >= 2:
        score += 0.20

    # Anticlimactic ending phrases
    anticlimactic_re = re.compile(
        r"\b(?:and\s+(?:that'?s?\s+(?:all|it)|nothing|nothing\s+(?:else|more)))\b"
        r"|\b(?:i\s+(?:couldn'?t\s+find|found\s+nothing|had\s+nothing))\b"
        r"|\b(?:still\s+couldn'?t\s+find\s+a\s+reason)\b"
        r"|\b(?:and\s+forgot\s+the\s+punchline\s+existed)\b",
        re.IGNORECASE
    )
    anti_hits = list(anticlimactic_re.finditer(s))
    if anti_hits and setup_hits:
        score += 0.30
        for m in anti_hits[:1]:
            evidence.append(_snip(text, m))

    # Long text without clear punch words (when setups are present)
    if len(lines) >= 6 and setup_hits:
        has_punch = bool(re.search(
            r'\b(?:dead|bodied|murdered|killed|done|over|finished|erased|destroyed|demolished)\b', s
        ))
        if not has_punch:
            score += 0.20
            evidence.append("Extended setup without clear punchline landing")

    return min(score, 1.0), evidence[:3]


def _detect_backcourt_violation(text: str) -> Tuple[float, List[str]]:
    """Backcourt Violation: over-explaining, dragging, excessive filler."""
    s = _norm(text)
    lines = _lines(text)
    score, evidence = 0.0, []

    filler_patterns = [
        r"\byou\s+know\s+(?:what\s+i\s+mean|what\s+i'?m\s+saying)\b",
        r"\b(?:like\s+i\s+said|as\s+i\s+said|i\s+said\s+it\s+before)\b",
        r"\band\s+(?:stuff|things|whatnot)\b",
        r"\b(?:basically|essentially|literally|honestly)\s+",
        r"\byou\s+feel\s+me\b|\byou\s+know\s+what\s+i'?m\s+sayin\b",
    ]
    sc, ms = _score_patterns(s, filler_patterns, 0.15, 0.35)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m))

    # Over-explaining the punch before throwing it
    explaining_patterns = [
        r"\b(?:this\s+bar\s+(?:gon'?\s+)?(?:hit|land)|let\s+me\s+explain\s+(?:why|the))\b",
        r"\b(?:before\s+i\s+(?:punch|hit|land)|let\s+me\s+break\s+down\s+why)\b",
        r"\b(?:i'?m\s+about\s+to\s+punch\s+you,?\s+but\s+before)\b",
    ]
    sc2, ms2 = _score_patterns(s, explaining_patterns, 0.25, 0.45)
    score += sc2
    for m in ms2[:1]:
        if len(evidence) < 3:
            evidence.append(_snip(text, m))

    # Repeated line openings
    if len(lines) >= 4:
        openings = [" ".join(ln.split()[:3]).lower() for ln in lines]
        repeated = len(openings) - len(set(openings))
        if repeated >= 3:
            score += 0.25
            evidence.append(f"Repeated line openings: {repeated} instances")
        elif repeated >= 2:
            score += 0.12

    # High proportion of filler-only lines
    filler_set = {'yeah', 'ok', 'uh', 'um', 'right', 'so', 'and', 'like', 'you know'}
    if lines:
        filler_lines = sum(
            1 for ln in lines
            if ln and len(set(ln.lower().split()) - filler_set) < 3
        )
        ratio = filler_lines / len(lines)
        if ratio > 0.35:
            score += 0.20
            evidence.append(f"Filler-heavy lines: {filler_lines}/{len(lines)} ({ratio:.0%})")

    return min(score, 1.0), evidence[:3]


def _detect_forced_angle(text: str) -> Tuple[float, List[str]]:
    """Charge / Forced Angle: weak ill-fitting angle repeated without escalation."""
    s = _norm(text)
    score, evidence = 0.0, []

    forced_patterns = [
        r"\b(?:kind\s+of\s+like|sort\s+of\s+like|almost\s+like|a\s+bit\s+like)\b",
        r"\bin\s+a\s+(?:way|sense)\b|\bin\s+some\s+ways\b|\bloosely\b",
        r"\b(?:you\s+could\s+(?:say|argue|make\s+the\s+case))\b",
        r"\b(?:if\s+you\s+(?:think|look)\s+at\s+it\s+(?:a\s+)?(?:that|this|certain)\s+way)\b",
        r"\b(?:i'?m\s+(?:reaching|stretching)|that'?s?\s+a\s+(?:stretch|reach))\b",
    ]
    sc, ms = _score_patterns(s, forced_patterns, 0.20, 0.55)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m))

    # Repeated weak angle: same thin accusation 3+ times
    lines = _lines(text)
    if len(lines) >= 4:
        common_forced = re.findall(
            r"\b(?:you\s+broke|you\s+fake|you\s+trash|you\s+wack)\b", s
        )
        if len(common_forced) >= 3:
            score += 0.30
            evidence.append(f'Repeated forced angle: {len(common_forced)}x')

    # Long setup with no punch word
    if len(s) > 150:
        has_punch = bool(re.search(
            r'\b(?:dead|over|done|finished|bodied|murdered|destroyed|erased)\b', s
        ))
        if not has_punch:
            score += 0.20
            evidence.append("Long setup without a clear punch landing")

    return min(score, 1.0), evidence[:3]


def _detect_travel(text: str) -> Tuple[float, List[str]]:
    """Travel / Stumble: verbal trip OR structural restart that breaks momentum."""
    s = _norm(text)
    score, evidence = 0.0, []

    # Verbal stumble markers (um, uh, pause, ellipsis, repeated word)
    stumble_patterns = [
        r"(?:\[(?:um+|uh+|er+|erm+)\]|\bum+\b|\buh+\b|\berm+\b)",
        r"\[(?:pause|hesitation|stumble|trip|stutters?|stuttering)\]",
        r"\b(?:wait|hold\s+on|hold\s+up)\b",
        r"(?:\.{3,})",
        r"\b(\w+)\s+\1\b",  # repeated word (word word)
    ]
    count = 0
    for p in stumble_patterns:
        for m in re.finditer(p, s, re.IGNORECASE):
            count += 1
            if len(evidence) < 2:
                evidence.append(_snip(text, m, pad=20))
            break  # one match per pattern

    score += min(count * 0.25, 0.60)

    # Structural restart markers (more severe form)
    restart_patterns = [
        r"\b(?:let\s+me\s+(?:restart|start\s+(?:over|again)|try\s+again|go\s+back))\b",
        r"\b(?:started?\s+(?:over|again)|restarted?)\b",
        r"\b(?:i\s+(?:messed\s+up|made\s+a\s+mistake|went\s+the\s+wrong\s+(?:way|direction)))\b",
        r"\[(?:restart|restarted?|starts?\s+over|structural\s+break|travel)\]",
        r"\b(?:that\s+wasn'?t\s+(?:right|supposed\s+to|intentional))\b",
    ]
    sc2, ms2 = _score_patterns(s, restart_patterns, 0.35, 0.75)
    score += sc2
    for m in ms2[:1]:
        if len(evidence) < 3:
            evidence.append(_snip(text, m, pad=20))

    return min(score, 1.0), evidence[:3]


def _detect_double_dribble(text: str) -> Tuple[float, List[str]]:
    """Double Dribble: verbatim or near-verbatim bar repetition."""
    import difflib
    lines = _lines(text)
    score, evidence = 0.0, []

    # Exact repeated lines (original logic)
    line_counts = Counter(ln.lower().strip() for ln in lines if len(ln.split()) > 3)
    exact_repeated = [(ln, cnt) for ln, cnt in line_counts.items() if cnt > 1]
    if exact_repeated:
        score += min(len(exact_repeated) * 0.45, 0.95)
        for ln, cnt in exact_repeated[:2]:
            evidence.append(f'Repeated ({cnt}x): "{ln[:80]}"')

    # Fuzzy near-duplicate lines (catches "is" vs "are", minor word changes)
    sig_lines = [(i, ln) for i, ln in enumerate(lines) if len(ln.split()) > 4]
    fuzzy_pairs = []
    for a in range(len(sig_lines)):
        for b in range(a + 1, len(sig_lines)):
            i, la = sig_lines[a]
            j, lb = sig_lines[b]
            ratio = difflib.SequenceMatcher(
                None, la.lower().strip(), lb.lower().strip()
            ).ratio()
            if ratio >= 0.80 and (i, j) not in [(p[0], p[1]) for p in fuzzy_pairs]:
                fuzzy_pairs.append((i, j, ratio, la, lb))

    if fuzzy_pairs:
        score += min(len(fuzzy_pairs) * 0.35, 0.60)
        for i, j, ratio, la, lb in fuzzy_pairs[:2]:
            if len(evidence) < 3:
                evidence.append(
                    f'Near-repeat (similarity {ratio:.0%}): '
                    f'"{la[:50]}" ≈ "{lb[:50]}"'
                )

    return min(score, 1.0), evidence[:3]


def _detect_choke(text: str) -> Tuple[float, List[str]]:
    """Turnover / Choke: complete breakdown — forgot lyrics, restarted, blanked out."""
    s = _norm(text)
    score, evidence = 0.0, []

    choke_patterns = [
        r"\[(?:forgot|choke|choked|forgetting|lost|restarted?|start(?:s?\s+)?over)\]",
        r"\b(?:forgot\s+(?:my|the)\s+(?:bar|line|lyrics?|words?))\b",
        r"\b(?:let\s+me\s+(?:start\s+over|try\s+again|go\s+back|restart))\b",
        r"\b(?:completely\s+(?:lost|forgot|blanked?|froze))\b",
        r"\b(?:mind\s+went\s+blank|blanked?\s+out|went\s+blank)\b",
        r"(?:\.\.\.\s*){3,}",
    ]
    sc, ms = _score_patterns(s, choke_patterns, 0.4, 0.90)
    score += sc
    for m in ms[:3]:
        evidence.append(_snip(text, m, pad=20))

    return min(score, 1.0), evidence[:3]


# ---- Fouls ----

def _detect_out_of_bounds(text: str) -> Tuple[float, List[str]]:
    """Out of Bounds: breaking round structure — mid-round format violation."""
    s = _norm(text)
    score, evidence = 0.0, []

    patterns = [
        r"\[(?:breaks?\s+structure|breaks?\s+format|breaks?\s+rules?|disruptive|out\s+of\s+bounds)\]",
        r"\b(?:breaking\s+(?:the\s+)?(?:rules?|format|structure|pattern))\b",
        r"\b(?:that'?s?\s+not\s+allowed|you\s+can'?t\s+do\s+that|against\s+(?:the\s+)?rules?)\b",
        r"\b(?:yo,?\s+who\s+(?:got|has)\s+my\s+(?:phone|keys?|bag))\b",
        r"\b(?:went\s+off\s+topic|(?:stop|stops)\s+(?:mid|mid[\s-]round)\s+to)\b",
    ]
    sc, ms = _score_patterns(s, patterns, 0.30, 0.80)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m, pad=20))

    return min(score, 1.0), evidence[:3]


def _detect_technical_foul(text: str) -> Tuple[float, List[str]]:
    """Technical Foul: rule violation — props, refusing instructions, mic violations."""
    s = _norm(text)
    score, evidence = 0.0, []

    patterns = [
        r"\[(?:prop(?:s)?|technical\s+foul|rule\s+violation|refused?|mic\s+violation)\]",
        r"\b(?:pull(?:s|ed)?\s+out\s+a\s+prop|using?\s+props?)\b",
        r"\b(?:refused?\s+(?:to\s+)?(?:start|stop|follow|rap)|won'?t\s+(?:start|follow\s+instructions?))\b",
        r"\b(?:technical\s+foul|tech(?:nical)?\s+violation|rule\s+break)\b",
        r"\b(?:grab(?:s|bed)?\s+(?:the\s+)?mic\s+(?:aggressively|hard)|snatched?\s+(?:the\s+)?mic)\b",
        r"\b(?:this\s+(?:is\s+)?(?:your|his)\s+paperwork)\b",
    ]
    sc, ms = _score_patterns(s, patterns, 0.40, 0.90)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m, pad=25))

    return min(score, 1.0), evidence[:3]


def _detect_physical_contact(text: str) -> Tuple[float, List[str]]:
    """Defensive Foul: physical contact with opponent."""
    s = _norm(text)
    score, evidence = 0.0, []

    patterns = [
        r"\[(?:physical\s+contact|pushed?|shoved?|touched?|grabbed?|flagrant)\]",
        r"\b(?:flagrant|physical\s+(?:contact|altercation|confrontation))\b",
        r"\b(?:pushed?|shoved?|grabbed?|got\s+physical|touched)\s+\w+\b",
    ]
    sc, ms = _score_patterns(s, patterns, 0.55, 0.90)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m, pad=20))

    return min(score, 1.0), evidence[:3]


def _detect_goaltending(text: str) -> Tuple[float, List[str]]:
    """Goaltending: excessive talking through opponent's round (unsportsmanlike)."""
    s = _norm(text)
    score, evidence = 0.0, []

    patterns = [
        r"\[(?:talks?\s+over|speaks?\s+over|interrupts?|cutting\s+(?:off|in)|goaltending)\]",
        r"\b(?:stop\s+(?:interrupting|cutting\s+(?:me|him|them)\s+off))\b",
        r"\b(?:let\s+him\s+(?:finish|speak|talk)|it'?s?\s+(?:my|his|their)\s+time)\b",
        r"\b(?:talking\s+(?:through|over)\s+(?:him|me|them|opponent))\b",
        r"\b(?:that'?s?\s+a\s+lie[!]?\s+that'?s?\s+a\s+lie)\b",
        r"\b(?:interrupting\s+every\s+bar|shouting\s+(?:through|during)\s+(?:his|their|opponent'?s?))\b",
    ]
    sc, ms = _score_patterns(s, patterns, 0.50, 0.90)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m, pad=20))

    return min(score, 1.0), evidence[:3]


def _detect_boundary_violation(text: str) -> Tuple[float, List[str]]:
    """Boundary Violation / Flagrant: serious rule or personal boundary breach."""
    s = _norm(text)
    score, evidence = 0.0, []

    patterns = [
        r"\[(?:boundary\s+violation|flagrant|major\s+violation|disqualified|dq|ejection)\]",
        r"\b(?:disqualified|dq'?d?|major\s+violation|flagrant\s+(?:foul|1|2|3))\b",
        r"\b(?:crossed\s+(?:the\s+)?line|went\s+too\s+far|beyond\s+(?:the\s+)?(?:limit|boundary|line))\b",
        r"\b(?:ejected|ejection|removed\s+from\s+(?:the\s+)?(?:stage|event|battle))\b",
        r"\b(?:hate\s+speech|real[\s-]life\s+(?:threats?|harm)|threatened?\s+(?:real|outside))\b",
    ]
    sc, ms = _score_patterns(s, patterns, 0.55, 0.90)
    score += sc
    for m in ms[:2]:
        evidence.append(_snip(text, m, pad=20))

    return min(score, 1.0), evidence[:3]


# ============================================================================
# Detector Registry
# ============================================================================

# For the 4 legacy skills: wrap existing float-returning functions
def _wrap_legacy(fn):
    """Adapt a legacy float-only detector to return (float, [])."""
    def wrapper(text):
        return float(fn(text)), []
    return wrapper


_DETECTORS = {
    # Legacy (reuse existing implementations from rap_techniques.py)
    "full_court_shot":      _wrap_legacy(detect_full_court_shot),
    "slam_dunk":            _wrap_legacy(detect_slam_dunk),
    "half_court_shot":      _wrap_legacy(detect_half_court_shot),
    "alley_oop":            _detect_alley_oop,
    # Highlights
    "and_1":                _detect_and_1,
    "fast_break":           _detect_fast_break,
    "three_pointer":        _detect_three_pointer,
    "euro_step":            _detect_euro_step,
    "steal":                _detect_steal,
    "crossover":            _detect_crossover,
    "hook_shot":            _detect_hook_shot,
    "out_the_gate":         _detect_out_the_gate,
    "fourth_quarter":       _detect_fourth_quarter,
    "spin_move":            _detect_spin_move,
    "post_move":            _detect_post_move,
    "pick_and_roll":        _detect_pick_and_roll,
    "isolation":            _detect_isolation,
    "breakaway":            _detect_breakaway,
    "midrange":             _detect_midrange,
    "layup":                _detect_layup,
    "rebound":              _detect_rebound,
    "no_look_pass":         _detect_no_look_pass,
    "floater":              _detect_floater,
    # Mistakes
    "offensive_foul":       _detect_offensive_foul,
    "carry":                _detect_carry,
    "backcourt_violation":  _detect_backcourt_violation,
    "forced_angle":         _detect_forced_angle,
    "travel":               _detect_travel,
    "double_dribble":       _detect_double_dribble,
    "choke":                _detect_choke,
    # Fouls
    "out_of_bounds":        _detect_out_of_bounds,
    "technical_foul":       _detect_technical_foul,
    "physical_contact":     _detect_physical_contact,
    "goaltending":          _detect_goaltending,
    "boundary_violation":   _detect_boundary_violation,
}


# ============================================================================
# Main Public API
# ============================================================================

def detect_skills(
    text: str,
    threshold: float = 0.4,
    negative_threshold: float = 0.5,
) -> SkillScanResult:
    """
    Multi-label skill detection for a piece of battle rap text.

    Args:
        text:               Lyrics or battle transcript to analyse.
        threshold:          Min confidence to report a positive/highlight skill.
        negative_threshold: Min confidence to report a mistake or foul.

    Returns:
        SkillScanResult with all detected skills, structured output, and evidence.
    """
    if not isinstance(text, str):
        text = str(text)

    detections: List[SkillDetection] = []

    for key, sid, name, points, category, default_thr in SKILL_REGISTRY:
        detector = _DETECTORS.get(key)
        if detector is None:
            continue

        eff_thr = negative_threshold if points < 0 else threshold
        eff_thr = max(eff_thr, default_thr)

        try:
            confidence, evidence = detector(text)
        except Exception:
            continue

        if confidence >= eff_thr:
            direction = "positive" if points >= 0 else "negative"
            detections.append(SkillDetection(
                skill_id=sid,
                skill_name=name,
                confidence=round(confidence, 4),
                points=points,
                direction=direction,
                category=category,
                evidence=evidence,
            ))

    detections.sort(key=lambda d: (d.direction != "positive", -abs(d.confidence)))

    highlights = [d for d in detections if d.direction == "positive"]
    mistakes   = [d for d in detections if d.direction == "negative"]
    total_score = round(sum(d.points for d in detections), 2)

    return SkillScanResult(
        detections=detections,
        total_score=total_score,
        highlights=highlights,
        mistakes=mistakes,
    )


def scan_round(text: str, threshold: float = 0.4) -> dict:
    """
    Analyse a full round of battle rap.
    Returns a JSON-serialisable dict.
    """
    return detect_skills(text, threshold=threshold).to_dict()


# ============================================================================
# Per-Line Scoring (Live API Interface)
# ============================================================================

@dataclass
class LineSummary:
    """Score breakdown for a single line of battle rap text."""
    line_number: int          # 1-indexed position in the round
    text: str                 # Original line text
    skills: List[SkillDetection]   # Skills attributed to this line
    line_score: float         # Sum of points for skills on this line

    def to_dict(self) -> dict:
        return {
            "line_number": self.line_number,
            "text": self.text,
            "skills": [dataclasses.asdict(s) for s in self.skills],
            "line_score": self.line_score,
        }


@dataclass
class RoundSummary:
    """Complete per-line scoring summary for one round."""
    round_number: int | None       # e.g. 1, 2, 3 (None if not provided)
    battler: str | None            # Battler name (None if not provided)
    lines: List[LineSummary]       # Per-line breakdown
    all_detections: List[SkillDetection]   # All detected skills (round level)
    total_score: float
    highlights: List[SkillDetection]
    deductions: List[SkillDetection]

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "battler": self.battler,
            "total_score": self.total_score,
            "lines": [ln.to_dict() for ln in self.lines],
            "highlights": [dataclasses.asdict(d) for d in self.highlights],
            "deductions": [dataclasses.asdict(d) for d in self.deductions],
        }


def _best_line_for_detection(detection: SkillDetection, lines: List[str]) -> int:
    """
    Return the 0-indexed line number that best matches this detection's evidence.
    Returns -1 for round-level skills with no line match.

    Strategy: extract word-level fragments from evidence (ignoring newlines),
    then score each line by how many fragments it contains.
    """
    if not detection.evidence or not lines:
        return -1

    hit_counts = [0] * len(lines)

    for ev in detection.evidence:
        # Split evidence on whitespace/newlines; try 3-word windows as probes
        ev_words = ev.lower().split()
        if not ev_words:
            continue
        # Build probes: single words (>=5 chars) and consecutive 2-word pairs
        probes = [w for w in ev_words if len(w) >= 5]
        probes += [" ".join(ev_words[i:i+2]) for i in range(len(ev_words) - 1)]

        for probe in probes:
            for i, line in enumerate(lines):
                if probe in line.lower():
                    hit_counts[i] += 1

    best = max(range(len(lines)), key=lambda i: hit_counts[i])
    return best if hit_counts[best] > 0 else -1


def score_round(
    text: str,
    round_number: int = None,
    battler: str = None,
    threshold: float = 0.4,
    negative_threshold: float = 0.5,
) -> RoundSummary:
    """
    Score a full round of battle rap with per-line attribution.

    Designed for live transcription pipelines: pass in one battler's
    complete round transcript and receive a structured breakdown of
    every detected skill, the line it fired on, and the reason.

    Args:
        text:               Full round transcript (one battler's turn).
        round_number:       Optional round number (1, 2, 3 …).
        battler:            Optional battler name or identifier.
        threshold:          Min confidence for positive skills (default 0.4).
        negative_threshold: Min confidence for negative skills (default 0.5).

    Returns:
        RoundSummary with per-line breakdown and round totals.
    """
    # Full-round detection (handles multi-line skills like fast break, breakaway)
    result = detect_skills(text, threshold=threshold,
                           negative_threshold=negative_threshold)

    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Build per-line containers
    line_summaries: List[LineSummary] = [
        LineSummary(line_number=i + 1, text=ln, skills=[], line_score=0.0)
        for i, ln in enumerate(raw_lines)
    ]

    # Attribute each detection to its best-matching line
    round_level: List[SkillDetection] = []  # skills with no line match
    for detection in result.detections:
        idx = _best_line_for_detection(detection, raw_lines)
        if idx >= 0:
            line_summaries[idx].skills.append(detection)
            line_summaries[idx].line_score = round(
                line_summaries[idx].line_score + detection.points, 2
            )
        else:
            # Round-level skill — attach to first line as fallback
            if line_summaries:
                line_summaries[0].skills.append(detection)
                line_summaries[0].line_score = round(
                    line_summaries[0].line_score + detection.points, 2
                )
            else:
                round_level.append(detection)

    return RoundSummary(
        round_number=round_number,
        battler=battler,
        lines=line_summaries,
        all_detections=result.detections,
        total_score=result.total_score,
        highlights=result.highlights,
        deductions=result.mistakes,
    )


def score_round_json(
    text: str,
    round_number: int = None,
    battler: str = None,
    threshold: float = 0.4,
    negative_threshold: float = 0.5,
) -> dict:
    """
    JSON-serialisable wrapper for score_round().
    Suitable for direct use with a live transcription API response.
    """
    return score_round(
        text,
        round_number=round_number,
        battler=battler,
        threshold=threshold,
        negative_threshold=negative_threshold,
    ).to_dict()


# ============================================================================
# Backward-Compatible Wrapper
# ============================================================================

def detect_rap_techniques(texts):
    """
    Drop-in replacement for the same function in rap_techniques.py.
    Returns (n, 4) csr_matrix with columns:
        [0] full_court_shot  [1] slam_dunk  [2] half_court_shot  [3] alley_oop

    Accepts a list of strings OR a single string.
    """
    if isinstance(texts, str):
        texts = [texts]

    rows = []
    for text in texts:
        s = text if isinstance(text, str) else ""
        rows.append([
            float(detect_full_court_shot(s)),
            float(detect_slam_dunk(s)),
            float(detect_half_court_shot(s)),
            float(detect_alley_oop(s)),
        ])
    return sparse.csr_matrix(np.asarray(rows, dtype=float))


# ============================================================================
# CLI
# ============================================================================

def _cli():
    import argparse

    # Fix Windows console encoding
    import io
    if hasattr(sys.stdout, 'buffer') and sys.stdout.encoding and \
            sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    ap = argparse.ArgumentParser(
        description="FCPBRL Skill Detection Engine — multi-label battle rap skill scanner"
    )
    group = ap.add_mutually_exclusive_group()
    group.add_argument("text", nargs="?", help="Inline text to analyse")
    group.add_argument("--file", "-f", help="Path to lyrics/transcript file")
    ap.add_argument("--threshold", "-t", type=float, default=0.4,
                    help="Min confidence threshold for positive skills (default: 0.4)")
    ap.add_argument("--neg-threshold", type=float, default=0.5,
                    help="Min confidence for negative skills/fouls (default: 0.5)")
    ap.add_argument("--json", "-j", action="store_true",
                    help="Output raw JSON instead of formatted text")
    ap.add_argument("--score-round", "-r", action="store_true",
                    help="Per-line scoring mode (for live transcription)")
    ap.add_argument("--round-number", type=int, default=None,
                    help="Round number to include in output (e.g. 1, 2, 3)")
    ap.add_argument("--battler", type=str, default=None,
                    help="Battler name to include in output")
    ap.add_argument("--list-skills", action="store_true",
                    help="Print all registered skills and exit")
    args = ap.parse_args()

    if args.list_skills:
        print(f"{'ID':<6} {'Name':<28} {'Pts':>6}  {'Category'}")
        print("-" * 60)
        for key, sid, name, pts, cat, thr in SKILL_REGISTRY:
            sign = "+" if pts >= 0 else ""
            print(f"{sid:<6} {name:<28} {sign}{pts:>5.2f}  {cat}")
        return

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    elif args.text:
        text = args.text
    else:
        ap.print_help()
        sys.exit(1)

    if args.score_round:
        summary = score_round(
            text,
            round_number=args.round_number,
            battler=args.battler,
            threshold=args.threshold,
            negative_threshold=args.neg_threshold,
        )
        if args.json:
            print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        else:
            hdr = "FCPBRL ROUND SCORE"
            if summary.battler:
                hdr += f" — {summary.battler}"
            if summary.round_number is not None:
                hdr += f"  (Round {summary.round_number})"
            print("=" * 70)
            print(hdr)
            print("=" * 70)
            for ln in summary.lines:
                if ln.skills:
                    print(f"\nLine {ln.line_number:>3}: {ln.text[:80]}")
                    for sk in ln.skills:
                        sign = "+" if sk.points >= 0 else ""
                        reason = sk.evidence[0][:70] if sk.evidence else "—"
                        print(f"          [{sk.skill_id}] {sk.skill_name:<25} "
                              f"{sign}{sk.points:.2f}  conf={sk.confidence:.0%}")
                        print(f"          reason: {reason}")
            print()
            print(f"TOTAL SCORE: {summary.total_score:+.2f}  "
                  f"({len(summary.highlights)} highlights / "
                  f"{len(summary.deductions)} deductions)")
        return

    result = detect_skills(text, threshold=args.threshold,
                            negative_threshold=args.neg_threshold)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print("=" * 70)
        print("FCPBRL SKILL SCAN")
        print("=" * 70)
        print(str(result))
        print()
        print(f"[{len(result.detections)} skill(s) detected across "
              f"{len(result.highlights)} highlight(s) / {len(result.mistakes)} deduction(s)]")


if __name__ == "__main__":
    _cli()
