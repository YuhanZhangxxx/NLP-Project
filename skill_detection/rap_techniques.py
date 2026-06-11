"""
Rap technique detection module.
Each technique has its own detector function for maintainability and extension.
"""
import re
import numpy as np
from scipy import sparse

# Try to import NLTK for POS tagging (Method 2: POS-based detection)
try:
    import nltk
    from nltk import pos_tag, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    # Ensure required NLTK data is downloaded
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            try:
                nltk.download('punkt', quiet=True)
            except:
                pass
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        try:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        except:
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                try:
                    nltk.download('averaged_perceptron_tagger', quiet=True)
                except:
                    pass
except ImportError:
    NLTK_AVAILABLE = False


# ============================================================================
# Full-Court Shot (+5.0 Points) - high-risk, high-reward bar
# ============================================================================

# Rare strong words (high weight, clearly indicating Full-Court Shot)
FULL_COURT_STRONG_KEYWORDS = {
    'documentary', 'investigation', 'autopsy', 'missing persons',
    'museum', 'exhibit', 'exhibition', 'gallery', 'showcase',
    'footage', 'premiere', 'premiered', 'aired', 'broadcast',
    'disappearance', 'vanished', 'erased', 'flatlined', 'folded', 'logged off'
}

# Medium-strength words (medium weight)
FULL_COURT_MEDIUM_KEYWORDS = {
    'camera', 'clip', 'clipped', 'replay', 'recording', 'video', 'tape', 'film',
    'display', 'installation', 'opened', 'opening', 'unveiled', 'debuted', 'launched',
    'spirit', 'energy', 'soul', 'ego', 'presence', 'essence', 'hype',
    'disappeared', 'faded', 'gone', 'empty', 'body', 'left', 'went'
}

# Generic words (low weight or potentially penalized)
FULL_COURT_WEAK_KEYWORDS = {
    'moment', 'scene', 'shot', 'frame', 'got', 'made', 'created', 'built',
    'released', 'published', 'streamed', 'pulled', 'grabbed', 'took'
}

# Victim-state keywords (at least one category must appear)
VICTIM_STATE_KEYWORDS = {
    # Disappearance / death / collapse
    'vanished', 'disappeared', 'erased', 'flatlined', 'folded', 'logged off',
    'died', 'dead', 'gone', 'faded', 'crumbled', 'collapsed', 'vanish',
    # Humiliation imagery / states
    'empty chair', 'autopsy', 'missing persons', 'wanted', 'judgment', 'verdict',
    'coroner', 'morgue', 'body bag', 'crime scene', 'evidence',
    # Abstract concepts (that can be crushed)
    'confidence', 'ego', 'pride', 'bravado', 'hype', 'aura', 'momentum',
    'tough talk', 'voice cracked', 'stumbled', 'paused', 'folded', 'froze',
    # Actions / states (indicating being crushed)
    'repossessed', 'stolen', 'deleted', 'rerouted', 'packed', 'shipped',
    'rage quit', 'witness protection', 'mission failure', 'loss',
    # Environment / atmosphere shifts (indicating opponent is suppressed)
    'went quiet', 'went silent', 'silence', 'quiet', 'stunned', 'shook'
}

# Capture/construction patterns (compact single-line form to avoid regex whitespace issues; subject "I" may be omitted)
FULL_COURT_CREATION_PATTERNS = [
    r'\b(pulled|grabbed|captured|caught|snapped|clipped|recorded|ran|ran\s+the|got|saved)\s+(the|that|this|a|an)?\s*\w*\s*(footage|replay|recording|video|tape|film|moment|shot|scene|clip|feed)',  # Added got, saved, feed
    r'\b(pulled|grabbed|captured|caught|snapped|clipped|recorded|got)\s+(the|that|this|a|an)?\s*(gps\s+history)',
    r'\b(built|constructed|assembled|erected|created)\s+(the|a|an)?\s*\w*\s*(museum|exhibit|exhibition|display|gallery|showcase|installation)',
    r'\b(printed|wrote|filed|issued|opened|logged|tracked|made|created|pushed|posted|curated|read)\s+(the|a|an)?\s*\w*\s*(receipts|paperwork|chart|report|bulletin|ticket|update|order|reel|lesson|timeline|exhibit|chapter|study|screen\s+capture|verdict|alert)',
    r'\b(pulled|grabbed|captured|caught|snapped|clipped|recorded|printed|wrote|filed|issued|opened|logged|tracked|made|created|pushed|posted|curated|read|got)\s+(the|that|this|a|an)?\s*(gps\s+history)',
]

# Direct capture signals (no following footage-type noun required)
FULL_COURT_DIRECT_CAPTURE_PATTERNS = [
    r'\b(screen\s+recorded|screen\s+captured|screen\s+capture|recorded)\b',  # Direct hit on screen recorded/captured
]

# Title-naming patterns (improved: no reliance on quotes/capitalization; boundaries use connectors/end-of-string)
FULL_COURT_TITLE_PATTERNS = [
    r'\b(titled|named|called|labeled|tagged)\s+(it|this|that|the)\s+([^\s]+(?:\s+[^\s]+)*?)(?:\s+(?:and|then|so|$))',  # Capture up to and/then/so/end
    r'\b(titled|named|called|labeled|tagged)\s+(it|this|that|the)\s+([^\s]+(?:\s+[^\s]+)*?)(?=\s+(?:and|then|so|dropped|released|opened|published|aired|premiered|streamed|uploaded|unveiled|put\s+out|broadcast|$))',  # Stricter boundary, including aired, unveiled, put out, broadcast
    r'\b(titled|named|called|labeled|tagged)\s+(it|this|that|the)\s+\w+',  # Fallback: simple match
]

# Release/opening patterns (compact single-line form; added series/season/cut/report/case file)
FULL_COURT_RELEASE_PATTERNS = [
    r'\b(dropped|released|unveiled|debuted|aired|broadcast|streamed|premiered|uploaded|published|put\s+out)\s+(the|a|an)?\s*\w*\s*(documentary|investigation|film|video|tape|episode|show|report|findings|case\s+file|season|evidence|patch\s+notes|logs|screenshot|travel\s+doc|storm\s+report|unboxing|mission\s+documentary|educational\s+documentary|directors\s+cut|series|trial\s+documentary|full\s+series|cut|case|doc|file|special)',  # Added put out, special
    r'\b(opened|opening|launched|presented|attached|included|showed|pinned|set|played|held|posted|unveiled)\s+(the|a|an)?\s*\w*\s*(exhibit|exhibition|display|museum|gallery|show|conference|documentary|evidence|logs|screenshot|receipts|everywhere|update|on\s+repeat|it\s+downtown|it)',  # Added unveiled it
]

# Victim-state patterns (must appear)
VICTIM_STATE_PATTERNS = [
    # Direct disappearance/death vocabulary
    r'\b(vanished|disappeared|erased|flatlined|folded|logged\s+off|died|dead|gone|faded|crumbled|collapsed|vanish)',
    # Humiliation imagery
    r'\b(empty\s+chair|autopsy|missing\s+persons|wanted|judgment|verdict|coroner|morgue|body\s+bag|crime\s+scene)',
    # Abstract concept + disappearance action (stricter match to avoid things like "energy was different")
    r'\b(spirit|energy|soul|ego|presence|essence|hype|confidence|pride|bravado|aura|momentum|tough\s+talk|voice|bars|act|legend|performance|scoreboard)\s+(left|went|fled|vanished|disappeared|faded|was\s+(already\s+)?gone|is\s+gone|got\s+(stolen|deleted|rerouted|repossessed|packed|shipped)|already\s+gone|cracked|stumbled|paused|folded|rage\s+quit|ended|turned\s+into|did\s+not\s+even\s+need)',
    r'\b(confidence|ego|pride|bravado|hype|aura|momentum|tough\s+talk|voice|bars|act|legend|performance)\s+(vanish|vanished|disappeared|gone|stolen|deleted|rerouted|repossessed|cracked|stumbled|paused|folded|rage\s+quit|ended|turned\s+into)',
    # Avoid matching weak expressions like "energy was different"
    r'\b(energy|vibe|momentum)\s+was\s+(different|off|strange)',  # This pattern is for exclusion, not matching
    # Disappearance + media
    r'\b(erased|vanished|disappeared|faded|gone)\s+(on|from|in)\s+(camera|film|tape|video|recording)',
    # Transformed into evidence/files
    r'\b(turned|converted|transformed)\s+\w+\s+(into|to)\s+(evidence|case\s+file|paperwork|report|document)',
    # Sent/entered protection/failure
    r'\b(sent|went|got)\s+\w+\s+(into|to)\s+(witness\s+protection|evidence|mission\s+failure)',
    # Specific crushing expressions
    r'\b(confidence|ego|pride)\s+rage\s+quit',
    r'\b(scoreboard|stage)\s+(did\s+not\s+even\s+need|turned\s+into)',
    r'\b(momentum|act)\s+got\s+(rerouted|packed|shipped)',
    r'\b(way\s+you\s+faded|class\s+is\s+gonna\s+study\s+that\s+loss)',
    r'\b(stage|courtroom)\s+turned\s+into',
    r'\b(momentum)\s+got\s+rerouted',
    r'\b(way\s+you\s+faded|you\s+faded)',
    # Direct action (you + crushing verb)
    r'\b(you|your)\s+(stumbled|froze|paused|folded|crashed|went\s+quiet|vanished|disappeared|erased|faded|gone)',
    r'\b(you|your)\s+(confidence|ego|pride|bravado|hype|aura|momentum)\s+(crashed|vanished|disappeared|erased|faded|gone|stolen|deleted)',
]

# Penalty words (presence triggers score reduction, especially inside titles)
VAGUENESS_PENALTY_WORDS = {
    'something', 'stuff', 'maybe', 'interesting', 'chapter', 'timeline', 
    'highlights', 'things', 'whatever', 'kinda', 'sort of', 'a moment',
    'a thing', 'a chapter', 'the timeline', 'the exhibit', 'the highlights',
    'watch this', 'news today', 'big event', 'evidence', 'missing', 'update complete',
    'bug found', 'need help', 'travel time', 'money talk', 'lessons',
    'camera time', 'a moment in time', 'some kind of', 'somehow', 'did a thing',
    'felt like', 'somewhere', 'was loud', 'exists', 'changed', 'moved around',
    'felt expensive', 'showed everything', 'so deep', 'needed', 'could measure',
    'was long', 'was off', 'was different', 'was strange'
}

def detect_victim_state_by_pos(text: str) -> bool:
    """
    Method 2: Detect victim-state semantics via POS tagging.

    Detection patterns:
    1. Past-tense verbs (VBD/VBN) indicating disappearance/failure actions
    2. Certain adjectives (JJ) indicating failure states
    3. Certain nouns (NN) representing failure/disappearance concepts
    4. Specific grammar structure: you/your + past-tense verb

    Returns:
        bool: Whether victim-state semantics were detected
    """
    if not NLTK_AVAILABLE:
        return False
    
    try:
        # Tokenization and POS tagging
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)

        # Verbs denoting victim-state semantics (base form and past/past-participle)
        defeat_verbs = {
            # Base form
            'vanish', 'disappear', 'erase', 'fade', 'collapse', 'crumble',
            'fold', 'fail', 'lose', 'fall', 'crash', 'break', 'destroy',
            'eliminate', 'defeat', 'overcome', 'conquer', 'crush', 'stumble',
            'freeze', 'pause', 'stop', 'end', 'die', 'flatline', 'quit',
            # Past tense / past participle
            'vanished', 'disappeared', 'erased', 'faded', 'collapsed', 'crumbled',
            'folded', 'failed', 'lost', 'fell', 'crashed', 'broke', 'broken',
            'destroyed', 'eliminated', 'defeated', 'overcame', 'overcome', 'conquered',
            'crushed', 'stumbled', 'froze', 'frozen', 'paused', 'stopped', 'ended',
            'died', 'dead', 'flatlined', 'quit', 'quitted'
        }
        
        # Adjectives denoting victim-state semantics
        defeat_adjectives = {
            'gone', 'dead', 'empty', 'lost', 'defeated', 'broken', 'crushed',
            'destroyed', 'eliminated', 'finished', 'over', 'done', 'stunned',
            'shocked', 'silent', 'quiet'
        }
        
        # Nouns denoting victim-state semantics
        defeat_nouns = {
            'loss', 'defeat', 'failure', 'collapse', 'end', 'death', 'autopsy',
            'evidence', 'verdict', 'judgment', 'missing', 'disappearance'
        }
        
        # Check POS patterns
        for i, (word, pos) in enumerate(pos_tags):
            word_lower = word.lower()

            # 1. Past-tense (VBD) or past-participle (VBN) verbs indicating disappearance/failure
            if pos in ['VBD', 'VBN'] and word_lower in defeat_verbs:
                # Check context: preceded by you/your, or followed by a crushing complement
                if i > 0:
                    prev_word = pos_tags[i-1][0].lower()
                    if prev_word in ['you', 'your']:
                        return True
                if i < len(pos_tags) - 1:
                    next_pos = pos_tags[i+1][1]
                    if next_pos in ['NN', 'NNS', 'DT']:  # Followed by noun or determiner
                        return True
                return True

            # 2. Adjective (JJ) indicating failure state
            if pos == 'JJ' and word_lower in defeat_adjectives:
                # Check whether inside "was/got/became + adjective" structure
                if i > 0:
                    prev_word = pos_tags[i-1][0].lower()
                    if prev_word in ['was', 'got', 'became', 'turned', 'went']:
                        return True
                return True

            # 3. Noun (NN/NNS) indicating failure/disappearance concept
            if pos in ['NN', 'NNS'] and word_lower in defeat_nouns:
                # Check whether inside "into/to + noun" or "the/a/an + noun" structure
                if i > 0:
                    prev_word = pos_tags[i-1][0].lower()
                    if prev_word in ['into', 'to', 'the', 'a', 'an']:
                        return True
                return True

            # 4. Specific structure: you/your + past-tense verb
            if word_lower in ['you', 'your'] and i < len(pos_tags) - 1:
                next_word, next_pos = pos_tags[i+1]
                if next_pos in ['VBD', 'VBN'] and next_word.lower() in defeat_verbs:
                    return True

        # 5. Check sequential pattern: abstract concept + past-tense verb
        for i in range(len(pos_tags) - 1):
            word1, pos1 = pos_tags[i]
            word2, pos2 = pos_tags[i+1]

            # Abstract noun (NN) + past-tense verb (VBD/VBN)
            abstract_nouns = {'confidence', 'ego', 'pride', 'hype', 'aura', 'momentum', 'spirit', 'energy'}
            if (pos1 in ['NN', 'NNS'] and word1.lower() in abstract_nouns and
                pos2 in ['VBD', 'VBN'] and word2.lower() in defeat_verbs):
                return True

        return False

    except Exception:
        # If NLTK processing fails, return False (fallback to Method 1)
        return False

def normalize_text(text: str) -> str:
    """
    Text normalization:
    1. Lowercase everything
    2. Merge hyphens: screen-recorded -> screen recorded
    3. Collapse multiple spaces into one
    4. Remove quotes (to handle unpunctuated text)
    """
    s = text.lower()
    # Merge hyphenated words
    s = re.sub(r'([a-z]+)-([a-z]+)', r'\1 \2', s)
    # Collapse multiple spaces into one
    s = re.sub(r'\s+', ' ', s)
    # Remove quotes
    s = s.replace('"', '').replace("'", '')
    return s.strip()

def detect_full_court_shot(text: str) -> float:
    """
    Detect the Full-Court Shot technique.
    High-risk, high-reward bar: an unexpected knockout that converts an event into media content.

    Core requirements:
    1. Must contain victim-state semantics
    2. Must contain the full pipeline (creation + naming + release)
    3. The title/naming step is the key highlight

    Grammatical features (strongest signals):
    - Triple-verb construction: I + VERB1 + OBJ1 + VERB2 + OBJ2 + VERB3 + OBJ3
    - Two-part structure: time frame + opponent state + I + triple action chain
    - Connector preference: and, and then, so I
    - Person configuration: you/your (victim) + I (producer)
    """
    # Text normalization
    s_lower = normalize_text(text)
    score = 0.0

    # ========================================================================
    # Hard gate 1: victim-state semantics must appear
    # ========================================================================
    has_victim_state = False

    # ========================================================================
    # Method 1: keyword-based detection (original method)
    # ========================================================================
    # Check victim-state keywords (supports multi-word phrases)
    for kw in VICTIM_STATE_KEYWORDS:
        if ' ' in kw:
            # Multi-word phrase: use regex match
            if re.search(r'\b' + re.escape(kw) + r'\b', s_lower):
                has_victim_state = True
                break
        else:
            # Single word: direct lookup
            if kw in s_lower:
                has_victim_state = True
                break

    # Check victim-state patterns
    if not has_victim_state:
        for pattern in VICTIM_STATE_PATTERNS:
            # Skip exclusion patterns (used for negative checks)
            if r'was\s+(different|off|strange)' in pattern:
                continue
            if re.search(pattern, s_lower):
                has_victim_state = True
                break

    # ========================================================================
    # Method 2: POS-based detection (new method, parallel to Method 1)
    # ========================================================================
    if not has_victim_state and NLTK_AVAILABLE:
        has_victim_state = detect_victim_state_by_pos(text)

    # If victim-state semantics still not found, check whether the title contains victim-state vocabulary.
    # This is especially important for concise forms (omitting subject "I"); the title itself may carry victim-state semantics.
    if not has_victim_state:
        # Extract the title text
        title_text = ""
        for pattern in FULL_COURT_TITLE_PATTERNS:
            match = re.search(pattern, s_lower)
            if match:
                # Try to extract the title from matched groups (if the pattern has capture groups)
                if len(match.groups()) >= 3:
                    # The third group is typically the title
                    title_text = match.group(3).strip()
                elif len(match.groups()) >= 1:
                    # If there's only one group, it may be the title
                    title_text = match.group(1).strip()

                # If group-based extraction fails, extract starting from match.end()
                if not title_text or len(title_text) < 2:
                    title_start = match.end()
                    # Find the title end (next verb or connector)
                    title_end_match = re.search(r'\s+(and|then|so|dropped|released|opened|published|aired|premiered|streamed|uploaded|$)', s_lower[title_start:])
                    if title_end_match:
                        title_text = s_lower[title_start:title_start + title_end_match.start()].strip()
                    else:
                        # If no end marker is found, take to end of sentence or up to 50 characters
                        remaining_text = s_lower[title_start:]
                        # Use the next major verb as boundary
                        next_verb_match = re.search(r'\s+(dropped|released|opened|published|aired|premiered|streamed|uploaded)', remaining_text)
                        if next_verb_match:
                            title_text = remaining_text[:next_verb_match.start()].strip()
                        else:
                            title_text = remaining_text[:50].strip()  # up to 50 characters
                break

        # Check whether the title contains victim-state vocabulary
        if title_text:
            # Check victim-state keywords within the title (including full phrases and single words)
            title_victim_keywords = {
                'disappearance', 'disappeared', 'vanished', 'erased', 'folded',
                'logout', 'heist', 'fold', 'autopsy', 'missing', 'exact frame',
                'where', 'went', 'fall', 'exhibit', 'guilty', 'wrong turn',
                'severe', 'collapse', 'return', 'sender', 'abort', 'fold on',
                'the logout', 'energy heist', 'the fold', 'live disappearance',
                'rap autopsy', 'exact frame', 'where his', 'where your',
                'the fall of', 'exhibit a', 'missing since', 'wrong turn',
                'severe confidence', 'return to sender', 'abort confidence',
                'fold on beat'
            }
            # First check full multi-word phrases
            for kw in title_victim_keywords:
                if ' ' in kw:
                    # Multi-word phrase: use regex match
                    if re.search(r'\b' + re.escape(kw) + r'\b', title_text):
                        has_victim_state = True
                        break
                else:
                    # Single word: direct lookup
                    if kw in title_text:
                        has_victim_state = True
                        break

    # Exclude weak expressions (e.g. "energy was different") - these don't count as victim-state semantics
    weak_expressions = [
        r'\b(energy|vibe|momentum)\s+was\s+(different|off|strange)',
        r'\b(moment|performance)\s+(was|did)\s+(loud|a\s+thing)',
        r'\b(crowd|room)\s+(reacted|felt)\s+(somehow|some\s+kind\s+of)',
        r'\b(room|stage|moment)\s+felt\s+(some\s+kind\s+of|like)',
        r'\b(energy|vibe|momentum)\s+(was|moved)\s+(different|around|somewhere)',
        r'\b(performance|moment)\s+did\s+a\s+thing',
        r'\b(moment|footage)\s+was\s+(loud|exists)',
        r'\b(round|things)\s+(was|were)\s+(intense|changed)',
        r'\b(confidence|vibe)\s+had\s+a\s+moment',
        r'\b(something)\s+was\s+off',
        r'\b(energy)\s+moved\s+around',
        r'\b(performance)\s+felt\s+expensive',
        r'\b(cameras)\s+showed\s+everything',
        r'\b(story)\s+was\s+so\s+deep',
        r'\b(highlight\s+reel)\s+was\s+long',
        r'\b(stage)\s+felt\s+like\s+history',
        r'\b(hype)\s+was\s+somewhere',
    ]
    # Any ONE weak phrase cancels the victim-state claim — first match wins.
    for weak_pattern in weak_expressions:
        if re.search(weak_pattern, s_lower):
            has_victim_state = False
            break
    
    # Hard gate: must have victim-state semantics to continue (removed pipeline-only fallback to avoid false positives)
    # Only sentences with genuine victim-state semantics pass; pipeline alone is insufficient
    if not has_victim_state:
        return 0.0

    # ========================================================================
    # Hard gate 2: pipeline detection (promoted to a hard-rule fallback)
    # ========================================================================
    # Detect the creation step (including direct-capture signals)
    has_creation = False
    for pattern in FULL_COURT_CREATION_PATTERNS:
        if re.search(pattern, s_lower):
            has_creation = True
            break

    # Detect direct-capture signals (e.g. screen recorded; no following footage-type noun required)
    if not has_creation:
        for pattern in FULL_COURT_DIRECT_CAPTURE_PATTERNS:
            if re.search(pattern, s_lower):
                has_creation = True
                break

    # Detect the naming step (the title is the key highlight)
    has_naming = False
    for pattern in FULL_COURT_TITLE_PATTERNS:
        if re.search(pattern, s_lower):
            has_naming = True
            break

    # Detect the release step
    has_release = False
    for pattern in FULL_COURT_RELEASE_PATTERNS:
        if re.search(pattern, s_lower):
            has_release = True
            break

    # Pipeline hard rule: if all three steps are present, give a high score (victim-state must already be present)
    if has_creation and has_naming and has_release:
        # Full three-step pipeline + victim-state, directly assign high score
        score = 0.85  # Hard-rule fallback to ensure sentences with full pipeline and victim-state are not missed
    elif has_creation and has_naming:
        # Two-step pipeline (including naming), moderate base score
        score = 0.4
    else:
        # No complete pipeline or missing naming, return low score
        if not (has_creation and has_naming):
            return 0.0
    
    # ========================================================================
    # Weighted keyword scoring (rare words weighted high, generic words low)
    # ========================================================================
    # Rare strong words (high weight)
    strong_count = sum(1 for kw in FULL_COURT_STRONG_KEYWORDS if kw in s_lower)
    score += min(strong_count * 0.12, 0.25)  # 0.12 per match, capped at 0.25

    # Medium-strength words (medium weight)
    medium_count = sum(1 for kw in FULL_COURT_MEDIUM_KEYWORDS if kw in s_lower)
    score += min(medium_count * 0.05, 0.15)  # 0.05 per match, capped at 0.15

    # Generic words (low weight, potentially penalized)
    weak_count = sum(1 for kw in FULL_COURT_WEAK_KEYWORDS if kw in s_lower)
    if weak_count > 3:  # Too many generic words: likely ordinary narration
        score -= min((weak_count - 3) * 0.03, 0.1)  # Apply penalty

    # Weak-object penalty (timeline/highlights/chapter, etc., to avoid false positives)
    weak_objects = {'timeline', 'highlights', 'chapter', 'story', 'part', 'segment'}
    weak_object_count = sum(1 for obj in weak_objects if obj in s_lower)
    if weak_object_count > 0:
        # If only weak objects with no strong media/exhibit objects, lower the cap
        strong_media_objects = {'documentary', 'investigation', 'report', 'exhibit', 'exhibition', 'museum', 'film', 'video', 'case file'}
        has_strong_object = any(obj in s_lower for obj in strong_media_objects)
        if not has_strong_object:
            score = min(score, 0.5)  # Weak object without strong object: cap at 0.5
    
    # ========================================================================
    # Negative penalty: vagueness-word detection (especially inside titles)
    # ========================================================================
    # Check vagueness words inside the title (most severe; penalize heavily or reject)
    title_vagueness = False
    title_vagueness_words = []
    for pattern in FULL_COURT_TITLE_PATTERNS:
        match = re.search(pattern, s_lower)
        if match:
            # Extract the title text (from titled/named/called up to and/then/so/end)
            title_start = match.end()
            # Find title end position (and/then/so/end)
            title_end_match = re.search(r'\s+(and|then|so|dropped|released|opened|published|aired|$)', s_lower[title_start:])
            if title_end_match:
                title_text = s_lower[title_start:title_start + title_end_match.start()]
            else:
                title_text = s_lower[title_start:title_start + 50]  # up to 50 characters

            # Check whether the title contains vagueness words
            for word in VAGUENESS_PENALTY_WORDS:
                if word in title_text:
                    title_vagueness = True
                    title_vagueness_words.append(word)
                    break
            if title_vagueness:
                break

    # Vagueness word in the title: apply a large penalty or reject outright
    if title_vagueness:
        score -= 0.6  # Large penalty for vagueness word in title (raised from 0.4 to 0.6)
        # If score drops too low, reject outright
        if score < 0.3:
            return 0.0

    # Check vagueness words elsewhere in the text
    vagueness_count = sum(1 for word in VAGUENESS_PENALTY_WORDS if word in s_lower)
    if vagueness_count > 0 and not title_vagueness:
        score -= min(vagueness_count * 0.2, 0.4)  # Vagueness words outside the title (raised from 0.15 to 0.2)

    # Check weak-expression patterns (apply penalty even if victim-state check passed)
    weak_expression_penalties = [
        (r'\b(some\s+kind\s+of|somehow|did\s+a\s+thing|felt\s+like|was\s+off|was\s+different)', 0.3),
        (r'\b(made\s+content\s+about|noticed\s+it|saw\s+it)', 0.2),
        (r'\b(felt|reacted|moved|changed|existed|showed|needed|could\s+measure|was\s+long)', 0.15),
    ]
    for weak_pattern, penalty in weak_expression_penalties:
        if re.search(weak_pattern, s_lower):
            score -= penalty
            break
    
    # ========================================================================
    # Grammatical feature detection (strongest signals) - based on linguistic analysis
    # ========================================================================

    # 1. Detect the "triple-verb construction": I + VERB1 + OBJ1 + VERB2 + OBJ2 + VERB3 + OBJ3
    # This is the most characteristic grammatical feature of Full-Court Shot
    # Also supports subject-omitted concise forms: VERB1 + OBJ1 + VERB2 + OBJ2 + VERB3 + OBJ3
    verb1_pattern = r'\b(pulled|grabbed|captured|caught|snapped|clipped|recorded|ran|got|printed|wrote|filed|issued|opened|logged|tracked|made|created|pushed|posted|built|constructed|assembled|erected)\s+'
    verb2_pattern = r'\b(titled|named|called|labeled|tagged)\s+(it|this|that|the)\s+'
    verb3_pattern = r'\b(dropped|released|unveiled|debuted|aired|broadcast|streamed|premiered|uploaded|published|opened|launched|presented)\s+'
    
    # Detect the triple-verb pattern (allowing connectors and a few other words in between)
    # Supports both with and without subject "I"
    triple_verb_pattern_with_i = r'\bi\s+' + verb1_pattern + r'[^.]*?' + verb2_pattern + r'[^.]*?' + verb3_pattern
    triple_verb_pattern_no_i = verb1_pattern + r'[^.]*?' + verb2_pattern + r'[^.]*?' + verb3_pattern  # Subject "I" omitted

    if re.search(triple_verb_pattern_with_i, s_lower, re.IGNORECASE) or re.search(triple_verb_pattern_no_i, s_lower, re.IGNORECASE):
        score += 0.3  # Triple-verb construction is a strong signal

    # 2. Detect time-frame openings (makes sentence read more like news broadcast)
    time_frame_patterns = [
        r'\b(by\s+the\s+end\s+of|the\s+moment|as\s+soon\s+as|after\s+that\s+round|mid\s+battle|soon\s+as|the\s+second|when\s+the|that\s+round|one\s+beat\s+switch\s+later|by\s+round\s+(two|three|four|five|\d+)|once)',
    ]
    has_time_frame = any(re.search(p, s_lower) for p in time_frame_patterns)
    if has_time_frame:
        score += 0.1
    
    # 3. Detect two-part structure: time frame + opponent state + I + triple action chain
    # Detect "your X was Y" or "your X vanished/disappeared" followed by "I + action"
    two_part_structure = re.search(
        r'(?:by\s+the\s+end|the\s+moment|after|mid|when|that\s+round|soon\s+as).*?your\s+\w+\s+(?:was|got|left|went|vanished|disappeared|erased|faded|gone|got\s+(?:stolen|deleted|rerouted|repossessed|packed|shipped))[^.]*?\bi\s+',
        s_lower,
        re.IGNORECASE
    )
    if two_part_structure:
        score += 0.15  # Two-part structure is a characteristic feature

    # 4. Detect connector preferences (and, and then, so I)
    connector_patterns = [
        r'\band\s+and\s+and',  # Repeated and
        r'\band\s+then',  # and then
        r'\bso\s+i\s+',  # so I
        r'\band\s+i\s+',  # and I
    ]
    connector_count = sum(1 for p in connector_patterns if re.search(p, s_lower))
    if connector_count > 0:
        score += min(connector_count * 0.05, 0.1)  # Connectors reinforce pipeline feel

    # 5. Detect person configuration: you/your (victim) + I (producer)
    has_you = bool(re.search(r'\b(you|your)\s+', s_lower))
    has_i = bool(re.search(r'\bi\s+', s_lower))
    if has_you and has_i:
        score += 0.1  # Person configuration creates pressure

    # 6. Detect title grammar features (noun-phrase or clause-abbreviated titles)
    title_grammar_patterns = [
        r'where\s+(his|the|your)\s+\w+\s+went',  # Where His X Went
        r'the\s+day\s+(he|you|they)\s+\w+',  # The Day He Y
        r'live\s+\w+',  # Live Noun
        r'missing\s+since\s+\w+',  # Missing Since Noun
        r'\w+\s+autopsy',  # X Autopsy
        r'\w+\s+heist',  # X Heist
        r'the\s+fall\s+of',  # The Fall Of
        r'exhibit\s+[a-z]',  # Exhibit A
        r'the\s+exact\s+frame',  # The Exact Frame
    ]
    title_grammar_count = sum(1 for p in title_grammar_patterns if re.search(p, s_lower))
    if title_grammar_count > 0:
        score += min(title_grammar_count * 0.08, 0.15)  # Title grammar features

    # ========================================================================
    # Extra bonus items
    # ========================================================================
    # Detect the strong "disappearance + media" combination
    if re.search(r'\b(erased|vanished|disappeared|faded|gone)\s+(on|from|in)\s+(camera|film|tape|video|recording)', s_lower):
        score += 0.1

    # Detect unexpected turns
    if re.search(r'\b(but|yet|however|though|although)\s+', s_lower):
        score += 0.05

    # Clamp score to a reasonable range
    return max(0.0, min(score, 1.0))


# ============================================================================
# Slam Dunk (+4.25 Points) - concussive knockout bar
# ============================================================================

SLAM_DUNK_KEYWORDS = {
    'legendary', 't-shirt', 'remember', 'die', 'death', 'kill', 'murder', 'dead',
    'only', 'every time', 'somebody', 'gotta', 'gonna', 'must', 'first',
    'devastating', 'haymaker', 'instant', 'explosive', 'reaction'
}

SLAM_DUNK_PATTERNS = [
    r'\bonly\s+\w+\s+on\s+(a|the)\s+\w+',
    r'\bevery\s+time\s+they?\s+\w+',
    r'\bsomebody\s+(gotta|gonna|must|has\s+to)\s+\w+',
    r'\b(gotta|gonna|must)\s+(die|kill|murder)',
    r'\b(legendary|famous|great)\s+.*\s+(but|yet|only)',
]

def detect_slam_dunk(text: str) -> float:
    """
    Detect the Slam Dunk technique.
    Concussive knockout bar, instant explosive reaction.
    """
    s_lower = text.lower()
    score = 0.0

    # Keyword match
    keyword_count = sum(1 for kw in SLAM_DUNK_KEYWORDS if kw in s_lower)
    score += min(keyword_count * 0.12, 0.4)

    # Pattern match
    pattern_matches = sum(1 for pattern in SLAM_DUNK_PATTERNS if re.search(pattern, s_lower))
    score += min(pattern_matches * 0.25, 0.5)

    # Detect strong contrast structures
    if re.search(r'\b(only|just|merely)\s+\w+\s+.*\s+(but|yet|however)', s_lower):
        score += 0.2

    return min(score, 1.0)


# ============================================================================
# Half-Court Shot (+3.75 Points) - creative risk that lands
# ============================================================================

HALF_COURT_KEYWORDS = {
    'trauma', 'triumph', 'flipped', 'turned', 'transformed', 'rearrange', 'direction',
    'crooked', 'compass', 'deep', 'crowd', 'check', 'cheer', 'creative', 'risk',
    'lands', 'perfectly', 'philosophy', 'metaphor', 'abstract'
}

HALF_COURT_PATTERNS = [
    r'\b(flipped|turned|transformed|changed)\s+\w+\s+(into|to|from)',
    r'\b(bar|line|verse)\s+so\s+deep',
    r'\b(rearrange|change|shift|alter)\s+\w+\s+(direction|course|path)',
    r'\bcrooked\s+\w+',
    r'\b(crowd|audience|people)\s+(ain\'?t|don\'?t)\s+know\s+if',
    r'\b(should|could|might)\s+(cheer|applaud|react)\s+.*\s+or\s+(check|help|worry)',
]

def detect_half_court_shot(text: str) -> float:
    """
    Detect the Half-Court Shot technique.
    Creative risk that lands with a perfect finish.
    """
    s_lower = text.lower()
    score = 0.0

    # Keyword match
    keyword_count = sum(1 for kw in HALF_COURT_KEYWORDS if kw in s_lower)
    score += min(keyword_count * 0.15, 0.4)

    # Pattern match
    pattern_matches = sum(1 for pattern in HALF_COURT_PATTERNS if re.search(pattern, s_lower))
    score += min(pattern_matches * 0.2, 0.4)

    # Detect emotional transformation (trauma -> triumph)
    if re.search(r'\b(trauma|pain|hurt|suffering)\s+.*\s+(into|to|became)\s+(triumph|victory|success|win)', s_lower):
        score += 0.2

    return min(score, 1.0)


# ============================================================================
# Alley-Oop/Assist (+3.5 Points) - multi-person coordination
# ============================================================================

ALLEY_OOP_KEYWORDS = {
    'line', 'him', 'up', 'cool', 'team', 'partner', 'assist', 'tag', 'together',
    'sync', 'crowd', 'synced', 'moment', 'eruption', 'coordinate', 'collaborate'
}

ALLEY_OOP_PATTERNS = [
    r'[\'"]\s*\w+[\'"]\s*:',
    r'\b(line|set|get|put)\s+(him|them|it)\s+up',
    r'\bcool\s*[—\-–]\s*(i\'?ll|i\s+will)',
    r'\b(partner|teammate|crew)\s*:',
    r'\b(tag|team|together|with)\s+\w+',
]

def detect_alley_oop(text: str) -> float:
    """
    Detect the Alley-Oop/Assist technique.
    Multi-person coordination, team collaboration.
    """
    s_lower = text.lower()
    score = 0.0

    # Keyword match
    keyword_count = sum(1 for kw in ALLEY_OOP_KEYWORDS if kw in s_lower)
    score += min(keyword_count * 0.12, 0.3)

    # Pattern match
    pattern_matches = sum(1 for pattern in ALLEY_OOP_PATTERNS if re.search(pattern, s_lower))
    score += min(pattern_matches * 0.3, 0.5)

    # Detect quoted dialogue (marker of multi-person coordination)
    quoted_dialogue = len(re.findall(r'[\'"].*?[\'"]', text))
    if quoted_dialogue >= 2:
        score += 0.2

    # Detect imperative language
    if re.search(r'\b(line|set|get|put)\s+(him|them|it|you)\s+up', s_lower):
        score += 0.2

    return min(score, 1.0)


# ============================================================================
# Main function: detect all techniques
# ============================================================================

def detect_rap_techniques(texts):
    """
    Detect all rap technique features.

    Returns an (n, 4) sparse matrix where each column is the intensity score (0-1) for a technique:
    - Column 0: Full-Court Shot
    - Column 1: Slam Dunk
    - Column 2: Half-Court Shot
    - Column 3: Alley-Oop/Assist

    Args:
        texts: list of text strings

    Returns:
        scipy.sparse.csr_matrix: sparse matrix of shape (len(texts), 4)
    """
    # Handle single string input (not just lists)
    if isinstance(texts, str):
        texts = [texts]
    rows = []
    
    for text in texts:
        s = text if isinstance(text, str) else ""
        
        # Call each technique's detector function
        full_court_score = detect_full_court_shot(s)
        slam_dunk_score = detect_slam_dunk(s)
        half_court_score = detect_half_court_shot(s)
        alley_oop_score = detect_alley_oop(s)
        
        rows.append([full_court_score, slam_dunk_score, half_court_score, alley_oop_score])
    
    A = np.asarray(rows, dtype=float)
    return sparse.csr_matrix(A)


# ============================================================================
# Technique info (for display and analysis)
# ============================================================================

TECHNIQUE_INFO = {
    "full_court_shot": {
        "name": "Full-Court Shot",
        "points": "+5.0 Points",
        "description": "High-risk, high-reward bar: an unexpected knockout"
    },
    "slam_dunk": {
        "name": "Slam Dunk",
        "points": "+4.25 Points",
        "description": "Concussive knockout bar with instant explosive reaction"
    },
    "half_court_shot": {
        "name": "Half-Court Shot",
        "points": "+3.75 Points",
        "description": "Creative risk that lands with a perfect finish"
    },
    "alley_oop": {
        "name": "Alley-Oop/Assist",
        "points": "+3.5 Points",
        "description": "Multi-person coordination and team collaboration"
    }
}

def get_technique_info(technique_key: str) -> dict:
    """Get information for a given technique."""
    return TECHNIQUE_INFO.get(technique_key, {})

