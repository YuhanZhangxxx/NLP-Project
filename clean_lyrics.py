# clean_lyrics.py
import re, sys, pathlib

REMOVE_LINE_PATTERNS = [
    r'^\s*\[[^\]]+\]\s*$',       # [Intro] [Verse 1: ...] [Chorus] [DJ Clue] [Outro]
    r'^\s*\*.*\*\s*$',           # *Evil laughing*  *giving shoutouts*
    r'^\s*See .* near .*$',      # ticket ads
    r'^\s*Get tickets .*$',      # ticket ad continuation
]

def clean_lyrics(text: str) -> str:
    lines = text.splitlines()
    out = []
    skip_recs = False  # skip "You might also like" section until blank line

    for raw in lines:
        line = raw.rstrip()

        if re.match(r'^\s*You might also like\s*$', line, flags=re.I):
            skip_recs = True
            continue
        if skip_recs:
            if line.strip() == "":
                skip_recs = False
            continue

        if any(re.match(p, line, flags=re.I) for p in REMOVE_LINE_PATTERNS):
            continue

        # drop all-caps short shouts (like DESERT STORM!!)
        if re.match(r'^[^a-z]*[A-Z][A-Z \-!?.\'/]*$', line) and len(line.split()) <= 4:
            continue

        # optional: uncomment next line to also remove parenthesis-only ad-libs
        # if re.match(r'^\s*\(.+\)\s*$', line): continue

        # light normalization
        line = re.sub(r'\s+,', ',', line)
        line = re.sub(r'\(\s+', '(', line)
        line = re.sub(r'\s+\)', ')', line)

        if line.strip():
            out.append(line)

    return "\n".join(out)

def process_path(path: pathlib.Path):
    if path.is_file():
        txt = path.read_text(encoding='utf-8', errors='ignore')
        path.with_name(path.stem + '.clean.txt').write_text(clean_lyrics(txt), encoding='utf-8')
    else:
        for f in path.rglob('*.txt'):
            txt = f.read_text(encoding='utf-8', errors='ignore')
            f.with_name(f.stem + '.clean.txt').write_text(clean_lyrics(txt), encoding='utf-8')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_lyrics.py <file_or_folder>")
        sys.exit(1)
    process_path(pathlib.Path(sys.argv[1]))
