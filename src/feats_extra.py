import numpy as np
from scipy import sparse

PUNCTS = set(".,!?;:\'\"-—()[]{}/*&^%$#@`~")

def text_stats(texts):
    """Returns (n,7) sparse features: length, line count, avg line length, token count, unique token ratio, repeat line ratio, punctuation ratio."""
    rows = []
    for t in texts:
        s = t if isinstance(t, str) else ""
        n_chars = len(s) or 1
        lines = [ln for ln in s.splitlines() if ln.strip()!=""]
        n_lines = max(len(lines), 1)
        avg_line = len(s)/n_lines
        toks = s.split()
        n_tok = len(toks) or 1
        uniq_tok_ratio = len(set(toks))/n_tok
        repeat_line_ratio = 1.0 - (len(set(lines))/n_lines)
        punct = sum(ch in PUNCTS for ch in s)/n_chars
        rows.append([n_chars, n_lines, avg_line, n_tok, uniq_tok_ratio, repeat_line_ratio, punct])
    A = np.asarray(rows, dtype=float)
    return sparse.csr_matrix(A)

# 向后兼容：从 skill_detection 文件夹导入
import sys
from pathlib import Path
skill_dir = Path(__file__).parent.parent / "skill_detection"
if str(skill_dir) not in sys.path:
    sys.path.insert(0, str(skill_dir))
from rap_techniques import detect_rap_techniques