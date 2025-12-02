#!/usr/bin/env python
"""
Rap Battle/Song Judge with quality score, detailed reasoning, and confidence breakdown.
Analyzes rap battles and rap songs to judge WHY they're good or bad, not just the prediction.
Provides insights on bars, wordplay, flow, and delivery quality.
"""
import argparse
import joblib
import numpy as np
import sys
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from feats_extra import PUNCTS

def clean_lyrics(text: str) -> str:
    """
    Clean lyrics: remove ads, section markers, recommendations, transcription artifacts, etc.
    Improved version with better artifact removal.
    """
    lines = text.splitlines()
    out = []
    skip_recs = False  # skip "You might also like" section until blank line
    
    # Patterns to remove completely
    REMOVE_LINE_PATTERNS = [
        r'^\s*\[[^\]]+\]\s*$',  # [Intro], [Verse 1], etc.
        r'^\s*\*.*\*\s*$',  # *Evil laughing*
        r'^\s*See .* near .*$',  # ticket ads
        r'^\s*Get tickets .*$',  # ticket ads
        r'^\s*We\'ll be right back\.?\s*$',  # transcription artifact
        r'^\s*Thank you for watching\.?\s*$',  # transcription artifact
        r'^\s*See you next time\.?\s*$',  # transcription artifact
        r'^\s*Do not censor\.?\s*$',  # transcription artifact
        r'^\s*Use natural casing and punctuation\.?\s*$',  # transcription artifact
        r'^\s*Outro Music\s*$',  # transcription artifact
        r'^\s*Turn me up in headphones\.?\s*$',  # transcription artifact
        r'^\s*Let\'s go!?\s*$',  # transcription artifact
        r'^\s*Peace out\.?\s*$',  # transcription artifact
        r'^\s*Be safe\.?\s*$',  # transcription artifact
        r'^\s*Ladies and gentlemen\.?\s*$',  # transcription artifact
    ]
    
    # Common transcription artifacts (partial matches)
    ARTIFACT_PATTERNS = [
        r'^.*We\'ll be right back.*$',
        r'^.*Do not censor.*$',
        r'^.*Use natural casing.*$',
        r'^.*Thank you for watching.*$',
    ]
    
    for raw in lines:
        line = raw.rstrip()
        
        # Skip "You might also like" section
        if re.match(r'^\s*You might also like\s*$', line, flags=re.I):
            skip_recs = True
            continue
        if skip_recs:
            if line.strip() == "":
                skip_recs = False
            continue
        
        # Remove lines matching exact patterns
        if any(re.match(p, line, flags=re.I) for p in REMOVE_LINE_PATTERNS):
            continue
        
        # Remove lines containing transcription artifacts
        if any(re.search(p, line, flags=re.I) for p in ARTIFACT_PATTERNS):
            continue
        
        # Drop all-caps short shouts (like DESERT STORM!!)
        if re.match(r'^[^a-z]*[A-Z][A-Z \-!?.\'/]*$', line) and len(line.split()) <= 4:
            continue
        
        # Skip very short lines that are likely artifacts
        if len(line.strip()) < 3:
            continue
        
        # Light normalization
        line = re.sub(r'\s+,', ',', line)
        line = re.sub(r'\(\s+', '(', line)
        line = re.sub(r'\s+\)', ')', line)
        
        if line.strip():
            out.append(line)
    
    return "\n".join(out)

def read_text(p: Path, preprocess=True):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = p.read_text(encoding=enc, errors="ignore")
            # Clean lyrics first (remove ads, etc.)
            text = clean_lyrics(text)
            # Match training preprocessing: replace newlines/tabs with spaces
            if preprocess:
                text = re.sub(r"[\r\n\t]+", " ", text).strip()
            return text
        except Exception:
            pass
    text = p.read_text(errors="ignore")
    # Clean lyrics first
    text = clean_lyrics(text)
    # Match training preprocessing: replace newlines/tabs with spaces
    if preprocess:
        text = re.sub(r"[\r\n\t]+", " ", text).strip()
    return text

def calculate_quality_score(p_good):
    """
    Convert probability to quality score (0-100).
    - 0-50: Bad (0-50% good probability)
    - 50-100: Good (50-100% good probability)
    """
    quality = p_good * 100
    return int(round(quality))

def get_confidence_level(p_good):
    diff = abs(p_good - 0.5)  # Distance from uncertain (0.5)
    if diff >= 0.4:
        return "Very High"
    elif diff >= 0.25:
        return "High"
    elif diff >= 0.15:
        return "Moderate"
    else:
        return "Low"

def get_quality_rating(quality_score):
    """Get human-readable quality rating."""
    if quality_score >= 80:
        return "Excellent"
    elif quality_score >= 65:
        return "Good"
    elif quality_score >= 50:
        return "Decent"
    elif quality_score >= 35:
        return "Below Average"
    elif quality_score >= 20:
        return "Poor"
    else:
        return "Very Poor"

def analyze_text_stats(text):
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    toks = text.split()
    chars = len(text)
    
    stats = {
        "total_lines": len(lines),
        "total_words": len(toks),
        "total_chars": chars,
        "avg_line_length": chars / max(len(lines), 1),
        "unique_word_ratio": len(set(toks)) / max(len(toks), 1),
        "repetition_ratio": 1.0 - (len(set(lines)) / max(len(lines), 1)),
        "punct_ratio": sum(ch in PUNCTS for ch in text) / max(chars, 1)
    }
    
    insights = []
    if stats["unique_word_ratio"] > 0.7:
        insights.append("[+] Strong wordplay variety (diverse vocabulary, creative bars)")
    elif stats["unique_word_ratio"] < 0.4:
        insights.append("[-] Weak wordplay (repetitive words, limited vocabulary)")
    
    if stats["repetition_ratio"] > 0.3:
        insights.append("[-] High bar repetition (recycled lines, lack of originality)")
    elif stats["repetition_ratio"] < 0.1:
        insights.append("[+] Fresh bars throughout (low repetition, original content)")
    
    if stats["avg_line_length"] > 50:
        insights.append("[+] Substantial bars (good depth, well-developed lines)")
    elif stats["avg_line_length"] < 20:
        insights.append("[-] Short bars (may lack punchlines or substance)")
    
    return stats, insights

def get_feature_contributions(pipe, text, top_n=15):
    """Extract feature contributions, supports both simple pipeline and FeatureUnion pipeline (v2_plus)"""
    clf = None
    feats_union = None
    vec = None
    
    # Check if this is a v2_plus model (uses FeatureUnion)
    for name, step in pipe.named_steps.items():
        if isinstance(step, FeatureUnion):
            feats_union = step
        elif isinstance(step, (LogisticRegression, MultinomialNB)):
            clf = step
        elif isinstance(step, TfidfVectorizer):
            vec = step
    
    if clf is None:
        return [], []
    
    # Get classes
    classes = list(getattr(clf, "classes_", ["bad", "good"]))
    i_good = classes.index("good") if "good" in classes else 1
    i_bad = classes.index("bad") if "bad" in classes else 0
    
    good_features = []
    bad_features = []
    
    # Handle FeatureUnion structure (v2_plus model)
    if feats_union is not None and isinstance(clf, LogisticRegression):
        coef = clf.coef_[0]
        # Make sure coefficients point toward 'good'
        if classes[i_good] != "good":
            coef = -coef
        
        # Calculate offset for each feature block
        offset = 0
        for name, transformer in feats_union.transformer_list:
            Xb = transformer.transform([text])
            dim = Xb.shape[1]
            sl = slice(offset, offset + dim)
            coef_slice = coef[sl]
            
            # Handle word and char n-grams
            if isinstance(transformer, TfidfVectorizer):
                feature_names = transformer.get_feature_names_out()
                x = Xb.toarray().ravel()
                idx = np.flatnonzero(x)
                
                for i in idx:
                    contrib = x[i] * coef_slice[i]
                    feat_name = feature_names[i]
                    if contrib > 0:
                        good_features.append((feat_name, contrib))
                    else:
                        bad_features.append((feat_name, abs(contrib)))
            
            offset += dim
    
    # Handle simple pipeline (v1 model)
    elif vec is not None:
        X = vec.transform([text])
        x = X.toarray().ravel()
        feature_names = vec.get_feature_names_out()
        
        if isinstance(clf, LogisticRegression):
            coef_good = clf.coef_[0] if i_good == 1 else -clf.coef_[0]
            coef_bad = -coef_good
            idx = np.flatnonzero(x)
            for i in idx:
                contrib_good = x[i] * coef_good[i]
                contrib_bad = x[i] * coef_bad[i]
                if contrib_good > 0:
                    good_features.append((feature_names[i], contrib_good))
                if contrib_bad > 0:
                    bad_features.append((feature_names[i], contrib_bad))
        
        elif isinstance(clf, MultinomialNB):
            log_prob_good = clf.feature_log_prob_[i_good]
            log_prob_bad = clf.feature_log_prob_[i_bad]
            
            idx = np.flatnonzero(x)
            for i in idx:
                log_diff = log_prob_good[i] - log_prob_bad[i]
                contrib = x[i] * log_diff
                
                if contrib > 0:
                    good_features.append((feature_names[i], contrib))
                else:
                    bad_features.append((feature_names[i], abs(contrib)))
    
    # Sort and get top N
    good_features.sort(key=lambda x: x[1], reverse=True)
    bad_features.sort(key=lambda x: x[1], reverse=True)
    
    return good_features[:top_n], bad_features[:top_n]

def score_lines(pipe, original_text, th_good=0.5, min_line_length=5):
    """
    Score individual lines in a song/rap battle.
    Returns list of (line_number, line_text, p_good, quality_score, verdict) tuples.
    """
    lines = [ln.strip() for ln in original_text.splitlines() if ln.strip()]
    line_scores = []
    
    clf = pipe.named_steps.get("clf", None)
    classes = list(getattr(clf, "classes_", [])) if clf else ["bad", "good"]
    
    if "good" in classes and "bad" in classes:
        i_good = classes.index("good")
        i_bad = classes.index("bad")
    else:
        i_bad, i_good = 0, 1
    
    for idx, line in enumerate(lines, 1):
        # Skip very short lines (likely not meaningful bars)
        if len(line.split()) < min_line_length:
            continue
        
        # Preprocess line to match training format
        import re
        line_processed = re.sub(r"[\r\n\t]+", " ", line).strip()
        
        # Get prediction for this line
        try:
            proba = pipe.predict_proba([line_processed])[0]
            p_good = float(proba[i_good])
            quality_score = calculate_quality_score(p_good)
            verdict = "good" if p_good >= th_good else "bad"
            line_scores.append((idx, line, p_good, quality_score, verdict))
        except Exception:
            # Skip lines that can't be processed
            continue
    
    return line_scores

def format_output(file_path, text, original_text, pipe, th_good=0.5, show_lines=False):
    # Get prediction and probabilities
    proba = pipe.predict_proba([text])[0]
    clf = pipe.named_steps.get("clf", None)
    classes = list(getattr(clf, "classes_", [])) if clf else ["bad", "good"]
    
    if "good" in classes and "bad" in classes:
        i_good = classes.index("good")
        i_bad = classes.index("bad")
    else:
        i_bad, i_good = 0, 1
    
    p_good = float(proba[i_good])
    p_bad = float(proba[i_bad])
    pred = "good" if p_good >= th_good else "bad"
    
    # Calculate metrics
    quality_score = calculate_quality_score(p_good)
    quality_rating = get_quality_rating(quality_score)
    confidence_level = get_confidence_level(p_good)
    
    # Get feature contributions
    good_features, bad_features = get_feature_contributions(pipe, text, top_n=12)
    
    # Analyze text statistics (use original text with newlines for accurate stats)
    text_stats_dict, text_insights = analyze_text_stats(original_text)
    
    # Format output
    output = []
    output.append("=" * 70)
    output.append(f"RAP BATTLE/SONG JUDGE: {Path(file_path).name}")
    output.append("=" * 70)
    output.append("")
    
    # Overall verdict
    output.append("[OVERALL VERDICT]")
    output.append("-" * 70)
    output.append(f"Judgment: {pred.upper()}")
    output.append(f"Quality Score: {quality_score}/100 ({quality_rating})")
    output.append(f"Confidence: {confidence_level} ({p_good*100:.1f}% good, {p_bad*100:.1f}% bad)")
    output.append("")
    
    # Why it's good/bad
    output.append("[WHY THIS RAP IS " + pred.upper() + "]")
    output.append("-" * 70)
    
    if pred == "good":
        if good_features:
            output.append("Top bars/wordplay contributing to GOOD rating:")
            for i, (feat, contrib) in enumerate(good_features[:8], 1):
                output.append(f"  {i:2d}. '{feat}' (+{contrib:.4f})")
        else:
            output.append("  (Feature analysis not available)")
        
        if bad_features:
            output.append("")
            output.append("Bars/patterns that slightly hurt the rating:")
            for i, (feat, contrib) in enumerate(bad_features[:5], 1):
                output.append(f"  {i:2d}. '{feat}' (-{contrib:.4f})")
    else:
        if bad_features:
            output.append("Top bars/patterns contributing to BAD rating:")
            for i, (feat, contrib) in enumerate(bad_features[:8], 1):
                output.append(f"  {i:2d}. '{feat}' (+{contrib:.4f})")
        
        if good_features:
            output.append("")
            output.append("Bars/wordplay that slightly help the rating:")
            for i, (feat, contrib) in enumerate(good_features[:5], 1):
                output.append(f"  {i:2d}. '{feat}' (+{contrib:.4f})")
    
    output.append("")
    
    # Line-level analysis
    if show_lines:
        line_scores = score_lines(pipe, original_text, th_good)
        if line_scores:
            output.append("[LINE-BY-LINE ANALYSIS]")
            output.append("-" * 70)
            # Sort by quality score (best first)
            line_scores_sorted = sorted(line_scores, key=lambda x: x[3], reverse=True)
            
            output.append("Top 10 Best Lines:")
            for idx, line_text, p_good, q_score, verdict in line_scores_sorted[:10]:
                output.append(f"  Line {idx:3d} [{q_score:3d}/100] ({p_good*100:5.1f}% good): {line_text[:60]}...")
            
            output.append("")
            output.append("Top 10 Worst Lines:")
            for idx, line_text, p_good, q_score, verdict in line_scores_sorted[-10:]:
                output.append(f"  Line {idx:3d} [{q_score:3d}/100] ({p_good*100:5.1f}% good): {line_text[:60]}...")
            
            # Summary stats
            avg_line_score = sum(q for _, _, _, q, _ in line_scores) / len(line_scores)
            good_lines = sum(1 for _, _, _, _, v in line_scores if v == "good")
            output.append("")
            output.append(f"Line-Level Summary: {len(line_scores)} lines analyzed")
            output.append(f"  Average Line Score: {avg_line_score:.1f}/100")
            output.append(f"  Good Lines: {good_lines}/{len(line_scores)} ({good_lines/len(line_scores)*100:.1f}%)")
            output.append("")
    
    # Text statistics
    output.append("[RAP STATISTICS]")
    output.append("-" * 70)
    output.append(f"  Total Bars/Lines: {text_stats_dict['total_lines']}")
    output.append(f"  Total Words: {text_stats_dict['total_words']}")
    output.append(f"  Average Bar Length: {text_stats_dict['avg_line_length']:.1f} characters")
    output.append(f"  Vocabulary Diversity: {text_stats_dict['unique_word_ratio']*100:.1f}% unique words")
    output.append(f"  Repetition Rate: {text_stats_dict['repetition_ratio']*100:.1f}% repeated bars")
    output.append("")
    
    if text_insights:
        output.append("[RAP QUALITY INSIGHTS]")
        for insight in text_insights:
            output.append(f"  {insight}")
        output.append("")
    
    # Confidence breakdown
    output.append("[JUDGE CONFIDENCE]")
    output.append("-" * 70)
    if p_good >= 0.7:
        output.append(f"  Strongly confident this rap is GOOD ({p_good*100:.1f}%)")
        output.append("  -> Strong bars, good wordplay, solid delivery")
    elif p_good >= 0.6:
        output.append(f"  Confident this rap is GOOD ({p_good*100:.1f}%)")
        output.append("  -> Decent performance overall")
    elif p_good >= 0.4:
        output.append(f"  Uncertain - close call ({p_good*100:.1f}% good, {p_bad*100:.1f}% bad)")
        output.append("  -> Mixed quality, could go either way")
    elif p_bad >= 0.6:
        output.append(f"  Confident this rap is BAD ({p_bad*100:.1f}%)")
        output.append("  -> Weak bars, poor delivery, or lack of substance")
    else:
        output.append(f"  Strongly confident this rap is BAD ({p_bad*100:.1f}%)")
        output.append("  -> Very weak performance, major issues detected")
    output.append("")
    output.append("=" * 70)
    
    return "\n".join(output)

def main():
    ap = argparse.ArgumentParser(
        description="Rap battle/song judge with quality score and detailed analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze from file
  python src/analyze_song.py --model models/song_lr_v2_plus_20251026.joblib --file lyrics.txt
  
  # Direct text input (simple text)
  python src/analyze_song.py --model models/song_lr_v2_plus_20251026.joblib --text "Your rap lyrics here"
  
  # Read from stdin (recommended for lyrics with special characters)
  cat lyrics.txt | python src/analyze_song.py --model models/song_lr_v2_plus_20251026.joblib
  
  # Or use heredoc (good for multi-line text)
  python src/analyze_song.py --model models/song_lr_v2_plus_20251026.joblib << 'EOF'
  Your lyrics content
  EOF
  
  # Interactive mode (recommended for lyrics with special characters)
  python src/analyze_song.py --model models/song_lr_v2_plus_20251026.joblib --interactive
  # Then paste lyrics directly, press Ctrl+D when done
  
  # Show line-by-line analysis
  python src/analyze_song.py --model models/song_lr_v2_plus_20251026.joblib --text "Your lyrics" --lines
        """
    )
    ap.add_argument("--model", required=True, help="Path to trained model (.joblib)")
    ap.add_argument("--file", help="Path to rap battle/song lyrics file (.txt)")
    ap.add_argument("--text", help="Rap lyrics text (alternative to --file)")
    ap.add_argument("--interactive", "-i", action="store_true",
                    help="Interactive mode: paste lyrics (press Ctrl+D or Ctrl+Z when done)")
    ap.add_argument("--th_good", type=float, default=0.5, 
                    help="Probability threshold for 'good' classification (default: 0.5)")
    ap.add_argument("--lines", action="store_true",
                    help="Show line-by-line analysis")
    args = ap.parse_args()
    
    # Check arguments
    if not args.file and not args.text:
        # If neither --file nor --text provided, try reading from stdin
        if not sys.stdin.isatty():
            # Has stdin (pipe or redirect)
            args.text = sys.stdin.read()
        elif args.interactive:
            # Interactive mode: prompt user for lyrics
            print("=" * 70, file=sys.stderr)
            print("🎤 Interactive Mode: Paste your lyrics (can include any special characters)", file=sys.stderr)
            print("", file=sys.stderr)
            print("Tips:", file=sys.stderr)
            print("  • Paste multi-line lyrics directly (supports quotes, brackets, etc.)", file=sys.stderr)
            print("  • When done, press Ctrl+D (Mac/Linux) or Ctrl+Z+Enter (Windows)", file=sys.stderr)
            print("  • Or type 'END' on a separate line to finish", file=sys.stderr)
            print("=" * 70, file=sys.stderr)
            print("", file=sys.stderr)
            
            lines = []
            try:
                while True:
                    line = input()
                    if line.strip() == "END":
                        break
                    lines.append(line)
            except EOFError:
                # Ctrl+D was pressed
                pass
            except KeyboardInterrupt:
                print("\nInput cancelled", file=sys.stderr)
                sys.exit(1)
            
            if not lines:
                print("Error: No lyrics entered", file=sys.stderr)
                sys.exit(1)
            
            args.text = "\n".join(lines)
        else:
            print("Error: Must provide either --file, --text, --interactive, or pipe text via stdin", file=sys.stderr)
            print("", file=sys.stderr)
            print("Usage:", file=sys.stderr)
            print("  1. From file: --file lyrics.txt", file=sys.stderr)
            print("  2. Direct input: --text \"your lyrics\"", file=sys.stderr)
            print("  3. Interactive mode: --interactive (recommended for lyrics with special characters)", file=sys.stderr)
            print("  4. Pipe input: echo 'lyrics' | python ...", file=sys.stderr)
            sys.exit(1)
    
    if args.file and args.text:
        print("Error: Cannot use both --file and --text. Choose one.", file=sys.stderr)
        sys.exit(1)
    
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    
    # If using v2_plus model, add src directory to path (model depends on feats_extra module)
    if "v2_plus" in str(model_path):
        src_dir = Path(__file__).parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
    
    # Read text
    if args.text:
        # Use provided text directly, clean first
        original_text = clean_lyrics(args.text)
        text = re.sub(r"[\r\n\t]+", " ", original_text).strip()
        file_path = Path("<stdin>")  # Filename for display
    else:
        # Read from file (read_text will clean internally)
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            sys.exit(1)
        original_text = read_text(file_path, preprocess=False)
        text = read_text(file_path, preprocess=True)
    
    pipe = joblib.load(model_path)
    analysis = format_output(file_path, text, original_text, pipe, args.th_good, show_lines=args.lines)
    print(analysis)

if __name__ == "__main__":
    main()
