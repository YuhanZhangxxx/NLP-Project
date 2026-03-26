#!/usr/bin/env python
"""Simple lyrics judge - just input lyrics text"""
import argparse
import pathlib
import joblib
import glob
import warnings

# Ignore version warnings
warnings.filterwarnings('ignore', category=UserWarning)

def find_latest_model():
    """Automatically find the latest model file"""
    models_dir = pathlib.Path(__file__).parent / "models"

    # Search in subdirs: lr_v2_plus, lr_v1, nb_v1 (priority order)
    for subdir in ["lr_v2_plus", "lr_v1", "nb_v1"]:
        subpath = models_dir / subdir
        if not subpath.exists():
            continue
        matches = list(subpath.glob("*.joblib"))
        if matches:
            return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    return None

def main():
    ap = argparse.ArgumentParser(
        description="Quick judge for lyrics quality - just input lyrics text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python judge.py "Yo I'm spitting fire with every line"
  python judge.py "Your rap lyrics here"
        """
    )
    ap.add_argument("text", nargs="+", help="Lyrics text (can be wrapped in quotes)")
    ap.add_argument("--model", help="Specify model path (optional, defaults to auto-selecting latest model)")
    ap.add_argument("--th_good", type=float, default=0.5, help="Probability threshold for 'good' classification (default: 0.5)")
    args = ap.parse_args()
    
    # Merge all text arguments (supports space-separated lyrics)
    lyrics = " ".join(args.text)
    
    # Select model
    if args.model:
        model_path = pathlib.Path(args.model)
    else:
        model_path = find_latest_model()
        if not model_path:
            print("Error: Could not find model file. Please ensure there are .joblib model files in models/ directory.", file=__import__('sys').stderr)
            return 1
        print(f"[Using model: {model_path.name}]")
    
    if not model_path.exists():
        print(f"Error: Model file does not exist: {model_path}", file=__import__('sys').stderr)
        return 1
    
    # Load model and predict
    try:
        # If using v2_plus model, need to add src directory to path (model depends on feats_extra module)
        import sys
        if "v2_plus" in str(model_path):
            src_dir = pathlib.Path(__file__).parent / "src"
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
        
        pipe = joblib.load(model_path)
        proba = pipe.predict_proba([lyrics])[0]
        
        # Get class indices
        clf = getattr(pipe, "named_steps", {}).get("clf", None)
        classes = list(getattr(clf, "classes_", []))
        if classes and ("good" in classes and "bad" in classes):
            i_good = classes.index("good")
            i_bad = classes.index("bad")
        else:
            i_bad, i_good = 0, 1
        
        p_good = float(proba[i_good])
        p_bad = float(proba[i_bad])
        pred = "good" if p_good >= args.th_good else "bad"
        
        # Output results
        print(f"\nResult: {pred.upper()}")
        print(f"Probability: good={p_good:.3f}, bad={p_bad:.3f}")
        print(f"Threshold: {args.th_good}")
        
        return 0
    except Exception as e:
        print(f"Error: Prediction failed - {e}", file=__import__('sys').stderr)
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

