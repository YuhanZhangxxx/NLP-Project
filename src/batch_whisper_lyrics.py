import argparse, os
from pathlib import Path
import torch
import whisper
from tqdm import tqdm

AUDIO_EXTS = {'.mp3','.wav','.m4a','.flac','.aac','.wma','.ogg','.opus','.mp4','.mov'}

def list_audio_files(p: Path):
    if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
        return [p]
    return [f for f in p.rglob('*') if f.is_file() and f.suffix.lower() in AUDIO_EXTS]

def main():
    ap = argparse.ArgumentParser(description="Batch Whisper -> one-line-per-segment lyrics (.txt per song)")
    ap.add_argument("--input", required=True, help="audio file or folder")
    ap.add_argument("--output", required=True, help="output folder for .txt lyrics")
    ap.add_argument("--model", default="large-v3", help="tiny/base/small/medium/large-v3/turbo")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu","auto"])
    ap.add_argument("--language", default="en", help="en/English, zh/Chinese, etc.")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if (args.device in ("cuda","auto") and torch.cuda.is_available()) else "cpu"
    fp16   = (device == "cuda")

    print(f"[INFO] Loading model {args.model} on {device} (fp16={fp16}) ...")
    try:
        model = whisper.load_model(args.model, device=device)
    except RuntimeError as e:
        # auto-downgrade (your 3070 has 8GB, large-v3 sometimes OOM)
        if "out of memory" in str(e).lower() and args.model.lower() == "large-v3":
            print("[WARN] OOM with large-v3; falling back to 'medium' ...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = whisper.load_model("medium", device=device)
        else:
            raise

    files = list_audio_files(in_path)
    if not files:
        print("[WARN] No audio files found."); return

    for f in tqdm(files, desc="Transcribing"):
        try:
            result = model.transcribe(
                str(f), language=args.language, task="transcribe",
                verbose=False, fp16=fp16, temperature=0.0
            )
            segments = result.get("segments", [])
            lines = [ (s.get("text") or "").strip() for s in segments ]
            lines = [ ln for ln in lines if ln ]
            out_path = out_dir / (f.stem + ".txt")
            with open(out_path, "w", encoding="utf-8") as w:
                for ln in lines:
                    w.write(ln + "\n")
        except Exception as e:
            print(f"[ERR] {f} -> {e}")

    print(f"[DONE] Wrote {len(files)} files to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
