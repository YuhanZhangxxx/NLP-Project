#!/usr/bin/env python
import argparse, pathlib, csv, sys, os
from faster_whisper import WhisperModel

AUDIO_EXTS = {".wav",".mp3",".m4a",".mp4",".mov",".flac",".ogg",".webm"}

def iter_audio_files(p: pathlib.Path):
    if p.is_file():
        return [p] if p.suffix.lower() in AUDIO_EXTS else []
    files = []
    for ext in AUDIO_EXTS:
        files.extend(p.rglob(f"*{ext}"))
    return sorted(files)

def main():
    ap = argparse.ArgumentParser(description="Batch transcribe audio/video to CSV using faster-whisper")
    ap.add_argument("--in", dest="inp", required=True, help="Audio file or directory containing audio files")
    ap.add_argument("--out", default="data/raw/transcripts.csv", help="Where to write CSV (id,text,label)")
    ap.add_argument("--label", default="", help="Optional default label for all rows (good/bad). Leave empty to label later.")
    ap.add_argument("--model_size", default="base", help="Model size: tiny, base, small, medium, large-v3, etc.")
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda/directml")
    ap.add_argument("--compute", default="int8_float16", help="int8_float16/float16/float32; use float32 on pure CPU")
    ap.add_argument("--language", default=None, help="e.g., en, zh; if omitted, auto-detect")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--vad", action="store_true", help="Enable VAD-based segmentation to reduce hallucinations")
    ap.add_argument("--model_dir", default=None, help="Custom model cache dir (optional)")
    args = ap.parse_args()

    inp = pathlib.Path(args.inp)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = iter_audio_files(inp)
    if not files:
        print(f"No audio files found under: {inp}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading faster-whisper model: size={args.model_size}, device={args.device}, compute={args.compute}")
    model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute, download_root=args.model_dir)

    written = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","text","label"])
        for path in files:
            print(f"Transcribing: {path}")
            segments, info = model.transcribe(str(path),
                                              language=args.language,
                                              beam_size=args.beam,
                                              vad_filter=args.vad)
            text = " ".join(seg.text.strip() for seg in segments if seg.text)
            sample_id = path.stem
            w.writerow([sample_id, text, args.label])
            written += 1

    print(f"Wrote {written} rows to {out}. Now add/verify labels and continue with split & training.")

if __name__ == "__main__":
    main()
