import argparse
from pathlib import Path
from faster_whisper import WhisperModel

AUDIO_EXTS = {".mp3",".wav",".m4a",".flac",".aac",".wma",".ogg",".opus",".mp4",".mov"}

def list_audio_files(root: Path):
    if root.is_file() and root.suffix.lower() in AUDIO_EXTS:
        return [root]
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]

def main():
    ap = argparse.ArgumentParser(description="Fast Whisper batch -> one line per segment (.txt per song)")
    ap.add_argument("--input",  required=True, help="audio file or folder")
    ap.add_argument("--output", required=True, help="output folder for .txt lyrics")
    ap.add_argument("--model",  default="large-v3", help="tiny/base/small/medium/large-v3/turbo")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu","auto"])
    ap.add_argument("--language", default="en", help="language code, e.g., en")
    ap.add_argument("--compute_type", default="float16", help="float16|int8_float16|int8|float32")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if (args.device in ("cuda","auto")) else "cpu"
    # 3070(8GB) usually runs large-v3 + float16; if not, use int8_float16 or switch to medium
    compute_type = args.compute_type

    print(f"[INFO] Loading {args.model} on {device}, compute_type={compute_type} ...")
    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    except Exception as e:
        # auto-downgrade when low on memory/VRAM
        msg = str(e).lower()
        if "out of memory" in msg or "cuda" in msg:
            print("[WARN] OOM/init failure; falling back to medium,float16 ...")
            model = WhisperModel("medium", device=device, compute_type="float16")
        else:
            raise

    files = list_audio_files(in_path)
    if not files:
        print("[WARN] No audio found."); return

    prompt = (
        "Transcribe as English rap lyrics. Keep slang, profanity, repetitions, and ad-libs. "
        "Do not censor. Use natural casing and punctuation."
    )

    for idx, f in enumerate(files, 1):
        out_txt = out_dir / (f.stem + ".txt")
        print(f"[{idx}/{len(files)}] {f.name} -> {out_txt.name}")

        # VAD filters silence/noise; beam_search improves accuracy; 30s chunks avoid long audio lag
        segments, info = model.transcribe(
            str(f),
            language=args.language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            initial_prompt=prompt,
            chunk_length=30,          # seconds; more stable and smooth
            without_timestamps=True   # text only
        )

        lines = []
        for seg in segments:
            txt = (seg.text or "").strip()
            if txt:
                lines.append(txt)

        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"[DONE] Wrote {len(files)} files to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
