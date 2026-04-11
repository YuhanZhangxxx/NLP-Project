#!/usr/bin/env python
"""
Transcribe audio using the OpenAI Whisper API (cloud).

Usage:
    python src/transcribe_whisper_api.py --in audio.mp3 --api-key sk-...
    python src/transcribe_whisper_api.py --in audio_folder/ --out data/transcripts/
    python src/transcribe_whisper_api.py --in audio.mp3 --format verbose_json  # with timestamps
    python src/transcribe_whisper_api.py --in audio.mp3  # uses WHISPER_API_KEY env var
"""
import argparse
import json
import os
import sys
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".mp4", ".mov", ".flac", ".ogg", ".webm"}
MAX_BYTES = 25 * 1024 * 1024  # Whisper API hard limit: 25 MB


def iter_audio_files(p: Path):
    if p.is_file():
        return [p] if p.suffix.lower() in AUDIO_EXTS else []
    return sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTS)


def transcribe_file(client, path: Path, language: str, response_format: str):
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"File exceeds 25 MB limit: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    with open(path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=language or None,
            response_format=response_format,
        )
    return result


def save_result(result, out_path: Path, fmt: str):
    if fmt == "verbose_json":
        # Build clean JSON with segments + timestamps
        segments = [
            {
                "id": seg.id,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            }
            for seg in result.segments
        ]
        payload = {
            "language": result.language,
            "duration": round(result.duration, 2),
            "text": result.text.strip(),
            "segments": segments,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        text = result if isinstance(result, str) else result.text
        out_path.write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Transcribe audio via OpenAI Whisper API")
    ap.add_argument("--in", dest="inp", required=True, help="Audio file or directory")
    ap.add_argument("--out", default=None, help="Output directory (default: same dir as input)")
    ap.add_argument("--api-key", default=None, help="OpenAI API key (falls back to WHISPER_API_KEY env var)")
    ap.add_argument("--language", default=None, help="Language code e.g. en, zh (omit for auto-detect)")
    ap.add_argument("--format", dest="fmt", default="text", choices=["text", "srt", "vtt", "verbose_json"],
                    help="Response format (default: text). verbose_json saves as .json with timestamps.")
    args = ap.parse_args()

    # Resolve API key: --api-key > WHISPER_API_KEY > OPENAI_API_KEY
    api_key = args.api_key or os.environ.get("WHISPER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No API key provided. Use --api-key or set WHISPER_API_KEY in .env.", file=sys.stderr)
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    inp = Path(args.inp)
    files = iter_audio_files(inp)
    if not files:
        print(f"ERROR: No audio files found under: {inp}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out) if args.out else (inp.parent if inp.is_file() else inp)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext_map = {"text": ".txt", "srt": ".srt", "vtt": ".vtt", "verbose_json": ".json"}
    ext = ext_map[args.fmt]

    for path in files:
        print(f"Transcribing: {path.name} ({path.stat().st_size / 1e6:.1f} MB) ...", end=" ", flush=True)
        try:
            result = transcribe_file(client, path, args.language, args.fmt)
            out_path = out_dir / (path.stem + ext)
            save_result(result, out_path, args.fmt)
            print(f"-> {out_path}")
        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)

    print(f"Done. {len(files)} file(s) written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
