#!/usr/bin/env python
"""End-to-end pipeline: Transcribe audio → Clean lyrics → Judge quality"""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(
        description="End-to-end: Transcribe audio with Whisper, clean lyrics, then judge quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe and judge a single audio file
  python scripts/transcribe_and_judge.py --audio audio/song.mp3

  # Use medium model for better quality
  python scripts/transcribe_and_judge.py --audio audio/song.mp3 --model medium

  # Get detailed analysis
  python scripts/transcribe_and_judge.py --audio audio/song.mp3 --detailed
        """
    )
    ap.add_argument("--audio", required=True, help="Audio file path (.mp3, .wav, etc.)")
    ap.add_argument("--model", default="medium", help="Whisper model: medium/large-v3 (default: medium)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="Device for transcription")
    ap.add_argument("--language", default="en", help="Language code (default: en)")
    ap.add_argument("--output-dir", default="data/transcribed_lyrics", help="Output directory for transcripts")
    ap.add_argument("--detailed", action="store_true", help="Show detailed analysis with feature contributions")
    ap.add_argument("--judge-model", help="Path to judge model (auto-selects latest if not specified)")
    ap.add_argument("--keep-transcript", action="store_true", help="Keep transcript files after judging")
    args = ap.parse_args()
    
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70, file=sys.stderr)
    print("END-TO-END PIPELINE: Audio -> Transcription -> Cleaning -> Judgment", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Audio file: {audio_path}", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Step 1: Transcribe
    print("[1/4] Transcribing audio with Whisper...", file=sys.stderr)
    try:
        result = subprocess.run(
            [
                sys.executable, "src/batch_whisper_lyrics.py",
                "--input", str(audio_path),
                "--output", str(output_dir),
                "--model", args.model,
                "--device", args.device,
                "--language", args.language
            ],
            capture_output=True,
            text=True,
            check=True
        )
        print("  Transcription complete", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"  Transcription failed: {e}", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)
    
    # Find transcript file
    transcript_file = output_dir / f"{audio_path.stem}.txt"
    if not transcript_file.exists():
        print(f"Error: Transcript file not found: {transcript_file}", file=sys.stderr)
        sys.exit(1)
    
    print(f"  Transcript saved: {transcript_file}", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Step 2: Clean lyrics
    print("[2/4] Cleaning lyrics...", file=sys.stderr)
    try:
        result = subprocess.run(
            [sys.executable, "clean_lyrics.py", str(transcript_file)],
            capture_output=True,
            text=True,
            check=True
        )
        print("  Cleaning complete", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"  Cleaning failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    cleaned_file = transcript_file.parent / f"{transcript_file.stem}.clean.txt"
    if not cleaned_file.exists():
        print(f"Error: Cleaned file not found: {cleaned_file}", file=sys.stderr)
        sys.exit(1)
    
    print(f"  Cleaned file: {cleaned_file}", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Step 3: Judge
    print("[3/4] Judging lyrics quality...", file=sys.stderr)
    
    # Read cleaned lyrics
    lyrics_text = cleaned_file.read_text(encoding="utf-8")
    
    # Determine judge command
    if args.detailed:
        judge_cmd = [
            sys.executable, "src/analyze_song.py",
            "--file", str(cleaned_file)
        ]
        if args.judge_model:
            judge_cmd.extend(["--model", args.judge_model])
        else:
            # Auto-select latest model (search in lr_v2_plus subdir)
            models_dir = Path("models")
            latest_model = None
            lr2_dir = models_dir / "lr_v2_plus"
            if lr2_dir.exists():
                matches = sorted(lr2_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
                if matches:
                    latest_model = matches[0]
            if latest_model:
                judge_cmd.extend(["--model", str(latest_model)])
            else:
                print("  Warning: No model found, using default", file=sys.stderr)
    else:
        judge_cmd = [sys.executable, "judge.py"]
        if args.judge_model:
            judge_cmd.extend(["--model", args.judge_model])
        judge_cmd.append(lyrics_text)
    
    try:
        result = subprocess.run(
            judge_cmd,
            input=lyrics_text if not args.detailed else None,
            capture_output=True,
            text=True,
            check=False
        )
        print("  Judgment complete", file=sys.stderr)
        print("", file=sys.stderr)
        
        # Print judgment output
        print("=" * 70, file=sys.stderr)
        print("JUDGMENT RESULT", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(result.stdout)
        if result.stderr and "InconsistentVersionWarning" not in result.stderr:
            print(result.stderr, file=sys.stderr)
    except Exception as e:
        print(f"  Judgment failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Step 4: Cleanup (optional)
    if not args.keep_transcript:
        print("", file=sys.stderr)
        print("[4/4] Cleaning up temporary files...", file=sys.stderr)
        transcript_file.unlink(missing_ok=True)
        print("  Cleanup complete", file=sys.stderr)
    else:
        print("", file=sys.stderr)
        print("[4/4] Keeping transcript files (--keep-transcript)", file=sys.stderr)
        print(f"  Transcript: {transcript_file}", file=sys.stderr)
        print(f"  Cleaned: {cleaned_file}", file=sys.stderr)
    
    print("", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("END-TO-END PIPELINE COMPLETE", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

if __name__ == "__main__":
    main()

