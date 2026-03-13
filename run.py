"""
Command-line interface for the transcription/diarization/segmentation pipeline.

Usage:
    python run.py interview.mp4
    python run.py podcast.wav --whisper-model medium.en --max-speakers 3
    python run.py lecture.mp3 --no-llm --output ./results
"""

import argparse
import sys

from config import (
    PipelineConfig, AudioConfig, TranscriptionConfig,
    DiarizationConfig, AlignmentConfig, SegmentationConfig,
    DeviceStrategy,
)
from pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Audio -> Transcription -> Diarization -> Coherent Sections"
    )
    parser.add_argument("media_path", help="Path to audio or video file")
    parser.add_argument(
        "-o", "--output", default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--whisper-model", default="base.en",
        help="Whisper model size (default: base.en)"
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device (default: auto)"
    )
    parser.add_argument(
        "--language", default="en",
        help="Language code for transcription (default: en)"
    )
    parser.add_argument(
        "--min-speakers", type=int, default=None,
        help="Minimum expected speakers"
    )
    parser.add_argument(
        "--max-speakers", type=int, default=None,
        help="Maximum expected speakers"
    )
    parser.add_argument(
        "--llm-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="LLM for topic shift detection"
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Disable LLM topic detection (embedding-only segmentation)"
    )
    parser.add_argument(
        "--min-section-duration", type=float, default=10.0,
        help="Minimum section duration in seconds"
    )
    parser.add_argument(
        "--max-section-duration", type=float, default=300.0,
        help="Maximum section duration in seconds"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # Build config
    config = PipelineConfig(
        audio=AudioConfig(),
        transcription=TranscriptionConfig(
            model_name=args.whisper_model,
            device=DeviceStrategy(args.device),
            language=args.language,
        ),
        diarization=DiarizationConfig(
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        ),
        alignment=AlignmentConfig(),
        segmentation=SegmentationConfig(
            llm_model=args.llm_model if not args.no_llm else "",
            llm_fallback_model="" if args.no_llm else "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            min_segment_duration_seconds=args.min_section_duration,
            max_segment_duration_seconds=args.max_section_duration,
        ),
        output_dir=args.output,
        log_level=args.log_level,
    )

    # Override LLM weight if disabled
    if args.no_llm:
        config.segmentation.llm_shift_weight = 0.0

    def progress(update):
        pct = update.get("progress", 0)
        status = update.get("status", "")
        filled = pct // 5
        bar = "\u2588" * filled + "\u2591" * (20 - filled)
        print(f"\r  [{bar}] {pct:3d}% {status}", end="", flush=True)
        if pct >= 100:
            print()

    pipe = None
    try:
        pipe = Pipeline(config)
        result = pipe.process(args.media_path, progress_callback=progress)
        paths = pipe.save_results(result)

        print(f"\nResults:")
        print(f"  Duration:    {result.metadata['duration_seconds']:.1f}s")
        print(f"  Speakers:    {result.metadata['num_speakers']}")
        print(f"  Sentences:   {result.metadata['num_sentences']}")
        print(f"  Sections:    {result.metadata['num_sections']}")
        print(f"\nFiles:")
        for label, path in paths.items():
            print(f"  {label:12s}: {path}")

    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if pipe is not None:
            pipe.cleanup()


if __name__ == "__main__":
    main()
