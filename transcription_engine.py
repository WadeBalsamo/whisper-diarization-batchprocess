"""
Production transcription engine wrapping OpenAI Whisper.

Handles chunked processing for long audio, device management, and structured
output with word-level timestamps.
"""

import logging
from typing import Dict, Any, Optional, List, Callable

import numpy as np
import torch
import whisper

from config import DeviceStrategy

logger = logging.getLogger(__name__)


class TranscriptionResult:
    """Structured transcription output."""

    def __init__(self, segments: List[Dict], text: str, language: str,
                 duration: float, model_name: str):
        self.segments = segments
        self.text = text
        self.language = language
        self.duration = duration
        self.model_name = model_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segments": self.segments,
            "text": self.text,
            "metadata": {
                "language": self.language,
                "duration": self.duration,
                "model": self.model_name,
                "segment_count": len(self.segments),
            }
        }


class Transcriber:
    """Whisper-based transcription with word-level timestamps."""

    def __init__(self, config):
        self.config = config
        self.device = self._resolve_device()
        self.model = self._load_model()

    def _resolve_device(self) -> str:
        strategy = self.config.device
        if strategy == DeviceStrategy.AUTO:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return strategy.value

    def _load_model(self) -> whisper.Whisper:
        logger.info(
            f"Loading Whisper model '{self.config.model_name}' on {self.device}"
        )
        model = whisper.load_model(
            self.config.model_name,
            device=self.device,
            in_memory=True,
        )
        logger.info("Whisper model loaded")
        return model

    def transcribe(
        self, audio: np.ndarray, sample_rate: int,
        progress_callback: Optional[Callable] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio array into structured segments with word timestamps.
        """
        audio = audio.astype(np.float32)
        duration = len(audio) / sample_rate

        logger.info(f"Transcribing {duration:.1f}s of audio")

        options = self._build_options()

        if duration > self.config.chunk_length_seconds:
            raw_result = self._transcribe_chunked(
                audio, sample_rate, options, progress_callback
            )
        else:
            raw_result = self.model.transcribe(audio, **options)
            if progress_callback:
                progress_callback({"progress": 100, "status": "complete"})

        segments = self._format_segments(raw_result)

        return TranscriptionResult(
            segments=segments,
            text=raw_result.get("text", ""),
            language=raw_result.get("language", self.config.language or "unknown"),
            duration=duration,
            model_name=self.config.model_name,
        )

    def _build_options(self) -> dict:
        opts = {
            "verbose": False,
            "temperature": self.config.temperature,
            "beam_size": self.config.beam_size,
            "condition_on_previous_text": self.config.condition_on_previous_text,
            "no_speech_threshold": self.config.no_speech_threshold,
            "compression_ratio_threshold": self.config.compression_ratio_threshold,
            "word_timestamps": self.config.word_timestamps,
            "fp16": (self.device == "cuda"),
        }
        if self.config.language:
            opts["language"] = self.config.language
        return opts

    def _transcribe_chunked(
        self, audio: np.ndarray, sample_rate: int,
        options: dict, progress_callback: Optional[Callable]
    ) -> dict:
        """Process long audio in overlapping chunks to avoid memory issues."""
        chunk_samples = self.config.chunk_length_seconds * sample_rate
        overlap_samples = int(self.config.chunk_overlap_seconds * sample_rate)
        step = chunk_samples - overlap_samples
        total = len(audio)

        chunks = []
        for start in range(0, total, step):
            end = min(start + chunk_samples, total)
            chunks.append((audio[start:end].astype(np.float32), start / sample_rate))

        logger.info(f"Processing {len(chunks)} chunks")

        all_segments = []
        detected_language = None

        for idx, (chunk_audio, offset) in enumerate(chunks):
            result = self.model.transcribe(chunk_audio, **options)

            if idx == 0 and result.get("language"):
                detected_language = result["language"]
                if "language" not in options or options["language"] is None:
                    options["language"] = detected_language

            for seg in result.get("segments", []):
                if not seg.get("text", "").strip():
                    continue

                seg["start"] += offset
                seg["end"] += offset

                if "words" in seg:
                    for w in seg["words"]:
                        w["start"] += offset
                        w["end"] += offset

                # Deduplicate overlap regions
                if all_segments:
                    last = all_segments[-1]
                    if (seg["start"] < last["end"]
                            and seg["text"].strip() == last["text"].strip()):
                        continue

                all_segments.append(seg)

            if progress_callback:
                progress_callback({
                    "progress": int((idx + 1) / len(chunks) * 100),
                    "status": f"chunk {idx+1}/{len(chunks)}",
                })

        return {
            "text": " ".join(s.get("text", "") for s in all_segments),
            "segments": all_segments,
            "language": detected_language or self.config.language,
        }

    def _format_segments(self, raw: dict) -> List[Dict[str, Any]]:
        """Normalize raw Whisper output into clean segment dicts."""
        segments = []
        for idx, seg in enumerate(raw.get("segments", [])):
            text = seg.get("text", "").strip()
            if not text:
                continue

            entry = {
                "id": idx + 1,
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": text,
            }

            if self.config.word_timestamps and "words" in seg:
                entry["words"] = [
                    {
                        "word": w.get("word", "").strip(),
                        "start": round(w.get("start", 0), 3),
                        "end": round(w.get("end", 0), 3),
                        "probability": round(w.get("probability", 1.0), 4),
                    }
                    for w in seg["words"]
                    if w.get("word", "").strip()
                ]

            segments.append(entry)

        return segments

    def cleanup(self):
        """Release model resources."""
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Transcriber cleaned up")
