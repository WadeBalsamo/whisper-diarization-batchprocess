"""
Speaker diarization using Pyannote Audio 3.x.

Produces speaker-labeled time intervals from audio. Replaces the NeMo MSDD
backend from the original diarize.py while maintaining the same downstream
interface (start_ms, end_ms, speaker) triples for the alignment stage.
"""

import logging
import os
import tempfile
from typing import List, Tuple, Optional, Dict

import numpy as np

logger = logging.getLogger(__name__)


class SpeakerSegment:
    """A contiguous interval assigned to one speaker."""
    __slots__ = ("start", "end", "speaker")

    def __init__(self, start: float, end: float, speaker: str):
        self.start = start
        self.end = end
        self.speaker = speaker

    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "speaker": self.speaker,
        }


class DiarizationResult:
    """Container for diarization output."""

    def __init__(self, speaker_segments: List[SpeakerSegment],
                 num_speakers: int):
        self.speaker_segments = speaker_segments
        self.num_speakers = num_speakers

    @property
    def speaker_labels(self) -> List[str]:
        return sorted(set(s.speaker for s in self.speaker_segments))

    def get_speaker_at(self, time_seconds: float) -> Optional[str]:
        """Find the speaker active at a given timestamp."""
        for seg in self.speaker_segments:
            if seg.start <= time_seconds <= seg.end:
                return seg.speaker
        return None

    def to_millis_triples(self) -> List[Tuple[int, int, str]]:
        """Convert to (start_ms, end_ms, speaker) format for alignment.

        This matches the interface expected by SpeakerAligner, analogous to
        the speaker_ts format from the original NeMo RTTM parsing.
        """
        return [
            (int(s.start * 1000), int(s.end * 1000), s.speaker)
            for s in self.speaker_segments
        ]

    def speaker_durations(self) -> Dict[str, float]:
        durations: Dict[str, float] = {}
        for seg in self.speaker_segments:
            durations[seg.speaker] = durations.get(seg.speaker, 0) + seg.duration()
        return durations

    def identify_primary_speaker(self) -> str:
        """Return speaker with most total speaking time."""
        durations = self.speaker_durations()
        return max(durations, key=durations.get)


class Diarizer:
    """Speaker diarization using Pyannote Audio."""

    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self):
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise RuntimeError(
                "pyannote.audio not installed. "
                "Install: pip install pyannote.audio"
            )

        logger.info(f"Loading diarization model: {self.config.model_name}")

        kwargs = {}
        if self.config.auth_token:
            kwargs["use_auth_token"] = self.config.auth_token

        try:
            self.pipeline = Pipeline.from_pretrained(
                self.config.model_name, **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            raise

        # Move to GPU if available
        import torch
        if torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))
            logger.info("Diarization pipeline moved to CUDA")

        logger.info("Diarization pipeline loaded")

    def diarize(
        self, audio: np.ndarray, sample_rate: int
    ) -> DiarizationResult:
        """
        Run speaker diarization on audio.

        Args:
            audio: mono float32 audio array
            sample_rate: sample rate in Hz

        Returns:
            DiarizationResult with speaker segments
        """
        import soundfile as sf

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            sf.write(tmp_path, audio, sample_rate, subtype="FLOAT")

            logger.info("Running speaker diarization...")

            params = {}
            if self.config.min_speakers is not None:
                params["min_speakers"] = self.config.min_speakers
            if self.config.max_speakers is not None:
                params["max_speakers"] = self.config.max_speakers

            annotation = self.pipeline(tmp_path, **params)

            segments = []
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                if turn.duration < self.config.min_segment_duration:
                    continue
                segments.append(SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                ))

            segments.sort(key=lambda s: s.start)

            num_speakers = len(set(s.speaker for s in segments))
            logger.info(
                f"Diarization complete: {num_speakers} speakers, "
                f"{len(segments)} segments"
            )

            return DiarizationResult(segments, num_speakers)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def cleanup(self):
        """Release model resources."""
        del self.pipeline
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
