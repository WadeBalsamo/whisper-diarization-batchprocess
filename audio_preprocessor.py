"""
Handles extraction of audio from any media container, resampling, channel
mixing, and optional preprocessing (normalization, high-pass filtering).

Uses ffmpeg/ffprobe for broad format support without loading full files into
memory via Python libraries.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Extract and preprocess audio from arbitrary media files."""

    def __init__(self, config):
        self.config = config
        self._validate_ffmpeg()

    def _validate_ffmpeg(self):
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, check=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError(
                "ffmpeg not found. Install it: https://ffmpeg.org/download.html"
            )

    def extract_audio(self, media_path: str) -> Tuple[np.ndarray, int]:
        """
        Extract mono float32 audio at target sample rate from any media file.

        Returns:
            (audio_array, sample_rate)
        """
        media_path = str(Path(media_path).resolve())
        sr = self.config.target_sample_rate

        logger.info(f"Extracting audio from {Path(media_path).name} at {sr}Hz")

        cmd = [
            "ffmpeg", "-i", media_path,
            "-vn",  # drop video
        ]

        if self.config.high_pass_hz > 0:
            cmd += ["-af", f"highpass=f={self.config.high_pass_hz}"]

        cmd += [
            "-acodec", "pcm_f32le",
            "-ar", str(sr),
            "-ac", str(self.config.target_channels),
            "-f", "f32le",
            "pipe:1"
        ]

        result = subprocess.run(cmd, capture_output=True, check=False)

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")
            raise RuntimeError(f"ffmpeg failed: {stderr[:500]}")

        audio = np.frombuffer(result.stdout, dtype=np.float32).copy()

        duration_hours = len(audio) / sr / 3600
        if duration_hours > self.config.max_duration_hours:
            raise ValueError(
                f"Audio duration {duration_hours:.1f}h exceeds "
                f"max {self.config.max_duration_hours}h"
            )

        if self.config.normalize_audio:
            audio = self._normalize(audio)

        logger.info(
            f"Extracted {len(audio)/sr:.1f}s of audio "
            f"(peak={np.abs(audio).max():.3f})"
        )
        return audio, sr

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Peak-normalize to [-0.95, 0.95] with slight headroom."""
        peak = np.abs(audio).max()
        if peak > 0 and peak != 1.0:
            audio = audio / peak * 0.95
        return audio

    def get_media_info(self, media_path: str) -> dict:
        """Probe media file for metadata using ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(media_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {"valid": False, "error": result.stderr[:300]}

        probe = json.loads(result.stdout)

        audio_stream = next(
            (s for s in probe.get("streams", [])
             if s.get("codec_type") == "audio"),
            None
        )

        if not audio_stream:
            return {"valid": False, "error": "No audio stream found"}

        return {
            "valid": True,
            "duration": float(probe["format"].get("duration", 0)),
            "sample_rate": int(audio_stream.get("sample_rate", 0)),
            "channels": int(audio_stream.get("channels", 0)),
            "codec": audio_stream.get("codec_name", "unknown"),
            "format": probe["format"].get("format_name", "unknown"),
        }

    def save_audio_to_wav(
        self, audio: np.ndarray, sample_rate: int,
        output_path: Optional[str] = None
    ) -> str:
        """Save audio array to WAV file (needed by some diarization backends)."""
        import soundfile as sf

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = tmp.name
            tmp.close()

        sf.write(output_path, audio, sample_rate, subtype="FLOAT")
        logger.info(f"Saved audio to {output_path}")
        return output_path
