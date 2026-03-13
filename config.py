"""
Pipeline configuration with sensible defaults and environment variable overrides.

All configuration is centralized here using dataclasses. Each pipeline stage
has its own config section, and the top-level PipelineConfig composes them.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class DeviceStrategy(Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


@dataclass
class AudioConfig:
    """Audio preprocessing configuration."""
    target_sample_rate: int = 16000
    target_channels: int = 1  # mono
    target_format: str = "f32le"
    normalize_audio: bool = True
    high_pass_hz: int = 80  # filter rumble below 80Hz
    max_duration_hours: float = 4.0


@dataclass
class TranscriptionConfig:
    """Whisper transcription configuration."""
    model_name: str = "base.en"
    device: DeviceStrategy = DeviceStrategy.AUTO
    language: str = "en"
    word_timestamps: bool = True
    beam_size: int = 5
    temperature: float = 0.0
    condition_on_previous_text: bool = True
    no_speech_threshold: float = 0.6
    compression_ratio_threshold: float = 2.4
    chunk_length_seconds: int = 1800
    chunk_overlap_seconds: float = 0.5


@dataclass
class DiarizationConfig:
    """Pyannote speaker diarization configuration."""
    model_name: str = "pyannote/speaker-diarization-3.1"
    auth_token: Optional[str] = field(
        default_factory=lambda: os.environ.get("HF_TOKEN")
    )
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    min_segment_duration: float = 0.5


@dataclass
class AlignmentConfig:
    """Word-to-speaker alignment configuration."""
    word_anchor: str = "start"  # "start", "mid", "end"
    max_words_in_sentence: int = 50
    restore_punctuation: bool = True
    punctuation_model: str = "kredor/punctuate-all"


@dataclass
class SegmentationConfig:
    """Semantic segmentation configuration."""
    # Embedding model for semantic similarity
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # LLM for topic shift detection
    llm_model: str = "meta-llama/Llama-3.1-8B"
    llm_use_4bit: bool = True
    llm_fallback_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Segmentation parameters
    min_segment_sentences: int = 3
    max_segment_sentences: int = 40
    min_segment_duration_seconds: float = 10.0
    max_segment_duration_seconds: float = 300.0

    # Adaptive threshold parameters
    similarity_window_size: int = 5
    base_similarity_threshold: float = 0.65
    adaptive_threshold_percentile: float = 25.0  # breakpoints at bottom 25%

    # Silence-based cues
    long_pause_threshold_seconds: float = 2.0
    pause_weight: float = 0.3

    # Speaker turn weight
    speaker_turn_weight: float = 0.2

    # LLM topic shift weight
    llm_shift_weight: float = 0.5

    # Final merge pass
    merge_if_similarity_above: float = 0.80


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)

    output_dir: str = "./output"
    log_level: str = "INFO"

    def resolve_device(self) -> torch.device:
        strategy = self.transcription.device
        if strategy == DeviceStrategy.AUTO:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(strategy.value)
