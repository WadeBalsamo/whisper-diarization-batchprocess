"""
End-to-end pipeline: media file -> coherent speaker-labeled sections.

Orchestrates all stages in sequence with progress reporting and structured
output in multiple formats (JSON, SRT, plain text).
"""

import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, Any, Optional

import srt as srt_lib

from config import PipelineConfig
from audio_preprocessor import AudioPreprocessor
from transcription_engine import Transcriber
from diarization_engine import Diarizer
from speaker_aligner import SpeakerAligner
from semantic_segmenter import SemanticSegmenter

logger = logging.getLogger(__name__)


class PipelineResult:
    """Complete pipeline output."""

    def __init__(
        self,
        sections: list,
        sentences: list,
        transcription_result: Any,
        diarization_result: Any,
        metadata: dict,
    ):
        self.sections = sections
        self.sentences = sentences
        self.transcription = transcription_result
        self.diarization = diarization_result
        self.metadata = metadata

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "sections": [s.to_dict() for s in self.sections],
            "sentences": [s.to_dict() for s in self.sentences],
            "speaker_summary": self._speaker_summary(),
        }

    def _speaker_summary(self) -> dict:
        if not self.diarization:
            return {}
        durations = self.diarization.speaker_durations()
        primary = self.diarization.identify_primary_speaker()
        return {
            "num_speakers": self.diarization.num_speakers,
            "primary_speaker": primary,
            "speaker_durations_seconds": {
                k: round(v, 1) for k, v in durations.items()
            },
        }

    def to_srt(self) -> str:
        """Generate SRT subtitle file from sentences."""
        subtitles = []
        for section in self.sections:
            for sent in section.sentences:
                subtitle = srt_lib.Subtitle(
                    index=len(subtitles) + 1,
                    start=timedelta(seconds=sent.start),
                    end=timedelta(seconds=sent.end),
                    content=f"[{sent.speaker}] {sent.text}",
                )
                subtitles.append(subtitle)
        return srt_lib.compose(subtitles)

    def to_section_srt(self) -> str:
        """Generate SRT with one subtitle per section."""
        subtitles = []
        for section in self.sections:
            text_preview = section.text[:200]
            if len(section.text) > 200:
                text_preview += "..."
            subtitle = srt_lib.Subtitle(
                index=section.section_id,
                start=timedelta(seconds=section.start),
                end=timedelta(seconds=section.end),
                content=(
                    f"[Section {section.section_id}: "
                    f"{section.topic_summary}]\n"
                    f"[{section.primary_speaker}] "
                    f"{text_preview}"
                ),
            )
            subtitles.append(subtitle)
        return srt_lib.compose(subtitles)


class Pipeline:
    """
    Full processing pipeline.

    Usage:
        config = PipelineConfig()
        pipe = Pipeline(config)
        result = pipe.process("interview.mp4")
        pipe.save_results(result)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._setup_logging()

        logger.info("Initializing pipeline components...")

        self.preprocessor = AudioPreprocessor(self.config.audio)
        self.transcriber = Transcriber(self.config.transcription)
        self.diarizer = Diarizer(self.config.diarization)
        self.aligner = SpeakerAligner(self.config.alignment)
        self.segmenter = SemanticSegmenter(self.config.segmentation)

        logger.info("Pipeline ready")

    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

    def process(
        self,
        media_path: str,
        progress_callback: Optional[Callable] = None,
    ) -> PipelineResult:
        """
        Process a media file through the complete pipeline.

        Stages:
          1. Audio extraction & preprocessing
          2. Transcription (Whisper)
          3. Speaker diarization (Pyannote)
          4. Word-to-speaker alignment
          5. Semantic segmentation

        Args:
            media_path: Path to audio/video file
            progress_callback: Optional callback receiving {progress, status}

        Returns:
            PipelineResult with sections, sentences, and metadata
        """
        media_path = str(Path(media_path).resolve())
        logger.info(f"Processing: {Path(media_path).name}")

        # Validate
        info = self.preprocessor.get_media_info(media_path)
        if not info.get("valid"):
            raise ValueError(f"Invalid media file: {info.get('error')}")

        # Stage 1: Audio extraction
        self._report(progress_callback, 5, "Extracting audio")
        audio, sr = self.preprocessor.extract_audio(media_path)

        # Stage 2: Transcription
        self._report(progress_callback, 15, "Transcribing")
        transcription = self.transcriber.transcribe(audio, sr)
        logger.info(
            f"Transcription: {len(transcription.segments)} segments, "
            f"language={transcription.language}"
        )

        # Stage 3: Diarization
        self._report(progress_callback, 45, "Diarizing speakers")
        diarization = self.diarizer.diarize(audio, sr)
        logger.info(
            f"Diarization: {diarization.num_speakers} speakers, "
            f"{len(diarization.speaker_segments)} turns"
        )

        # Stage 4: Alignment
        self._report(progress_callback, 70, "Aligning speakers to words")
        speaker_triples = diarization.to_millis_triples()
        sentences = self.aligner.align(
            transcription.segments, speaker_triples
        )
        logger.info(f"Alignment: {len(sentences)} labeled sentences")

        # Stage 5: Segmentation
        self._report(progress_callback, 80, "Detecting topic boundaries")
        sections = self.segmenter.segment(sentences)
        logger.info(f"Segmentation: {len(sections)} coherent sections")

        self._report(progress_callback, 100, "Complete")

        metadata = {
            "source_file": Path(media_path).name,
            "duration_seconds": round(transcription.duration, 1),
            "language": transcription.language,
            "whisper_model": transcription.model_name,
            "num_speakers": diarization.num_speakers,
            "num_sentences": len(sentences),
            "num_sections": len(sections),
        }

        return PipelineResult(
            sections=sections,
            sentences=sentences,
            transcription_result=transcription,
            diarization_result=diarization,
            metadata=metadata,
        )

    def save_results(
        self, result: PipelineResult, output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """Save all outputs to disk."""
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}

        # Full JSON
        json_path = out / "result.json"
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        paths["json"] = str(json_path)

        # Sentence-level SRT
        srt_path = out / "transcript.srt"
        with open(srt_path, "w") as f:
            f.write(result.to_srt())
        paths["srt"] = str(srt_path)

        # Section-level SRT
        section_srt_path = out / "sections.srt"
        with open(section_srt_path, "w") as f:
            f.write(result.to_section_srt())
        paths["section_srt"] = str(section_srt_path)

        # Plain text with speaker labels
        txt_path = out / "transcript.txt"
        with open(txt_path, "w") as f:
            for section in result.sections:
                f.write(
                    f"\n{'='*60}\n"
                    f"Section {section.section_id}: {section.topic_summary}\n"
                    f"[{section.start:.1f}s - {section.end:.1f}s] "
                    f"Primary: {section.primary_speaker}\n"
                    f"{'='*60}\n\n"
                )
                for sent in section.sentences:
                    f.write(f"  [{sent.speaker}] {sent.text}\n")
        paths["txt"] = str(txt_path)

        logger.info(f"Results saved to {out}")
        return paths

    def _report(self, callback, progress, status):
        if callback:
            callback({"progress": progress, "status": status})

    def cleanup(self):
        """Release all model resources."""
        self.transcriber.cleanup()
        self.diarizer.cleanup()
        self.segmenter.cleanup()
        logger.info("Pipeline cleaned up")
