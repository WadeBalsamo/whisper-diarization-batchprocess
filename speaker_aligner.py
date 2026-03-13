"""
Aligns word-level transcription timestamps with speaker diarization segments.
Handles punctuation restoration and sentence boundary realignment.

Core alignment logic is adapted from the original diarize.py functions:
  - get_words_speaker_mapping -> _map_words_to_speakers
  - get_realigned_ws_mapping_with_punctuation -> _realign_at_boundaries
  - get_sentences_speaker_mapping -> _group_into_sentences

Refactored into a class with clear method boundaries and typed interfaces.
"""

import logging
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AlignedWord:
    """A word with timestamp and speaker assignment."""
    word: str
    start: float  # seconds
    end: float    # seconds
    speaker: str
    probability: float = 1.0


@dataclass
class LabeledSentence:
    """A sentence with speaker label and timing."""
    text: str
    speaker: str
    start: float
    end: float
    words: List[AlignedWord] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "word_count": len(self.words),
        }


class SpeakerAligner:
    """
    Aligns transcription words with diarization speaker labels.

    Pipeline:
      1. Extract words with timestamps from transcription
      2. Map each word to a speaker using diarization timestamps
      3. Restore punctuation if enabled
      4. Realign speaker labels at sentence boundaries
      5. Group into speaker-labeled sentences
    """

    SENTENCE_ENDINGS = ".?!"

    def __init__(self, config):
        self.config = config
        self.punct_model = None

        if config.restore_punctuation:
            self._load_punctuation_model()

    def _load_punctuation_model(self):
        try:
            from deepmultilingualpunctuation import PunctuationModel
            self.punct_model = PunctuationModel(
                model=self.config.punctuation_model
            )
            logger.info("Punctuation restoration model loaded")
        except ImportError:
            logger.warning(
                "deepmultilingualpunctuation not installed. "
                "Punctuation restoration disabled."
            )
        except Exception as e:
            logger.warning(f"Failed to load punctuation model: {e}")

    def align(
        self,
        transcription_segments: List[Dict[str, Any]],
        speaker_segments: List[Tuple[int, int, str]],
    ) -> List[LabeledSentence]:
        """
        Main alignment entry point.

        Args:
            transcription_segments: Whisper output segments with 'words' field
            speaker_segments: (start_ms, end_ms, speaker_label) triples

        Returns:
            List of LabeledSentence objects
        """
        # Step 1: Flatten all words
        words = self._extract_words(transcription_segments)
        if not words:
            logger.warning("No words extracted from transcription")
            return []

        # Step 2: Map words to speakers
        word_speaker_map = self._map_words_to_speakers(words, speaker_segments)

        # Step 3: Restore punctuation
        if self.punct_model and self.config.restore_punctuation:
            word_speaker_map = self._restore_punctuation(word_speaker_map)

        # Step 4: Realign at sentence boundaries
        word_speaker_map = self._realign_at_boundaries(word_speaker_map)

        # Step 5: Group into sentences
        sentences = self._group_into_sentences(word_speaker_map)

        logger.info(f"Alignment produced {len(sentences)} labeled sentences")
        return sentences

    def _extract_words(
        self, segments: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Extract flat word list with timestamps from transcription segments."""
        words = []
        for seg in segments:
            for w in seg.get("words", []):
                word_text = w.get("word", "").strip()
                if not word_text:
                    continue
                words.append({
                    "word": word_text,
                    "start": w.get("start", seg.get("start", 0)),
                    "end": w.get("end", seg.get("end", 0)),
                    "probability": w.get("probability", 1.0),
                })
        return words

    def _get_anchor_time(self, start: float, end: float) -> float:
        """Get anchor timestamp for word-to-speaker matching.

        Mirrors get_word_ts_anchor from diarize.py.
        """
        anchor = self.config.word_anchor
        if anchor == "end":
            return end
        elif anchor == "mid":
            return (start + end) / 2
        return start

    def _map_words_to_speakers(
        self,
        words: List[Dict],
        speaker_segments: List[Tuple[int, int, str]],
    ) -> List[Dict[str, Any]]:
        """Map each word to the speaker active at that word's anchor time.

        Adapted from get_words_speaker_mapping in diarize.py. Uses a sliding
        pointer over sorted speaker segments for O(n) alignment.
        """
        if not speaker_segments:
            return [{**w, "speaker": "SPEAKER_00"} for w in words]

        result = []
        spk_idx = 0
        s, e, sp = speaker_segments[0]

        for w in words:
            anchor_ms = int(self._get_anchor_time(w["start"], w["end"]) * 1000)

            # Advance speaker pointer to the interval containing anchor
            while anchor_ms > e and spk_idx < len(speaker_segments) - 1:
                spk_idx += 1
                s, e, sp = speaker_segments[spk_idx]

            # If past the last segment, extend it (same as original)
            if spk_idx == len(speaker_segments) - 1 and anchor_ms > e:
                e = int(w["end"] * 1000)

            result.append({
                "word": w["word"],
                "start": w["start"],
                "end": w["end"],
                "speaker": sp,
                "probability": w.get("probability", 1.0),
            })

        return result

    def _restore_punctuation(
        self, word_map: List[Dict]
    ) -> List[Dict]:
        """Use punctuation model to add sentence-ending punctuation.

        Adapted from the punctuation restoration block in diarize_audio.
        """
        words_only = [w["word"] for w in word_map]

        try:
            labeled = self.punct_model.predict(words_only)
        except Exception as exc:
            logger.warning(f"Punctuation restoration failed: {exc}")
            return word_map

        ending_puncts = set(self.SENTENCE_ENDINGS)
        model_puncts = set(".,;:!?")
        is_acronym = lambda x: bool(re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x))

        for entry, (_, punct_label) in zip(word_map, labeled):
            word = entry["word"]
            if (
                word
                and punct_label in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += punct_label
                if word.endswith(".."):
                    word = word.rstrip(".")
                entry["word"] = word

        return word_map

    def _realign_at_boundaries(
        self, word_map: List[Dict]
    ) -> List[Dict]:
        """
        When a sentence straddles a speaker boundary, assign the whole
        sentence to the majority speaker. Prevents mid-sentence speaker
        switches caused by diarization imprecision.

        Direct refactoring of get_realigned_ws_mapping_with_punctuation
        from diarize.py.
        """
        max_words = self.config.max_words_in_sentence
        words_list = [w["word"] for w in word_map]
        speaker_list = [w["speaker"] for w in word_map]
        n = len(word_map)

        def is_sentence_end(idx: int) -> bool:
            if idx < 0 or idx >= n:
                return False
            return (
                words_list[idx]
                and words_list[idx][-1] in self.SENTENCE_ENDINGS
            )

        def find_sentence_start(idx: int) -> int:
            left = idx
            while (
                left > 0
                and idx - left < max_words
                and speaker_list[left - 1] == speaker_list[left]
                and not is_sentence_end(left - 1)
            ):
                left -= 1
            if left == 0 or is_sentence_end(left - 1):
                return left
            return -1

        def find_sentence_end(idx: int, budget: int) -> int:
            right = idx
            while (
                right < n
                and right - idx < budget
                and not is_sentence_end(right)
            ):
                right += 1
            if right == n - 1 or is_sentence_end(right):
                return right
            return -1

        k = 0
        while k < n:
            if (
                k < n - 1
                and speaker_list[k] != speaker_list[k + 1]
                and not is_sentence_end(k)
            ):
                left = find_sentence_start(k)
                right = (
                    find_sentence_end(k, max_words - k + left - 1)
                    if left > -1 else -1
                )

                if min(left, right) == -1:
                    k += 1
                    continue

                span_speakers = speaker_list[left:right + 1]
                majority = max(set(span_speakers), key=span_speakers.count)

                if span_speakers.count(majority) >= len(span_speakers) // 2:
                    speaker_list[left:right + 1] = [majority] * (
                        right - left + 1
                    )
                    k = right

            k += 1

        # Apply realigned speakers back
        realigned = []
        for i, entry in enumerate(word_map):
            new_entry = entry.copy()
            new_entry["speaker"] = speaker_list[i]
            realigned.append(new_entry)

        return realigned

    def _group_into_sentences(
        self, word_map: List[Dict]
    ) -> List[LabeledSentence]:
        """Group word-speaker mappings into sentences, splitting on speaker
        change or sentence-ending punctuation.

        Adapted from get_sentences_speaker_mapping in diarize.py.
        """
        if not word_map:
            return []

        sentences = []
        current_words: List[Dict] = []
        current_speaker = word_map[0]["speaker"]

        for entry in word_map:
            # Speaker change -> close current sentence
            if entry["speaker"] != current_speaker and current_words:
                sentences.append(self._build_sentence(
                    current_words, current_speaker
                ))
                current_words = []
                current_speaker = entry["speaker"]

            current_words.append(entry)

            # Sentence-ending punctuation -> close sentence
            if (
                entry["word"]
                and entry["word"][-1] in self.SENTENCE_ENDINGS
                and current_words
            ):
                sentences.append(self._build_sentence(
                    current_words, current_speaker
                ))
                current_words = []

        # Flush remainder
        if current_words:
            sentences.append(self._build_sentence(
                current_words, current_speaker
            ))

        return sentences

    def _build_sentence(
        self, words: List[Dict], speaker: str
    ) -> LabeledSentence:
        text = " ".join(w["word"] for w in words).strip()
        aligned_words = [
            AlignedWord(
                word=w["word"],
                start=w["start"],
                end=w["end"],
                speaker=w["speaker"],
                probability=w.get("probability", 1.0),
            )
            for w in words
        ]
        return LabeledSentence(
            text=text,
            speaker=speaker,
            start=words[0]["start"],
            end=words[-1]["end"],
            words=aligned_words,
        )
