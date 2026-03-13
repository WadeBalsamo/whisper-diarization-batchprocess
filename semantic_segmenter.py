"""
Intelligent segmentation using:
  1. Sentence-level embeddings for local similarity scoring
  2. Adaptive thresholding on a similarity curve
  3. LLM-based topic shift detection at candidate boundaries
  4. Pause and speaker-turn signals as supplementary cues
  5. Merge pass to rejoin oversplit segments

No hardcoded word lists. Boundaries emerge from the embedding geometry
and LLM judgment.
"""

import logging
import math
from typing import List, Tuple, Any
from dataclasses import dataclass

import numpy as np
from scipy.signal import argrelmin

logger = logging.getLogger(__name__)


@dataclass
class CoherentSection:
    """A topically coherent section of the transcript."""
    section_id: int
    sentences: List[Any]  # List of LabeledSentence
    start: float
    end: float
    primary_speaker: str
    topic_summary: str = ""
    coherence_score: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)

    def to_dict(self) -> dict:
        speaker_counts: dict = {}
        for s in self.sentences:
            speaker_counts[s.speaker] = speaker_counts.get(s.speaker, 0) + 1
        return {
            "section_id": self.section_id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "primary_speaker": self.primary_speaker,
            "speaker_distribution": speaker_counts,
            "sentence_count": self.sentence_count,
            "topic_summary": self.topic_summary,
            "coherence_score": round(self.coherence_score, 4),
            "text": self.text,
        }


class SemanticSegmenter:
    """
    Segments a sequence of labeled sentences into coherent topical sections
    using embedding similarity analysis and LLM-based topic shift detection.
    """

    def __init__(self, config):
        self.config = config
        self.embedding_model = None
        self.llm = None
        self.llm_tokenizer = None
        self._load_embedding_model()
        self._load_llm()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_embedding_model(self):
        """Load sentence-transformer for embedding generation."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install: pip install sentence-transformers"
            )

        logger.info(
            f"Loading embedding model: {self.config.embedding_model}"
        )
        self.embedding_model = SentenceTransformer(
            self.config.embedding_model
        )
        logger.info("Embedding model loaded")

    def _load_llm(self):
        """Load a causal LM for topic shift detection.

        Tries the primary model first, then the fallback. If both fail
        (or both are empty strings indicating --no-llm), falls back to
        embedding-only segmentation.
        """
        import torch

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            logger.warning(
                "transformers not installed — using embedding-only segmentation"
            )
            return

        models_to_try = [
            m for m in [self.config.llm_model, self.config.llm_fallback_model]
            if m  # skip empty strings
        ]

        for model_name in models_to_try:
            try:
                logger.info(f"Loading LLM for topic detection: {model_name}")
                self.llm_tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

                load_kwargs = {"trust_remote_code": True}

                if self.config.llm_use_4bit and torch.cuda.is_available():
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    load_kwargs["device_map"] = "auto"
                else:
                    load_kwargs["torch_dtype"] = (
                        torch.float16 if torch.cuda.is_available()
                        else torch.float32
                    )

                self.llm = AutoModelForCausalLM.from_pretrained(
                    model_name, **load_kwargs
                )

                if not torch.cuda.is_available():
                    self.llm = self.llm.to("cpu")

                logger.info(f"LLM loaded: {model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue

        logger.warning(
            "No LLM loaded — falling back to embedding-only segmentation"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(
        self, sentences: List[Any]
    ) -> List[CoherentSection]:
        """
        Segment a sequence of LabeledSentence objects into coherent sections.

        Pipeline:
          1. Embed all sentences
          2. Compute pairwise sequential similarity curve
          3. Compute pause signal curve
          4. Compute speaker-turn signal curve
          5. Identify candidate boundaries via adaptive thresholding
          6. Score candidates with LLM topic-shift detection
          7. Select final boundaries
          8. Build sections
          9. Merge pass to rejoin oversplit sections
         10. Generate topic summaries

        Returns:
            List of CoherentSection objects
        """
        if not sentences:
            return []

        n = len(sentences)
        logger.info(f"Segmenting {n} sentences into coherent sections")

        if n < self.config.min_segment_sentences:
            return [self._make_section(0, sentences, 1.0)]

        # Step 1: Embed
        texts = [s.text for s in sentences]
        embeddings = self._embed(texts)

        # Step 2: Sequential similarity curve
        sim_curve = self._compute_similarity_curve(embeddings)

        # Step 3: Pause signal
        pause_curve = self._compute_pause_curve(sentences)

        # Step 4: Speaker turn signal
        turn_curve = self._compute_speaker_turn_curve(sentences)

        # Step 5: Combined boundary signal (low = likely boundary)
        boundary_signal = self._combine_signals(
            sim_curve, pause_curve, turn_curve
        )

        # Step 6: Candidate boundaries via adaptive threshold
        candidates = self._find_candidate_boundaries(
            boundary_signal, sentences
        )

        logger.info(f"Found {len(candidates)} candidate boundaries")

        # Step 7: Score with LLM
        if self.llm is not None:
            scored = self._score_with_llm(candidates, sentences)
        else:
            scored = [(idx, 0.5) for idx in candidates]

        # Step 8: Select final boundaries
        final_boundaries = self._select_boundaries(
            scored, boundary_signal, sentences
        )

        logger.info(f"Selected {len(final_boundaries)} final boundaries")

        # Step 9: Build sections
        sections = self._build_sections(final_boundaries, sentences, embeddings)

        # Step 10: Merge pass
        sections = self._merge_small_sections(sections, embeddings, sentences)

        # Step 11: Summaries
        sections = self._generate_summaries(sections)

        logger.info(
            f"Segmentation complete: {len(sections)} coherent sections"
        )
        return sections

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Encode texts into normalized embedding vectors."""
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.config.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings)

    # ------------------------------------------------------------------
    # Signal curves
    # ------------------------------------------------------------------

    def _compute_similarity_curve(
        self, embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute local semantic similarity between consecutive windows.

        For each position i, computes cosine similarity between the mean
        embedding of a window ending at i and a window starting at i+1.
        Low values indicate potential topic shifts.

        Returns array of length n-1 (one value per gap between sentences).
        """
        n = len(embeddings)
        w = self.config.similarity_window_size
        sims = np.zeros(n - 1)

        for i in range(n - 1):
            left_start = max(0, i - w + 1)
            left_emb = embeddings[left_start:i + 1].mean(axis=0)

            right_end = min(n, i + 1 + w)
            right_emb = embeddings[i + 1:right_end].mean(axis=0)

            # Cosine similarity (already L2-normalized)
            sims[i] = float(np.dot(left_emb, right_emb))

        return sims

    def _compute_pause_curve(self, sentences: List[Any]) -> np.ndarray:
        """
        Compute pause duration signal between consecutive sentences.
        Longer pauses -> lower values (more likely boundary).

        Returns array of length n-1, values in [0, 1].
        """
        n = len(sentences)
        pauses = np.zeros(n - 1)

        for i in range(n - 1):
            gap = max(0.0, sentences[i + 1].start - sentences[i].end)
            thresh = self.config.long_pause_threshold_seconds
            # Sigmoid: large gaps -> value near 0 (likely boundary)
            pauses[i] = 1.0 / (1.0 + math.exp((gap - thresh) / (thresh / 3)))

        return pauses

    def _compute_speaker_turn_curve(
        self, sentences: List[Any]
    ) -> np.ndarray:
        """
        Speaker change signal: 0.0 where speaker changes, 1.0 where same.

        Returns array of length n-1.
        """
        n = len(sentences)
        turns = np.ones(n - 1)

        for i in range(n - 1):
            if sentences[i].speaker != sentences[i + 1].speaker:
                turns[i] = 0.0

        return turns

    def _combine_signals(
        self,
        similarity: np.ndarray,
        pause: np.ndarray,
        speaker_turn: np.ndarray,
    ) -> np.ndarray:
        """
        Combine signals into a single boundary likelihood curve.
        Lower values = more likely boundary.
        """
        w_pause = self.config.pause_weight
        w_turn = self.config.speaker_turn_weight
        w_sim = 1.0 - w_pause - w_turn

        if w_sim < 0:
            logger.warning(
                f"pause_weight ({w_pause}) + speaker_turn_weight ({w_turn}) > 1.0; "
                f"clamping similarity weight to 0"
            )
            total = w_pause + w_turn
            w_pause /= total
            w_turn /= total
            w_sim = 0.0

        return (
            w_sim * similarity
            + w_pause * pause
            + w_turn * speaker_turn
        )

    # ------------------------------------------------------------------
    # Candidate boundary detection
    # ------------------------------------------------------------------

    def _find_candidate_boundaries(
        self, signal: np.ndarray, sentences: List[Any]
    ) -> List[int]:
        """
        Find candidate boundary positions using adaptive thresholding.

        A candidate boundary at index i means a potential break BETWEEN
        sentence i and sentence i+1.
        """
        n = len(signal)

        threshold = np.percentile(
            signal, self.config.adaptive_threshold_percentile
        )

        local_min_indices: set = set()
        if n > 4:
            mins = argrelmin(signal, order=2)[0]
            local_min_indices = set(mins.tolist())

        candidates = []
        last_boundary_idx = -1
        for i in range(n):
            # Force a boundary if the current segment exceeds max duration
            if last_boundary_idx >= 0:
                seg_start_time = sentences[last_boundary_idx + 1].start
            else:
                seg_start_time = sentences[0].start
            seg_end_time = sentences[i].end
            force = (
                seg_end_time - seg_start_time
                > self.config.max_segment_duration_seconds
                and i - (last_boundary_idx if last_boundary_idx >= 0 else -1)
                > self.config.min_segment_sentences
            )

            if force or signal[i] < threshold or i in local_min_indices:
                if self._boundary_satisfies_constraints(
                    i, candidates, sentences
                ):
                    candidates.append(i)
                    last_boundary_idx = i

        return candidates

    def _boundary_satisfies_constraints(
        self, idx: int, existing: List[int], sentences: List[Any]
    ) -> bool:
        """Check that a boundary at idx satisfies min/max segment size."""
        min_sents = self.config.min_segment_sentences
        n = len(sentences)

        if idx < min_sents - 1:
            return False
        if idx > n - min_sents - 1:
            return False

        # Must be at least min_sents away from previous boundary
        if existing and idx - existing[-1] < min_sents:
            return False

        # Duration check
        if existing:
            seg_start = sentences[existing[-1] + 1].start
        else:
            seg_start = sentences[0].start

        seg_end = sentences[idx].end
        if seg_end - seg_start < self.config.min_segment_duration_seconds:
            return False

        return True

    # ------------------------------------------------------------------
    # LLM topic shift scoring
    # ------------------------------------------------------------------

    def _score_with_llm(
        self, candidates: List[int], sentences: List[Any]
    ) -> List[Tuple[int, float]]:
        """
        For each candidate boundary, compute a perplexity-based topic
        shift score using the LLM.
        """
        import torch

        scored = []
        device = next(self.llm.parameters()).device

        for idx in candidates:
            before_start = max(0, idx - 3)
            before_text = " ".join(
                s.text for s in sentences[before_start:idx + 1]
            )

            after_end = min(len(sentences), idx + 5)
            after_text = " ".join(
                s.text for s in sentences[idx + 1:after_end]
            )

            shift_score = self._llm_topic_shift_score(
                before_text, after_text, device
            )
            scored.append((idx, shift_score))

        return scored

    def _llm_topic_shift_score(
        self, text_before: str, text_after: str, device
    ) -> float:
        """
        Use perplexity-based scoring: if text_after is surprising given
        text_before as context, there's likely a topic shift.

        Returns a score in [0, 1] where higher = more likely topic shift.
        """
        import torch

        max_ctx = 300
        text_before = text_before[-max_ctx:]
        text_after = text_after[:max_ctx]

        prompt = f"{text_before} {text_after}"

        try:
            inputs = self.llm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            before_ids = self.llm_tokenizer(
                text_before,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            ctx_len = before_ids.input_ids.shape[1]

            with torch.no_grad():
                outputs = self.llm(**inputs)
                logits = outputs.logits

            if logits.shape[1] <= ctx_len + 1:
                return 0.5

            shift_logits = logits[:, ctx_len:-1, :].contiguous()
            shift_labels = inputs["input_ids"][:, ctx_len + 1:].contiguous()

            if shift_logits.shape[1] == 0 or shift_labels.shape[1] == 0:
                return 0.5

            min_len = min(shift_logits.shape[1], shift_labels.shape[1])
            shift_logits = shift_logits[:, :min_len, :]
            shift_labels = shift_labels[:, :min_len]

            loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            perplexity = torch.exp(loss).item()

            # Calibrated sigmoid: midpoint ~15, scale ~8
            shift_prob = 1.0 / (1.0 + math.exp(-(perplexity - 15) / 8))

            return float(np.clip(shift_prob, 0.0, 1.0))

        except Exception as e:
            logger.warning(f"LLM scoring failed: {e}")
            return 0.5

    # ------------------------------------------------------------------
    # Boundary selection
    # ------------------------------------------------------------------

    def _select_boundaries(
        self,
        scored_candidates: List[Tuple[int, float]],
        boundary_signal: np.ndarray,
        sentences: List[Any],
    ) -> List[int]:
        """
        Select final boundaries from scored candidates.
        Combines the embedding-based signal with LLM shift scores.
        """
        if not scored_candidates:
            return []

        w_llm = self.config.llm_shift_weight
        w_emb = 1.0 - w_llm

        final_scores = []
        for idx, llm_score in scored_candidates:
            emb_score = 1.0 - boundary_signal[idx]
            combined = w_emb * emb_score + w_llm * llm_score
            final_scores.append((idx, combined))

        # Greedy selection: highest-scoring first, respecting spacing
        final_scores.sort(key=lambda x: x[1], reverse=True)

        selected: List[int] = []
        for idx, score in final_scores:
            if score < 0.4:
                continue

            too_close = any(
                abs(idx - sel_idx) < self.config.min_segment_sentences
                for sel_idx in selected
            )

            if not too_close:
                selected.append(idx)

        selected.sort()
        return selected

    # ------------------------------------------------------------------
    # Section construction
    # ------------------------------------------------------------------

    def _build_sections(
        self,
        boundaries: List[int],
        sentences: List[Any],
        embeddings: np.ndarray,
    ) -> List[CoherentSection]:
        """Build CoherentSection objects from boundary indices."""
        sections = []
        n = len(sentences)

        starts = [0] + [b + 1 for b in boundaries]
        ends = [b + 1 for b in boundaries] + [n]

        for sec_idx, (s, e) in enumerate(zip(starts, ends)):
            if s >= e:
                continue

            seg_sentences = sentences[s:e]
            seg_embeddings = embeddings[s:e]

            # Internal coherence: mean cosine similarity to centroid
            if len(seg_embeddings) > 1:
                centroid = seg_embeddings.mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid /= norm
                coherence = float(np.mean([
                    np.dot(emb, centroid) for emb in seg_embeddings
                ]))
            else:
                coherence = 1.0

            speaker_counts: dict = {}
            for sent in seg_sentences:
                speaker_counts[sent.speaker] = (
                    speaker_counts.get(sent.speaker, 0) + 1
                )
            primary = max(speaker_counts, key=speaker_counts.get)

            section = CoherentSection(
                section_id=sec_idx + 1,
                sentences=seg_sentences,
                start=seg_sentences[0].start,
                end=seg_sentences[-1].end,
                primary_speaker=primary,
                coherence_score=coherence,
            )
            sections.append(section)

        return sections

    # ------------------------------------------------------------------
    # Merge pass
    # ------------------------------------------------------------------

    def _merge_small_sections(
        self,
        sections: List[CoherentSection],
        embeddings: np.ndarray,
        sentences: List[Any],
    ) -> List[CoherentSection]:
        """Merge sections that are too short or too similar to neighbors."""
        if len(sections) <= 1:
            return sections

        merged = [sections[0]]

        for section in sections[1:]:
            prev = merged[-1]

            should_merge = (
                section.sentence_count < self.config.min_segment_sentences
                or section.duration < self.config.min_segment_duration_seconds
            )

            if not should_merge:
                prev_emb = self._section_embedding(prev, sentences, embeddings)
                curr_emb = self._section_embedding(
                    section, sentences, embeddings
                )
                sim = float(np.dot(prev_emb, curr_emb))
                if sim > self.config.merge_if_similarity_above:
                    should_merge = True

            if should_merge:
                merged[-1] = CoherentSection(
                    section_id=prev.section_id,
                    sentences=prev.sentences + section.sentences,
                    start=prev.start,
                    end=section.end,
                    primary_speaker=prev.primary_speaker,
                    coherence_score=(
                        prev.coherence_score + section.coherence_score
                    ) / 2,
                )
            else:
                merged.append(section)

        for i, sec in enumerate(merged):
            sec.section_id = i + 1

        return merged

    def _section_embedding(
        self,
        section: CoherentSection,
        all_sentences: List[Any],
        all_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute mean normalized embedding for a section."""
        # Build an identity map for fast lookup
        sentence_id_map = {id(s): i for i, s in enumerate(all_sentences)}
        indices = [
            sentence_id_map[id(sent)]
            for sent in section.sentences
            if id(sent) in sentence_id_map
        ]

        if not indices:
            return np.zeros(all_embeddings.shape[1])

        embs = all_embeddings[indices]
        mean_emb = embs.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        return mean_emb / norm if norm > 0 else mean_emb

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def _generate_summaries(
        self, sections: List[CoherentSection]
    ) -> List[CoherentSection]:
        """Generate brief topic summaries for each section using LLM."""
        if self.llm is None:
            for section in sections:
                if section.sentences:
                    first = section.sentences[0].text
                    section.topic_summary = (
                        first[:100] + "..." if len(first) > 100 else first
                    )
            return sections

        import torch
        device = next(self.llm.parameters()).device

        for section in sections:
            text_sample = section.text[:500]
            prompt = (
                f"Briefly summarize the main topic of this passage "
                f"in one short phrase (under 10 words):\n\n"
                f"\"{text_sample}\"\n\n"
                f"Topic:"
            )

            try:
                inputs = self.llm_tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=600,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.llm.generate(
                        **inputs,
                        max_new_tokens=20,
                        temperature=0.3,
                        do_sample=True,
                        pad_token_id=self.llm_tokenizer.pad_token_id,
                    )

                generated = self.llm_tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()

                summary = generated.split("\n")[0].strip().strip('"').strip("'")
                section.topic_summary = summary[:80]

            except Exception as e:
                logger.warning(f"Summary generation failed: {e}")
                if section.sentences:
                    section.topic_summary = section.sentences[0].text[:80]

        return sections

    def _make_section(
        self, start_idx: int, sentences: List[Any], coherence: float
    ) -> CoherentSection:
        """Convenience to wrap all sentences in a single section."""
        speaker_counts: dict = {}
        for s in sentences:
            speaker_counts[s.speaker] = speaker_counts.get(s.speaker, 0) + 1
        primary = (
            max(speaker_counts, key=speaker_counts.get)
            if speaker_counts else "UNKNOWN"
        )

        return CoherentSection(
            section_id=1,
            sentences=sentences,
            start=sentences[0].start,
            end=sentences[-1].end,
            primary_speaker=primary,
            coherence_score=coherence,
        )

    def cleanup(self):
        """Release model resources."""
        del self.embedding_model
        if self.llm is not None:
            del self.llm
            del self.llm_tokenizer
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
