"""
src/agent/grounder.py
─────────────────────
GrounderAgent: links [^N] footnotes in the report back to exact character
spans in source chunks using difflib fuzzy matching.

Known limitation (documented in README and AgentReport):
The difflib SequenceMatcher threshold (0.6) is calibrated for paraphrased
or condensed claims, not verbatim quotes. A claim can score above the threshold
and be "grounded" even if it's slightly inaccurate — citation_coverage measures
traceability, not truth. This is explicitly noted in EvaluationResult.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import List, Tuple

from src.models.schemas import Chunk, CitationLink

logger = logging.getLogger(__name__)

# Regex to find [^N] footnote markers in the report text
_FOOTNOTE_RE = re.compile(r"\[\^(\d+)\]")
# Minimum similarity ratio for a claim to be considered grounded
_SIMILARITY_THRESHOLD = 0.6
# Max chars to the left/right of the citation marker to extract as the claim
_CLAIM_WINDOW = 300


class GrounderAgent:
    """
    Links report footnote citations back to character spans in source chunks.
    """

    def ground(
        self,
        report_text: str,
        all_chunks: List[Chunk],
    ) -> Tuple[str, List[CitationLink]]:
        """
        Find all [^N] footnote markers in the report, attempt to match each
        to a character span in one of the source chunks via difflib, and
        build CitationLink objects.

        The report text is returned unchanged — grounding only produces
        the CitationLink index that the UI uses to highlight source spans.

        Args:
            report_text: The raw Markdown report from SynthesizerAgent.
            all_chunks: All chunks available in the store for this query.

        Returns:
            Tuple of:
              - report_text: Unchanged (returned for workflow convenience).
              - citations: List of CitationLink objects, one per matched footnote.
        """
        if not all_chunks:
            logger.warning("GrounderAgent: no chunks provided — returning empty citations")
            return report_text, []

        citations: List[CitationLink] = []
        matched_footnotes: set[int] = set()

        for match in _FOOTNOTE_RE.finditer(report_text):
            footnote_id = int(match.group(1))
            if footnote_id in matched_footnotes:
                continue  # only link each footnote once

            # Extract claim context: window of chars before the [^N] marker
            claim_start = max(0, match.start() - _CLAIM_WINDOW)
            claim_text = report_text[claim_start:match.start()].strip()

            # Find the best matching chunk for this claim
            best_link = self._find_best_match(footnote_id, claim_text, all_chunks)
            if best_link:
                citations.append(best_link)
                matched_footnotes.add(footnote_id)
            else:
                logger.debug(
                    f"GrounderAgent: could not ground footnote [^{footnote_id}] "
                    f"(claim='{claim_text[:80]}…')"
                )

        logger.info(
            f"GrounderAgent: matched {len(citations)}/{len(matched_footnotes | {int(m.group(1)) for m in _FOOTNOTE_RE.finditer(report_text)})} footnotes"
        )
        return report_text, citations

    def _find_best_match(
        self,
        footnote_id: int,
        claim_text: str,
        chunks: List[Chunk],
    ) -> CitationLink | None:
        """
        Use difflib to find the chunk with the highest similarity to the claim.

        Returns a CitationLink if similarity ≥ _SIMILARITY_THRESHOLD, else None.
        """
        if not claim_text:
            return None

        best_ratio = 0.0
        best_chunk: Chunk | None = None
        best_char_start = 0
        best_char_end = 0

        for chunk in chunks:
            matcher = difflib.SequenceMatcher(
                None, claim_text.lower(), chunk.text.lower(), autojunk=False
            )
            
            # matcher.ratio() uses len(a)+len(b), which unfairly penalizes when the
            # chunk is much larger than the claim (which is always true here).
            # Instead, we calculate the percentage of claim characters that exist
            # in the chunk as contiguous blocks (size >= 5 to filter noise).
            blocks = matcher.get_matching_blocks()
            matched_chars = sum(b.size for b in blocks if b.size >= 5)
            ratio = matched_chars / len(claim_text) if claim_text else 0.0

            if ratio > best_ratio:
                best_ratio = ratio
                best_chunk = chunk
                # Find the best matching block within the chunk for the char span
                best_block = max(
                    matcher.get_matching_blocks(), key=lambda b: b.size, default=None
                )
                if best_block and best_block.size > 0:
                    # b.b is the start of the match in chunk.text
                    # Offset by chunk.char_start to get absolute document position
                    best_char_start = chunk.char_start + best_block.b
                    best_char_end = chunk.char_start + best_block.b + best_block.size
                else:
                    best_char_start = chunk.char_start
                    best_char_end = chunk.char_end

        if best_ratio < _SIMILARITY_THRESHOLD or best_chunk is None:
            return None

        excerpt = best_chunk.text[
            best_char_start - best_chunk.char_start:
            best_char_end - best_chunk.char_start
        ].strip()

        return CitationLink(
            footnote_id=footnote_id,
            chunk_id=best_chunk.chunk_id,
            char_start=best_char_start,
            char_end=best_char_end,
            excerpt=excerpt,
        )
