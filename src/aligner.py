"""Alignment and Merging module for AI Minutes.

Aligns Whisper speech segments with Nemotron diarization speaker segments
using temporal overlap (IoU / overlap duration) and merges consecutive
utterances by the same speaker.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AlignedSegment:
    """Represents a finalized utterance with speaker label and timestamps."""

    speaker: str
    start: float
    end: float
    text: str
    confidence_overlap: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def format_seconds_to_hms(seconds: float) -> str:
    """Format seconds into HH:MM:SS format."""
    total_sec = max(0, int(round(seconds)))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def compute_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculate overlap duration in seconds between two intervals."""
    overlap = min(end1, end2) - max(start1, start2)
    return max(0.0, overlap)


class TimestampAligner:
    """Aligns ASR segments and speaker diarization intervals."""

    def __init__(
        self,
        default_speaker: str = "話者 0",
        merge_max_gap_seconds: float = 2.5,
    ) -> None:
        """Initialize aligner.

        Args:
            default_speaker: Speaker name used when no overlap is found.
            merge_max_gap_seconds: Maximum silence gap (seconds) between
                consecutive segments of the same speaker to merge.
        """
        self.default_speaker = default_speaker
        self.merge_max_gap_seconds = merge_max_gap_seconds

    def align(
        self,
        whisper_segments: List[Dict[str, Any]],
        diarization_segments: List[Dict[str, Any]],
    ) -> List[AlignedSegment]:
        """Assign the best-matching speaker to each Whisper segment.

        Args:
            whisper_segments: List of dicts with 'start', 'end', 'text'.
            diarization_segments: List of dicts with 'start', 'end', 'speaker'.

        Returns:
            List of AlignedSegment objects.
        """
        aligned: List[AlignedSegment] = []

        if not whisper_segments:
            return aligned

        if not diarization_segments:
            logger.warning("話者分離結果が空のため、すべての発言に既定話者を割り当てます。")
            return [
                AlignedSegment(
                    speaker=self.default_speaker,
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"],
                    confidence_overlap=0.0,
                )
                for seg in whisper_segments
            ]

        for w_seg in whisper_segments:
            w_start = float(w_seg["start"])
            w_end = float(w_seg["end"])
            w_text = w_seg["text"]
            w_dur = max(0.001, w_end - w_start)

            # Accumulate overlap for each speaker
            speaker_overlaps: Dict[str, float] = {}

            for d_seg in diarization_segments:
                d_start = float(d_seg["start"])
                d_end = float(d_seg["end"])
                spk = str(d_seg["speaker"])

                ov = compute_overlap(w_start, w_end, d_start, d_end)
                if ov > 0:
                    speaker_overlaps[spk] = speaker_overlaps.get(spk, 0.0) + ov

            best_speaker = self.default_speaker
            best_overlap = 0.0

            if speaker_overlaps:
                # Speaker with highest overlap duration
                best_speaker, best_overlap = max(
                    speaker_overlaps.items(), key=lambda item: item[1]
                )
            else:
                # Fallback: find temporally closest diarization segment
                closest_seg = min(
                    diarization_segments,
                    key=lambda d: min(
                        abs(float(d["start"]) - w_end),
                        abs(float(d["end"]) - w_start),
                    ),
                )
                best_speaker = str(closest_seg["speaker"])
                best_overlap = 0.0

            overlap_ratio = min(1.0, best_overlap / w_dur)

            aligned.append(
                AlignedSegment(
                    speaker=best_speaker,
                    start=w_start,
                    end=w_end,
                    text=w_text,
                    confidence_overlap=round(overlap_ratio, 3),
                )
            )

        return aligned

    def merge_consecutive(
        self,
        aligned_segments: List[AlignedSegment],
    ) -> List[AlignedSegment]:
        """Merge consecutive segments spoken by the same speaker.

        Args:
            aligned_segments: List of AlignedSegment objects.

        Returns:
            Merged list of AlignedSegment objects.
        """
        if not aligned_segments:
            return []

        merged: List[AlignedSegment] = []
        current = AlignedSegment(
            speaker=aligned_segments[0].speaker,
            start=aligned_segments[0].start,
            end=aligned_segments[0].end,
            text=aligned_segments[0].text,
            confidence_overlap=aligned_segments[0].confidence_overlap,
        )

        for next_seg in aligned_segments[1:]:
            # If same speaker and gap within threshold
            gap = next_seg.start - current.end
            if next_seg.speaker == current.speaker and gap <= self.merge_max_gap_seconds:
                # Merge texts: append cleanly
                # Check whether to add space (e.g. alphanumeric or Japanese)
                current_text = current.text.rstrip()
                next_text = next_seg.text.lstrip()

                if current_text and next_text:
                    # If both ends are ASCII/English words, add a space
                    if current_text[-1].isascii() and next_text[0].isascii():
                        current.text = f"{current_text} {next_text}"
                    else:
                        current.text = f"{current_text}{next_text}"
                else:
                    current.text = current_text + next_text

                current.end = max(current.end, next_seg.end)
                current.confidence_overlap = round(
                    (current.confidence_overlap + next_seg.confidence_overlap) / 2.0, 3
                )
            else:
                merged.append(current)
                current = AlignedSegment(
                    speaker=next_seg.speaker,
                    start=next_seg.start,
                    end=next_seg.end,
                    text=next_seg.text,
                    confidence_overlap=next_seg.confidence_overlap,
                )

        merged.append(current)
        return merged

    def process(
        self,
        whisper_segments: List[Dict[str, Any]],
        diarization_segments: List[Dict[str, Any]],
    ) -> List[AlignedSegment]:
        """Execute full alignment and merging pipeline.

        Args:
            whisper_segments: Output from Whisper ASR.
            diarization_segments: Output from Nemotron Diarization.

        Returns:
            List of final merged AlignedSegments.
        """
        aligned = self.align(whisper_segments, diarization_segments)
        merged = self.merge_consecutive(aligned)
        return merged
