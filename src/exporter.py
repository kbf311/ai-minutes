"""Result Exporter module for AI Minutes.

Formats aligned segments into Markdown, TXT, JSON, and SRT files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.aligner import AlignedSegment, format_seconds_to_hms


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds into SubRip (SRT) format HH:MM:SS,mmm."""
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class ResultExporter:
    """Exports transcribed and diarized results to various formats."""

    @staticmethod
    def to_formatted_lines(segments: List[AlignedSegment]) -> List[str]:
        """Convert segments into standard line format.

        Format: [HH:MM:SS - HH:MM:SS] 話者 X: 発話内容
        """
        lines: List[str] = []
        for seg in segments:
            start_str = format_seconds_to_hms(seg.start)
            end_str = format_seconds_to_hms(seg.end)
            line = f"[{start_str} - {end_str}] {seg.speaker}: {seg.text}"
            lines.append(line)
        return lines

    @classmethod
    def save_markdown(
        cls,
        segments: List[AlignedSegment],
        output_path: str | Path,
        title: str = "会議議事録（話者分離・文字起こし結果）",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Export results as a formatted Markdown file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [f"# {title}\n"]

        if metadata:
            lines.append("## 実行メタデータ")
            for k, v in metadata.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        lines.append("## 議事録本文\n")
        formatted = cls.to_formatted_lines(segments)
        for line in formatted:
            lines.append(f"{line}\n")

        content = "\n".join(lines)
        output_path.write_text(content, encoding="utf-8")
        return output_path

    @classmethod
    def save_text(
        cls,
        segments: List[AlignedSegment],
        output_path: str | Path,
    ) -> Path:
        """Export results as a plain text file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        formatted = cls.to_formatted_lines(segments)
        content = "\n".join(formatted) + "\n"
        output_path.write_text(content, encoding="utf-8")
        return output_path

    @classmethod
    def save_json(
        cls,
        segments: List[AlignedSegment],
        output_path: str | Path,
        metadata: Optional[Dict[str, Any]] = None,
        raw_whisper: Optional[List[Dict[str, Any]]] = None,
        raw_diarization: Optional[List[Dict[str, Any]]] = None,
    ) -> Path:
        """Export results as structured JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data: Dict[str, Any] = {
            "metadata": metadata or {},
            "utterances": [seg.to_dict() for seg in segments],
        }
        if raw_whisper is not None:
            data["raw_whisper_segments"] = raw_whisper
        if raw_diarization is not None:
            data["raw_diarization_segments"] = raw_diarization

        content = json.dumps(data, ensure_ascii=False, indent=2)
        output_path.write_text(content, encoding="utf-8")
        return output_path

    @classmethod
    def save_srt(
        cls,
        segments: List[AlignedSegment],
        output_path: str | Path,
    ) -> Path:
        """Export results as SubRip (SRT) subtitle file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        blocks: List[str] = []
        for i, seg in enumerate(segments, start=1):
            s_time = format_srt_timestamp(seg.start)
            e_time = format_srt_timestamp(seg.end)
            text = f"[{seg.speaker}] {seg.text}"
            blocks.append(f"{i}\n{s_time} --> {e_time}\n{text}\n")

        content = "\n".join(blocks)
        output_path.write_text(content, encoding="utf-8")
        return output_path

    @classmethod
    def export_all(
        cls,
        segments: List[AlignedSegment],
        base_output_path: str | Path,
        title: str = "会議議事録",
        metadata: Optional[Dict[str, Any]] = None,
        raw_whisper: Optional[List[Dict[str, Any]]] = None,
        raw_diarization: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Path]:
        """Export to all standard formats (.md, .txt, .json, .srt)."""
        base_path = Path(base_output_path)
        stem = base_path.with_suffix("")

        saved_files = {
            "markdown": cls.save_markdown(segments, stem.with_suffix(".md"), title, metadata),
            "text": cls.save_text(segments, stem.with_suffix(".txt")),
            "json": cls.save_json(segments, stem.with_suffix(".json"), metadata, raw_whisper, raw_diarization),
            "srt": cls.save_srt(segments, stem.with_suffix(".srt")),
        }
        return saved_files
