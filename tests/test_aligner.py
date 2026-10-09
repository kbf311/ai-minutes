"""Unit tests for TimestampAligner and ResultExporter."""

import json
from pathlib import Path

from src.aligner import AlignedSegment, TimestampAligner, compute_overlap, format_seconds_to_hms
from src.exporter import ResultExporter, format_srt_timestamp


def test_compute_overlap():
    # Overlap between [1.0, 5.0] and [2.0, 4.0] is 2.0
    assert compute_overlap(1.0, 5.0, 2.0, 4.0) == 2.0
    # Overlap between [1.0, 3.0] and [3.0, 5.0] is 0.0
    assert compute_overlap(1.0, 3.0, 3.0, 5.0) == 0.0
    # No overlap
    assert compute_overlap(1.0, 2.0, 3.0, 4.0) == 0.0
    # Partial overlap
    assert compute_overlap(1.0, 4.0, 2.0, 6.0) == 2.0


def test_format_seconds_to_hms():
    assert format_seconds_to_hms(0) == "00:00:00"
    assert format_seconds_to_hms(2) == "00:00:02"
    assert format_seconds_to_hms(65) == "00:01:05"
    assert format_seconds_to_hms(3665) == "01:01:05"


def test_format_srt_timestamp():
    assert format_srt_timestamp(2.5) == "00:00:02,500"
    assert format_srt_timestamp(65.123) == "00:01:05,123"


def test_aligner_logic():
    aligner = TimestampAligner()

    whisper_segments = [
        {"start": 1.0, "end": 4.0, "text": "本日の定例ミーティングを始めます。"},
        {"start": 5.0, "end": 8.0, "text": "よろしくお願いします。"},
    ]

    diarization_segments = [
        {"start": 0.8, "end": 4.2, "speaker": "話者 0"},
        {"start": 4.9, "end": 8.1, "speaker": "話者 1"},
    ]

    aligned = aligner.align(whisper_segments, diarization_segments)
    assert len(aligned) == 2
    assert aligned[0].speaker == "話者 0"
    assert aligned[0].text == "本日の定例ミーティングを始めます。"
    assert aligned[1].speaker == "話者 1"
    assert aligned[1].text == "よろしくお願いします。"


def test_aligner_consecutive_merge():
    aligner = TimestampAligner(merge_max_gap_seconds=2.0)

    whisper_segments = [
        {"start": 1.0, "end": 3.0, "text": "こんにちは。"},
        {"start": 3.5, "end": 6.0, "text": "本日の議題について説明します。"},
        {"start": 7.0, "end": 9.0, "text": "承知いたしました。"},
    ]

    diarization_segments = [
        {"start": 0.5, "end": 6.2, "speaker": "話者 0"},
        {"start": 6.8, "end": 9.5, "speaker": "話者 1"},
    ]

    merged = aligner.process(whisper_segments, diarization_segments)
    assert len(merged) == 2
    assert merged[0].speaker == "話者 0"
    assert merged[0].start == 1.0
    assert merged[0].end == 6.0
    assert "こんにちは。" in merged[0].text
    assert "本日の議題について説明します。" in merged[0].text

    assert merged[1].speaker == "話者 1"
    assert merged[1].start == 7.0
    assert merged[1].end == 9.0


def test_exporter_formats(tmp_path: Path):
    segments = [
        AlignedSegment(speaker="話者 0", start=2.0, end=8.0, text="本日の定例ミーティングを始めます。"),
        AlignedSegment(speaker="話者 1", start=9.0, end=15.0, text="よろしくお願いします。議題の1点目についてですが…"),
    ]

    lines = ResultExporter.to_formatted_lines(segments)
    assert lines[0] == "[00:00:02 - 00:00:08] 話者 0: 本日の定例ミーティングを始めます。"
    assert lines[1] == "[00:00:09 - 00:00:15] 話者 1: よろしくお願いします。議題の1点目についてですが…"

    saved = ResultExporter.export_all(segments, tmp_path / "test_minutes")
    assert saved["markdown"].exists()
    assert saved["text"].exists()
    assert saved["json"].exists()
    assert saved["srt"].exists()

    md_content = saved["markdown"].read_text(encoding="utf-8")
    assert "[00:00:02 - 00:00:08] 話者 0: 本日の定例ミーティングを始めます。" in md_content

    json_content = json.loads(saved["json"].read_text(encoding="utf-8"))
    assert len(json_content["utterances"]) == 2
    assert json_content["utterances"][0]["speaker"] == "話者 0"
