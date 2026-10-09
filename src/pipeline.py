"""Full Pipeline for AI Minutes.

Coordinates recording (or loading audio), Whisper transcription,
Nemotron speaker diarization, timestamp alignment, and exporting.
"""

from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.aligner import AlignedSegment, TimestampAligner
from src.audio_recorder import AudioRecorder
from src.diarizer import NemotronDiarizer
from src.exporter import ResultExporter
from src.transcriber import WhisperTranscriber

logger = logging.getLogger(__name__)


class AIMinutesPipeline:
    """End-to-end pipeline for audio minutes generation."""

    def __init__(
        self,
        whisper_model: str = "large-v3-turbo",
        diarizer_model: str = "nvidia/Nemotron-3-Diarization",
        language: str = "ja",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        output_dir: str | Path = "output",
    ) -> None:
        """Initialize the pipeline.

        Args:
            whisper_model: Whisper model size or HuggingFace repo.
            diarizer_model: Nemotron diarization model repo.
            language: Speech recognition language ('ja').
            device: 'cuda' or 'cpu'. If None, auto-detected.
            compute_type: ASR compute type ('float16' / 'int8').
            output_dir: Directory where generated files are saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.language = language
        self.transcriber = WhisperTranscriber(
            model_size_or_path=whisper_model,
            device=device,
            compute_type=compute_type,
            language=language,
        )
        self.diarizer = NemotronDiarizer(
            model_id=diarizer_model,
            device=device,
        )
        self.aligner = TimestampAligner()

    def process_audio_file(
        self,
        audio_path: str | Path,
        base_output_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process an existing audio file and generate minutes.

        Args:
            audio_path: Path to the audio file (WAV).
            base_output_name: Base filename for output files (without extension).

        Returns:
            Dictionary containing processed segments and paths of saved files.
        """
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if base_output_name is None:
            base_output_name = f"minutes_{audio_path.stem}_{timestamp_str}"

        print("\n" + "=" * 55)
        print(f"🚀 AI Minutes 処理開始: {audio_path.name}")
        print("=" * 55)

        start_time = time.time()

        # Step 1: Transcribe with Whisper
        whisper_segments = self.transcriber.transcribe(audio_path)

        # Step 2: Diarize with Nemotron-3
        diarization_segments = self.diarizer.diarize(audio_path)

        # Step 3: Align and merge
        print("🔗 タイムスタンプの照合と話者マージを実行中...")
        aligned_segments = self.aligner.process(whisper_segments, diarization_segments)
        print(f"✅ 整形完了: 全 {len(aligned_segments)} 件の発言ブロックを生成しました。")

        # Step 4: Export results
        output_base_path = self.output_dir / base_output_name
        metadata = {
            "処理日時": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "入力音声": audio_path.name,
            "Whisperモデル": self.transcriber.model_size_or_path,
            "Diarizationモデル": self.diarizer.model_id,
            "実行デバイス (ASR)": self.transcriber.device,
            "実行デバイス (分離)": self.diarizer.device,
            "言語": self.language,
            "所要時間": f"{time.time() - start_time:.1f} 秒",
        }

        saved_files = ResultExporter.export_all(
            segments=aligned_segments,
            base_output_path=output_base_path,
            title=f"会議議事録 - {audio_path.stem}",
            metadata=metadata,
            raw_whisper=whisper_segments,
            raw_diarization=diarization_segments,
        )

        total_elapsed = time.time() - start_time
        print("\n" + "=" * 55)
        print("🎉 議事録作成が完了しました！")
        print(f"⏱️  総処理時間: {total_elapsed:.1f} 秒")
        print("📁 保存されたファイル:")
        for fmt, fpath in saved_files.items():
            print(f"   [{fmt.upper()}] {fpath}")
        print("=" * 55 + "\n")

        # Display preview in console
        print("--- [議事録プレビュー] ---")
        lines = ResultExporter.to_formatted_lines(aligned_segments)
        if not lines:
            print("  （発話内容が検出されませんでした）")
        else:
            for line in lines[:10]:
                print(f"  {line}")
            if len(lines) > 10:
                print(f"  ... 他 {len(lines) - 10} 件")
        print("--------------------------\n")

        return {
            "segments": aligned_segments,
            "saved_files": saved_files,
            "metadata": metadata,
        }

    def record_and_process(
        self,
        device_index: Optional[int] = None,
        base_output_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record audio from microphone/input device, then process minutes.

        Args:
            device_index: Audio device index.
            base_output_name: Base filename for output files.

        Returns:
            Dictionary containing processed segments and saved file paths.
        """
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if base_output_name is None:
            base_output_name = f"minutes_rec_{timestamp_str}"

        wav_path = self.output_dir / f"{base_output_name}.wav"

        recorder = AudioRecorder(device_index=device_index)
        saved_wav = recorder.record_interactive(wav_path)

        return self.process_audio_file(saved_wav, base_output_name=base_output_name)
