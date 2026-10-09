"""Speech-to-Text (ASR) module for AI Minutes.

Uses faster-whisper (CTranslate2 backend) to transcribe audio
with automatic CUDA (float16) to CPU (int8) fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    """Check if CUDA is available via torch or ctranslate2."""
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except Exception:
        pass

    try:
        import ctranslate2

        supported_types = ctranslate2.get_supported_compute_types("cuda")
        return len(supported_types) > 0
    except Exception:
        pass

    return False


class WhisperTranscriber:
    """Transcription using faster-whisper."""

    def __init__(
        self,
        model_size_or_path: str = "large-v3-turbo",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: str = "ja",
    ) -> None:
        """Initialize Whisper model.

        Args:
            model_size_or_path: faster-whisper model name (default: 'large-v3-turbo').
            device: 'cuda' or 'cpu'. If None, auto-detected.
            compute_type: 'float16', 'int8', 'float32', etc. If None, auto-selected.
            language: Target language code (default: 'ja').
        """
        self.model_size_or_path = model_size_or_path
        self.language = language

        if device is None:
            self.device = "cuda" if is_cuda_available() else "cpu"
        else:
            self.device = device

        if compute_type is None:
            if self.device == "cuda":
                self.compute_type = "float16"
            else:
                self.compute_type = "int8"
        else:
            self.compute_type = compute_type

        logger.info(
            "WhisperTranscriber configured: model=%s, device=%s, compute_type=%s, language=%s",
            self.model_size_or_path,
            self.device,
            self.compute_type,
            self.language,
        )
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        print(
            f"🔄 Whisper モデル読み込み中... ({self.model_size_or_path}, デバイス: {self.device}, 計算型: {self.compute_type})"
        )
        try:
            self._model = WhisperModel(
                self.model_size_or_path,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as e:
            if self.device == "cuda":
                logger.warning("CUDAでのWhisper起動に失敗しました。CPU (int8) にフォールバックします: %s", e)
                print("⚠️  CUDAエラーのため、WhisperをCPUモードにフォールバックします...")
                self.device = "cpu"
                self.compute_type = "int8"
                self._model = WhisperModel(
                    self.model_size_or_path,
                    device="cpu",
                    compute_type="int8",
                )
            else:
                raise e

    def transcribe(
        self,
        audio_path: str | Path,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> List[Dict[str, Any]]:
        """Transcribe an audio file into timed text segments.

        Args:
            audio_path: Path to the WAV/audio file.
            beam_size: Beam search width (default: 5).
            vad_filter: Enable Silero VAD to filter out non-speech segments.

        Returns:
            List of segments: [{'start': float, 'end': float, 'text': str}, ...]
        """
        self._load_model()
        audio_path = str(Path(audio_path).resolve())

        print(f"🎙️  音声認識（ASR）を実行中... [{audio_path}]")
        segments_gen, info = self._model.transcribe(
            audio_path,
            language=self.language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=False,
        )

        results: List[Dict[str, Any]] = []
        for seg in segments_gen:
            text = seg.text.strip()
            if not text:
                continue
            results.append(
                {
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": text,
                }
            )

        print(f"✅ 音声認識完了: {len(results)} 件のセグメントを抽出しました。")
        return results
