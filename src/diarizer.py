"""Speaker Diarization module for AI Minutes.

Uses nvidia/Nemotron-3-Diarization via Hugging Face transformers
to determine 'who spoke when' with up to 8 speakers.
Supports automatic CUDA to CPU fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "nvidia/Nemotron-3-Diarization"


class NemotronDiarizer:
    """Speaker Diarization using nvidia/Nemotron-3-Diarization."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        speaker_prefix: str = "話者 ",
    ) -> None:
        """Initialize Nemotron diarizer.

        Args:
            model_id: Hugging Face model repository ID.
            device: 'cuda' or 'cpu'. If None, auto-detected.
            speaker_prefix: Prefix for speaker label (e.g. '話者 ' -> '話者 0').
        """
        self.model_id = model_id
        self.speaker_prefix = speaker_prefix
        self.device = device
        self._processor = None
        self._model = None

    def _init_device(self) -> str:
        if self.device:
            return self.device

        import torch

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        return self.device

    def _load_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        import torch
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor

        device = self._init_device()
        print(f"🔄 Nemotron-3-Diarization モデル読み込み中... ({self.model_id}, デバイス: {device})")

        self._processor = AutoProcessor.from_pretrained(self.model_id)

        try:
            if device == "cuda":
                self._model = AutoModelForAudioFrameClassification.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            else:
                self._model = AutoModelForAudioFrameClassification.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                ).to("cpu")
        except Exception as e:
            if device == "cuda":
                logger.warning("CUDAでのNemotron読み込みに失敗しました。CPUにフォールバックします: %s", e)
                print("⚠️  CUDAエラーのため、NemotronをCPUモードにフォールバックします...")
                self.device = "cpu"
                self._model = AutoModelForAudioFrameClassification.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                ).to("cpu")
            else:
                raise e

        self._model.eval()

    def diarize(self, audio_path: str | Path) -> List[Dict[str, Any]]:
        """Perform speaker diarization on a 16kHz audio file.

        Args:
            audio_path: Path to the WAV file.

        Returns:
            List of diarization segments:
            [{'speaker': '話者 0', 'start': 0.5, 'end': 4.2}, ...]
        """
        import torch

        self._load_model()
        audio_path = Path(audio_path).resolve()

        print(f"👥 話者分離（Diarization）を実行中... [{audio_path.name}]")

        # Load audio data
        # Target sampling rate from processor feature extractor
        target_sr = getattr(self._processor.feature_extractor, "sampling_rate", 16000)

        data, sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            # Convert multi-channel to mono if necessary
            data = np.mean(data, axis=1)

        if sr != target_sr:
            logger.warning("音声のサンプリングレートが %d Hz です (期待値: %d Hz)", sr, target_sr)

        # Prepare inputs
        inputs = self._processor(data, sampling_rate=target_sr, return_tensors="pt")
        target_device = self._model.device
        model_dtype = next(self._model.parameters()).dtype

        inputs = {
            k: v.to(device=target_device, dtype=model_dtype if v.is_floating_point() else None)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            outputs = self._model(**inputs)
            logits = outputs.logits

        # Extract speaker dictionary using processor helper
        attention_mask = inputs.get("attention_mask", None)
        extracted = self._processor.extract_speaker_dict(logits, attention_mask)

        # extracted is a list of segments for batch index 0
        raw_segments = extracted[0] if isinstance(extracted, list) and len(extracted) > 0 else []

        segments: List[Dict[str, Any]] = []
        for item in raw_segments:
            # Handle different key naming conventions gracefully
            spk_id = item.get("Speaker", item.get("speaker", 0))
            start_sec = float(item.get("Start", item.get("start", 0.0)))
            end_sec = float(item.get("End", item.get("end", 0.0)))

            if end_sec <= start_sec:
                continue

            spk_label = f"{self.speaker_prefix}{spk_id}"
            segments.append(
                {
                    "speaker": spk_label,
                    "speaker_id": spk_id,
                    "start": round(start_sec, 2),
                    "end": round(end_sec, 2),
                }
            )

        # Sort by start time
        segments.sort(key=lambda x: x["start"])
        print(f"✅ 話者分離完了: {len(segments)} 件の発話区間を検出しました。")
        return segments
