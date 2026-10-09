"""Audio recording module for AI Minutes.

Captures audio from an input device (microphone/stereo mixer)
and saves it as 16kHz mono 16-bit PCM WAV.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
SUBTYPE = "PCM_16"


def list_input_devices() -> List[Dict[str, Any]]:
    """List all available audio input devices.

    Returns:
        List of dicts containing device index, name, hostapi, and max input channels.
    """
    devices = sd.query_devices()
    input_devices: List[Dict[str, Any]] = []

    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            name = dev.get("name", f"Device {idx}")
            # Ensure name can be printed cleanly
            input_devices.append(
                {
                    "index": idx,
                    "name": name,
                    "hostapi": dev.get("hostapi", 0),
                    "channels": dev.get("max_input_channels", 1),
                    "default_samplerate": dev.get("default_samplerate", 44100),
                }
            )
    return input_devices


def print_input_devices() -> None:
    """Print available audio input devices in a user-friendly format."""
    devices = list_input_devices()
    default_input = sd.default.device[0]

    print("\n--- 利用可能なオーディオ入力デバイス一覧 ---")
    if not devices:
        print("  ※ 入力デバイスが見つかりませんでした。")
        return

    for dev in devices:
        is_default = " [既定]" if dev["index"] == default_input else ""
        print(
            f"  [{dev['index']:2d}] {dev['name']}{is_default} "
            f"(最大入力ch: {dev['channels']})"
        )
    print("-------------------------------------------\n")


class AudioRecorder:
    """Records audio from a specified input device to a 16kHz mono WAV file."""

    def __init__(
        self,
        device_index: Optional[int] = None,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
    ) -> None:
        """Initialize recorder.

        Args:
            device_index: Audio device index (None for default system device).
            sample_rate: Target sample rate in Hz (default: 16000).
            channels: Target channel count (default: 1 for mono).
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._is_recording = False
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback from sounddevice stream."""
        if status:
            logger.warning("Audio callback status: %s", status)
        if self._is_recording:
            self._q.put(indata.copy())

    def record_interactive(
        self, output_path: str | Path, stop_event: Optional[threading.Event] = None
    ) -> Path:
        """Start recording and wait for user to press Enter or until stop_event is set.

        Args:
            output_path: Path to save the resulting WAV file.
            stop_event: Optional threading.Event to signal stop programmatically.

        Returns:
            Path to the saved WAV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._is_recording = True
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

        print("\n==========================================")
        print("🎙️  録音を開始しました")
        print(f"    デバイスID: {self.device_index if self.device_index is not None else '既定'}")
        print(f"    サンプリング: {self.sample_rate} Hz (モノラル)")
        print(f"    保存先: {output_path}")
        print("    👉 [Enter] キーを押すと録音を停止します...")
        print("==========================================\n")

        # Open InputStream
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            device=self.device_index,
            channels=self.channels,
            dtype="int16",
            callback=self._audio_callback,
        )

        frames: List[np.ndarray] = []
        start_time = time.time()

        def _wait_for_input():
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
            self._is_recording = False

        input_thread = threading.Thread(target=_wait_for_input, daemon=True)

        with self._stream:
            input_thread.start()
            try:
                while self._is_recording:
                    if stop_event and stop_event.is_set():
                        self._is_recording = False
                        break
                    try:
                        data = self._q.get(timeout=0.1)
                        frames.append(data)
                    except queue.Empty:
                        continue
            except KeyboardInterrupt:
                print("\n[Ctrl+C] 録音中断シグナルを受信しました。")
                self._is_recording = False

        # Drain any remaining frames
        while not self._q.empty():
            try:
                frames.append(self._q.get_nowait())
            except queue.Empty:
                break

        elapsed = time.time() - start_time
        print(f"\n⏹️  録音を停止しました (録音時間: {elapsed:.1f} 秒)")

        if not frames:
            logger.warning("録音データが空です。")
            # Create a 0.5s silence file to prevent crashes
            empty_data = np.zeros((int(self.sample_rate * 0.5), self.channels), dtype=np.int16)
            sf.write(str(output_path), empty_data, self.sample_rate, subtype=SUBTYPE)
            return output_path

        audio_data = np.concatenate(frames, axis=0)
        sf.write(str(output_path), audio_data, self.sample_rate, subtype=SUBTYPE)
        print(f"💾 WAVファイルを保存しました: {output_path} ({len(audio_data) / self.sample_rate:.1f}秒)\n")
        return output_path
