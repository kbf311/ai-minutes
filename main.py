"""Main CLI entry point for AI Minutes.

Allows interactive recording, audio file processing, and device inspection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_recorder import list_input_devices, print_input_devices
from src.pipeline import AIMinutesPipeline


def interactive_menu() -> None:
    """Run interactive CLI menu."""
    while True:
        print("\n" + "=" * 50)
        print("🎙️   AI Minutes - 話者分離付き議事録作成システム")
        print("=" * 50)
        print(" 1. 録音を開始して議事録を作成 (マイク入力)")
        print(" 2. 音声ファイルを指定して議事録を作成 (.wav)")
        print(" 3. 録音デバイス（マイク）一覧を表示")
        print(" 4. 終了")
        print("=" * 50)

        choice = input("番号を選択してください [1-4]: ").strip()

        if choice == "1":
            print("\n--- 録音デバイスの選択 ---")
            print("入力デバイスIDを入力してください (空欄で既定デバイス): ")
            dev_in = input("デバイスID [Enterで既定]: ").strip()
            dev_idx: Optional[int] = int(dev_in) if dev_in.isdigit() else None

            pipeline = AIMinutesPipeline()
            try:
                pipeline.record_and_process(device_index=dev_idx)
            except Exception as e:
                print(f"\n❌ エラーが発生しました: {e}\n")

        elif choice == "2":
            file_path = input("\n音声ファイルのパスを入力してください: ").strip().strip('"').strip("'")
            if not file_path:
                print("※ パスが入力されませんでした。")
                continue
            p = Path(file_path)
            if not p.exists():
                print(f"❌ ファイルが存在しません: {p}")
                continue

            pipeline = AIMinutesPipeline()
            try:
                pipeline.process_audio_file(p)
            except Exception as e:
                print(f"\n❌ エラーが発生しました: {e}\n")

        elif choice == "3":
            print_input_devices()

        elif choice == "4" or choice.lower() in ["exit", "q", "quit"]:
            print("終了します。")
            break
        else:
            print("無効な選択です。1〜4の番号を入力してください。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Minutes: 音声録音・話者分離付き文字起こしシステム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="実行コマンド")

    # Command: record
    rec_parser = subparsers.add_parser("record", help="マイクから音声を録音して議事録を作成")
    rec_parser.add_argument("--device", type=int, default=None, help="オーディオ入力デバイスID")
    rec_parser.add_argument("--model", type=str, default="large-v3-turbo", help="Whisperモデル名 (例: large-v3-turbo, medium, small)")
    rec_parser.add_argument("--device-type", type=str, default=None, choices=["cuda", "cpu"], help="推論デバイス (cuda または cpu)")
    rec_parser.add_argument("--output-dir", type=str, default="output", help="出力保存先ディレクトリ")

    # Command: file
    file_parser = subparsers.add_parser("file", help="既存の音声ファイルを文字起こしして議事録を作成")
    file_parser.add_argument("audio_path", type=str, help="音声ファイルパス (WAV形式推奨)")
    file_parser.add_argument("--model", type=str, default="large-v3-turbo", help="Whisperモデル名 (例: large-v3-turbo, medium, small)")
    file_parser.add_argument("--device-type", type=str, default=None, choices=["cuda", "cpu"], help="推論デバイス (cuda または cpu)")
    file_parser.add_argument("--output-dir", type=str, default="output", help="出力保存先ディレクトリ")

    # Command: devices
    subparsers.add_parser("devices", help="利用可能なオーディオ入力デバイス一覧を表示")

    args = parser.parse_args()

    if args.command is None:
        interactive_menu()
    elif args.command == "devices":
        print_input_devices()
    elif args.command == "record":
        pipeline = AIMinutesPipeline(
            whisper_model=args.model,
            device=args.device_type,
            output_dir=args.output_dir,
        )
        pipeline.record_and_process(device_index=args.device)
    elif args.command == "file":
        pipeline = AIMinutesPipeline(
            whisper_model=args.model,
            device=args.device_type,
            output_dir=args.output_dir,
        )
        pipeline.process_audio_file(args.audio_path)


if __name__ == "__main__":
    main()
