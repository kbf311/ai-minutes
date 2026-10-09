# 実装確認および不明点・仕様確認書

## 1. 作成したアプリケーションの概要

`docs/requirements.md` および `docs/note.md` の要件に基づき、Windows環境で動作するスタンドアロン型音声議事録作成システム **「AI Minutes」** を実装しました。

### 実装完了コンポーネント一覧
| モジュール | ファイルパス | 主な役割 |
|---|---|---|
| **録音モジュール** | [`src/audio_recorder.py`](file:///c:/Work/Projects/kbf311/ai-minutes/src/audio_recorder.py) | `sounddevice` / `soundfile` を使用した 16kHz モノラル WAV (PCM 16-bit) 録音、入力デバイス一覧取得、Enterキーによる開始・停止制御 |
| **音声認識モジュール** | [`src/transcriber.py`](file:///c:/Work/Projects/kbf311/ai-minutes/src/transcriber.py) | `faster-whisper` による日本語音声認識。CUDA fp16 自動検出および CPU int8 フォールバック対応 |
| **話者分離モジュール** | [`src/diarizer.py`](file:///c:/Work/Projects/kbf311/ai-minutes/src/diarizer.py) | `nvidia/Nemotron-3-Diarization`（Transformers）による最大8話者のタイムスタンプ付き発話区間判定 |
| **アライメント・マージ** | [`src/aligner.py`](file:///c:/Work/Projects/kbf311/ai-minutes/src/aligner.py) | Whisper発話とNemotron話者区間の時間的重複（IoU / 重複秒数）計算、同一話者の連続発言マージ |
| **エクスポートモジュール** | [`src/exporter.py`](file:///c:/Work/Projects/kbf311/ai-minutes/src/exporter.py) | Markdown (`.md`), テキスト (`.txt`), 字幕 (`.srt`), 詳細構造化データ (`.json`) の一括出力 |
| **統合パイプライン** | [`src/pipeline.py`](file:///c:/Work/Projects/kbf311/ai-minutes/src/pipeline.py) | 録音・ファイル読み込みから各モデル推論、ファイル保存までを一気通貫で処理 |
| **CLIインターフェース** | [`main.py`](file:///c:/Work/Projects/kbf311/ai-minutes/main.py) | 対話型メニュー（1〜4選択）およびコマンドライン引数指定対応 |
| **起動ランチャー** | [`run.bat`](file:///c:/Work/Projects/kbf311/ai-minutes/run.bat) | Windows用ダブルクリック起動バッチファイル（UTF-8設定済み） |

---

## 2. ご確認・ご相談事項（不明点・環境差異）

実装にあたり、実行環境および今後の運用に関して以下の点をご確認いただけますと幸いです。

### ① 実行環境のGPU（ハードウェア）について
- **現状**: 現在の実行環境を調査したところ、GPUは **`Intel(R) Iris(R) Xe Graphics`（統合グラフィックス）** が搭載されており、NVIDIA CUDA GPU（RTX等）は未検出でした。
- **対応方針**: 本システムでは、CUDAが検出されない場合に自動的に **CPUモード**（Whisper: `int8`、Nemotron-3: `float32`）へ安全にフォールバックするよう実装済みです。
- **ご確認点**:
  - 本番運用は別の NVIDIA GPU 搭載PC（またはGPUクラウド）で行われますでしょうか？
  - それとも、現在のPCのCPU環境を主用途として運用されますでしょうか？（CPU運用の場合は、モデルサイズとして `medium` や `small` を併用すると処理速度が大幅に向上します）

### ② 初回モデルダウンロードとモデルサイズの既定値について
- **仕様**:
  - Whisper: 既定モデル `large-v3-turbo`（約1.5GB）
  - Nemotron-3: `nvidia/Nemotron-3-Diarization`（約200MB）
  - 初回実行時に Hugging Face から自動ダウンロードされます（トークン認証不要）。
- **ご確認点**:
  - コマンドライン引数 `--model large-v3-turbo`（または `medium`, `small`, `base`, `tiny`）で自由に切り替え可能にしていますが、既定モデルは要件書通り `large-v3-turbo` のままでよろしいでしょうか？

### ③ UI（操作形態）の次期フェーズ拡張について
- 要件書には「初期フェーズは CLI スクリプトで実装、簡易 GUI（gradio / PyQt）も考慮」と記載されております。
- 現在は直感的に使える対話型 CLI（番号を選んで実行するメニュー方式）を実装しております。
- **ご確認点**:
  - 次のステップとして、**PySide6 によるデスクトップ GUI** や **ブラウザで動く Web UI (Gradio / FastAPI)** の実装をご希望されますでしょうか？

### ④ 同一話者の連続発言マージのしきい値について
- 同一話者が短い間隔で連続して話した場合、1つの発言ブロックに結合して読みやすく整形します。
- 現在のデフォルト設定は **無音間隔 2.5秒以内** の連続発言を自動結合する設計にしています。会議のテンポや発言スタイルに応じて調整可能です。

---

## 3. アプリケーションの実行方法

### 方法 A: バッチファイルから起動（最も簡単）
エクスプローラーから `run.bat` をダブルクリック、またはターミナルで実行：
```cmd
run.bat
```
対話メニューが表示され、番号（1: 録音して議事録作成、2: 既存音声ファイル指定、3: マイク一覧、4: 終了）で操作できます。

### 方法 B: コマンドライン引数による実行
```bash
# 1. 接続されているマイク・入力デバイスの一覧を表示
python main.py devices

# 2. マイクから録音して議事録を作成 (Enterキーで録音停止)
python main.py record

# 3. 入力デバイスを指定して録音 (例: デバイスID 1)
python main.py record --device 1

# 4. 既存の音声ファイルから議事録を作成
python main.py file sample_meeting.wav

# 5. モデルサイズを指定して高速処理 (例: medium または small)
python main.py file sample_meeting.wav --model medium
```

### 出力ファイルについて
処理完了後、`output/` フォルダ配下に以下のファイルが一括生成されます：
- `minutes_xxx.md` : タイムスタンプ・話者名・メタデータ付き Markdown
- `minutes_xxx.txt` : シンプルなテキストファイル
- `minutes_xxx.srt` : 動画・音声プレイヤー用字幕ファイル
- `minutes_xxx.json` : 各発話の詳細データ（区間・重複度・未マージ詳細）
