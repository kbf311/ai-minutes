# 開発タスク指示書：Nemotron-3-Diarization と Whisper による話者分離付き文字起こし

## 1. 概要と重要前提
本タスクでは、Windows ローカル環境でマイク音声の録音から「話者分離付き文字起こし」までを行う Python スクリプトを作成します。

### 【極めて重要な前提知識（モデルの役割分担）】
1. **`nvidia/Nemotron-3-Diarization` について**:
   - これはテキスト生成や音声認識（ASR）を行う LLM ではなく、**「話者ダイアライゼーション（誰がいつ話したかのタイムスタンプ判定）専用」の 0.1B（100M）軽量モデル** です。
   - モデル単体では文字起こしを行いません。
   - クラスタリング方式ではなく、**Sortformer（End-to-End）アーキテクチャ** を採用しており、最大8話者に対応し、重複発話（被り）も判定可能です。
   - Hugging Face の `nvidia/Nemotron-3-Diarization` として公開されており、Gated model ではないためトークン認証なしでロード可能です。
   - 最新の `transformers`（GitHub main ブランチ対応）の `AutoModel` / 専用クラスまたは pipeline で利用します。

2. **音声認識（ASR）について**:
   - 文字起こし自体は `faster-whisper`（CTranslate2 バックエンド、`large-v3` または `large-v3-turbo`）を併用します。

3. **統合（アライメント）方針**:
   - `faster-whisper` から「発話テキストとタイムスタンプ `[start, end]`」を取得します。
   - `Nemotron-3-Diarization` から「話者ラベルとタイムスタンプ区間 `[start, end, speaker_id]`」を取得します。
   - 両者の時間的重なり（Overlap / IoU）を計算し、テキストセグメントに対して最も重複時間の長い話者 ID を割り当てて整形出力します。

---

## 2. 実装要件

### A. 録音モジュール (`sounddevice`, `soundfile`)
- 16kHz, 1ch（モノラル）, PCM 16-bit の WAV 形式で録音・保存すること。
  （※両モデルのネイティブ入力形式に合わせ、再サンプリング負荷を省くため）
- Enter キーで開始・停止（または任意キー制御）ができるシンプルな CLI 実装とすること。

### B. 話者分離モジュール (`Nemotron-3-Diarization`)
- 音声ファイルを読み込み、話者ごとの発話区間リストを出力すること。
  出力期待例: `[{"speaker": "Speaker 0", "start": 0.5, "end": 4.2}, ...]`
- CUDA（GPU）が利用可能な場合は GPU を使い、なければ CPU にフォールバックすること。

### C. 文字起こしモジュール (`faster-whisper`)
- 音声を日本語（`language="ja"`）で文字起こしし、セグメント単位のテキストと `start`, `end` を取得すること。
- GPU（`device="cuda"`, `compute_type="float16"`）を基本とし、CPU 時は `int8` に切り替えること。

### D. マージロジックと結果出力
- 各テキストセグメントに対して、重なる話者セグメントを計算して話者を同定すること。
- 同一話者の連続発言は読みやすく 1 つの発言ブロックにまとめること。
- 結果はコンソール表示に加え、タイムスタンプ付きのテキストファイル（または Markdown）に保存すること。
  例: `[00:01:23 - 00:01:30] 話者 0: こんにちは、本日の議題です。`

---

## 3. 注意点・エラー回避
- `pyannote.audio` のような Hugging Face トークン入力（`use_auth_token`）を要求するコードは書かないこと。
- `transformers` から Nemotron-3 を呼び出す際、未対応の古いリリース版を想定したコードではなく、標準的な PyTorch テンソル入力および最新 Hugging Face インターフェースに準拠した実装にすること。
- 依存関係は `pip install torch torchaudio`, `pip install git+https://github.com/huggingface/transformers.git`, `pip install faster-whisper sounddevice soundfile` を想定すること。