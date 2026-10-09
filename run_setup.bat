@echo off
cd /d %~dp0

:: 1. venvフォルダがなければ作成する
if not exist venv (
    echo "venvフォルダを作成中..."
    python -m venv venv
)

:: 2. 仮想環境を有効化
echo "仮想環境を有効化"
call venv\Scripts\activate

:: 3. pipのアップグレード
echo "pipのアップグレード"
python -m pip install --upgrade pip

:: 4. インストールするファイルの選択
echo.
echo ============================================================
echo  どちらのモードでインストールしますか？
echo ============================================================
echo  [1] requirements.txt
echo      - パッケージを最新バージョンでインストールします
echo      - 新機能を使いたい場合や、通常はこちらを推奨
echo.
echo  [2] requirements-lock.txt
echo      - 動作確認済みのバージョンに固定してインストールします
echo      - エラーを避けて、確実に動かしたい場合はこちらを推奨
echo ============================================================
echo  [1] または [2] を入力して Enter を押してください。
echo.

:choice
set SELECT=1
set /p SELECT="番号を入力してください (1 or 2) [Enterで1を選択]: "

if "%SELECT%"=="1" goto req
if "%SELECT%"=="2" goto lock
echo [エラー] 1か2を入力してください。
goto choice

:req
set TARGET_FILE=requirements.txt
goto install

:lock
set TARGET_FILE=requirements-lock.txt
goto install

:install
:: 5. パッケージのインストール
echo.
if exist %TARGET_FILE% (
    echo "%TARGET_FILE% からパッケージをインストール中..."
    pip install -r %TARGET_FILE%
    
    :: 1 を選択し、かつ requirements-lock.txt が存在しない場合は作成する
    if "%SELECT%"=="1" (
        if not exist requirements-lock.txt (
            echo.
            echo "requirements-lock.txt が存在しないため、現在の環境から作成します..."
            pip freeze > requirements-lock.txt
            echo "requirements-lock.txt を作成しました。"
        )
    )
) else (
    echo [エラー] %TARGET_FILE% が見つかりませんでした。
)

echo.
echo "セットアップ完了"
pause