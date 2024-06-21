# ベースイメージとしてPythonの公式イメージを使用
FROM python:3.8-slim

# 作業ディレクトリの設定
WORKDIR /app

# 依存関係ファイルをコピーし、依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのファイルをコピー
COPY . .

# アプリケーションを実行するためのコマンド
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
