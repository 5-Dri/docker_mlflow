FROM python:3.11

# パッケージ更新して脆弱性に対応
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# MLflowのみインストール
RUN pip install --no-cache-dir mlflow

WORKDIR /mlflow
EXPOSE 5000
