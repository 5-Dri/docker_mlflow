FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

# ポート開放
EXPOSE 5000

# apt-get update 実行 & 不要なキャッシュ削除
RUN apt-get update && \
    apt-get install -y git python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# pip のセットアップ
RUN python3 -m ensurepip && pip3 install --upgrade pip --no-cache-dir

# PyTorch, PyG, 依存ライブラリのインストール
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir && \
    pip3 install torch_geometric networkx numpy hydra-optuna-sweeper mlflow seaborn --no-cache-dir && \
    pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.3.0+cu118.html --no-cache-dir

# 作業ディレクトリの設定
WORKDIR /root/prog



