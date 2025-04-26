import torch

# バージョン確認
print("PyTorch Version:", torch.__version__)

# CUDA (GPU) が使えるか確認（CPU版ならFalseでOK）
print("Is CUDA available?", torch.cuda.is_available())

# デバイスタイプの確認
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# 簡単なテンソル演算テスト
x = torch.rand(3, 3)
y = torch.rand(3, 3)
z = x + y
print("Tensor addition result:\n", z)
