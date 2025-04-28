import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # GUIなしバックエンド
import matplotlib.pyplot as plt
import random

# ==== 保存ディレクトリを作成 ====
save_dir = './result'
os.makedirs(save_dir, exist_ok=True)

# ==== 間違った画像保存用ディレクトリを作成 ====
save_unabled_dir = './result_unabled'
os.makedirs(save_unabled_dir, exist_ok=True)

# ==== 1. データセットの準備 ====
transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# ==== 2. 3層ニューラルネットワーク（MLP）の定義 ====
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),               # 28x28 → 784
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)            # 出力は10クラス
        )

    def forward(self, x):
        return self.model(x)

# モデル作成
model = MLP()

# ==== 3. 損失関数と最適化手法 ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==== 4. 学習ループ ====
for epoch in range(5):  # 5エポック学習
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/5] Loss: {running_loss / len(trainloader):.4f}')

# ==== 5. テスト精度を計算 ====
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

# ==== 6. テスト画像から8枚を推論・保存 ====

# テストデータから1バッチ取得
testiter = iter(testloader)
images, labels = next(testiter)

# 0〜63の中からランダムに8個インデックスを選ぶ
indices = random.sample(range(len(images)), 8)

# 選んだランダムなインデックスで推論・保存
for idx, random_idx in enumerate(indices):
    image = images[random_idx].unsqueeze(0)  # (1,1,28,28)
    label = labels[random_idx]

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # 画像をプロットして保存
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"True: {label.item()}, Pred: {predicted.item()}")
    plt.axis('off')
    plt.savefig(f'{save_dir}/mnist_prediction_random_{idx}.png')
    plt.close()

# ==== 7. テストセット全体から失敗した画像を保存 ====

model.eval()
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(testloader):
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for i in range(images.size(0)):
            if predicted[i].item() != labels[i].item():
                plt.imshow(images[i].squeeze(), cmap='gray')
                plt.title(f"True: {labels[i].item()}, Pred: {predicted[i].item()}")
                plt.axis('off')
                plt.savefig(f'{save_unabled_dir}/wrong_batch{batch_idx}_img{i}.png')
                plt.close()


