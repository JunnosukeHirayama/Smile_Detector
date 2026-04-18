# データの可視化
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# データのロード
data_path = Path('data') / 'facial_keypoints.json'
with open(data_path, 'r', encoding='utf-8') as fp:
    data = json.load(fp)

train_data = data['train']

# 笑顔と非笑顔のサンプルをそれぞれ2つずつ取得
smiles = [item for item in train_data if item[-1]['smile'] == True][:2]
not_smiles = [item for item in train_data if item[-1]['smile'] == False][:2]
samples_to_plot = smiles + not_smiles

# グラフ描画の作成
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, sample in zip(axes, samples_to_plot):
    sample_id = sample[0]
    is_smile = sample[-1]['smile']
    pts = np.array(sample[1:16], dtype=float)

    ax.scatter(pts[:, 0], pts[:, 1], c='blue', s=50)

    for i, (x, y) in enumerate(pts):
        ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', color='red', fontsize=10)

    ax.invert_yaxis()
    ax.axis('equal')

    # タイトルにIDと笑顔判定を表示
    ax.set_title(f"ID: {sample_id} | Smile: {is_smile}", fontsize=14)

plt.tight_layout()
plt.show()