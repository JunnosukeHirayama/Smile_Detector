# モデルの学習及び推論関数の設定
import math
import numpy as np
from sklearn.linear_model import LogisticRegression


# モデルの学習
# グローバル変数として学習済みモデルを保持
_smile_model = None

def train_model():
    """Trainデータを使ってロジックを学習させる"""
    import json
    from pathlib import Path

    # データの読み込み
    data_path = Path('data') / 'facial_keypoints.json'
    with open(data_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    train_data = data['train']
    X_train = []
    y_train = []

    for item in train_data:
        face = item[:-1]
        is_smile = item[-1]['smile']
        X_train.append(normalize_and_extract(face))
        y_train.append(is_smile)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # ロジスティック回帰モデルの構築と学習
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model

def normalize_and_extract(facial_data: list) -> np.ndarray:
    """顔の座標データを正規化し、特徴量ベクトルに変換する"""
    # 座標部分のみを抽出
    pts = np.array(facial_data[1:16], dtype=float)

    # 平行移動
    nose = pts[10]
    pts = pts - nose

    # スケーリング
    left_eye, right_eye = pts[0], pts[1]
    eye_dist = np.linalg.norm(left_eye - right_eye)
    if eye_dist > 0:
        pts = pts / eye_dist

    # 回転（両目を水平に）
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.arctan2(dy, dx)

    # 回転行列 R
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array(((c, -s), (s, c)))
    pts = np.dot(pts, R.T)

    # 追加の特徴量
    mouth_left, mouth_right = pts[11], pts[12]
    upper_lip, lower_lip = pts[13], pts[14]

    width = np.linalg.norm(mouth_left - mouth_right)
    height = np.linalg.norm(upper_lip - lower_lip)
    lip_center_y = (upper_lip[1] + lower_lip[1]) / 2.0
    mouth_corner_y = (mouth_left[1] + mouth_right[1]) / 2.0
    curve = lip_center_y - mouth_corner_y

    # 特徴量の結合（計33次元）
    features = pts.flatten()
    extra_features = np.array([width, height, curve])

    return np.concatenate([features, extra_features])

# 初回呼び出し時にモデルを学習
try:
    _smile_model = train_model()
    print("モデルの学習が完了しました")
except Exception as e:
    print(f"モデルの学習に失敗しました: {e}")

# 推論関数

def smile_predict(facial_data: list) -> bool:
    """Return True if the facial data is classified as smiling."""
    global _smile_model
    if _smile_model is None:
        return False

    # データを正規化
    features = normalize_and_extract(facial_data)

    # モデルによる予測
    pred = _smile_model.predict([features])[0]

    return bool(pred)