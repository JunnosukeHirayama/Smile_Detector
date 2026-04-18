# 正答率計算用
import json
from pathlib import Path


try:
    # ローカル環境
    from smile_predict import smile_predict
except ImportError:
    # Colab環境
    if 'smile_predict' not in globals():
        print("【警告】smile_predict関数が見つかりません。先にモデル学習のコードを実行してください。")

def main():
    # データのロード
    data_path = Path('data') / 'facial_keypoints.json'
    
    # ファイルが存在しない場合のエラーハンドリングを追加
    if not data_path.exists():
        print(f"【エラー】データファイルが見つかりません: {data_path}")
        return

    with open(data_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    test_data = data['test']

    # 評価指標の初期化
    correct_count = 0
    total_count = len(test_data)

    # 推論と精度評価
    for sample in test_data:
        face_input = sample[:-1]      # 特徴量（座標データ）
        gt_bool = sample[-1]['smile'] # 正解ラベル (Ground Truth)

        # モデルによる予測
        pred_bool = smile_predict(face_input)

        # 正誤判定
        if pred_bool == gt_bool:
            correct_count += 1

    # 結果の算出と出力
    accuracy = correct_count / total_count

    print("=== Evaluation Results (評価結果) ===")
    print(f"Total Samples : {total_count}")
    print(f"Correct       : {correct_count}")
    print(f"Accuracy      : {accuracy * 100:.1f} %")

if __name__ == "__main__":
    main()