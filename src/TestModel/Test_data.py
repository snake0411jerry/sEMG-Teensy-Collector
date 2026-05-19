import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# 參數設定
# ==========================================
# 請替換為您合併好的 CSV 檔案路徑
CSV_PATH = r"D:\project\NCU\114\Jumior\AI Project\Dataset\Code\Test_video\CSV\KUAN_TEST_0513_Segment11-20.csv"
# 設定圖片輸出的資料夾
OUTPUT_DIR = r"D:\project\NCU\114\Jumior\AI Project\Dataset\Code\Test_video\img"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"📂 開始讀取資料: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# ==========================================
# 定義需要比較的特徵清單
# ==========================================
# 這裡列出基本的特徵名稱 (不包含 .1 後綴)
feature_names = [
    'Shoulder_Y_norm', 'Shoulder_Y_vel', 'Shoulder_Y_acc',
    'Shoulder_Z_norm', 'Shoulder_Z_vel', 'Shoulder_Z_acc',
    'Knee_Y_norm', 'Knee_Y_vel', 'Knee_Y_acc',
    'Knee_Z_norm', 'Knee_Z_vel', 'Knee_Z_acc',
    'Ankle_Y_norm', 'Ankle_Y_vel', 'Ankle_Y_acc',
    'Ankle_Z_norm', 'Ankle_Z_vel', 'Ankle_Z_acc',
    'Knee_Angle_norm', 'Knee_Angle_vel'
]

print(f"📊 準備繪製 {len(feature_names)} 組特徵對比圖...")

# ==========================================
# 批次繪圖並儲存
# ==========================================
for feature in feature_names:
    col_opencv = feature        # 原本 OpenCV (TRC) 的欄位名稱
    col_mediapipe = feature + '.1' # Mediapipe 附加進來的欄位名稱
    
    # 檢查欄位是否存在
    if col_opencv not in df.columns or col_mediapipe not in df.columns:
        print(f"⚠️ 找不到特徵 {feature}，跳過繪圖。")
        continue

    # 建立圖表
    plt.figure(figsize=(10, 5))
    
    # 畫出 OpenCV 數據 (藍色)
    plt.plot(df[col_opencv], label='OpenCV (Ground Truth)', color='blue', alpha=0.7, linewidth=1.5)
    
    # 畫出 Mediapipe 數據 (橘色)
    plt.plot(df[col_mediapipe], label='Mediapipe (Predicted)', color='orange', alpha=0.8, linestyle='--', linewidth=2)
    
    plt.title(f'Comparison: {feature}')
    plt.xlabel('Frame')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # 儲存圖片
    save_path = os.path.join(OUTPUT_DIR, f"{feature}_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close() # 關閉圖表以釋放記憶體

print(f"✅ 所有特徵對比圖已成功儲存至: {OUTPUT_DIR} 資料夾！")