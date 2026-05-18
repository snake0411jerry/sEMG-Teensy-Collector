import numpy as np
import pandas as pd

# ==========================================
# 1. 讀取合併好的多模態訓練資料
# ==========================================
file_path = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0507\Excel_0507\squat_a_combined_training_data.csv"
df = pd.read_csv(file_path)

# --- 動態分離 X (特徵) 與 Y (標籤) ---
# 找出所有不是 'EMG_' 開頭的欄位當作 X
feature_cols = [col for col in df.columns if not col.startswith('EMG_')]
# 指定 EMG 欄位當作 Y
label_cols = ['EMG_Main', 'EMG_Compass']

X_data = df[feature_cols].values  # 形狀會是 (N, 特徵數量，例如 18)
Y_data = df[label_cols].values    # 形狀會是 (N, 2) 因為有兩個通道

print(f"✅ 成功載入資料！總幀數 (N): {len(df)}")
print(f"👉 X 骨架特徵數量: {len(feature_cols)} 個")
print(f"👉 Y EMG 預測通道: {len(label_cols)} 個 ({label_cols})\n")

# ==========================================
# 2. 設定滑動時間窗參數
# ==========================================
window_size = 30  # 每次看 30 幀 (以 60FPS 來說就是 0.5 秒)
step_size = 1     # 每次往右滑動 1 幀 (1 代表最高重疊率，能產生最多訓練資料)

X_windows = []
Y_labels = []

# ==========================================
# 3. 開始切割 (Sliding Window)
# ==========================================
# 迴圈走到 N - window_size 就必須停下，否則最後一個視窗會裝不滿 30 幀
for i in range(0, len(X_data) - window_size + 1, step_size):
    
    # 切出 30 幀的骨架特徵矩陣: 從第 i 筆取到第 i+30 筆
    window_X = X_data[i : i + window_size, :]
    
    # 決定預測目標 Y (Many-to-One 架構)
    # 這裡我們取這 30 幀的「最後一幀」對應的 2 個 EMG 數值作為預測目標
    target_Y = Y_data[i + window_size - 1, :]
    
    X_windows.append(window_X)
    Y_labels.append(target_Y)

# ==========================================
# 4. 轉換為深度學習框架 (PyTorch / TensorFlow) 看得懂的 Tensor
# ==========================================
X_tensor = np.array(X_windows)
Y_tensor = np.array(Y_labels)

print("-" * 40)
print("🚀 Tensor 打包完成！準備餵給模型：")
print(f"X 輸入矩陣形狀: {X_tensor.shape} --> (Batch_Size, Time_Steps, Features)")
print(f"Y 標籤矩陣形狀: {Y_tensor.shape} --> (Batch_Size, 2個 EMG 通道)")
print("-" * 40)

# (選用) 如果你使用 PyTorch，下一步通常是這樣轉成 Torch Tensor：
# import torch
# X_torch = torch.tensor(X_tensor, dtype=torch.float32)
# Y_torch = torch.tensor(Y_tensor, dtype=torch.float32)