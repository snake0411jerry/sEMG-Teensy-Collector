import pandas as pd
import numpy as np

# 1. 讀取與初步清理 (與之前相同)
file_path = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0507\Excel_0507\squat-general.trc.csv"
df = pd.read_csv(file_path, skiprows=4)

columns_to_keep = {
    'Y5': 'Shoulder_Y', 'Z5': 'Shoulder_Z',
    'Y13': 'Hip_Y', 'Z13': 'Hip_Z',
    'Y14': 'Knee_Y', 'Z14': 'Knee_Z',
    'Y15': 'Ankle_Y', 'Z15': 'Ankle_Z' # 這裡多抓了腳踝的 Z 軸備用
}
df_clean = df[list(columns_to_keep.keys())].rename(columns=columns_to_keep).astype(float)

# ==========================================
# 🎯 C802 計畫書標準：座標正規化與物理運動學特徵
# ==========================================

# FPS 設定 (用來計算真實物理時間差 dt)
FPS = 60
dt = 1.0 / FPS

# --- 步驟一：身高估算 (Height Estimation) ---
# 為了做身高正規化，我們需要抓取受試者「站直時」的身體長度
# 這裡取整個序列中，肩膀與腳踝 Y 軸差距的最大值，當作該受試者的參考身高 (Reference Height)
ref_height = (df_clean['Shoulder_Y'] - df_clean['Ankle_Y']).max()
print(f"估算受試者參考身高比例為: {ref_height:.4f}")

# 準備一個清單來裝我們要處理的關節
joints = ['Shoulder', 'Knee', 'Ankle']
axes = ['Y', 'Z']

for joint in joints:
    for axis in axes:
        col_name = f"{joint}_{axis}"
        root_col = f"Hip_{axis}"  # 根節點設定為 Hip
        
        # --- 步驟二：原點置中 (Root-centered) ---
        # 所有關節減去 Hip 的座標
        centered_col = f"{col_name}_centered"
        df_clean[centered_col] = df_clean[col_name] - df_clean[root_col]
        
        # --- 步驟三：身高正規化 (Height-normalized) ---
        # 置中後的座標，除以受試者身高
        norm_col = f"{col_name}_norm"
        df_clean[norm_col] = df_clean[centered_col] / ref_height
        
        # --- 步驟四：正規化速度 (Velocity, 1階差分) ---
        # v = (後一幀正規化座標 - 前一幀正規化座標) / dt
        # 單位變成：(身高比例 / 秒)
        vel_col = f"{col_name}_vel"
        df_clean[vel_col] = df_clean[norm_col].diff() / dt
        df_clean[vel_col] = df_clean[vel_col].fillna(0) # 第一筆補零
        
        # --- 步驟五：正規化加速度 (Acceleration, 2階差分) ---
        # a = (後一幀速度 - 前一幀速度) / dt
        # 單位變成：(身高比例 / 秒^2)
        acc_col = f"{col_name}_acc"
        df_clean[acc_col] = df_clean[vel_col].diff() / dt
        df_clean[acc_col] = df_clean[acc_col].fillna(0)

# 因為 Hip 自己減自己會變成 0，速度加速度也是 0，所以我們就不特別算 Hip 的衍伸特徵了。

# 篩選出最終要送進 ST-GCN 的正規化矩陣 (只保留 norm, vel, acc 結尾的欄位)
final_columns = [col for col in df_clean.columns if col.endswith(('_norm', '_vel', '_acc'))]
df_stgcn_ready = df_clean[final_columns]

print("\n🔥 符合 ST-GCN 標準的物理引導特徵矩陣 (前5筆):")
print(df_stgcn_ready.head())

# 輸出檔案
output_path = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0507\Excel_0507\squat_stgcn_features.csv"
df_stgcn_ready.to_csv(output_path, index=False)