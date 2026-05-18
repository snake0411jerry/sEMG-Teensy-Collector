import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 告訴 Pandas：請跳過前 4 行，把第 5 行當作真正的欄位名稱
df = pd.read_csv(r"D:\project\NCU\114\Jumior\AI Project\Dataset\0512\OPENCAP\OpenCapData_0512_SHIH\OpenCapData_411bdac8-937e-49ef-90a2-208c1d13187b\MarkerData\general1-10.trc.csv", skiprows=4)

# 這時候再來抓 Y5，通常就抓得到了！
frame_shoulder_y = df['Y5'].values
# 假設你已經讀取了兩個 CSV 的欄位變成 numpy array
emg_signal = pd.read_csv(r"D:\project\NCU\114\Jumior\AI Project\Dataset\0512\OPENCAP\OpenCapData_0512_SHIH\OpenCapData_411bdac8-937e-49ef-90a2-208c1d13187b\EMGData\SHI_MIN_0512_training_1-10.csv")['Main_raw'].values

# --- 步驟 1：手動填入你觀察圖表找出的「有效動作區間 index」 ---
# 這些數字請你從 Excel 或圖表上大約抓一下
FRAME_START = 282      # 骨架第一次深蹲開始的 index
FRAME_END = 2751      # 骨架最後一次深蹲結束的 index

EMG_START = 255       # EMG 第一次發力開始的 index
EMG_END = 3801        # EMG 最後一次發力結束的 index

# --- 步驟 2：裁切頭尾多餘的空白數據 ---
frame_cut = frame_shoulder_y[FRAME_START : FRAME_END]
emg_cut = emg_signal[EMG_START : EMG_END]

print(f"裁切後 骨架長度: {len(frame_cut)}")
print(f"裁切後 EMG長度: {len(emg_cut)}")

# --- 步驟 3：重採樣 (將 EMG 壓縮/拉伸至與骨架相同長度) ---
# 建立舊的時間軸 (0 到 1)
old_time_axis = np.linspace(0, 1, len(emg_cut))
# 建立新的時間軸 (0 到 1，但切分的份數等於骨架的長度)
new_time_axis = np.linspace(0, 1, len(frame_cut))

# 使用線性插值，計算出對齊後的新 EMG 數值
emg_aligned = np.interp(new_time_axis, old_time_axis, emg_cut)

print(f"對齊後 最終骨架長度: {len(frame_cut)}")
print(f"對齊後 最終EMG長度: {len(emg_aligned)}")

# --- 步驟 4：畫圖檢查對齊結果 (這一步非常重要！) ---
fig, ax1 = plt.subplots(figsize=(12, 5))

# 畫骨架高度 (藍線)
ax1.plot(frame_cut, color='blue', label='Shoulder Height')
ax1.set_ylabel('Height')

# 建立共用 X 軸的第二個 Y 軸來畫 EMG (紅線)
ax2 = ax1.twinx()
ax2.plot(emg_aligned, color='red', alpha=0.6, label='EMG Aligned')
ax2.set_ylabel('EMG Intensity')

plt.title("Feature Aligned Data (Frames vs EMG)")
fig.legend(loc="upper right")
plt.show()

# 最後你可以把 frame_cut 和 emg_aligned 合併存成一個新的 CSV 拿去訓練！