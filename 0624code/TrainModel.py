import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, BatchNormalization, GlobalAveragePooling1D, Concatenate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 讀取資料與前處理
# ==========================================
data_dir = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Combined"
file_paths = glob.glob(os.path.join(data_dir, "*_Combined_Features.csv"))
train_files, val_files = train_test_split(file_paths, test_size=0.2, random_state=42)

# 定義特徵欄位名稱
static_cols = ['Subj_Age', 'Subj_Height_m', 'Subj_Weight_norm', 'Subj_Gender', 'Load_1RM_Ratio']
label_emg_cols = ['EMG_Main_MVC', 'EMG_Compass_MVC']
# 💡 更新：定義三個代償標籤的欄位名稱
label_comp_cols = ['Comp_Heel_Raise', 'Comp_Knee_Valgus', 'Comp_Trunk_Lean']

def plot_yolo_style_results(history, save_path='results.png'):
    h = history.history
    epochs = range(1, len(h['loss']) + 1)
    
    # 建立 2x5 的網格畫布
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # 定義第一列 (Train) 的對應數據與標題
    row1_metrics = [
        ('train/total_loss', 'loss'),
        ('train/emg_loss', 'out_emg_loss'),
        ('train/comp_loss', 'out_comp_loss'),
        ('metrics/train_emg_mae', 'out_emg_mae'),
        ('metrics/train_comp_acc', 'out_comp_accuracy')
    ]
    
    # 定義第二列 (Val) 的對應數據與標題
    row2_metrics = [
        ('val/total_loss', 'val_loss'),
        ('val/emg_loss', 'val_out_emg_loss'),
        ('val/comp_loss', 'val_out_comp_loss'),
        ('metrics/val_emg_mae', 'val_out_emg_mae'),
        ('metrics/val_comp_acc', 'val_out_comp_accuracy')
    ]
    
    # 輔助函數：畫單一圖表面板
    def plot_panel(ax, title, data):
        # 1. 原始數據 (藍色實線點狀)
        ax.plot(epochs, data, marker='.', markersize=6, linestyle='-', 
                linewidth=1.5, color='#1f77b4', label='results')
        
        # 2. 平滑數據 (橘色虛線，使用 pandas rolling 計算移動平均)
        smooth_data = pd.Series(data).rolling(window=3, min_periods=1).mean()
        ax.plot(epochs, smooth_data, linestyle=':', linewidth=2, 
                color='#ff7f0e', alpha=0.8, label='smooth')
        
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # 移除頂部與右側的邊框框線，讓圖表更乾淨 (YOLO 風格)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 僅在第一個圖表顯示圖例
        if title == 'train/total_loss':
            ax.legend(loc='upper right', frameon=True)

    # 繪製第一列 (Train)
    for i, (title, key) in enumerate(row1_metrics):
        if key in h:
            plot_panel(axes[0, i], title, h[key])
        
    # 繪製第二列 (Val)
    for i, (title, key) in enumerate(row2_metrics):
        if key in h:
            plot_panel(axes[1, i], title, h[key])
        
    # 自動排版並儲存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 訓練結果圖表已儲存至目前目錄下的: {save_path}")
    plt.show()

def load_data(files):
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        
        # 🌟 標籤工程：根據第二步算出的幾何特徵設定閾值 (閾值可依據實際實驗標準調整)
        # 1. 墊腳尖：左腳或右腳跟抬高程度大於身高比例的某個閾值 (示範設為 0.02)
        df['Comp_Heel_Raise'] = ((df['L_Heel_Rise_norm'] > 0.02) | (df['R_Heel_Rise_norm'] > 0.02)).astype(float)
        
        # 2. 膝蓋內凹：雙膝距離/雙踝距離小於 0.92 (代表膝蓋分得不夠開，往內縮)
        df['Comp_Knee_Valgus'] = (df['Knee_Ankle_Ratio_norm'] < 0.92).astype(float)
        
        # 3. 背部前傾：前傾角度大於 40 度 (40.0 / 180.0)
        df['Comp_Trunk_Lean'] = (df['Trunk_Lean_Angle_norm'] > (40.0 / 180.0)).astype(float)
        
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True), dfs

train_df_all, train_dfs = load_data(train_files)
val_df_all, val_dfs = load_data(val_files)

# 動態特徵 (排除靜態、標籤、不相關欄位)
# 💡 更新：在 exclude 中加入 'Comp_'，防止新建立的標籤欄位變成了訓練特徵
exclude = ['EMG_', 'Knee_Toe_Diff', 'vel', 'acc', 'Target', 'Unnamed', 'Subj_', 'Comp_']
ts_cols = [col for col in train_df_all.columns if not any(k in col for k in exclude)]

print(f"訓練使用的動態特徵數量: {len(ts_cols)}")
print(f"動態特徵欄位包含: {ts_cols}")

# 建立 Scaler
scaler_ts = StandardScaler().fit(train_df_all[ts_cols].values)
scaler_static = StandardScaler().fit(train_df_all[static_cols].values)

# ==========================================
# 2. 切割滑動窗口 (Sliding Window)
# ==========================================
WINDOW_SIZE = 40
STEP_SIZE = 5

def create_multi_modal_windows(df_list):
    X_ts, X_static, Y_emg, Y_comp = [], [], [], []
    for df in df_list:
        ts_data = scaler_ts.transform(df[ts_cols].values)
        static_data = scaler_static.transform(df[static_cols].values)
        emg_data = df[label_emg_cols].values 
        comp_data = df[label_comp_cols].values # 💡 更新：此處將一次取出三個標籤的數值 (Shape: N x 3)
        
        for j in range(0, len(ts_data) - WINDOW_SIZE, STEP_SIZE):
            X_ts.append(ts_data[j : j + WINDOW_SIZE, :])
            target_idx = j + WINDOW_SIZE - 1 
            X_static.append(static_data[target_idx, :]) 
            Y_emg.append(emg_data[target_idx, :])
            Y_comp.append(comp_data[target_idx, :]) # 💡 存入的會是大小為 3 的 One-hot 類似陣列 (e.g. [0, 1, 0])
            
    return np.array(X_ts), np.array(X_static), np.array(Y_emg), np.array(Y_comp)

X_train_ts, X_train_stat, Y_train_emg, Y_train_comp = create_multi_modal_windows(train_dfs)
X_val_ts, X_val_stat, Y_val_emg, Y_val_comp = create_multi_modal_windows(val_dfs)

print(f"訓練樣本數: {len(X_train_ts)} | 驗證樣本數: {len(X_val_ts)}")
print(f"代償標籤訓練集形狀: {Y_train_comp.shape}") # 預期形狀會是 (樣本數, 3)

# ==========================================
# 3. 🧠 建構多模態多任務模型 (升級為多標籤分類)
# ==========================================
# 輸入 1: 連續骨架訊號
input_ts = Input(shape=(WINDOW_SIZE, len(ts_cols)), name='ts_input')
# 輸入 2: 受試者靜態特徵
input_static = Input(shape=(len(static_cols),), name='static_input')

# --- 時間序列特徵萃取 (1D-CNN / TCN) ---
x_ts = Conv1D(64, 3, padding='causal', activation='relu')(input_ts)
x_ts = BatchNormalization()(x_ts)
x_ts = Conv1D(64, 3, padding='causal', dilation_rate=2, activation='relu')(x_ts)
ts_features = GlobalAveragePooling1D()(x_ts)

# --- 特徵融合層 (Late Fusion) ---
fused_features = Concatenate()([ts_features, input_static])
fused_features = Dense(64, activation='relu')(fused_features)
fused_features = Dropout(0.2)(fused_features)

# --- 任務 1: EMG %MVC 預測 (回歸) ---
emg_branch = Dense(32, activation='relu')(fused_features)
# 💡 確保這裡的 activation 已經改為 'linear'
out_emg = Dense(2, activation='linear', name='out_emg')(emg_branch) 

# ==========================================
# 👇👇👇 你的新程式碼貼在這裡 👇👇👇
# ==========================================
# --- 任務 2: 三大代償動作辨識 (多標籤分類) ---
comp_branch = Dense(64, activation='relu')(fused_features) # 加大神經元，增強分類學習能力
comp_branch = Dropout(0.3)(comp_branch) # 加入 Dropout 穩定驗證集表現，防止後期跳水
out_comp = Dense(3, activation='sigmoid', name='out_comp')(comp_branch)
# ==========================================
# 👆👆👆 你的新程式碼貼在這裡 👆👆👆
# ==========================================

# 將兩個輸出端綁定到模型上
model = tf.keras.Model(inputs=[input_ts, input_static], outputs=[out_emg, out_comp])

# ==========================================
# 4. 模型編譯與訓練
# ==========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), 
    # 💡 在多標籤分類中，三個獨立二元分類的總 Loss 依然是 binary_crossentropy
    loss={'out_emg': 'mse', 'out_comp': 'binary_crossentropy'},
    loss_weights={'out_emg': 3.0, 'out_comp': 1.0}, # 可依訓練後 regression 與 classification 的收斂狀況調整權重
    metrics={'out_emg': 'mae', 'out_comp': 'accuracy'}
)

model.summary()

print("\n🚀 開始訓練多模態多標籤模型 (Multi-Modal Multi-Label Training)...")

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=5, 
    min_lr=1e-5, 
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=10, 
    restore_best_weights=True
)

history = model.fit(
    {'ts_input': X_train_ts, 'static_input': X_train_stat}, 
    {'out_emg': Y_train_emg, 'out_comp': Y_train_comp},
    epochs=50, 
    batch_size=32,
    validation_data=(
        {'ts_input': X_val_ts, 'static_input': X_val_stat}, 
        {'out_emg': Y_val_emg, 'out_comp': Y_val_comp}
    ),
    callbacks=[early_stopping, reduce_lr], 
    verbose=1
)

# 呼叫函數畫圖
plot_yolo_style_results(history, save_path='my_model_results.png')