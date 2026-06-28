import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, BatchNormalization, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split 
import joblib
import tensorflow.keras.backend as K

def pearson_mse_loss(y_true, y_pred):
    # 1. 計算標準 MSE
    mse = K.mean(K.square(y_true - y_pred))
    
    # 2. 計算 Pearson Correlation
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym, axis=0)
    r_den = K.sqrt(K.sum(K.square(xm), axis=0) * K.sum(K.square(ym), axis=0))
    r = r_num / (r_den + K.epsilon())
    
    # 3. 取兩通道的平均相關係數
    mean_r = K.mean(r)
    
    # 4. 混合 Loss：希望 MSE 越小越好，同時 r 越接近 1 越好 (所以用 1 - r)
    # 這裡的 0.5 可以依據訓練狀況微調 (例如調成 0.2 或 1.0)
    return mse + 0.5 * (1 - mean_r)
# ==========================================
# 1. 讀取資料與標籤前處理 (🔥 改回：檔案級別的隨機切分)
# ==========================================
data_dir = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0529\Combined"
file_pattern = os.path.join(data_dir, "*_Combined_Features.csv")
file_paths = glob.glob(file_pattern)

if not file_paths:
    raise ValueError(f"在 {data_dir} 找不到任何檔案，請檢查路徑！")

# 🔥 關鍵修改：將「檔案路徑清單」隨機打散並切分 (80% 訓練, 20% 驗證)
# random_state=42 確保每次跑切分的檔案組合是一樣的，方便除錯
train_files, val_files = train_test_split(file_paths, test_size=0.2, random_state=42)

print(f"📂 總共找到了 {len(file_paths)} 個檔案！")
print(f"   - 🏋️ 隨機分配 {len(train_files)} 個檔案至【訓練集】")
print(f"   - 🧪 隨機分配 {len(val_files)} 個檔案至【驗證集】")

def load_and_preprocess(file_list):
    dfs = []
    for fp in file_list:
        df = pd.read_csv(fp)
        
        # 計算代償動作機率標籤 (閾值 > 30度)
        COMPENSATE_THRESHOLD = 30.0 / 180.0
        df['Comp_Prob_Target'] = (df['Knee_Toe_Diff_norm'] > COMPENSATE_THRESHOLD).astype(float)
        dfs.append(df)
    return dfs

train_dfs = load_and_preprocess(train_files)
val_dfs = load_and_preprocess(val_files)

combined_train_df = pd.concat(train_dfs, ignore_index=True)

# 🔥 修正資料洩漏：必須排除標籤欄位 'Comp_Prob_Target' 以及索引欄位 'Unnamed: 0'
exclude_keywords = ['EMG_', 'Knee_Toe_Diff', 'Comp_Prob_Target', 'Unnamed: 0']
feature_cols = [col for col in combined_train_df.columns if not any(k in col for k in exclude_keywords)]

print(f"✅ 修正後 - 最終使用的特徵數量: {len(feature_cols)}")
# ⚠️ 請確認印出來的數量是 11 個，且裡面沒有 Comp_Prob_Target
print(f"🔍 特徵清單範例: {feature_cols[:5]}...")
label_emg_cols = ['EMG_Main_MVC', 'EMG_Compass_MVC'] # 預測絕對發力值
label_comp_col = ['Comp_Prob_Target']

# Scaler 只能使用「訓練集」數據來 fit
scaler_x = StandardScaler()
scaler_x.fit(combined_train_df[feature_cols].values)

scaler_y_emg = MinMaxScaler() # 配合 sigmoid 使用
scaler_y_emg.fit(combined_train_df[label_emg_cols].values)

# ==========================================
# 2. 滑動時間窗切割 
# ==========================================
window_size = 40  
step_size = 5     

def create_windows(df_list):
    X_list, Y_emg_list, Y_comp_list = [], [], []
    segments_lengths = []
    
    for df in df_list:
        X_data = scaler_x.transform(df[feature_cols].values)
        Y_emg_data = scaler_y_emg.transform(df[label_emg_cols].values)
        Y_comp_data = df[label_comp_col].values 
        
        count = 0
        for j in range(0, len(X_data) - window_size, step_size):
            X_list.append(X_data[j : j + window_size, :])
            target_idx = j + window_size - 1 
            Y_emg_list.append(Y_emg_data[target_idx, :])
            Y_comp_list.append(Y_comp_data[target_idx, :])
            count += 1
        segments_lengths.append(count) 
        
    return np.array(X_list), np.array(Y_emg_list), np.array(Y_comp_list), segments_lengths

print("\n⏳ 正在進行時間窗切割...")
X_train, Y_train_emg, Y_train_comp, _ = create_windows(train_dfs)
X_val, Y_val_emg, Y_val_comp, val_segments_lengths = create_windows(val_dfs)

print(f"✅ 訓練集樣本數 (Windows): {len(X_train)}")
print(f"✅ 驗證集樣本數 (Windows): {len(X_val)}")

# ==========================================
# 3. 建模與訓練 (多任務輸出 Multi-task)
# ==========================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = Add()([x, inputs]) 
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x) 
    return Add()([x, res]) 

print("\n🧠 正在建構 多任務 TCN + Transformer 模型...")
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# --- 共享特徵提取層 (Shared Representation) ---
x = Conv1D(filters=64, kernel_size=3, padding='causal', dilation_rate=1, activation='relu')(inputs)
x = BatchNormalization()(x)
shared_features = Dropout(0.1)(x)

# 🔥 任務 1 專屬大腦 (EMG 分支)
emg_x = Conv1D(filters=64, kernel_size=3, padding='causal', dilation_rate=2, activation='relu')(shared_features)
emg_x = transformer_encoder(emg_x, head_size=64, num_heads=2, ff_dim=128, dropout=0.1)
emg_x = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(emg_x)
emg_branch = Dense(units=32, activation='relu')(emg_x)
out_emg = Dense(units=2, activation='sigmoid', name='out_emg')(emg_branch) # 使用 sigmoid 鎖定在 0~1

# 🔥 任務 2 專屬大腦 (代償分支)
comp_x = GlobalAveragePooling1D()(shared_features)
comp_branch = Dense(units=16, activation='relu')(comp_x)
out_comp = Dense(units=1, activation='sigmoid', name='out_comp')(comp_branch)

model = tf.keras.Model(inputs=inputs, outputs=[out_emg, out_comp])

custom_adam = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=custom_adam, 
    loss={
        'out_emg': pearson_mse_loss,  # 🚀 使用混合損失函數
        'out_comp': 'binary_crossentropy'  
    },
    loss_weights={'out_emg': 10.0, 'out_comp': 0.1}, 
    metrics={'out_emg': 'mae', 'out_comp': 'accuracy'}
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train, 
    {'out_emg': Y_train_emg, 'out_comp': Y_train_comp},
    epochs=100,
    batch_size=32,
    validation_data=(X_val, {'out_emg': Y_val_emg, 'out_comp': Y_val_comp}),
    callbacks=[early_stop], 
    verbose=2 # 一行顯示一個 Epoch
)

# ==========================================
# 4. 預測與視覺化 (包含 MVC 絕對值與後處理斜率)
# ==========================================
predictions = model.predict(X_val)
pred_emg = predictions[0]  
pred_comp = predictions[1] 

num_plots = min(3, len(val_segments_lengths))
start_idx = 0

print(f"\n📈 準備繪製驗證集的 {num_plots} 個連續動作片段...")

for i in range(num_plots):
    seg_len = val_segments_lengths[i]
    end_idx = start_idx + seg_len
    
    # 1. 擷取原始預測片段 (MVC 絕對值)
    y_true_emg_seg = Y_val_emg[start_idx:end_idx]
    y_pred_emg_seg = pred_emg[start_idx:end_idx]
    
    y_true_comp_seg = Y_val_comp[start_idx:end_idx]
    y_pred_comp_seg = pred_comp[start_idx:end_idx]
    
    # 2. 後處理：計算斜率 (變化率) 
    true_emg_main_slope = np.diff(y_true_emg_seg[:, 0])
    pred_emg_main_slope = np.diff(y_pred_emg_seg[:, 0])
    
    true_emg_comp_slope = np.diff(y_true_emg_seg[:, 1])
    pred_emg_comp_slope = np.diff(y_pred_emg_seg[:, 1])

    # ----------------------------------------------------
    # 📊 圖表 A：原始預測結果 (MVC 與 代償機率)
    # ----------------------------------------------------
    plt.figure(figsize=(12, 10))
    
    # --- 主動肌 MVC ---
    plt.subplot(3, 1, 1)
    plt.plot(y_true_emg_seg[:, 0], label='True EMG_Main (%MVC)', color='blue', alpha=0.7)
    plt.plot(y_pred_emg_seg[:, 0], label='Predicted EMG_Main (sigmoid)', color='red', linestyle='--')
    plt.title(f'Validation Segment {i+1} - EMG Main Muscle Force (Absolute Amplitude)')
    plt.ylabel('EMG Amplitude [0, 1]')
    plt.legend()

    # --- 協同肌 MVC ---
    plt.subplot(3, 1, 2)
    plt.plot(y_true_emg_seg[:, 1], label='True EMG_Compass (%MVC)', color='green', alpha=0.7)
    plt.plot(y_pred_emg_seg[:, 1], label='Predicted EMG_Compass (sigmoid)', color='orange', linestyle='--')
    plt.title(f'Validation Segment {i+1} - EMG Compass Muscle Force (Absolute Amplitude)')
    plt.ylabel('EMG Amplitude [0, 1]')
    plt.legend()
    
    # --- 代償機率 ---
    plt.subplot(3, 1, 3)
    plt.plot(y_true_comp_seg, label='True Compensation (1=Yes, 0=No)', color='gray', alpha=0.5, drawstyle='steps-pre')
    plt.plot(y_pred_comp_seg, label='Predicted Probability (sigmoid)', color='purple')
    plt.axhline(y=0.5, color='r', linestyle=':', label='Threshold (0.5)')
    plt.title(f'Validation Segment {i+1} - Knee Valgus Compensation Probability')
    plt.xlabel('Time Windows')
    plt.ylabel('Probability [0, 1]')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------
    # 📊 圖表 B：🔥 後處理斜率對比圖 
    # ----------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # --- 主動肌 斜率 ---
    plt.subplot(2, 1, 1)
    plt.plot(true_emg_main_slope, label='True EMG_Main Slope (Post-processed)', color='blue', alpha=0.6)
    plt.plot(pred_emg_main_slope, label='Predicted EMG_Main Slope (Post-processed)', color='red', linestyle='-', alpha=0.8)
    plt.title(f'Validation Segment {i+1} - Post-processed Rate of Change (EMG Main Slope)')
    plt.ylabel('Rate of Change (Δ Amplitude)')
    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='--') 
    plt.legend()

    # --- 協同肌 斜率 ---
    plt.subplot(2, 1, 2)
    plt.plot(true_emg_comp_slope, label='True EMG_Compass Slope (Post-processed)', color='green', alpha=0.6)
    plt.plot(pred_emg_comp_slope, label='Predicted EMG_Compass Slope (Post-processed)', color='orange', linestyle='-', alpha=0.8)
    plt.title(f'Validation Segment {i+1} - Post-processed Rate of Change (EMG Compass Slope)')
    plt.xlabel('Time Windows (n-1)')
    plt.ylabel('Rate of Change (Δ Amplitude)')
    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='--') 
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    start_idx = end_idx


# ==========================================
# 4.5 最終驗證集整體評估成績單
# ==========================================
print("\n" + "="*50)
print("📊 隨機驗證集 整體評估成績單 (Validation Metrics)")
print("="*50)

all_true_slopes_main, all_pred_slopes_main = [], []
all_true_slopes_comp, all_pred_slopes_comp = [], []

start_idx = 0
for seg_len in val_segments_lengths:
    end_idx = start_idx + seg_len
    
    y_true_seg = Y_val_emg[start_idx:end_idx]
    y_pred_seg = pred_emg[start_idx:end_idx]
    
    all_true_slopes_main.extend(np.diff(y_true_seg[:, 0]))
    all_pred_slopes_main.extend(np.diff(y_pred_seg[:, 0]))
    all_true_slopes_comp.extend(np.diff(y_true_seg[:, 1]))
    all_pred_slopes_comp.extend(np.diff(y_pred_seg[:, 1]))
    
    start_idx = end_idx

# --- 1. EMG 絕對值 (MVC) 評估 ---
mae_main = mean_absolute_error(Y_val_emg[:, 0], pred_emg[:, 0])
mae_comp = mean_absolute_error(Y_val_emg[:, 1], pred_emg[:, 1])

print("💪 【EMG 絕對發力值 (Absolute MVC)】")
print(f"  - 主動肌 (Main) 平均誤差 (MAE): {mae_main:.4f} (約 {mae_main*100:.2f}%)")
print(f"  - 協同肌 (Comp) 平均誤差 (MAE): {mae_comp:.4f} (約 {mae_comp*100:.2f}%)")

# 🔥 這裡就是剛剛幫你補上的：絕對包絡線的皮爾森相關係數 (你的目標 > 0.85 就在這看！)
mvc_corr_main, _ = pearsonr(Y_val_emg[:, 0], pred_emg[:, 0])
mvc_corr_comp, _ = pearsonr(Y_val_emg[:, 1], pred_emg[:, 1])

print("\n🎯 【EMG 包絡線波形相似度 (Envelope Correlation)】")
print(f"  - 主動肌 (Main) 包絡線皮爾森相關係數: {mvc_corr_main:.4f}")
print(f"  - 協同肌 (Comp) 包絡線皮爾森相關係數: {mvc_corr_comp:.4f}")

# --- 2. 🔥 EMG 斜率 (變化率) 波形相似度評估 ---
slope_corr_main, _ = pearsonr(all_true_slopes_main, all_pred_slopes_main)
slope_corr_comp, _ = pearsonr(all_true_slopes_comp, all_pred_slopes_comp)

print("\n📈 【EMG 斜率/動態變化相似度 (Post-processed Slope Correlation)】")
print(f"  - 主動肌 (Main) 波形相似度 (Pearson r): {slope_corr_main:.4f}")
print(f"  - 協同肌 (Comp) 波形相似度 (Pearson r): {slope_corr_comp:.4f}")

# --- 3. 代償機率 (Compensation) 評估 ---
pred_comp_binary = (pred_comp > 0.5).astype(float)
comp_acc = accuracy_score(Y_val_comp, pred_comp_binary)

print("\n⚠️ 【代償動作辨識 (Compensation Detection)】")
print(f"  - 膝蓋內凹二元分類準確率 (Accuracy): {comp_acc*100:.2f}%")
print("="*50 + "\n")

# ==========================================
# 5. 儲存模型與正規化器 
# ==========================================
print("\n💾 正在儲存多任務模型 (實驗 C: 包含動力學輸入 Velocity/Acceleration)...")
# 🚀 存檔名稱更新為 withVA
model.save("teacher_multitask_random_withVA.keras")
joblib.dump(scaler_x, "scaler_x_multitask_random_withVA.pkl")
joblib.dump(scaler_y_emg, "scaler_y_emg_multitask_random_withVA.pkl") 
print("✅ 實驗 C 模型儲存完畢！")