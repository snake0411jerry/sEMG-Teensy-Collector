import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from tensorflow.keras.layers import BatchNormalization, Attention, Input, Concatenate, Flatten
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import joblib

# ==========================================
# 1. 讀取資料與受試者資料隔離 (Train: SHIH_MIN, YU_JIE, TEST1 / Val: KUAN)
# ==========================================
# 🔥 更新為新的資料路徑
data_dir = r"D:\project\NCU\114\Jumior\AI Project\Dataset\Code\0519Combined\Origin"
file_pattern = os.path.join(data_dir, "*_Combined_Features.csv")
file_paths = glob.glob(file_pattern)

if not file_paths:
    raise ValueError(f"在 {data_dir} 找不到任何檔案，請檢查路徑！")

# 🔥 新增 TEST1 的 MVC 數值
subject_mvc_map = {
    'KUAN':     {'MVC_Main': 920, 'MVC_Compass': 750},
    'SHIH_MIN': {'MVC_Main': 975, 'MVC_Compass': 774}, 
    'YU_JIE':   {'MVC_Main': 827, 'MVC_Compass': 608},
    'TEST1':    {'MVC_Main': 1018, 'MVC_Compass': 1019}
}

print(f"📂 總共找到了 {len(file_paths)} 個檔案準備進行處理！")

train_dfs = []
val_dfs = []

for fp in file_paths:
    df = pd.read_csv(fp)
    filename = os.path.basename(fp).upper()
    
    current_subj = None
    for subj in subject_mvc_map.keys():
        if subj in filename:
            current_subj = subj
            break
            
    if current_subj is None:
        print(f"⚠️ 略過未知受試者檔案: {filename}")
        continue
        
    df['Subject_MVC_Main'] = subject_mvc_map[current_subj]['MVC_Main']
    df['Subject_MVC_Compass'] = subject_mvc_map[current_subj]['MVC_Compass']
    
    # KUAN 放入驗證集，其餘 (SHIH_MIN, YU_JIE, TEST1) 放入訓練集
    if current_subj == 'TEST1':
        val_dfs.append(df)
    else:
        train_dfs.append(df)

if not train_dfs or not val_dfs:
    raise ValueError("訓練集或驗證集為空！請確認檔名與分類邏輯。")

# ==========================================
# 1.5 獨立訓練 Scaler (防止資料洩漏)
# ==========================================
train_combined_df = pd.concat(train_dfs, ignore_index=True)

feature_cols = [col for col in train_combined_df.columns if not col.startswith('EMG_')]
label_cols = ['EMG_Main_MVC', 'EMG_Compass_MVC']

scaler_x = StandardScaler()
scaler_x.fit(train_combined_df[feature_cols].values)

scaler_y = MinMaxScaler()
scaler_y.fit(train_combined_df[label_cols].values)

print(f"✅ 成功載入！訓練集檔案數: {len(train_dfs)}，驗證集 (KUAN) 檔案數: {len(val_dfs)}")

# ==========================================
# 2. 滑動時間窗切割
# ==========================================
window_size = 40  
step_size = 5     

def create_windows(df_list, dataset_name=""):
    X_list, Y_list = [], []
    segments = 0
    for df in df_list:
        X_scaled = scaler_x.transform(df[feature_cols].values)
        Y_scaled = scaler_y.transform(df[label_cols].values)
        
        for j in range(0, len(X_scaled) - window_size, step_size):
            X_list.append(X_scaled[j : j + window_size, :])
            Y_list.append(Y_scaled[j + window_size // 2, :])
        segments += 1
    
    X_arr = np.array(X_list)
    Y_arr = np.array(Y_list)
    print(f"  - {dataset_name} 共提取了 {len(X_arr)} 個時間窗 (來自 {segments} 個 Segment)")
    return X_arr, Y_arr

print("⏳ 正在進行時間窗切割...")
X_train, Y_train = create_windows(train_dfs, "訓練集 (Train)")
X_test, Y_test = create_windows(val_dfs, "驗證集 (Val/KUAN)")

# ==========================================
# 3. 建模與訓練
# ==========================================
print("\n🧠 正在建構 CNN-BiLSTM-Attention 模型...")
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# CNN 層 
x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
x = BatchNormalization()(x)

# 第一層 Bi-LSTM
x = Bidirectional(LSTM(units=64, return_sequences=True))(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x) 

# 第二層 Bi-LSTM 
lstm_out = Bidirectional(LSTM(units=32, return_sequences=True, dropout=0.1))(x)

# Attention 
attention_out = Attention()([lstm_out, lstm_out])
x = Flatten()(attention_out)

# 全連接層與輸出層
x = Dense(units=32, activation='relu')(x)
outputs = Dense(units=2, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# 宣告優化器
custom_adam = tf.keras.optimizers.Adam(learning_rate=0.001)

# 使用 Huber Loss
huber_loss = tf.keras.losses.Huber(delta=0.1) 

model.compile(optimizer=custom_adam, loss=huber_loss, metrics=['mae'])
model.summary()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.3, patience=5, min_lr=0.00001, verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 訓練模型
history = model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, Y_test), 
    callbacks=[early_stop, reduce_lr], 
    verbose=1
)

# ==========================================
# 4. 預測與視覺化 (針對未見過的 KUAN 資料)
# ==========================================
predictions = model.predict(X_test)

plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.plot(Y_test[:, 0], label='True EMG_Main (%MVC)', color='blue', alpha=0.7)
plt.plot(predictions[:, 0], label='Predicted EMG_Main', color='red', linestyle='--')
plt.title('Performance on Unseen Subject (KUAN) - EMG Main (Huber Loss)')
plt.ylabel('EMG (0~1)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(Y_test[:, 1], label='True EMG_Compass (%MVC)', color='green', alpha=0.7)
plt.plot(predictions[:, 1], label='Predicted EMG_Compass', color='orange', linestyle='--')
plt.title('Performance on Unseen Subject (KUAN) - EMG Compass (Huber Loss)')
plt.xlabel('Time Windows')
plt.ylabel('EMG (0~1)')
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 6. 計算預測評估指標 (Metrics)
# ==========================================
print("\n📊 驗證集 (KUAN 專屬資料) 最終評估成績單：")

y_true_main = Y_test[:, 0]
y_pred_main = predictions[:, 0]
y_true_comp = Y_test[:, 1]
y_pred_comp = predictions[:, 1]

mae_main = mean_absolute_error(y_true_main, y_pred_main)
mae_comp = mean_absolute_error(y_true_comp, y_pred_comp)
rmse_main = np.sqrt(mean_squared_error(y_true_main, y_pred_main))
rmse_comp = np.sqrt(mean_squared_error(y_true_comp, y_pred_comp))
r2_main = r2_score(y_true_main, y_pred_main)
r2_comp = r2_score(y_true_comp, y_pred_comp)
corr_main, _ = pearsonr(y_true_main, y_pred_main)
corr_comp, _ = pearsonr(y_true_comp, y_pred_comp)

print("\n💪 【EMG Main (主肌肉)】")
print(f"  - 波形相似度 (Pearson r) : {corr_main:.4f}")
print(f"  - 模型解釋力 (R² Score)  : {r2_main:.4f}")
print(f"  - 平均誤差   (MAE)       : {mae_main:.4f} (約 {mae_main*100:.2f}% MVC)")
print(f"  - 均方根誤差 (RMSE)      : {rmse_main:.4f}")

print("\n🧭 【EMG Compass (副肌肉)】")
print(f"  - 波形相似度 (Pearson r) : {corr_comp:.4f}")
print(f"  - 模型解釋力 (R² Score)  : {r2_comp:.4f}")
print(f"  - 平均誤差   (MAE)       : {mae_comp:.4f} (約 {mae_comp*100:.2f}% MVC)")
print(f"  - 均方根誤差 (RMSE)      : {rmse_comp:.4f}")

# ==========================================
# 7. 儲存模型與正規化器 
# ==========================================
print("\n💾 正在儲存模型與正規化器...")
model.save("lightweight_emg_model_huber.h5")
joblib.dump(scaler_x, "scaler_x_huber.pkl")
print("✅ 儲存完畢！")