import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from tensorflow.keras.layers import BatchNormalization,Attention, Input, Concatenate, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import joblib

def peak_weighted_mse(y_true, y_pred):
    # 確保資料型態一致，避免 TensorFlow 報錯
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 計算基礎的「均方誤差 (MSE)」：(真實值 - 預測值) 的平方
    squared_error = tf.square(y_true - y_pred)
    
    # ==========================================
    # 規則 A：高點加權 (針對肌肉明顯發力時)
    # ==========================================
    # 隨著真實值 (y_true) 越大，給予的權重就越高。
    # 如果 y_true 是 1.0 (100% MVC)，權重會是 1 + 10*1.0 = 11 倍！
    # 如果 y_true 是 0.0，權重就是 1 倍。
    peak_weight = 1.0 + 60.0 * y_true 
    
    # ==========================================
    # 規則 B：低點 (基準線) 加權 (針對肌肉放鬆時)
    # ==========================================
    base_ones = tf.ones_like(y_true) # 先準備一個全部都是 1 的底板
    
    # tf.where 的意思是：(條件, 條件成立時的值, 條件不成立時的值)
    # 條件：當真實值小於 0.05 (也就是 EMG 非常微弱，低於 5% MVC 時)
    # 成立：越接近 0，權重越高。例如 y_true=0 時，權重是 10*(0.05 - 0) + 1 = 1.5 倍
    # 不成立：權重維持 1 倍
    zero_weight = tf.where(
        y_true < 0.05, 
        10.0 * (0.05 - y_true) + 1.0, 
        base_ones
    )
    
    # ==========================================
    # 總結算：取兩者中「較大」的權重
    # ==========================================
    # 這會形成一個 "V" 型或 "U" 型的權重曲線：兩頭高、中間低
    final_weight = tf.maximum(peak_weight, zero_weight)
    
    # 將原始的誤差乘上我們精心設計的權重，然後取平均回傳
    return tf.reduce_mean(squared_error * final_weight)
# ==========================================
# 1. 讀取資料與正規化準備 (加入個人 MVC 特徵版)
# ==========================================
data_dir = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0513\Combined"
file_pattern = os.path.join(data_dir, "*_Combined_Features.csv")
file_paths = glob.glob(file_pattern)

if not file_paths:
    raise ValueError(f"在 {data_dir} 找不到任何檔案，請檢查路徑！")

# 🔥 關鍵修改 1：建立四位受試者的最大肌力 (MVC) 對照表
# ⚠️ 請務必將下方的數值，替換成您當初為這四人實際量測到的真實 MVC！
subject_mvc_map = {
    'KUAN':     {'MVC_Main': 920, 'MVC_Compass': 750},
    'SHIH_MIN': {'MVC_Main': 975, 'MVC_Compass': 774}, # 之前 SHIH_MIN 的數據
    'YU_JIE':     {'MVC_Main': 827, 'MVC_Compass': 608}
}

print(f"📂 總共找到了 {len(file_paths)} 個檔案準備進行訓練！")

all_dfs = []
for fp in file_paths:
    df = pd.read_csv(fp)
    
    # 從檔名辨識是哪一位受試者 (轉大寫避免大小寫差異)
    filename = os.path.basename(fp).upper()
    current_subj = None
    for subj in subject_mvc_map.keys():
        if subj in filename:
            current_subj = subj
            break
            
    if current_subj is None:
        raise ValueError(f"⚠️ 無法從檔名 {filename} 辨識出受試者，請確認檔名是否包含 KUAN, SHIH_MIN, YU_JIE或TEST_1！")
        
    # 🔥 關鍵修改 2：將該受試者的最大肌力新增為特徵 (X)
    df['Subject_MVC_Main'] = subject_mvc_map[current_subj]['MVC_Main']
    df['Subject_MVC_Compass'] = subject_mvc_map[current_subj]['MVC_Compass']
    
    all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)

# 動態分離 X (骨架特徵 + MVC 特徵) 與 Y (EMG標籤)
# 注意：我們剛加的 'Subject_MVC_Main' 等欄位不以 'EMG_' 開頭，所以會自動被歸類到 X 裡面！
feature_cols = [col for col in combined_df.columns if not col.startswith('EMG_')]
label_cols = ['EMG_Main_MVC', 'EMG_Compass_MVC']

# X 座標特徵使用標準化 (StandardScaler)
# 🔥 關鍵好處：這裡 StandardScaler 會自動把我們剛剛加入的 800, 600 這種大數值的 MVC，
# 跟其他骨架特徵一起縮放到平均為 0、標準差為 1 的範圍，非常利於神經網路訓練！
scaler_x = StandardScaler()
scaler_x.fit(combined_df[feature_cols].values)

# Y 標籤的縮放
scaler_y = MinMaxScaler()
scaler_y.fit(combined_df[label_cols].values)

# ==========================================
# 2. 滑動時間窗切割 (🚨 獨立在每個 Segment 內切割)
# ==========================================
window_size = 100
step_size = 5     

X_windows_list = []
Y_labels_list = []

print("⏳ 正在進行時間窗切割...")
for i, df in enumerate(all_dfs):
    # 分別轉換每個 Segment 的資料
    X_data = scaler_x.transform(df[feature_cols].values)
    Y_data = scaler_y.transform(df[label_cols].values)
    
    # 在該 Segment 內部切窗，確保不會跨界
    segments_extracted = 0
    for j in range(0, len(X_data) - window_size, step_size):
        # 假設延遲大約是 60ms (依據你的 FPS 換算幀數，假設 60FPS，1幀約 16.6ms，推移 3~4 幀)
        delay_frames = 4 
        window_X = X_data[j : j + window_size, :]
        target_Y = Y_data[j + window_size - 1, :] 
        
        X_windows_list.append(window_X)
        Y_labels_list.append(target_Y)
        segments_extracted += 1
        
    print(f"  - Segment {i+1} 提取了 {segments_extracted} 個時間窗")

X_tensor = np.array(X_windows_list)
Y_tensor = np.array(Y_labels_list)
print(f"✅ 總共生成了 {len(X_tensor)} 個時間窗樣本！")
# ==========================================
# 3. 資料切分、建模與訓練
# ==========================================
print("\n🔀 正在從總時間窗中隨機抽取 20% 作為測試集...")

# 🔥 關鍵修改：使用 train_test_split 進行隨機打亂與切分
# test_size=0.2 代表測試集佔 20%
# random_state=42 確保每次隨機抽取的結果都一樣，方便您重現實驗結果
X_train, X_test, Y_train, Y_test = train_test_split(
    X_tensor, Y_tensor, test_size=0.2, random_state=42
)

print(f"✅ 訓練集樣本數: {len(X_train)}")
print(f"✅ 測試集樣本數: {len(X_test)}")

print("\n🧠 正在建構 CNN-BiLSTM-Attention 模型...")
# 注意力機制通常建議使用 Functional API 寫法會比較好處理維度
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# CNN 層
x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu')(inputs)
x = BatchNormalization()(x)
x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

# 第一層 Bi-LSTM
x = Bidirectional(LSTM(units=64, return_sequences=True))(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

# 第二層 Bi-LSTM (為了 Attention，這裡必須保留 return_sequences=True)
lstm_out = Bidirectional(LSTM(units=32, return_sequences=True, dropout=0.2))(x)
# 使用自身對自身做 Attention (Self-Attention 概念)
attention_out = Attention()([lstm_out, lstm_out])

# 將結果展平以連接 Dense 層
x = Flatten()(attention_out)

# 全連接層與輸出層
x = Dense(units=32, activation='relu')(x)
outputs = Dense(units=2, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
# 編譯模型
custom_adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=custom_adam, loss=peak_weighted_mse, metrics=['mae'])
model.summary()
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.3, patience=5, min_lr=0.00001, verbose=1
)
# 提早停止機制
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
# 4. 預測與視覺化 (畫圖)
# ==========================================
predictions = model.predict(X_test)

plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.plot(Y_test[:, 0], label='True EMG_Main (%MVC)', color='blue', alpha=0.7)
plt.plot(predictions[:, 0], label='Predicted EMG_Main', color='red', linestyle='--')
plt.title('Optimized CNN-BiLSTM - EMG Main (Real Causality)')
plt.ylabel('EMG (0~1)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(Y_test[:, 1], label='True EMG_Compass (%MVC)', color='green', alpha=0.7)
plt.plot(predictions[:, 1], label='Predicted EMG_Compass', color='orange', linestyle='--')
plt.title('Optimized CNN-BiLSTM - EMG Compass (Real Causality)')
# 🔥 修改標籤文字，註明是隨機抽樣的結果
plt.xlabel('Time Windows (Tested on Random 20% Data)')
plt.ylabel('EMG (0~1)')
plt.legend()

plt.tight_layout()
plt.show()


# ==========================================
# 6. 計算預測評估指標 (Metrics)
# ==========================================
print("\n📊 測試集 (20% 未看過資料) 最終評估成績單：")

# 將資料分開計算：Main 肌肉與 Compass 肌肉
y_true_main = Y_test[:, 0]
y_pred_main = predictions[:, 0]

y_true_comp = Y_test[:, 1]
y_pred_comp = predictions[:, 1]

# 1. MAE (平均絕對誤差) - 越小越好 (例如 0.05 代表平均預測誤差只有 5% MVC)
mae_main = mean_absolute_error(y_true_main, y_pred_main)
mae_comp = mean_absolute_error(y_true_comp, y_pred_comp)

# 2. RMSE (均方根誤差) - 越小越好 (比 MAE 更嚴格，對極端誤差懲罰較重)
rmse_main = np.sqrt(mean_squared_error(y_true_main, y_pred_main))
rmse_comp = np.sqrt(mean_squared_error(y_true_comp, y_pred_comp))

# 3. R² (決定係數) - 越接近 1 越好 (通常大於 0.7 就算不錯的模型)
r2_main = r2_score(y_true_main, y_pred_main)
r2_comp = r2_score(y_true_comp, y_pred_comp)

# 4. Pearson r (皮爾森相關係數) - 越接近 1 越好 (代表波形起伏的相似度，EMG 論文最愛用！)
corr_main, _ = pearsonr(y_true_main, y_pred_main)
corr_comp, _ = pearsonr(y_true_comp, y_pred_comp)

# 印出結果
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
# 7. 儲存模型與正規化器 (為未來預測做準備)
# ==========================================
print("\n💾 正在儲存模型與正規化器...")

# 1. 儲存神經網路模型 (大腦)
model.save("lightweight_emg_model.h5")

# 2. 儲存 X 的正規化器 (量尺 - 這是必備的！未來的新資料必須用同一把量尺縮放)
joblib.dump(scaler_x, "scaler_x.pkl")

print("✅ 儲存完畢！您現在擁有 lightweight_emg_model.h5 與 scaler_x.pkl 兩個檔案了。")