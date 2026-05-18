import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# ==========================================
# 0. 重新定義自訂的損失函數 (載入模型時會用到)
# ==========================================
def peak_weighted_mse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    squared_error = tf.square(y_true - y_pred)
    peak_weight = 1.0 + 60.0 * y_true 
    base_ones = tf.ones_like(y_true)
    zero_weight = tf.where(y_true < 0.05, 10.0 * (0.05 - y_true) + 1.0, base_ones)
    final_weight = tf.maximum(peak_weight, zero_weight)
    return tf.reduce_mean(squared_error * final_weight)

# ==========================================
# 1. 載入我們的大腦 (Model) 與量尺 (Scaler)
# ==========================================
print("🧠 正在載入模型與正規化器...")
# 載入時告訴 TensorFlow 我們自訂的函數是什麼
model = tf.keras.models.load_model(
    "lightweight_emg_model.h5", 
    custom_objects={'peak_weighted_mse': peak_weighted_mse}
)
scaler_x = joblib.load("scaler_x.pkl")
print("✅ 載入成功！準備進行預測。")

# ==========================================
# 2. 讀取「全新」的骨架資料
# ==========================================
# 假設這是未來系統即時錄到、或全新的 CSV 檔案
new_data_path = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0513\Combined\KUAN_New_Action.csv" # 替換成您要測試的檔案路徑
df_new = pd.read_csv(new_data_path)

# 🔥 關鍵：提供這位受試者的 MVC 上限
# （未來如果是新受試者，只要幫他量一次這兩個數字填進來就好）
subject_mvc_main = 920    
subject_mvc_compass = 750 

df_new['Subject_MVC_Main'] = subject_mvc_main
df_new['Subject_MVC_Compass'] = subject_mvc_compass

# ==========================================
# 3. 資料前處理 (使用舊量尺)
# ==========================================
# 抓取特徵欄位 (排除原本檔案裡可能有的 EMG 標籤)
feature_cols = [col for col in df_new.columns if not col.startswith('EMG_')]

# 🚨 絕對不能用 fit_transform，只能用 transform！
# 因為我們必須用訓練時的那把「舊量尺」來衡量新資料
X_new_scaled = scaler_x.transform(df_new[feature_cols].values)

# ==========================================
# 4. 滑動時間窗切割 (Step=1 達成連續高解析度預測)
# ==========================================
window_size = 100
X_windows = []

print("⏳ 正在將新資料切成時間窗...")
# 為了畫出最平滑的連續預測曲線，推論時通常 step 設為 1 (每一偵都預測一次)
for i in range(len(X_new_scaled) - window_size + 1):
    X_windows.append(X_new_scaled[i : i + window_size])
    
X_tensor_new = np.array(X_windows)

# ==========================================
# 5. 進行預測！
# ==========================================
print("⚡ 模型正在進行預測...")
predictions = model.predict(X_tensor_new)

# 提取預測結果 (0~1 之間的 %MVC)
pred_main = predictions[:, 0]
pred_compass = predictions[:, 1]

print(f"✅ 預測完成！共產出了 {len(pred_main)} 筆預測數據。")

# ==========================================
# 6. 視覺化預測結果
# ==========================================
plt.figure(figsize=(12, 5))
plt.plot(pred_main, label='Predicted EMG Main (%MVC)', color='red')
plt.plot(pred_compass, label='Predicted EMG Compass (%MVC)', color='orange')
plt.title('Future Prediction from Skeleton Data Only')
plt.xlabel('Time (Frames)')
plt.ylabel('Predicted EMG (0~1 %MVC)')
plt.legend()
plt.tight_layout()
plt.show()

# 如果您想把預測結果存成新的 CSV：
df_result = pd.DataFrame({'Predicted_Main': pred_main, 'Predicted_Compass': pred_compass})
df_result.to_csv("Predicted_EMG.csv", index=False)