import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# 1. 讀取資料
# ==========================================
file_path = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0507\Excel_0507\squat_combined_training_data.csv"
df = pd.read_csv(file_path)

# 分離特徵 (X) 與 標籤 (Y)
# 隨機森林對沒正規化的資料適應力很強，但為了保險我們還是做一下
feature_cols = [col for col in df.columns if not col.startswith('EMG_')]
label_cols = ['EMG_Main', 'EMG_Compass']

X = df[feature_cols].values
Y = df[label_cols].values

# ==========================================
# 2. 資料切分 (維持時間順序)
# ==========================================
# 即使是隨機森林，測試「未來」的動作還是比較客觀
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]

# ==========================================
# 3. 建立並訓練隨機森林模型
# ==========================================
print("🌲 隨機森林正在生長中...")
# n_estimators: 森林裡要有幾棵樹 (100-200 適合小數據)
# random_state: 固定隨機種子，確保每次結果一樣
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# 訓練
rf_model.fit(X_train, Y_train)
print("✅ 訓練完成！")

# ==========================================
# 4. 預測與結果視覺化
# ==========================================
predictions = rf_model.predict(X_test)

plt.figure(figsize=(15, 8))

# 畫出 EMG_Main 的結果
plt.subplot(2, 1, 1)
plt.plot(Y_test[:, 0], label='True EMG_Main', color='blue', alpha=0.6)
plt.plot(predictions[:, 0], label='RF Predicted', color='red', linestyle='--')
plt.title('Random Forest - EMG Main Prediction (Small Data Test)')
plt.legend()

# 畫出 EMG_Compass 的結果
plt.subplot(2, 1, 2)
plt.plot(Y_test[:, 1], label='True EMG_Compass', color='green', alpha=0.6)
plt.plot(predictions[:, 1], label='RF Predicted', color='orange', linestyle='--')
plt.title('Random Forest - EMG Compass Prediction')
plt.legend()

plt.tight_layout()
plt.show()

# 印出特徵重要性 (這是隨機森林最強的功能！)
print("\n項目的特徵貢獻度 (Feature Importance):")
importance = rf_model.feature_importances_
for i, col in enumerate(feature_cols):
    print(f"{col}: {importance[i]:.4f}")