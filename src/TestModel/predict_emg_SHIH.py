import cv2
import numpy as np
import tensorflow as tf
import keras
import joblib
from collections import deque
from tensorflow.keras.layers import LSTM, Bidirectional, Attention
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
# ==========================================
# 1. 參數設定與載入模型
# ==========================================
MODEL_PATH = r"D:\project\NCU\114\Jumior\AI Project\Dataset\Code\Model\lightweight_emg_model.h5"
SCALER_PATH = r"D:\project\NCU\114\Jumior\AI Project\Dataset\Code\Model\scaler_x.pkl"
VIDEO_PATH = r"D:\project\NCU\114\Jumior\AI Project\Dataset\Code\Test_video\video\KUAN_TEST_0513_Segment11-20.mp4"
@keras.saving.register_keras_serializable()
class SafeLSTM(LSTM):
    def __init__(self, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(**kwargs)

custom_objs = {"LSTM": SafeLSTM, "Bidirectional": Bidirectional, "Attention": Attention}
model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objs) 
scaler_x = joblib.load(SCALER_PATH)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

FAKE_MVC_MAIN = 900
FAKE_MVC_COMPASS = 700

# ==========================================
# 2. 骨架特徵提取函數 (修正對齊 TRC 演算法)
# ==========================================
def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_spatial_vars_aligned(landmarks):
    # Mediapipe 的座標: x=水平, y=垂直向下, z=深度
    # 這裡直接轉出 NumPy 陣列
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    
    L_sh, R_sh = 11, 12
    L_hip, R_hip = 23, 24
    L_kn, R_kn = 25, 26
    L_an, R_an = 27, 28
    
    shoulder = (pts[L_sh] + pts[R_sh]) / 2.0
    hip      = (pts[L_hip] + pts[R_hip]) / 2.0
    knee     = (pts[L_kn] + pts[R_kn]) / 2.0
    ankle    = (pts[L_an] + pts[R_an]) / 2.0
    
    # 1. 膝蓋角度並除以 180.0 (對齊 TRC 的 Knee_Angle_norm)
    knee_angle_norm = float(calculate_angle(hip, knee, ankle)) / 180.0
    
    # 2. 計算 ref_height (肩膀Y - 腳踝Y 的距離，以估算身高比例)
    axis_Y, axis_Z = 1, 2
    ref_height = abs(shoulder[axis_Y] - ankle[axis_Y])
    if ref_height == 0: ref_height = 1e-5 # 避免除以 0
    
    # 3. 將座標以 Hip 為中心 (Centered) 並除以 ref_height (Norm)
    shoulder_norm_Y = (shoulder[axis_Y] - hip[axis_Y]) / ref_height
    shoulder_norm_Z = (shoulder[axis_Z] - hip[axis_Z]) / ref_height
    
    knee_norm_Y = (knee[axis_Y] - hip[axis_Y]) / ref_height
    knee_norm_Z = (knee[axis_Z] - hip[axis_Z]) / ref_height
    
    ankle_norm_Y = (ankle[axis_Y] - hip[axis_Y]) / ref_height
    ankle_norm_Z = (ankle[axis_Z] - hip[axis_Z]) / ref_height

    # 回傳純正規化後的 7 個變數
    return np.array([
        float(shoulder_norm_Y), float(shoulder_norm_Z),
        float(knee_norm_Y),     float(knee_norm_Z),
        float(ankle_norm_Y),    float(ankle_norm_Z),
        float(knee_angle_norm)
    ])

# ==========================================
# 3. 快速提取影片全部特徵 (修正 EMG 與微分)
# ==========================================
# 把假的 EMG 數值改成符合 %MVC 的合理範圍 (例如 0.0)
FAKE_MVC_MAIN_NORM = 0.0 
FAKE_MVC_COMPASS_NORM = 0.0

print("⏳ 正在快速解析影片骨架特徵...")
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps): fps = 30.0
dt = 1.0 / fps

pos_history = deque(maxlen=3)
all_frame_features = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_world_landmarks:
        # 使用剛剛修正過的新函數取得 Normalized 的座標點
        current_norm_pos = extract_spatial_vars_aligned(results.pose_world_landmarks)
        pos_history.append(current_norm_pos)
        
        if len(pos_history) == 3:
            p0, p1, p2 = pos_history
            
            # 對 Normalized 的值做微分，這樣 v 和 a 的尺度才會與 TRC 完全一致
            v1 = (p1 - p0) / dt
            v2 = (p2 - p1) / dt
            a2 = (v2 - v1) / dt
            
            frame_features = np.array([
                p2[0], v2[0], a2[0],  # Shoulder Y norm, vel, acc
                p2[1], v2[1], a2[1],  # Shoulder Z norm, vel, acc
                p2[2], v2[2], a2[2],  # Knee Y norm, vel, acc
                p2[3], v2[3], a2[3],  # Knee Z norm, vel, acc
                p2[4], v2[4], a2[4],  # Ankle Y norm, vel, acc
                p2[5], v2[5], a2[5],  # Ankle Z norm, vel, acc
                p2[6], v2[6],         # Knee Angle norm, vel (注意：無 a2)
                FAKE_MVC_MAIN_NORM, FAKE_MVC_COMPASS_NORM  # 放 0.0 而不是 900
            ])
            all_frame_features.append(frame_features)

cap.release()
print(f"✅ 影片解析完成，共提取 {len(all_frame_features)} 幀特徵！")

# ==========================================
# 4. 批次資料處理與神經網路預測
# ==========================================
print("🧠 正在進行模型批次預測...")
# 先將所有特徵縮放 (與你們 TrainModel.py 的邏輯完全一致)
all_features_np = np.array(all_frame_features)
all_scaled = scaler_x.transform(all_features_np)

window_size = 100
X_windows = []

# 切割成 100 幀的滑動視窗
for i in range(len(all_scaled) - window_size + 1):
    X_windows.append(all_scaled[i : i + window_size])

X_windows = np.array(X_windows)

predictions = model.predict(X_windows)
# 直接精準取出第 0 行 (Main) 與第 1 行 (Compass)
pred_main = predictions[:, 0]
pred_compass = predictions[:, 1]

print("✅ 預測完成！正在生成圖表...")

# ==========================================
# 5. 繪製精美的 EMG 趨勢圖表
# ==========================================
# 產生時間軸 (X軸)，從第 100 幀開始對應的秒數
time_axis = np.arange(len(pred_main)) * dt + (window_size * dt)

plt.figure(figsize=(12, 6))
plt.plot(time_axis, pred_main * 100, label='Predicted EMG Main (%MVC)', color='red', linewidth=2)
plt.plot(time_axis, pred_compass * 100, label='Predicted EMG Compass (%MVC)', color='green', linewidth=2)

plt.title('AI Predicted sEMG Muscle Activation over Squat Video', fontsize=16, fontweight='bold')
plt.xlabel('Time (Seconds)', fontsize=12)
plt.ylabel('Muscle Activation (% MVC)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

# 顯示圖表
plt.show()

# ==========================================
# 6. 輸出預測結果為 CSV 檔案 (符合附件格式)
# ==========================================
print("📝 正在生成並輸出 CSV 檔案...")

# 設定與附件完全一致的 22 個欄位名稱
columns = [
    'Shoulder_Y_norm', 'Shoulder_Y_vel', 'Shoulder_Y_acc', 
    'Shoulder_Z_norm', 'Shoulder_Z_vel', 'Shoulder_Z_acc', 
    'Knee_Y_norm', 'Knee_Y_vel', 'Knee_Y_acc', 
    'Knee_Z_norm', 'Knee_Z_vel', 'Knee_Z_acc', 
    'Ankle_Y_norm', 'Ankle_Y_vel', 'Ankle_Y_acc', 
    'Ankle_Z_norm', 'Ankle_Z_vel', 'Ankle_Z_acc', 
    'Knee_Angle_norm', 'Knee_Angle_vel', 
    'EMG_Main_Predicted', 'EMG_Compass_Predicted'
]

# 將所有擷取的骨架特徵轉換為 DataFrame
df_output = pd.DataFrame(all_frame_features, columns=columns)

# 因為 window_size = 100，前 99 幀沒有預測資料，我們用 NaN (空值) 來補齊長度
pad_length = window_size - 1
padded_pred_main = np.pad(pred_main, (pad_length, 0), constant_values=np.nan)
padded_pred_compass = np.pad(pred_compass, (pad_length, 0), constant_values=np.nan)

# 將假的 FAKE_MVC 覆蓋為 AI 實際預測的數值
df_output['EMG_Main_Predicted'] = padded_pred_main
df_output['EMG_Compass_Predicted'] = padded_pred_compass

# 設定您的輸出路徑
OUTPUT_CSV_PATH = r"D:\project\NCU\114\Jumior\AI Project\Dataset\Code\Test_video\CSV\KUAN_TEST_0513_Segment11-20.csv"

# 輸出為 CSV 檔案 (不包含 index)
df_output.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"✅ CSV 檔案已成功輸出至: {OUTPUT_CSV_PATH}")