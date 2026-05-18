import pandas as pd
import numpy as np
import scipy.signal as signal
import os

# ==========================================
# 參數設定區 (請根據實際路徑調整)
# ==========================================
# 已經轉換好的 EMG 檔案 (含有 Segment 1~5，且已轉為每偵)
emg_file_path = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0513\emg\KUAN_RAW_DATA_auto_calculated.csv"

# TRC 檔案列表 (對應 Segment 1~5)
trc_files = [
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0513\opoencap\KUAN\21-50\OpenCapData_6dd32c04-9aaa-446f-b58b-f56163bf7867\MarkerData\general1-10.trc.csv",
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0513\opoencap\KUAN\21-50\OpenCapData_6dd32c04-9aaa-446f-b58b-f56163bf7867\MarkerData\general11-20.trc.csv",
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0513\opoencap\KUAN\21-50\OpenCapData_6dd32c04-9aaa-446f-b58b-f56163bf7867\MarkerData\general21-30.trc.csv",
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0513\opoencap\KUAN\21-50\OpenCapData_6dd32c04-9aaa-446f-b58b-f56163bf7867\MarkerData\general31-40.trc.csv",
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0513\opoencap\KUAN\21-50\OpenCapData_6dd32c04-9aaa-446f-b58b-f56163bf7867\MarkerData\general41-50.trc.csv"
]

# 輸出儲存路徑
output_dir = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0513\Combined"

# ==========================================
# 演算法核心參數
# ==========================================
MVC_MAIN_RAW = 920
MVC_COMP_RAW = 750
BASELINE_FIXED = 462.0  # 🔥 固定基準值

FPS = 60.0              # 影片幀率
dt = 1.0 / FPS

# 🔥 關鍵修改：因為您的 EMG 檔案已經轉換為「每偵一筆」，
# 此時數據的「取樣率」等於影片的「幀率 (FPS)」！
EMG_FS = FPS           

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 輔助函式：計算膝蓋角度
# ==========================================
def calculate_angle(v1, v2):
    dot_product = np.sum(v1 * v2, axis=1)
    norm_v1 = np.linalg.norm(v1, axis=1)
    norm_v2 = np.linalg.norm(v2, axis=1)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

# ==========================================
# 第一步：讀取 EMG 檔案並獲取每組目標偵數
# ==========================================
try:
    df_emg_all = pd.read_csv(emg_file_path, encoding='utf-8-sig')
except UnicodeDecodeError:
    df_emg_all = pd.read_csv(emg_file_path, encoding='big5')

segment_target_frames = df_emg_all.groupby('Segment')['Frame'].max().to_dict()
print("🎯 各 Segment 目標偵數:", segment_target_frames)

# ==========================================
# 第二步：處理每個 Segment
# ==========================================
for segment_idx, trc_path in enumerate(trc_files, start=1):
    print(f"\n⏳ 開始處理 Segment {segment_idx} ...")
    
    if segment_idx not in segment_target_frames:
        print(f"⚠️ 找不到 Segment {segment_idx} 的 EMG 數據，跳過。")
        continue
        
    target_frames = int(segment_target_frames[segment_idx])
    
    # ----------------------------------------------------
    # 2.1 讀取與插值對齊 TRC 骨架數據
    # ----------------------------------------------------
    try:
        df_gen = pd.read_csv(trc_path, skiprows=4)
    except FileNotFoundError:
        print(f"⚠️ 找不到 TRC 檔案: {trc_path}，跳過此組。")
        continue

    df_gen = df_gen.rename(columns={'Unnamed: 0': 'Frame', 'Unnamed: 1': 'Time'})
    df_gen = df_gen.dropna(subset=['Frame'])
    df_gen['Frame'] = df_gen['Frame'].astype(int)
    
    start_frame = df_gen['Frame'].min()
    df_gen['Frame'] = df_gen['Frame'] - start_frame + 1
    
    original_frames = df_gen['Frame'].values
    target_frame_grid = np.linspace(1, target_frames, target_frames)
    resampled_data = {'Frame': np.arange(1, target_frames + 1)}
    
    for col in df_gen.columns:
        if col not in ['Frame', 'Time']:
            valid_y = pd.to_numeric(df_gen[col], errors='coerce').fillna(0).values
            resampled_data[col] = np.interp(target_frame_grid, original_frames, valid_y)
            
    df_resampled_skel = pd.DataFrame(resampled_data)
    
    # ----------------------------------------------------
    # 2.2 特徵工程：提取需要的骨架點與計算角度/速度/加速度
    # ----------------------------------------------------
    columns_to_keep = {
        'Y5': 'Shoulder_Y', 'Z5': 'Shoulder_Z',
        'Y13': 'Hip_Y', 'Z13': 'Hip_Z',
        'Y14': 'Knee_Y', 'Z14': 'Knee_Z',
        'Y15': 'Ankle_Y', 'Z15': 'Ankle_Z' 
    }
    df_clean = df_resampled_skel[list(columns_to_keep.keys())].rename(columns=columns_to_keep)
    
    v_thigh = np.array([df_clean['Hip_Z'] - df_clean['Knee_Z'], df_clean['Hip_Y'] - df_clean['Knee_Y']]).T
    v_shank = np.array([df_clean['Ankle_Z'] - df_clean['Knee_Z'], df_clean['Ankle_Y'] - df_clean['Knee_Y']]).T
    df_clean['Knee_Angle'] = calculate_angle(v_thigh, v_shank)
    
    ref_height = (df_clean['Shoulder_Y'] - df_clean['Ankle_Y']).max()
    for joint in ['Shoulder', 'Knee', 'Ankle']:
        for axis in ['Y', 'Z']:
            col_name = f"{joint}_{axis}"
            root_col = f"Hip_{axis}"
            
            centered_col = f"{col_name}_centered"
            df_clean[centered_col] = df_clean[col_name] - df_clean[root_col]
            
            norm_col = f"{col_name}_norm"
            df_clean[norm_col] = df_clean[centered_col] / ref_height
            
            vel_col = f"{col_name}_vel"
            df_clean[vel_col] = df_clean[norm_col].diff() / dt
            df_clean[vel_col] = df_clean[vel_col].fillna(0)
            
            acc_col = f"{col_name}_acc"
            df_clean[acc_col] = df_clean[vel_col].diff() / dt
            df_clean[acc_col] = df_clean[acc_col].fillna(0)

    df_clean['Knee_Angle_norm'] = df_clean['Knee_Angle'] / 180.0
    df_clean['Knee_Angle_vel'] = df_clean['Knee_Angle_norm'].diff() / dt
    df_clean['Knee_Angle_vel'] = df_clean['Knee_Angle_vel'].fillna(0)

    final_columns = [col for col in df_clean.columns if col.endswith(('_norm', '_vel', '_acc'))]
    df_final_features = df_clean[final_columns].copy()
    
    # ----------------------------------------------------
    # 2.3 提取對應的 EMG 數據並處理 MVC 平滑包絡線
    # ----------------------------------------------------
    df_emg_segment = df_emg_all[df_emg_all['Segment'] == segment_idx].copy().reset_index(drop=True)
    
    # 計算振幅 (固定基準值)
    mvc_main_amp = MVC_MAIN_RAW - BASELINE_FIXED
    mvc_comp_amp = MVC_COMP_RAW - BASELINE_FIXED
    
    # 🔥 關鍵修改：將 'Main_raw' 改為 'Raw0'，將 'Compasste1_raw' 改為 'Raw1'
    # (請依據您實際貼片的通道位置調整 Raw0 或 Raw1)
    emg_main_rectified = np.abs(df_emg_segment['Raw0'].values - BASELINE_FIXED) / mvc_main_amp
    emg_comp_rectified = np.abs(df_emg_segment['Raw1'].values - BASELINE_FIXED) / mvc_comp_amp
    
    # 低通濾波 (頻率設定為 5Hz，取樣率設定為 FPS)
    b, a = signal.butter(4, 5.0 / (EMG_FS / 2.0), btype='low')
    emg_main_mvc_env = signal.filtfilt(b, a, emg_main_rectified)
    emg_comp_mvc_env = signal.filtfilt(b, a, emg_comp_rectified)
    
    # 合併特徵
    df_final_features['EMG_Main_MVC'] = emg_main_mvc_env
    df_final_features['EMG_Compass_MVC'] = emg_comp_mvc_env
    
    # ----------------------------------------------------
    # 2.4 輸出該 Segment 的結果
    # ----------------------------------------------------
    output_filename = os.path.join(output_dir, f"KUAN_Segment_{segment_idx}_Combined_Features.csv")
    df_final_features.to_csv(output_filename, index=False)
    print(f"✅ Segment {segment_idx} 處理完成，輸出長度: {len(df_final_features)} 偵")
    print(f"📁 儲存至: {output_filename}")

print("\n🎉 全部 Segment 結合完成！")