import pandas as pd
import numpy as np
import os

# ==========================================
# 參數設定區
# ==========================================
# 已經轉換好的 EMG 檔案 (含有 Segment，且已轉為 60 FPS 包絡線)
emg_file_path = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Clean\KUAN_0622_CLEAN_SYNCED_Env_60FPS.csv"

# TRC 檔案列表 (對應 Segment 1~7)
trc_files = [
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Open\OpenCapData_KUAN\MarkerData\0.trc",   # Seg 1
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Open\OpenCapData_KUAN\MarkerData\20.trc",  # Seg 2
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Open\OpenCapData_KUAN\MarkerData\40.trc",  # Seg 3
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Open\OpenCapData_KUAN\MarkerData\50.trc",  # Seg 4
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Open\OpenCapData_KUAN\MarkerData\A.trc",   # Seg 5
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Open\OpenCapData_KUAN\MarkerData\B.trc",   # Seg 6
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Open\OpenCapData_KUAN\MarkerData\C.trc"    # Seg 7
]

# 輸出儲存路徑
output_dir = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Combined"

# ⚠️ 🌟 靜態特徵設定：加入 Squat_1RM_kg (深蹲最大重量)
# 性別: 0=女, 1=男
SUBJECT_INFO = {
    'Age': 21, 
    'Height_cm': 179.0, 
    'Weight_kg': 73.0, 
    'Gender': 1, 
    'Squat_1RM_kg': 90.0  # 👈 請依據受試者的實際 1RM 修改這個數字
}

# 🏋️ 🌟 負重設定 (各 Segment 對應的槓片負重量，單位: kg)
SEGMENT_LOADS_KG = {
    1: 0.0,   # 徒手
    2: 20.0,  # 空槓
    3: 40.0,  
    4: 50.0,  
    5: 0.0,  # 對應 A.trc (請視實際狀況修改)
    6: 0.0,  # 對應 B.trc (請視實際狀況修改)
    7: 0.0   # 對應 C.trc (請視實際狀況修改)
}

# ⚠️ MVC 基準值 (用於 EMG 正規化)
MVC_MAIN_CLEAN_MAX = 18.2 
MVC_COMP_CLEAN_MAX = 4.5

FPS = 60.0
dt = 1.0 / FPS
os.makedirs(output_dir, exist_ok=True)

def calculate_angle(v1, v2):
    dot_product = np.sum(v1 * v2, axis=1)
    norm_v1 = np.linalg.norm(v1, axis=1)
    norm_v2 = np.linalg.norm(v2, axis=1)
    cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

# 讀取 EMG 資料 (容錯處理編碼)
try:
    df_emg_all = pd.read_csv(emg_file_path, encoding='utf-8-sig')
except UnicodeDecodeError:
    df_emg_all = pd.read_csv(emg_file_path, encoding='big5')

segment_target_frames = df_emg_all.groupby('Segment')['Frame'].max().to_dict()

for segment_idx, trc_path in enumerate(trc_files, start=1):
    if segment_idx not in segment_target_frames: 
        print(f"⚠️ EMG 資料中沒有 Segment {segment_idx}，跳過。")
        continue
        
    target_frames = int(segment_target_frames[segment_idx])
    
    # ----------------------------------------------------
    # 2.1 讀取與插值對齊 TRC 骨架數據
    # ----------------------------------------------------
    try:
        df_gen = pd.read_csv(trc_path, sep='\t', skiprows=4)
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
        'X1': 'Neck_X', 'Y1': 'Neck_Y', 'Z1': 'Neck_Z',                 
        'Y5': 'Shoulder_Y', 'Z5': 'Shoulder_Z',         
        'X8': 'midHip_X', 'Y8': 'midHip_Y', 'Z8': 'midHip_Z',           
        'X9': 'RHip_X', 'Y9': 'RHip_Y', 'Z9': 'RHip_Z',                 
        'X10': 'RKnee_X', 'Y10': 'RKnee_Y', 'Z10': 'RKnee_Z',           
        'X11': 'RAnkle_X', 'Y11': 'RAnkle_Y', 'Z11': 'RAnkle_Z',        
        'X12': 'Hip_X', 'Y12': 'Hip_Y', 'Z12': 'Hip_Z',                 
        'X13': 'Knee_X', 'Y13': 'Knee_Y', 'Z13': 'Knee_Z',               
        'X14': 'Ankle_X', 'Y14': 'Ankle_Y', 'Z14': 'Ankle_Z',             
        'X15': 'BigToe_X', 'Y15': 'BigToe_Y', 'Z15': 'BigToe_Z',         
        'X16': 'SmallToe_X', 'Y16': 'SmallToe_Y', 'Z16': 'SmallToe_Z',   
        'Y17': 'LHeel_Y',                                               
        'Y18': 'RBigToe_Y',                                             
        'Y20': 'RHeel_Y'                                                
    }
    
    available_cols = {k: v for k, v in columns_to_keep.items() if k in df_resampled_skel.columns}
    df_clean = df_resampled_skel[list(available_cols.keys())].rename(columns=available_cols)

    df_clean['Toe_X'] = (df_clean['BigToe_X'] + df_clean['SmallToe_X']) / 2.0
    df_clean['Toe_Y'] = (df_clean['BigToe_Y'] + df_clean['SmallToe_Y']) / 2.0
    df_clean['Toe_Z'] = (df_clean['BigToe_Z'] + df_clean['SmallToe_Z']) / 2.0
    
    v_thigh = np.array([df_clean['Hip_Z'] - df_clean['Knee_Z'], df_clean['Hip_Y'] - df_clean['Knee_Y']]).T
    v_shank = np.array([df_clean['Ankle_Z'] - df_clean['Knee_Z'], df_clean['Ankle_Y'] - df_clean['Knee_Y']]).T
    df_clean['Knee_Angle'] = calculate_angle(v_thigh, v_shank)

    v_knee_dir = np.array([df_clean['Knee_X'] - df_clean['Hip_X'],
                           df_clean['Knee_Y'] - df_clean['Hip_Y'],
                           df_clean['Knee_Z'] - df_clean['Hip_Z']]).T
    
    v_toe_dir = np.array([df_clean['Toe_X'] - df_clean['Ankle_X'],
                          df_clean['Toe_Y'] - df_clean['Ankle_Y'],
                          df_clean['Toe_Z'] - df_clean['Ankle_Z']]).T
    
    df_clean['Knee_Toe_Angle_Diff'] = calculate_angle(v_knee_dir, v_toe_dir)
    ref_height = (df_clean['Shoulder_Y'] - df_clean['Ankle_Y']).max()
    
    # ---- 骨架代償特徵 ----
    df_clean['L_Heel_Rise'] = df_clean['LHeel_Y'] - df_clean['BigToe_Y']
    df_clean['R_Heel_Rise'] = df_clean['RHeel_Y'] - df_clean['RBigToe_Y']
    df_clean['L_Heel_Rise_norm'] = df_clean['L_Heel_Rise'] / ref_height
    df_clean['R_Heel_Rise_norm'] = df_clean['R_Heel_Rise'] / ref_height
    df_clean['L_Heel_Rise_vel'] = df_clean['L_Heel_Rise_norm'].diff() / dt
    df_clean['R_Heel_Rise_vel'] = df_clean['R_Heel_Rise_norm'].diff() / dt
    df_clean['L_Heel_Rise_vel'] = df_clean['L_Heel_Rise_vel'].fillna(0)
    df_clean['R_Heel_Rise_vel'] = df_clean['R_Heel_Rise_vel'].fillna(0)

    knee_dist_xz = np.sqrt((df_clean['Knee_X'] - df_clean['RKnee_X'])**2 + (df_clean['Knee_Z'] - df_clean['RKnee_Z'])**2)
    ankle_dist_xz = np.sqrt((df_clean['Ankle_X'] - df_clean['RAnkle_X'])**2 + (df_clean['Ankle_Z'] - df_clean['RAnkle_Z'])**2)
    df_clean['Knee_Ankle_Ratio_norm'] = knee_dist_xz / (ankle_dist_xz + 1e-5)
    df_clean['Knee_Ankle_Ratio_vel'] = df_clean['Knee_Ankle_Ratio_norm'].diff() / dt
    df_clean['Knee_Ankle_Ratio_vel'] = df_clean['Knee_Ankle_Ratio_vel'].fillna(0)

    trunk_vec_z = df_clean['Neck_Z'] - df_clean['midHip_Z']
    trunk_vec_y = df_clean['Neck_Y'] - df_clean['midHip_Y']
    df_clean['Trunk_Lean_Angle'] = np.degrees(np.arctan2(np.abs(trunk_vec_z), trunk_vec_y))
    df_clean['Trunk_Lean_Angle_norm'] = df_clean['Trunk_Lean_Angle'] / 180.0
    df_clean['Trunk_Lean_Angle_vel'] = df_clean['Trunk_Lean_Angle_norm'].diff() / dt
    df_clean['Trunk_Lean_Angle_vel'] = df_clean['Trunk_Lean_Angle_vel'].fillna(0)
    
    # 執行原有的動態特徵迴圈
    for joint in ['Shoulder', 'Knee', 'Ankle', 'Toe']:
        axes = ['Y', 'Z'] if joint == 'Shoulder' else ['X', 'Y', 'Z']
        for axis in axes: 
            col_name = f"{joint}_{axis}"
            root_col = f"Hip_{axis}" if axis in ['X', 'Y', 'Z'] else f"Hip_Y"
            
            if col_name in df_clean.columns and root_col in df_clean.columns:
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

    df_clean['Knee_Toe_Diff_norm'] = df_clean['Knee_Toe_Angle_Diff'] / 180.0
    df_clean['Knee_Toe_Diff_vel'] = df_clean['Knee_Toe_Diff_norm'].diff() / dt
    df_clean['Knee_Toe_Diff_vel'] = df_clean['Knee_Toe_Diff_vel'].fillna(0)
    
    # ----------------------------------------------------
    # 2.3 整合靜態特徵與 EMG 數據 (🌟 關鍵縮放邏輯在這裡)
    # ----------------------------------------------------
    df_final_features = df_clean[[col for col in df_clean.columns if col.endswith(('_norm', '_vel', '_acc'))]].copy()
    
    # 🌟 靜態特徵縮小至 0~2 的區間，有助於模型穩定訓練
    df_final_features['Subj_Age'] = SUBJECT_INFO['Age'] / 100.0             # 例：21 -> 0.21
    df_final_features['Subj_Height_m'] = SUBJECT_INFO['Height_cm'] / 100.0  # 例：179cm -> 1.79m
    df_final_features['Subj_Weight_norm'] = SUBJECT_INFO['Weight_kg'] / 100.0 # 例：73kg -> 0.73
    df_final_features['Subj_Gender'] = SUBJECT_INFO['Gender']
    
    # 🌟 計算「當下負重」佔「深蹲 1RM」的百分比 (Relative Intensity)
    current_load_kg = SEGMENT_LOADS_KG.get(segment_idx, 0.0)
    df_final_features['Load_1RM_Ratio'] = current_load_kg / SUBJECT_INFO['Squat_1RM_kg']
    
    # 提取該組的 EMG 資料
    df_emg_segment = df_emg_all[df_emg_all['Segment'] == segment_idx].copy().reset_index(drop=True)
    
    # 對齊長度
    min_len = min(len(df_final_features), len(df_emg_segment))
    df_final_features = df_final_features.iloc[:min_len].copy()
    
    # 計算 EMG %MVC
    emg_main_mvc = np.clip(df_emg_segment['Raw0'].values[:min_len] / MVC_MAIN_CLEAN_MAX, 0, 1.5)
    emg_comp_mvc = np.clip(df_emg_segment['Raw1'].values[:min_len] / MVC_COMP_CLEAN_MAX, 0, 1.5)
    
    df_final_features['EMG_Main_MVC'] = emg_main_mvc
    df_final_features['EMG_Compass_MVC'] = emg_comp_mvc
    
    # 儲存
    out_name = os.path.join(output_dir, f"KUAN_Seg_{segment_idx}_Combined_Features.csv")
    df_final_features.to_csv(out_name, index=False)
    
    load_ratio = df_final_features['Load_1RM_Ratio'].iloc[0]
    print(f"✅ Segment {segment_idx} (負重: {current_load_kg}kg, 1RM強度: {load_ratio*100:.1f}%) 處理完成！")

print("\n🚀 所有 TRC、靜態特徵與 EMG 特徵合併完成！")