import pandas as pd
import numpy as np
import os

# ==========================================
# 參數設定區 (已更新為 0622 處理完的資料)
# ==========================================
# 已經轉換好的 EMG 檔案 (含有 Segment，且已轉為 60 FPS 包絡線)
emg_file_path = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Clean\KUAN_0622_CLEAN_SYNCED_Env_60FPS.csv"

# TRC 檔案列表 (對應 Segment 1~5)
trc_files = [
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Open\OpenCapData_KUAN\MarkerData\0.trc",
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Open\OpenCapData_KUAN\MarkerData\20.trc",
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Open\OpenCapData_KUAN\MarkerData\40.trc",
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Open\OpenCapData_KUAN\MarkerData\50.trc",
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Open\OpenCapData_KUAN\MarkerData\A.trc",
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Open\OpenCapData_KUAN\MarkerData\B.trc",
    r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Open\OpenCapData_KUAN\MarkerData\C.trc"
]

# 輸出儲存路徑
output_dir = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Combined"

# ⚠️ 受試者靜態資料
# 性別: 0=女, 1=男
SUBJECT_INFO = {'Age': 21, 'Height_cm': 179.0, 'Weight_kg': 73.0, 'Gender': 1}

# ⚠️ MVC 基準值 (已填入剛剛程式算出來的 Global MVC 數值)
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
    # 💡 更新：擴展 columns_to_keep，納入背部前傾、雙側膝蓋與腳跟點
    columns_to_keep = {
        'X1': 'Neck_X', 'Y1': 'Neck_Y', 'Z1': 'Neck_Z',                 # 用於背部前傾
        'Y5': 'Shoulder_Y', 'Z5': 'Shoulder_Z',         
        'X8': 'midHip_X', 'Y8': 'midHip_Y', 'Z8': 'midHip_Z',           # 用於背部前傾基準
        'X9': 'RHip_X', 'Y9': 'RHip_Y', 'Z9': 'RHip_Z',                 
        'X10': 'RKnee_X', 'Y10': 'RKnee_Y', 'Z10': 'RKnee_Z',           # 用於膝蓋內凹
        'X11': 'RAnkle_X', 'Y11': 'RAnkle_Y', 'Z11': 'RAnkle_Z',         # 用於膝蓋內凹
        'X12': 'Hip_X', 'Y12': 'Hip_Y', 'Z12': 'Hip_Z',                 # 左髖 (維持原名作基準)
        'X13': 'Knee_X', 'Y13': 'Knee_Y', 'Z13': 'Knee_Z',               # 左膝 (維持原名作基準)
        'X14': 'Ankle_X', 'Y14': 'Ankle_Y', 'Z14': 'Ankle_Z',             # 左踝 (維持原名作基準)
        'X15': 'BigToe_X', 'Y15': 'BigToe_Y', 'Z15': 'BigToe_Z',         # 左大腳趾
        'X16': 'SmallToe_X', 'Y16': 'SmallToe_Y', 'Z16': 'SmallToe_Z',   # 左小腳趾
        'Y17': 'LHeel_Y',                                               # 左腳跟 (用於墊腳尖)
        'Y18': 'RBigToe_Y',                                             # 右大腳趾 (用於墊腳尖)
        'Y20': 'RHeel_Y'                                                # 右腳跟 (用於墊腳尖)
    }
    
    available_cols = {k: v for k, v in columns_to_keep.items() if k in df_resampled_skel.columns}
    df_clean = df_resampled_skel[list(available_cols.keys())].rename(columns=available_cols)

    # 利用 BigToe 與 SmallToe 計算腳掌前緣的中點
    df_clean['Toe_X'] = (df_clean['BigToe_X'] + df_clean['SmallToe_X']) / 2.0
    df_clean['Toe_Y'] = (df_clean['BigToe_Y'] + df_clean['SmallToe_Y']) / 2.0
    df_clean['Toe_Z'] = (df_clean['BigToe_Z'] + df_clean['SmallToe_Z']) / 2.0
    
    # 原始大腿與小腿的夾角 (計算 Y, Z 軸投影)
    v_thigh = np.array([df_clean['Hip_Z'] - df_clean['Knee_Z'], df_clean['Hip_Y'] - df_clean['Knee_Y']]).T
    v_shank = np.array([df_clean['Ankle_Z'] - df_clean['Knee_Z'], df_clean['Ankle_Y'] - df_clean['Knee_Y']]).T
    df_clean['Knee_Angle'] = calculate_angle(v_thigh, v_shank)

    # 向量 1：大腿 3D 指向
    v_knee_dir = np.array([df_clean['Knee_X'] - df_clean['Hip_X'],
                           df_clean['Knee_Y'] - df_clean['Hip_Y'],
                           df_clean['Knee_Z'] - df_clean['Hip_Z']]).T
    
    # 向量 2：腳掌 3D 指向
    v_toe_dir = np.array([df_clean['Toe_X'] - df_clean['Ankle_X'],
                          df_clean['Toe_Y'] - df_clean['Ankle_Y'],
                          df_clean['Toe_Z'] - df_clean['Ankle_Z']]).T
    
    df_clean['Knee_Toe_Angle_Diff'] = calculate_angle(v_knee_dir, v_toe_dir)
    
    ref_height = (df_clean['Shoulder_Y'] - df_clean['Ankle_Y']).max()
    
    # ====================================================
    # 🌟 新增：三大代償特徵工程運算 (確保結尾為 _norm / _vel)
    # ====================================================
    
    # ---- 1. 墊腳尖特徵 (Heel Raise) ----
    # 計算腳跟相對於大腳趾的垂直高度差 (平放時接近常數，墊腳尖時數值顯著正向增加)
    df_clean['L_Heel_Rise'] = df_clean['LHeel_Y'] - df_clean['BigToe_Y']
    df_clean['R_Heel_Rise'] = df_clean['RHeel_Y'] - df_clean['RBigToe_Y']
    
    # 進行身高比例正規化與速度計算
    df_clean['L_Heel_Rise_norm'] = df_clean['L_Heel_Rise'] / ref_height
    df_clean['R_Heel_Rise_norm'] = df_clean['R_Heel_Rise'] / ref_height # 已補上底線
    
    df_clean['L_Heel_Rise_vel'] = df_clean['L_Heel_Rise_norm'].diff() / dt
    df_clean['R_Heel_Rise_vel'] = df_clean['R_Heel_Rise_norm'].diff() / dt
    df_clean['L_Heel_Rise_vel'] = df_clean['L_Heel_Rise_vel'].fillna(0)
    df_clean['R_Heel_Rise_vel'] = df_clean['R_Heel_Rise_vel'].fillna(0)

    # ---- 2. 膝蓋內凹特徵 (Knee Valgus) ----
    # 在 X-Z 水平平面上，計算「雙膝距離」與「雙踝距離」
    knee_dist_xz = np.sqrt((df_clean['Knee_X'] - df_clean['RKnee_X'])**2 + (df_clean['Knee_Z'] - df_clean['RKnee_Z'])**2)
    ankle_dist_xz = np.sqrt((df_clean['Ankle_X'] - df_clean['RAnkle_X'])**2 + (df_clean['Ankle_Z'] - df_clean['RAnkle_Z'])**2)
    # 膝踝距離比：正常下蹲時比例應保持穩定，若內凹則比例會大幅小於 1 且快速下降
    df_clean['Knee_Ankle_Ratio_norm'] = knee_dist_xz / (ankle_dist_xz + 1e-5)
    df_clean['Knee_Ankle_Ratio_vel'] = df_clean['Knee_Ankle_Ratio_norm'].diff() / dt
    df_clean['Knee_Ankle_Ratio_vel'] = df_clean['Knee_Ankle_Ratio_vel'].fillna(0)

    # ---- 3. 背部前傾特徵 (Excessive Forward Trunk Lean) ----
    # 計算軀幹向量（midHip 到 Neck）在矢狀面 (Y-Z 平面) 的夾角
    trunk_vec_z = df_clean['Neck_Z'] - df_clean['midHip_Z']
    trunk_vec_y = df_clean['Neck_Y'] - df_clean['midHip_Y']
    # 使用 arctan2 計算與絕對垂直線的角度 (0度代表完全直立，角度越大代表前傾越嚴重)
    df_clean['Trunk_Lean_Angle'] = np.degrees(np.arctan2(np.abs(trunk_vec_z), trunk_vec_y))
    # 將角度正規化 (除以 180) 與計算角速度
    df_clean['Trunk_Lean_Angle_norm'] = df_clean['Trunk_Lean_Angle'] / 180.0
    df_clean['Trunk_Lean_Angle_vel'] = df_clean['Trunk_Lean_Angle_norm'].diff() / dt
    df_clean['Trunk_Lean_Angle_vel'] = df_clean['Trunk_Lean_Angle_vel'].fillna(0)

    # ====================================================
    
    # 執行原有的動態特徵迴圈 (Shoulder, Knee, Ankle, Toe 的中心化與速度計算)
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

    # 角度特徵正規化與速度 (原有的 Knee 角度特徵)
    df_clean['Knee_Angle_norm'] = df_clean['Knee_Angle'] / 180.0
    df_clean['Knee_Angle_vel'] = df_clean['Knee_Angle_norm'].diff() / dt
    df_clean['Knee_Angle_vel'] = df_clean['Knee_Angle_vel'].fillna(0)

    df_clean['Knee_Toe_Diff_norm'] = df_clean['Knee_Toe_Angle_Diff'] / 180.0
    df_clean['Knee_Toe_Diff_vel'] = df_clean['Knee_Toe_Diff_norm'].diff() / dt
    df_clean['Knee_Toe_Diff_vel'] = df_clean['Knee_Toe_Diff_vel'].fillna(0)
    
    # ----------------------------------------------------
    # 2.3 整合靜態特徵與 EMG 數據
    # ----------------------------------------------------
    # 自動篩選出所有以 _norm, _vel, _acc 結尾的欄位 (我們新加的特徵會自動被收進來)
    df_final_features = df_clean[[col for col in df_clean.columns if col.endswith(('_norm', '_vel', '_acc'))]].copy()
    
    df_final_features['Subj_Age'] = SUBJECT_INFO['Age']
    df_final_features['Subj_Height'] = SUBJECT_INFO['Height_cm']
    df_final_features['Subj_Weight'] = SUBJECT_INFO['Weight_kg']
    df_final_features['Subj_Gender'] = SUBJECT_INFO['Gender']
    
    # 提取該組的 EMG 資料
    df_emg_segment = df_emg_all[df_emg_all['Segment'] == segment_idx].copy().reset_index(drop=True)
    
    # ⚠️ 長度安全對齊：確保骨架特徵長度與 EMG 特徵長度完美一致
    min_len = min(len(df_final_features), len(df_emg_segment))
    df_final_features = df_final_features.iloc[:min_len].copy()
    
    # 計算 %MVC，並設定上限為 1.5 (150%)
    emg_main_mvc = np.clip(df_emg_segment['Raw0'].values[:min_len] / MVC_MAIN_CLEAN_MAX, 0, 1.5)
    emg_comp_mvc = np.clip(df_emg_segment['Raw1'].values[:min_len] / MVC_COMP_CLEAN_MAX, 0, 1.5)
    
    df_final_features['EMG_Main_MVC'] = emg_main_mvc
    df_final_features['EMG_Compass_MVC'] = emg_comp_mvc
    
    # 輸出合併檔案
    out_name = os.path.join(output_dir, f"KUAN_Seg_{segment_idx}_Combined_Features.csv")
    df_final_features.to_csv(out_name, index=False)
    print(f"✅ Segment {segment_idx} 處理完成 (包含靜態特徵、新代償特徵與 EMG_MVC)，已輸出至 {out_name}")

print("\n🚀 所有 TRC 與 EMG 特徵合併完成！")