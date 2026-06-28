import pandas as pd
import numpy as np
import scipy.signal as signal
import os
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

# --- 設定參數 ---
TARGET_FPS = 60.0  
EMG_FS = 1000.0

# 1. 原始資料路徑 (Input)
INPUT_FILE = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Raw\KUAN_0622_Raw_DATA.csv"

# 2. 輸出路徑 (Output) - 僅保留總檔輸出目錄
COMBINED_OUTPUT_DIR = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Clean"


def apply_dual_emg_pipeline(data, fs):
    """雙軌 EMG 處理：同時輸出給 FFT (保留高頻) 與給 AI (低通包絡線) 的資料"""
    # 1. 去直流偏移
    signal_centered = data - np.mean(data)
    
    # 2. 60Hz 陷波濾波器
    b_notch, a_notch = iirnotch(60.0, 30.0, fs)
    signal_notched = filtfilt(b_notch, a_notch, signal_centered)
    
    # 3. 20Hz 高通濾波器
    nyq = 0.5 * fs
    sos_high = butter(4, 20.0 / nyq, btype='high', output='sos')
    signal_high = sosfiltfilt(sos_high, signal_notched)
    
    # 🌟 分流點 1：保留 20Hz 以上的高頻震盪，這是給 FFT 分析疲勞用的
    signal_for_fft = signal_high 
    
    # 4. 全波整流
    signal_rectified = np.abs(signal_high)
    
    # 5. 5Hz 低通濾波器
    sos_low = butter(4, 5.0 / nyq, btype='low', output='sos')
    signal_envelope = sosfiltfilt(sos_low, signal_rectified)
    
    # 🌟 分流點 2：平滑的肌肉出力趨勢，這是給 AI 模型用的
    return signal_for_fft, signal_envelope


def process_and_sync_emg(df, fps, fs):
    print("⏳ 正在進行雙軌 EMG 處理 (AI 訓練用 60Hz + 疲勞分析用 1000Hz)...")
    data_cols = ['Raw0', 'Raw1', 'Raw2']
    
    # 在 1000Hz 狀態下先做好濾波與包絡線
    for col in data_cols:
        if col in df.columns:
            raw_data = pd.to_numeric(df[col], errors='coerce').fillna(0).values
            fft_data, env_data = apply_dual_emg_pipeline(raw_data, fs)
            df[f'{col}_FFT'] = fft_data
            df[f'{col}_Env'] = env_data

    # 尋找同步訊號切割點 (保留此邏輯以確保各組別時間軸對齊正確)
    start_indices = df[df['Time_ms'] == 1].index.tolist()
    start_indices.append(len(df)) 
    
    all_segments_env = []
    all_segments_fft = []
    
    for i in range(len(start_indices) - 1):
        start_idx = start_indices[i]
        end_idx = start_indices[i+1]
        
        segment_df = df.iloc[start_idx:end_idx].copy()
        max_ms = segment_df['Time_ms'].max()
        
        if max_ms <= 0 or pd.isna(max_ms): continue
            
        num_frames = int(round((max_ms / 1000.0) * fps))
        if num_frames == 0: num_frames = 1
            
        print(f"🎬 組別 {i+1}: 總時長 {max_ms} ms -> AI 降頻 {num_frames} 偵 / FFT 保持 {len(segment_df)} 筆")

        # 建立完美的理論時間軸
        num_raw_samples = len(segment_df)
        perfect_valid_x = np.linspace(0, max_ms, num_raw_samples)
        
        # ==========================================
        # 建立軌道 1：給 AI 用的 60FPS 降頻包絡線資料
        # ==========================================
        new_times = np.linspace(0, max_ms, num_frames)
        resampled_env = pd.DataFrame({
            'Segment': i + 1,
            'Frame': np.arange(1, num_frames + 1),
            'Time_ms': new_times
        })
        
        for col in data_cols:
            env_col = f'{col}_Env'
            if env_col in segment_df.columns:
                valid_y = segment_df[env_col].values
                resampled_env[col] = np.clip(np.interp(new_times, perfect_valid_x, valid_y), 0, None)
            
        if 'Marker' in segment_df.columns:
            valid_y_marker = pd.to_numeric(segment_df['Marker'], errors='coerce').fillna(0).values
            resampled_env['Marker'] = np.round(np.interp(new_times, perfect_valid_x, valid_y_marker)).astype(int)
            
        all_segments_env.append(resampled_env)

        # ==========================================
        # 建立軌道 2：給 FFT 用的 1000Hz 高通乾淨資料
        # ==========================================
        fft_df = pd.DataFrame({
            'Segment': i + 1,
            'Time_ms': perfect_valid_x # 使用修復過的平滑時間軸
        })
        for col in data_cols:
            fft_col = f'{col}_FFT'
            if fft_col in segment_df.columns:
                fft_df[col] = segment_df[fft_col].values
                
        if 'Marker' in segment_df.columns:
            fft_df['Marker'] = pd.to_numeric(segment_df['Marker'], errors='coerce').fillna(0).values
            
        all_segments_fft.append(fft_df)

    return all_segments_env, all_segments_fft


if __name__ == '__main__':
    try:
        df = pd.read_csv(INPUT_FILE, encoding='big5', low_memory=False)
    except:
        df = pd.read_csv(INPUT_FILE, encoding='cp950', low_memory=False)

    # 取得兩組不同的資料列表
    segmented_dfs_env, segmented_dfs_fft = process_and_sync_emg(df, TARGET_FPS, EMG_FS)

    if segmented_dfs_env and segmented_dfs_fft:
        os.makedirs(COMBINED_OUTPUT_DIR, exist_ok=True)

        # 📦 任務：合併總檔輸出 (檔名加上 Env_60FPS 與 FFT_1000Hz 區別)
        print("\n📦 開始產生合併總檔...")
        
        # 輸出 60Hz AI 訓練總檔
        combined_env_df = pd.concat(segmented_dfs_env, ignore_index=True)
        env_out_path = os.path.join(COMBINED_OUTPUT_DIR, 'KUAN_CLEAN_SYNCED_Env_60FPS.csv')
        combined_env_df.to_csv(env_out_path, index=False, encoding='utf-8-sig')
        
        # 輸出 1000Hz FFT 疲勞分析總檔
        combined_fft_df = pd.concat(segmented_dfs_fft, ignore_index=True)
        fft_out_path = os.path.join(COMBINED_OUTPUT_DIR, 'KUAN_CLEAN_SYNCED_FFT_1000Hz.csv')
        combined_fft_df.to_csv(fft_out_path, index=False, encoding='utf-8-sig')
        
        print(f"🎉 AI 總檔儲存至: {env_out_path}")
        print(f"🎉 FFT 總檔儲存至: {fft_out_path}")
            
        print("\n🚀 雙軌總檔處理與輸出皆已順利完成！")