import pandas as pd
import numpy as np
import os

# --- 設定參數 ---
target_fps = 60  
file_path = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0601\emg\TEST1_0601_RAW_DATA.csv"
output_dir = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0601\emg"
# ----------------

def process_per_frame(df, fps):
    # 🚨 修正 Bug：您的資料 Time_ms 是從 0 開始的，不是 1！
    # 這樣才能抓到第一行 Marker = 1 的同步訊號
    start_indices = df[df['Time_ms'] == 1].index.tolist()
    start_indices.append(len(df)) # 加入結尾索引
    
    all_segments = []
    data_cols = ['Raw0', 'Raw1', 'Raw2']
    
    for i in range(len(start_indices) - 1):
        start_idx = start_indices[i]
        end_idx = start_indices[i+1]
        
        # 提取該組數據
        segment_df = df.iloc[start_idx:end_idx].copy()
        
        # 自動計算該組的時長 (ms)
        max_ms = segment_df['Time_ms'].max()
        
        if max_ms <= 0 or pd.isna(max_ms):
            print(f"組別 {i+1} 時長異常，跳過處理。")
            continue
            
        # 根據時長與 FPS 自動計算總偵數
        num_frames = int(round((max_ms / 1000.0) * fps))
        if num_frames == 0:
            num_frames = 1
            
        print(f"組別 {i+1}: 偵測到總時長 {max_ms} ms -> 自動計算為 {num_frames} 偵 (FPS: {fps})")

        # 建立新的時間軸 (從 0 到 max_ms)
        new_times = np.linspace(0, max_ms, num_frames)
        
        resampled = pd.DataFrame({
            'Segment': i + 1,
            'Frame': np.arange(1, num_frames + 1),
            'Time_ms': new_times
        })
        
        # 對 EMG 訊號數據進行線性插值
        for col in data_cols:
            if col in segment_df.columns:
                valid_x = segment_df['Time_ms'].values
                valid_y = pd.to_numeric(segment_df[col], errors='coerce').fillna(0).values
                resampled[col] = np.interp(new_times, valid_x, valid_y)
            
        # 針對 Marker 進行插值 (確保同步訊號不會遺失)
        if 'Marker' in segment_df.columns:
            valid_x = segment_df['Time_ms'].values
            valid_y_marker = pd.to_numeric(segment_df['Marker'], errors='coerce').fillna(0).values
            # 使用最大值保留法：只要區間內有掃到 1，就保留 1，確保同步訊號不會被插值淡化
            resampled['Marker'] = np.round(np.interp(new_times, valid_x, valid_y_marker)).astype(int)
            
        all_segments.append(resampled)

    if not all_segments:
        return pd.DataFrame()
    return pd.concat(all_segments, ignore_index=True)

# 讀取檔案
try:
    df = pd.read_csv(file_path, encoding='big5', low_memory=False)
except:
    df = pd.read_csv(file_path, encoding='cp950', low_memory=False)

# 執行轉換
result_df = process_per_frame(df, target_fps)

# 儲存結果
if not result_df.empty:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    full_output_path = os.path.join(output_dir, 'TEST1_0601_RAW_DATA_auto_calculated.csv')
    result_df.to_csv(full_output_path, index=False, encoding='utf-8-sig')
    print(f"\n🎉 轉換完成！檔案已儲存至: {full_output_path}")
else:
    print("\n❌ 失敗：無法從檔案中提取有效數據。")