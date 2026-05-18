import pandas as pd
import numpy as np
import os

# --- 設定參數 ---
target_fps = 60  # 請根據您的影片設定 FPS (例如 30 或 60)
file_path = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0513\emg\KUAN_RAW_DATA.csv"
output_dir = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0513\emg"
# ----------------

def process_per_frame(df, fps):
    # 尋找每組數據的起始索引 (這份資料的 Time_ms 是從 1 開始)
    start_indices = df[df['Time_ms'] == 1].index.tolist()
    start_indices.append(len(df)) # 加入結尾索引
    
    all_segments = []
    
    # 定義需要轉換的數據欄位 (符合 CSV 內的欄位名稱)
    data_cols = ['Raw0', 'Raw1', 'Raw2']
    
    for i in range(len(start_indices) - 1):
        start_idx = start_indices[i]
        end_idx = start_indices[i+1]
        
        # 提取該組數據
        segment_df = df.iloc[start_idx:end_idx].copy()
        
        # --- 自動計算該組的時長 (ms) ---
        max_ms = segment_df['Time_ms'].max()
        
        if max_ms <= 0 or pd.isna(max_ms):
            print(f"組別 {i+1} 時長異常，跳過處理。")
            continue
            
        # --- 根據時長與 FPS 自動計算總偵數 ---
        num_frames = int(round((max_ms / 1000.0) * fps))
        
        if num_frames == 0:
            num_frames = 1 # 至少要有 1 偵
            
        print(f"組別 {i+1}: 偵測到總時長 {max_ms} ms -> 自動計算為 {num_frames} 偵 (FPS: {fps})")

        # 建立新的偵時間軸 (從 1 到 max_ms，平均分配給 num_frames 個點)
        new_times = np.linspace(1, max_ms, num_frames)
        
        # 建立存放結果的 DataFrame
        resampled = pd.DataFrame({
            'Segment': i + 1,
            'Frame': np.arange(1, num_frames + 1),
            'Time_ms': new_times
        })
        
        # 對訊號數據進行線性插值
        for col in data_cols:
            if col in segment_df.columns:
                valid_x = segment_df['Time_ms'].values
                valid_y = pd.to_numeric(segment_df[col], errors='coerce').fillna(0).values
                resampled[col] = np.interp(new_times, valid_x, valid_y)
            
        # 針對 Marker 進行插值並四捨五入成整數 (保持 0/1 狀態)
        if 'Marker' in segment_df.columns:
            valid_x = segment_df['Time_ms'].values
            valid_y_marker = pd.to_numeric(segment_df['Marker'], errors='coerce').fillna(0).values
            resampled['Marker'] = np.round(np.interp(new_times, valid_x, valid_y_marker)).astype(int)
            
        all_segments.append(resampled)

    if not all_segments:
        return pd.DataFrame()
    return pd.concat(all_segments, ignore_index=True)

# 讀取檔案
try:
    # 使用 low_memory=False 避免欄位型態警告
    df = pd.read_csv(file_path, encoding='big5', low_memory=False)
except:
    df = pd.read_csv(file_path, encoding='cp950', low_memory=False)

# 執行自動計算與轉換
result_df = process_per_frame(df, target_fps)

# 儲存結果
if not result_df.empty:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    full_output_path = os.path.join(output_dir, 'KUAN_RAW_DATA_auto_calculated.csv')
    result_df.to_csv(full_output_path, index=False, encoding='utf-8-sig')
    print(f"\n🎉 轉換完成！檔案已儲存至: {full_output_path}")
else:
    print("\n❌ 失敗：無法從檔案中提取有效數據。")