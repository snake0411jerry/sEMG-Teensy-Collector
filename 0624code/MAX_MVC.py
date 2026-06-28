import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

# ==========================================
# 參數設定區
# ==========================================
# ✅ 請確認這裡是您最原始的 1000Hz Raw Data 檔案路徑！
MVC_FILE_PATH = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0625\Raw\KUAN_0622_Raw_DATA.csv"
FS = 1000.0  # 硬體原始採樣率

# ==========================================
# 核心濾波函式 (已修復 SOS 浮點數精度問題)
# ==========================================
def apply_full_emg_pipeline(data, fs):
    """標準 EMG 處理流水線：去直流 -> 60Hz 陷波 -> 20Hz 高通 -> 全波整流 -> 5Hz 低通包絡線"""
    # 1. 去直流偏移
    signal_centered = data - np.mean(data)
    
    # 2. 60Hz 陷波濾波器
    b_notch, a_notch = iirnotch(60.0, 30.0, fs)
    signal_notched = filtfilt(b_notch, a_notch, signal_centered)
    
    # 3. 20Hz 高通濾波器
    nyq = 0.5 * fs
    sos_high = butter(4, 20.0 / nyq, btype='high', output='sos')
    signal_high = sosfiltfilt(sos_high, signal_notched)
    
    # 4. 全波整流
    signal_rectified = np.abs(signal_high)
    
    # 5. 5Hz 低通濾波器
    sos_low = butter(4, 5.0 / nyq, btype='low', output='sos')
    signal_envelope = sosfiltfilt(sos_low, signal_rectified)
    
    return signal_high, signal_envelope

# ==========================================
# 主程式：動態切割與視覺化
# ==========================================
def calculate_max_mvc_per_segment():
    print(f"📂 讀取 MVC 原始測試檔案中: {MVC_FILE_PATH}")
    try:
        df = pd.read_csv(MVC_FILE_PATH, encoding='big5', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(MVC_FILE_PATH, encoding='cp950', low_memory=False)
    except FileNotFoundError:
        print(f"❌ 找不到檔案：{MVC_FILE_PATH}，請確認路徑是否正確！")
        return

    # 💡 動態切割邏輯：尋找硬體重新計時的斷點 (Time_ms == 1)
    start_indices = df[df['Time_ms'] == 1].index.tolist()
    
    # 如果找不到 Time_ms == 1，代表可能是連續沒中斷的檔案，就把整份當作一組
    if not start_indices:
        start_indices = [0]
        
    start_indices.append(len(df)) 

    global_max_main = 0
    global_max_comp = 0
    best_seg_main = 0
    best_seg_comp = 0

    print("\n" + "="*40)
    print(" 📊 動態切割與各組別 MVC 運算報告")
    print("="*40)

    # 逐組運算
    for i in range(len(start_indices) - 1):
        seg_num = i + 1
        start_idx = start_indices[i]
        end_idx = start_indices[i+1]
        
        # 切割出該組別的資料
        seg_df = df.iloc[start_idx:end_idx].copy()
        
        # 如果該組資料太短 (例如雜訊斷點)，則跳過
        if len(seg_df) < FS: # 至少要有一秒的資料
            continue
            
        time_sec = seg_df['Time_ms'].values / 1000.0

        # --- 處理主動肌 (Raw0) ---
        raw0 = pd.to_numeric(seg_df['Raw0'], errors='coerce').fillna(0).values
        raw0_clean, raw0_env = apply_full_emg_pipeline(raw0, FS)
        seg_max_main = np.max(raw0_env)

        # --- 處理協同肌 (Raw1) ---
        raw1 = pd.to_numeric(seg_df['Raw1'], errors='coerce').fillna(0).values
        raw1_clean, raw1_env = apply_full_emg_pipeline(raw1, FS)
        seg_max_comp = np.max(raw1_env)

        # 更新全域最大值
        if seg_max_main > global_max_main:
            global_max_main = seg_max_main
            best_seg_main = seg_num
            
        if seg_max_comp > global_max_comp:
            global_max_comp = seg_max_comp
            best_seg_comp = seg_num

        max_ms = seg_df['Time_ms'].max()
        print(f"🎬 組別 {seg_num:02d} (時長 {max_ms/1000.0:.1f}s) | 主動肌 MVC: {seg_max_main:6.1f} | 協同肌 MVC: {seg_max_comp:6.1f}")

        # --- 繪製該組的圖表 ---
        plt.figure(figsize=(12, 6))
        
        # 主動肌子圖
        plt.subplot(2, 1, 1)
        plt.plot(time_sec, raw0_clean, color='blue', alpha=0.3, label='Filtered AC Signal (Raw0)')
        plt.plot(time_sec, raw0_env, color='red', linewidth=2, label='Smoothed Envelope')
        plt.axhline(y=seg_max_main, color='green', linestyle='--', linewidth=2, label=f'Max MVC: {seg_max_main:.1f}')
        plt.title(f'Segment {seg_num}: Main Muscle (Raw0)', fontsize=12, fontweight='bold')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        # 協同肌子圖
        plt.subplot(2, 1, 2)
        plt.plot(time_sec, raw1_clean, color='blue', alpha=0.3, label='Filtered AC Signal (Raw1)')
        plt.plot(time_sec, raw1_env, color='orange', linewidth=2, label='Smoothed Envelope')
        plt.axhline(y=seg_max_comp, color='green', linestyle='--', linewidth=2, label=f'Max MVC: {seg_max_comp:.1f}')
        plt.title(f'Segment {seg_num}: Compass Muscle (Raw1)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show(block=False) 

    # --- 輸出總結結果 ---
    print("\n" + "🔥"*20)
    print(" 🏆 最終最大自主收縮 (Global MVC) 結論")
    print("🔥"*20)
    print(f"▶ 主動肌 (Raw0) 總體最大數值: {global_max_main:.2f} (來自組別 {best_seg_main})")
    print(f"▶ 協同肌 (Raw1) 總體最大數值: {global_max_comp:.2f} (來自組別 {best_seg_comp})")
    print("-" * 40)
    print("💡 請將以上兩個數值填入您的 `emg_skelton_combined.py` 腳本中：")
    print(f"   MVC_MAIN_CLEAN_MAX = {global_max_main:.1f}")
    print(f"   MVC_COMP_CLEAN_MAX = {global_max_comp:.1f}")
    print("=" * 40 + "\n")
    
    # 讓程式暫停，直到使用者手動關閉所有圖表視窗
    print("👀 所有組別的波形圖已開啟，關閉所有圖表視窗即可結束程式...")
    plt.show()

if __name__ == '__main__':
    calculate_max_mvc_per_segment()