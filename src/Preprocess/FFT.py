import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 參數設定 ---
FILE_PATH = r"E:\SHIH_MIN_0615_RAW_DATA_Filtered.csv"
FS = 1000                             
CHANNEL = 'Raw1'                      

# --- SNR 計算區間設定 (單位：秒) ---
# 請根據你實際錄製的動作時間進行修改
REST_WINDOW = (20, 21)   # 擷取 1~3 秒作為「完全放鬆」的基線雜訊區間
ACTIVE_WINDOW = (24,25) # 擷取 19~21 秒作為「肌肉發力」的訊號區間

def calculate_rms(signal_array):
    """計算均方根值 (Root Mean Square)"""
    # 加上微小常數避免全為 0 時發生 log(0) 錯誤
    if len(signal_array) == 0: return 1e-6 
    return np.sqrt(np.mean(signal_array**2))

def analyze_emg_with_snr(file_path, channel, fs):
    # 1. 讀取 CSV 檔案
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"找不到檔案：{file_path}")
        return

    time_ms = df['Time_ms'].values
    time_sec = time_ms / 1000.0  
    raw_signal = df[channel].values

    # 2. 消除直流偏移 (DC Offset) - 這是計算正確 RMS 的關鍵！
    signal_centered = raw_signal - np.mean(raw_signal)

    # 3. 擷取特定區間的訊號並計算 RMS
    # 找出時間落在我們設定區間內的資料索引
    rest_mask = (time_sec >= REST_WINDOW[0]) & (time_sec <= REST_WINDOW[1])
    active_mask = (time_sec >= ACTIVE_WINDOW[0]) & (time_sec <= ACTIVE_WINDOW[1])

    rest_signal = signal_centered[rest_mask]
    active_signal = signal_centered[active_mask]

    rms_rest = calculate_rms(rest_signal)
    rms_active = calculate_rms(active_signal)

    # 計算 SNR (dB)
    snr_db = 20 * np.log10(rms_active / rms_rest)

    print(f"[{channel} 訊號品質報告]")
    print(f"放鬆區間 RMS: {rms_rest:.2f} (ADC 單位)")
    print(f"發力區間 RMS: {rms_active:.2f} (ADC 單位)")
    print(f"SNR (訊噪比): {snr_db:.2f} dB")

    # 4. 計算 FFT
    n_active = len(active_signal)
    freqs = np.fft.rfftfreq(n_active, d=1/FS)
    fft_magnitude = np.abs(np.fft.rfft(active_signal)) / n_active
    fft_magnitude[1:] = fft_magnitude[1:] * 2 

    # 5. 繪圖顯示
    plt.figure(figsize=(12, 8))

    # --- 上方子圖：時間域與 SNR 區間標示 ---
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(time_sec, raw_signal, color='blue', linewidth=0.5)
    ax1.axhline(y=np.mean(raw_signal), color='red', linestyle='--', label='Baseline')
    
    # 用綠色背景標示放鬆區間
    ax1.axvspan(REST_WINDOW[0], REST_WINDOW[1], color='green', alpha=0.2, label='Resting Window (Noise)')
    # 用橘色背景標示發力區間
    ax1.axvspan(ACTIVE_WINDOW[0], ACTIVE_WINDOW[1], color='orange', alpha=0.2, label='Active Window (Signal)')
    
    # 在圖表上顯示 SNR 數值
    ax1.text(0.02, 0.9, f"SNR: {snr_db:.1f} dB", transform=ax1.transAxes, 
             fontsize=14, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    ax1.set_title(f'EMG Time Domain Signal ({channel})')
    ax1.set_xlabel('Time (Seconds)')
    ax1.set_ylabel('ADC Value (0-1023)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- 下方子圖：頻率域 (FFT) ---
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(freqs, fft_magnitude, color='purple', linewidth=1)
    ax2.set_title('EMG Frequency Spectrum (FFT)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_xlim(0, 500) 
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    analyze_emg_with_snr(FILE_PATH, CHANNEL, FS)