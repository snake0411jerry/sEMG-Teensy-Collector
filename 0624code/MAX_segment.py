import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ==========================================
# 參數設定區
# ==========================================
# 🚨 設定剛剛輸出 Segment 檔案的資料夾路徑
DATA_DIR = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0622\Clean\Segmented\KUAN_0622_CLEAN_SEGMENTS"
# 尋找所有符合命名規則的 Segment 檔案
FILE_PATTERN = os.path.join(DATA_DIR, "KUAN_0622_CLEAN_Segment_*.csv")

# ==========================================
# 主程式：批次計算與視覺化
# ==========================================
def batch_calculate_mvc():
    # 取得所有 Segment 檔案的列表
    segment_files = glob.glob(FILE_PATTERN)
    
    if not segment_files:
        print(f"❌ 在 {DATA_DIR} 中找不到任何 Segment 檔案，請確認路徑與檔名！")
        return

    # 確保檔案按照 Segment 編號排序 (避免 1, 10, 2 這種字串排序問題)
    segment_files.sort(key=lambda x: int(x.split('_Segment_')[-1].replace('.csv', '')))

    print(f"🔍 總共找到 {len(segment_files)} 個 Segment 檔案，準備開始分析...\n")

    for file_path in segment_files:
        segment_num = file_path.split('_Segment_')[-1].replace('.csv', '')
        print(f"📂 正在分析: 組別 {segment_num} ({os.path.basename(file_path)})")
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
        except Exception as e:
            print(f"讀取檔案失敗 {file_path}: {e}")
            continue

        # 建立時間軸 (使用檔案內的 Time_ms)
        time_sec = df['Time_ms'].values / 1000.0

        # --- 處理主動肌 (Raw0) ---
        # 注意：這邊的 Raw0 已經是上一步處理好的「包絡線 (Envelope)」了
        raw0_env = pd.to_numeric(df['Raw0'], errors='coerce').fillna(0).values
        max_mvc_main = np.max(raw0_env)

        # --- 處理協同肌 (Raw1) ---
        raw1_env = pd.to_numeric(df['Raw1'], errors='coerce').fillna(0).values
        max_mvc_comp = np.max(raw1_env)

        # --- 輸出結果 ---
        print("-" * 40)
        print(f"  [組別 {segment_num}] MVC 計算結果")
        print(f"▶ 主動肌 (Raw0) 最大包絡線數值: {max_mvc_main:.2f}")
        print(f"▶ 協同肌 (Raw1) 最大包絡線數值: {max_mvc_comp:.2f}")
        print("-" * 40 + "\n")

        # --- 畫圖確認訊號品質 ---
        plt.figure(figsize=(14, 8))
        
        # 畫主動肌
        plt.subplot(2, 1, 1)
        plt.plot(time_sec, raw0_env, color='red', linewidth=2, label='Smoothed Envelope (60 FPS)')
        plt.axhline(y=max_mvc_main, color='green', linestyle='--', linewidth=2, label=f'Max MVC: {max_mvc_main:.1f}')
        plt.title(f'Segment {segment_num}: Main Muscle (Raw0) - MVC Test', fontsize=14, fontweight='bold')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        # 畫協同肌
        plt.subplot(2, 1, 2)
        plt.plot(time_sec, raw1_env, color='orange', linewidth=2, label='Smoothed Envelope (60 FPS)')
        plt.axhline(y=max_mvc_comp, color='green', linestyle='--', linewidth=2, label=f'Max MVC: {max_mvc_comp:.1f}')
        plt.title(f'Segment {segment_num}: Compass Muscle (Raw1) - MVC Test', fontsize=14, fontweight='bold')
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        
        # 存檔圖片，方便後續檢視，不阻擋程式運行
        img_out_path = file_path.replace('.csv', '_MVC_Plot.png')
        plt.savefig(img_out_path)
        plt.close() # 關閉畫布釋放記憶體
        print(f"📸 圖片已儲存至: {img_out_path}\n")

    print("🎉 所有 Segment 檔案分析完畢！")

if __name__ == '__main__':
    batch_calculate_mvc()