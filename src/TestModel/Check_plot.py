import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 設定檔案搜尋路徑與輸出路徑
# ==========================================
# 這是您上一階段輸出 _Predicted.csv 的資料夾
data_dir = r"D:\project\NCU\114\Jumior\AI Project\Dataset\Code\0519Combined\Predict"
search_pattern = os.path.join(data_dir, "*_Predicted.csv")
file_list = glob.glob(search_pattern)

# 自動建立一個新資料夾來存放這 20 張圖表
output_dir = os.path.join(data_dir, "Prediction_Plots")
os.makedirs(output_dir, exist_ok=True)

if not file_list:
    print("❌ 找不到任何 _Predicted.csv 檔案，請確認路徑！")
else:
    print(f"🔍 找到 {len(file_list)} 個預測檔案，準備開始繪製圖表...\n")

# ==========================================
# 2. 迴圈處理每個檔案並繪圖
# ==========================================
for file_path in file_list:
    filename = os.path.basename(file_path)
    print(f"📊 正在繪製圖表: {filename}")
    
    # 讀取包含預測結果的 CSV
    df = pd.read_csv(file_path)
    
    # 防呆機制：檢查必要的欄位是否存在
    required_cols = ['EMG_Main_MVC', 'Predicted_EMG_Main', 'EMG_Compass_MVC', 'Predicted_EMG_Compass']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"⚠️ {filename} 缺少必要欄位 {missing}，跳過此檔案。")
        continue
        
    # 建立圖表 (上下兩張子圖，尺寸 15x8)
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # ------------------------------------------
    # 上半部：繪製 Main 肌肉
    # ------------------------------------------
    axes[0].plot(df['EMG_Main_MVC'], label='True EMG Main (Real)', color='blue', alpha=0.7, linewidth=2)
    axes[0].plot(df['Predicted_EMG_Main'], label='Predicted EMG Main (AI)', color='red', linestyle='--', linewidth=2)
    axes[0].set_title(f'EMG Main Comparison: {filename.replace(".csv", "")}', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Muscle Activation (%MVC)', fontsize=12)
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # ------------------------------------------
    # 下半部：繪製 Compass 肌肉
    # ------------------------------------------
    axes[1].plot(df['EMG_Compass_MVC'], label='True EMG Compass (Real)', color='green', alpha=0.7, linewidth=2)
    axes[1].plot(df['Predicted_EMG_Compass'], label='Predicted EMG Compass (AI)', color='orange', linestyle='--', linewidth=2)
    axes[1].set_title(f'EMG Compass Comparison: {filename.replace(".csv", "")}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time (Frames)', fontsize=12)
    axes[1].set_ylabel('Muscle Activation (%MVC)', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # 自動調整排版，避免字體重疊
    plt.tight_layout()
    
    # ==========================================
    # 3. 儲存圖片並關閉畫布釋放記憶體
    # ==========================================
    # 將圖片名稱設定為原本的檔名加上 .png
    save_path = os.path.join(output_dir, filename.replace('.csv', '.png'))
    plt.savefig(save_path, dpi=150) # dpi=150 確保圖片有夠高的解析度
    plt.close() # 務必要 close，不然畫 20 張圖記憶體會爆掉

print("-" * 50)
print(f"🎉 恭喜！所有圖表皆已繪製完成！")
print(f"📂 請前往此資料夾查看您的圖表：{output_dir}")