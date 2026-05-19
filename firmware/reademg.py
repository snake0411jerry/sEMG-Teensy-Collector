import serial
import time

# ==========================================
# 設定區
# ==========================================
COM_PORT = 'COM3'            # ⚠️ 請改成你 Teensy 對應的 COM Port
BAUD_RATE = 115200           # 必須與 Arduino 程式碼設定的鮑率一致
OUTPUT_FILE = 'SHIH_MIN_Exported.csv' # 你想存存放電腦裡的檔名

# 🎯 定義要統計的肌電訊號門檻值清單
THRESHOLDS = [500, 600, 700, 800, 900]

print(f"嘗試連接至 {COM_PORT} ...")

try:
    # 開啟序列埠
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
    print("連線成功！正在等待 Teensy 傳送資料...\n")
except Exception as e:
    print(f"無法開啟 {COM_PORT}，請檢查：")
    print("1. 傳輸線有沒有接好")
    print("2. COM Port 號碼是否正確")
    print("3. ⚠️ Arduino IDE 的 Serial Monitor 必須關閉 (否則會搶佔通訊埠)！")
    exit()

is_recording = False
headers = []

# 用來動態記錄每個通道符合門檻值次數的字典
threshold_counts = {}
# 🎯 新增：用來記錄每個通道的「最大值」字典
max_values = {}

try:
    # 建立並開啟要寫入的 CSV 檔案
    with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as f:
        print(f"檔案 {OUTPUT_FILE} 已建立，等待數據中...")
        
        while True:
            # 讀取序列埠的一行文字，並進行解碼與去空白
            raw_data = ser.readline()
            if not raw_data:
                continue
                
            line = raw_data.decode('utf-8', errors='ignore').strip()
            
            # 將收到的原始訊息印在終端機
            print(f"[收到] {line}")

            # 💡 聰明過濾邏輯：
            # 當看到 Header (Time_ms開頭) 時，代表真正的資料要開始了
            if line.startswith("Time_ms"):
                is_recording = True
                f.write(line + "\n")
                print("--> 🎯 偵測到 CSV 標題，開始正式寫入檔案！")
                
                # 解析欄位名稱，用來動態初始化統計字典
                headers = [h.strip() for h in line.split(',')]
                # 排除第一欄 Time_ms，其餘為肌電訊號通道（如 Ch1, Ch2...）
                for channel in headers[1:]:
                    threshold_counts[channel] = {th: 0 for th in THRESHOLDS}
                    # 🎯 新增：初始化各通道的最大值為負無窮大
                    max_values[channel] = float('-inf') 
                continue
            
            # 當看到檔案結尾的 "=" 分隔線，代表資料傳完了
            if is_recording and ("===" in line or "檔案讀取完畢" in line):
                print("--> 🛑 偵測到結束標記，停止錄製！")
                break
            
            # 如果處於錄製狀態，就把這一行無條件寫進 CSV
            if is_recording:
                f.write(line + "\n")
                
                # 即時門檻值與最大值統計邏輯
                try:
                    data_values = [v.strip() for v in line.split(',')]
                    
                    # 💡 防錯機制：確保這行收到的資料欄位數量與 Header 完全一致才進行處理
                    if len(data_values) == len(headers):
                        # 從索引 1 開始（跳過第一欄 Time_ms）
                        for idx, val_str in enumerate(data_values[1:], start=1):
                            channel_name = headers[idx]
                            val = float(val_str)  # 轉換為數值以利大小比較
                            
                            # 🎯 新增：檢查並更新該通道的最大值
                            if val > max_values[channel_name]:
                                max_values[channel_name] = val
                            
                            # 檢查數值是否大於各個設定的門檻值
                            for th in THRESHOLDS:
                                if val > th:
                                    threshold_counts[channel_name][th] += 1
                                    
                except ValueError:
                    # 捕捉因硬體通訊突發干擾產生的斷字或非數字雜訊，避免程式中斷
                    print(f"⚠️ 收到不完整或異常資料，跳過此行統計：{line}")
                    continue

except KeyboardInterrupt:
    print("\n使用者手動中斷程式 (Ctrl+C)。")
finally:
    # 確保關閉序列埠與檔案
    ser.close()
    print(f"\n✅ 序列埠已關閉。資料已成功完整儲存至電腦：{OUTPUT_FILE}")
    
    # ==========================================
    # 🎯 錄製結束或中斷時，在終端機列印精美的統計報告 (包含最大值)
    # ==========================================
    if threshold_counts:
        print("\n" + "="*80)
        print("📊 各通道 sEMG 訊號強度門檻值與「最大值」統計報告")
        print("="*80)
        
        # 列印表頭
        header_str = f"{'通道名稱':<15}"
        for th in THRESHOLDS:
            header_str += f"{f'>{th}':>10}"
        header_str += f"{'Max(最大值)':>14}"  # 🎯 新增最大值標題
        print(header_str)
        print("-" * (15 + 10 * len(THRESHOLDS) + 14))
        
        # 依序列印每個通道的累計數據與最大值
        for channel in threshold_counts:
            row_str = f"{channel:<15}"
            for th in THRESHOLDS:
                count = threshold_counts[channel][th]
                row_str += f"{count:>10}"
                
            # 🎯 擷取並格式化最大值 (若無資料則顯示 N/A)
            max_val = max_values[channel]
            if max_val != float('-inf'):
                row_str += f"{max_val:>14.2f}" 
            else:
                row_str += f"{'N/A':>14}"
                
            print(row_str)
        print("="*80)