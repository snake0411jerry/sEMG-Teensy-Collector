import serial
import time

# ==========================================
# 設定區
# ==========================================
COM_PORT = 'COM3'            # ⚠️ 請改成你 Teensy 對應的 COM Port
BAUD_RATE = 115200           # 必須與 Arduino 程式碼設定的鮑率一致
OUTPUT_FILE = 'SHIH_MIN_Exported.csv' # 你想存放在電腦裡的檔名

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
            
            # 為了讓你知道有在運作，將收到的原始訊息印在終端機
            print(f"[收到] {line}")

            # 💡 聰明過濾邏輯：
            # 當看到 Header (Time_ms開頭) 時，代表真正的資料要開始了
            if line.startswith("Time_ms"):
                is_recording = True
                f.write(line + "\n")
                print("--> 🎯 偵測到 CSV 標題，開始正式寫入檔案！")
                continue
            
            # 當看到檔案結尾的 "=" 分隔線，代表資料傳完了
            if is_recording and ("===" in line or "檔案讀取完畢" in line):
                print("--> 🛑 偵測到結束標記，停止錄製！")
                break
            
            # 如果處於錄製狀態，就把這一行無條件寫進 CSV
            if is_recording:
                f.write(line + "\n")

except KeyboardInterrupt:
    print("\n使用者手動中斷程式 (Ctrl+C)。")
finally:
    # 確保關閉序列埠與檔案
    ser.close()
    print(f"\n✅ 序列埠已關閉。資料已成功完整儲存至電腦：{OUTPUT_FILE}")