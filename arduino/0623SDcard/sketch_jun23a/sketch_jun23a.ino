#include <SD.h>
#include <MTP_Teensy.h>

// --- 硬體腳位設定 (保留你原本的燈號腳位) ---
const int chipSelect = BUILTIN_SDCARD;      
const int LED_RED_PIN = 6;     
const int LED_EX1_PIN = 8;     
const int LED_EX2_PIN = 10;    

// ==========================================
// 錯誤閃燈模式 (SD卡沒插好時觸發)
// ==========================================
void errorMode() {
    while(1) {
        digitalWrite(LED_RED_PIN, !digitalRead(LED_RED_PIN));
        delay(100);
    }
}

// ==========================================
// 主程式 Setup
// ==========================================
void setup() {
    Serial.begin(115200);
    
    // 初始化燈號
    pinMode(LED_RED_PIN, OUTPUT);
    pinMode(LED_EX1_PIN, OUTPUT);
    pinMode(LED_EX2_PIN, OUTPUT);
    digitalWrite(LED_RED_PIN, LOW);
    digitalWrite(LED_EX1_PIN, LOW);
    digitalWrite(LED_EX2_PIN, LOW);

    // 給 Serial Monitor 一點時間連線 (最多等 3 秒)
    while (!Serial && millis() < 3000);

    Serial.println("====================================");
    Serial.println("  Teensy 4.1 純讀卡機模式啟動中...  ");
    Serial.println("====================================");

    // 1. 初始化 SD 卡
    if (!SD.begin(chipSelect)) {
        Serial.println("❌ SD 卡初始化失敗！請確認卡片是否插妥。");
        errorMode(); 
    }
    Serial.println("✅ SD 卡掛載成功！");
    
    // 2. 啟動 MTP 功能
    MTP.begin();
    MTP.addFilesystem(SD, "Teensy_EMG_Data"); // 這裡是你會在電腦檔案總管看到的磁碟名稱
    
    Serial.println("✅ MTP 服務已啟動！");
    Serial.println("👉 現在你可以打開電腦的「檔案總管」，直接存取 CSV 檔案了。");

    // 點亮其中一顆燈，表示系統正常運作中 (可依喜好更改)
    digitalWrite(LED_EX1_PIN, HIGH); 
}

// ==========================================
// 主程式 Loop
// ==========================================
void loop() {
    // 全力處理電腦端傳來的讀取/複製請求
    MTP.loop();
}