#include <SD.h>
#include <SPI.h>

// Teensy 4.1 專用：指定使用內建的 MicroSD 卡槽
const int chipSelect = BUILTIN_SDCARD;
const char* filename = "test_log.csv";

void setup() {
    Serial.begin(115200);
    
    // 等待序列埠連線 (最多等 3 秒，避免沒接電腦時卡死)
    while (!Serial && millis() < 3000); 

    Serial.println("=== Teensy 4.1 SD 卡寫入測試 ===");
    Serial.print("初始化 SD 卡... ");

    // 檢查 SD 卡是否正常讀取
    if (!SD.begin(chipSelect)) {
        Serial.println("失敗！請檢查 SD 卡是否插好或格式是否為 FAT32/exFAT。");
        while (true); // 發生錯誤，程式停在這裡
    }
    Serial.println("成功！");

    // 嘗試開啟檔案，如果檔案是空的，就先寫入 CSV 標題列
    File dataFile = SD.open(filename, FILE_WRITE);
    if (dataFile) {
        if (dataFile.size() == 0) {
            dataFile.println("Timestamp(ms),Message");
            Serial.println("-> 已建立新的 CSV 檔案並寫入標題列。");
        }
        dataFile.close(); // 寫完標題先關閉
    } else {
        Serial.println("錯誤：無法開啟檔案！");
    }

    Serial.println("\n✅ 系統就緒！");
    Serial.println("👉 請在上方的輸入框打字，按下 Enter 後會自動存入 SD 卡。");
    Serial.println("--------------------------------------------------");
}

void loop() {
    // 檢查序列埠是否收到電腦傳來的資料
    if (Serial.available() > 0) {
        
        // 讀取整串文字，直到遇到「換行符號」為止
        String inputStr = Serial.readStringUntil('\n');
        
        // 去除字串前後多餘的空白或隱藏字元 (例如 \r)
        inputStr.trim(); 

        // 確保不是空字串才執行寫入
        if (inputStr.length() > 0) {
            unsigned long currentTime = millis();

            // 開啟檔案準備寫入
            File dataFile = SD.open(filename, FILE_WRITE);

            if (dataFile) {
                // 寫入時間戳記
                dataFile.print(currentTime);
                dataFile.print(",");
                // 寫入你輸入的文字，並換行
                dataFile.println(inputStr);
                
                // ⚠️ 非常重要：寫入完畢立刻關閉檔案，確保資料存進記憶體！
                dataFile.close(); 

                // 回報給電腦螢幕看，確認有執行成功
                Serial.print("[成功存檔] 時間: ");
                Serial.print(currentTime);
                Serial.print(" ms | 內容: ");
                Serial.println(inputStr);
            } else {
                Serial.println("❌ 寫入失敗：無法開啟檔案！");
            }
        }
    }
}