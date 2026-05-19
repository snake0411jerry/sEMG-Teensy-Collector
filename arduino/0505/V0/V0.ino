#include <SD.h>
#include <SPI.h>

// --- 系統參數設定 ---
const int TARGET_SAMPLING_RATE = 1000;   // 保持 1000 Hz 採樣 (每 1ms 記錄一次)
const int SAMPLE_INTERVAL_US = 1000000 / TARGET_SAMPLING_RATE; 
const int VIDEO_FPS = 60;                // 設定影片幀率，用來換算 Frame

// --- 硬體腳位設定 ---
const int NUM_CHANNELS = 3; 
const int emgPins[NUM_CHANNELS] = {14, 15, 16}; 
const int chipSelect = BUILTIN_SDCARD;      
const int BTN_PIN = 2;         

// --- 燈號腳位 ---
const int LED_RED_PIN = 6;     
const int LED_EX1_PIN = 8;     
const int LED_EX2_PIN = 10;    

enum SystemState {
    STATE_IDLE,          
    STATE_RECORDING      
};
volatile SystemState currentState = STATE_IDLE;

File dataFile;

// --- 雙緩衝區 ---
const int BUFFER_SIZE = 500; 
struct EmgRecord {
    unsigned long time_ms;     // 記錄毫秒 (ms)
    unsigned long frameNumber; // 記錄換算後的 60FPS 幀數
    int raw[NUM_CHANNELS];
    int marker;
};

EmgRecord bufferA[BUFFER_SIZE];
EmgRecord bufferB[BUFFER_SIZE];
EmgRecord* volatile writeBuffer = bufferA; 
EmgRecord* volatile readBuffer = bufferB;  
volatile int writeIndex = 0;               
volatile bool bufferReadyForSD = false;    

volatile unsigned long sessionStartTime = 0; 
volatile bool isFirstSample = false;

// 儲存最新的數值供 Serial Plotter 使用
volatile int latestRaw[NUM_CHANNELS] = {0, 0, 0}; 

IntervalTimer samplingTimer;

bool lastButtonState = HIGH; 
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50; 

// --- Serial 輸出控制 ---
unsigned long lastSerialPrintTime = 0;
const int SERIAL_PRINT_INTERVAL = 10; // 每 10ms 輸出一次到 Serial Plotter (約 100 FPS)

// ==========================================
// 硬體中斷常式 (ISR) - 每 1ms 觸發
// ==========================================
void samplingISR() {
    if (currentState == STATE_IDLE) return;
    
    if (currentState == STATE_RECORDING) {
        EmgRecord& rec = writeBuffer[writeIndex];
        
        // 1. 計算經過的毫秒數
        unsigned long elapsed_ms = millis() - sessionStartTime;
        rec.time_ms = elapsed_ms;
        
        // 2. 換算成 60 FPS 的幀數 (每 1/60 秒 +1 Frame)，並強制從 1 開始
        rec.frameNumber = ((elapsed_ms * VIDEO_FPS) / 1000) + 1;
        
        // 讀取並儲存 RAW 值，同時更新給 Serial 顯示的變數
        for(int ch = 0; ch < NUM_CHANNELS; ch++) {
            int val = analogRead(emgPins[ch]);
            rec.raw[ch] = val;
            latestRaw[ch] = val; 
        }

        // 處理 Marker
        if (isFirstSample) {
            rec.marker = 1; 
            isFirstSample = false;
        } else {
            rec.marker = 0; 
        }

        writeIndex++;

        // 緩衝區滿交換
        if (writeIndex >= BUFFER_SIZE) {
            EmgRecord* temp = writeBuffer;
            writeBuffer = readBuffer;
            readBuffer = temp;
            writeIndex = 0;
            bufferReadyForSD = true; 
        }
    }
}

void errorMode() {
    while(1) {
        digitalWrite(LED_RED_PIN, !digitalRead(LED_RED_PIN));
        digitalWrite(LED_EX1_PIN, !digitalRead(LED_EX1_PIN));
        digitalWrite(LED_EX2_PIN, !digitalRead(LED_EX2_PIN));
        delay(100);
    }
}

// ==========================================
// 主程式 Setup
// ==========================================
void setup() {
    Serial.begin(115200);
    
    pinMode(BTN_PIN, INPUT_PULLUP); 
    pinMode(LED_RED_PIN, OUTPUT);
    pinMode(LED_EX1_PIN, OUTPUT);
    pinMode(LED_EX2_PIN, OUTPUT);

    // 初始熄燈
    digitalWrite(LED_RED_PIN, LOW);
    digitalWrite(LED_EX1_PIN, LOW);
    digitalWrite(LED_EX2_PIN, LOW);

    analogReadResolution(10);       
    analogReadAveraging(4);        

    if (!SD.begin(chipSelect)) errorMode(); 

    // 啟動定時器：每 1ms 觸發
    samplingTimer.begin(samplingISR, SAMPLE_INTERVAL_US); 
    
    Serial.print("Sampling started at Hz: ");
    Serial.println(TARGET_SAMPLING_RATE);
    Serial.println("Ready. Waiting for button press...");
}

// ==========================================
// 主程式 Loop
// ==========================================
void loop() {
    unsigned long currentMillis = millis();

    // ---------------------------------------------------------
    // 處理 Serial Plotter 顯示 (非阻塞設計，避免影響 1ms 中斷)
    // ---------------------------------------------------------
    if (currentMillis - lastSerialPrintTime >= SERIAL_PRINT_INTERVAL) {
        lastSerialPrintTime = currentMillis;

        if (currentState == STATE_IDLE) {
            // 待機狀態時，直接讀取腳位數值方便除錯與貼電極片
            Serial.print(analogRead(emgPins[0])); Serial.print(",");
            Serial.print(analogRead(emgPins[1])); Serial.print(",");
            Serial.println(analogRead(emgPins[2]));
        } else {
            // 錄製狀態時，拿取 ISR 內存好的最新數值，避免搶佔 ADC 資源
            Serial.print(latestRaw[0]); Serial.print(",");
            Serial.print(latestRaw[1]); Serial.print(",");
            Serial.println(latestRaw[2]);
        }
    }

    // ---------------------------------------------------------
    // 處理按鈕邏輯
    // ---------------------------------------------------------
    bool reading = digitalRead(BTN_PIN);
    if (reading != lastButtonState) lastDebounceTime = currentMillis;
    
    if ((currentMillis - lastDebounceTime) > debounceDelay) {
        static bool buttonPressed = false;
        if (reading == LOW && !buttonPressed) {
            buttonPressed = true; 
            
            if (currentState == STATE_IDLE) {
                // --- 開始錄製 ---
                digitalWrite(LED_RED_PIN, HIGH); 
                digitalWrite(LED_EX1_PIN, HIGH); 
                digitalWrite(LED_EX2_PIN, HIGH); 

                dataFile = SD.open("TEST1_MAX_RAW_DATA.csv", FILE_WRITE);
                if (dataFile) {
                    if (dataFile.size() == 0) {
                        // Header 更新
                        dataFile.println("Time_ms,Frame,Raw0,Raw1,Raw2,Marker");
                    }
                    writeIndex = 0;
                    bufferReadyForSD = false;
                    isFirstSample = true;
                    
                    // --- 按下按鈕的當下，記錄基準毫秒數 ---
                    sessionStartTime = millis(); 
                    
                    currentState = STATE_RECORDING; 
                    Serial.println("Recording Started!"); // 提示文字
                } else {
                    errorMode();
                }
            } 
            else if (currentState == STATE_RECORDING) {
                // --- 停止錄製 ---
                currentState = STATE_IDLE; 
                digitalWrite(LED_RED_PIN, LOW); 
                digitalWrite(LED_EX1_PIN, LOW); 
                digitalWrite(LED_EX2_PIN, LOW); 

                if (dataFile) {
                    // 寫入剩餘資料
                    for (int i = 0; i < writeIndex; i++) {
                        EmgRecord& rec = writeBuffer[i];
                        if (i == writeIndex - 1) rec.marker = 2; 
                        
                        dataFile.print(rec.time_ms); dataFile.print(",");
                        dataFile.print(rec.frameNumber); dataFile.print(",");
                        dataFile.print(rec.raw[0]); dataFile.print(",");
                        dataFile.print(rec.raw[1]); dataFile.print(",");
                        dataFile.print(rec.raw[2]); dataFile.print(",");
                        dataFile.println(rec.marker);
                    }
                    dataFile.close(); 
                    Serial.println("Stopped. Raw data saved.");
                }
            }
        } else if (reading == HIGH) {
            buttonPressed = false; 
        }
    }
    lastButtonState = reading;

    // ---------------------------------------------------------
    // 處理 SD 卡寫入
    // ---------------------------------------------------------
    if (currentState == STATE_RECORDING && bufferReadyForSD && dataFile) {
        for (int i = 0; i < BUFFER_SIZE; i++) {
            EmgRecord& rec = readBuffer[i];
            
            dataFile.print(rec.time_ms); dataFile.print(",");
            dataFile.print(rec.frameNumber); dataFile.print(",");
            dataFile.print(rec.raw[0]); dataFile.print(",");
            dataFile.print(rec.raw[1]); dataFile.print(",");
            dataFile.print(rec.raw[2]); dataFile.print(",");
            dataFile.println(rec.marker);
        }
        dataFile.flush(); 
        bufferReadyForSD = false; 
    }
}