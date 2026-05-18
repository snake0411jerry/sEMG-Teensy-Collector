// 定義 2 個通道的腳位
const int NUM_CHANNELS = 2;
const int emgPins[NUM_CHANNELS] = {A0, A1}; 

// 校準變數陣列
int max_analog_dta[NUM_CHANNELS] = {500, 500};  
int min_analog_dta[NUM_CHANNELS] = {100, 100};  
int static_analog_dta[NUM_CHANNELS] = {0, 0}; 

// 取得特定通道平滑數值
int getAnalog(int ch) {
    long sum = 0;
    for(int i = 0; i < 32; i++) {
        sum += analogRead(emgPins[ch]);
    }
    int dta = sum >> 5; 
    
    if(dta > max_analog_dta[ch]) max_analog_dta[ch] = dta;
    if(dta < min_analog_dta[ch]) min_analog_dta[ch] = dta;
    
    return dta;
}

void setup() {
    Serial.begin(115200);
    while (!Serial); 
    
    Serial.println("--- 兩通道初始化：請保持放鬆 ---");
    
    long sum[NUM_CHANNELS] = {0, 0};
    for(int i = 0; i < 1000; i++) {
        for(int ch = 0; ch < NUM_CHANNELS; ch++) {
            sum[ch] += getAnalog(ch);
        }
        if(i % 200 == 0) Serial.print("."); 
        delay(2);
    }
    
    Serial.println("\n各通道基準值:");
    for(int ch = 0; ch < NUM_CHANNELS; ch++) {
        static_analog_dta[ch] = sum[ch] / 1000;
        Serial.print("CH"); Serial.print(ch + 1); Serial.print(": "); 
        Serial.println(static_analog_dta[ch]);
    }
}

void loop() {
    float intensity[NUM_CHANNELS];

    for(int ch = 0; ch < NUM_CHANNELS; ch++) {
        int val = getAnalog(ch);
        if(val > static_analog_dta[ch]) {
            intensity[ch] = map(val, static_analog_dta[ch], max_analog_dta[ch], 0, 100);
        } else {
            intensity[ch] = 0; 
        }
        intensity[ch] = constrain(intensity[ch], 0, 100);
    }

    // 輸出繪圖標籤
// --- 進階版：同時支援繪圖與 CSV 錄製 ---
    Serial.print("Time_us:");     Serial.print(micros());    Serial.print(",");
    Serial.print("Biceps%:");     Serial.print(intensity[0]); Serial.print(",");
    Serial.print("Trapezius%:");  Serial.println(intensity[1]);
    delay(10); 
}