// 定義連接腳位
const int emgPin = A0; 

// 數值校準變數 [cite: 138, 140, 144]
int max_analog_dta = 500;   // 預設最大值
int min_analog_dta = 100;   // 預設最小值
int static_analog_dta = 0;  // 靜態基準值（肌肉放鬆時）

// 取得平滑後的類比數值 
int getAnalog(int pin) {
    long sum = 0;
    for(int i=0; i<32; i++) {
        sum += analogRead(pin);
    }
    int dta = sum >> 5; // 取 32 次採樣的平均值
    
    // 動態更新最大與最小值，以適應不同受試者的發力程度 [cite: 138, 140]
    if(dta > max_analog_dta) max_analog_dta = dta;
    if(dta < min_analog_dta) min_analog_dta = dta;
    
    return dta;
}

void setup() {
    Serial.begin(115200);
    while (!Serial); // 等待 Teensy 序列埠連線
    
    Serial.println("--- 系統初始化：請保持肌肉放鬆 5 秒 ---");
    
    long sum = 0;
    // 模擬原範例的倒數計時初始化邏輯 [cite: 151, 152]
    for(int i=0; i<1000; i++) {
        sum += getAnalog(emgPin);
        if(i % 200 == 0) Serial.print("."); 
        delay(2);
    }
    
    static_analog_dta = sum / 1000;
    Serial.println("\n初始化完成！");
    Serial.print("靜態基準值: "); Serial.println(static_analog_dta);
}

void loop() {
    int val = getAnalog(emgPin);
    
    // 計算發力百分比 (Intensity %)
    float intensity = 0;
    
    // 參考範例邏輯：以靜態基準值為中心計算 
    if(val > static_analog_dta) {
        intensity = map(val, static_analog_dta, max_analog_dta, 0, 100);
    } else {
        intensity = 0; // 低於基準值視為完全放鬆
    }

    // 限制在 0-100% 之間
    intensity = constrain(intensity, 0, 100);

    // 格式化輸出，方便 Serial Plotter 繪圖
    // 格式：標籤1:數值1,標籤2:數值2 (中間用空格或逗號隔開)
    Serial.print("Raw:");       Serial.print(val);
    Serial.print(" ");
    Serial.print("Static:");    Serial.print(static_analog_dta);
    Serial.print(" ");
    Serial.print("Intensity%:"); Serial.println(intensity);

    delay(10); // 100Hz 輸出頻率，適合觀察肌肉收縮曲線 [cite: 151]
}