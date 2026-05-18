# sEMG-Teensy-Collector

## 專案簡介
本專案為 6 通道表面肌電圖 (sEMG) 訊號擷取與分析系統。專案整合了 Teensy 4.1 進行硬體多通道訊號擷取、Python 訊號前處理，以及深度學習模型進行肌肉活動狀態預測。此外，專案也包含將肌電訊號與骨架數據（MediaPipe）進行同步整合的模組，以利後續的連動測試。

## 🗂️ 專案目錄結構

```text

sEMG-Teensy-Collector/ 
├── .vscode/                 # VS Code 編輯器工作區設定檔 
├── 0513Combined/            # 整合測試資料庫或特定進度備份 (2026/05/13) 
├── Model/                   # 存放訓練好的 AI 模型權重檔 (如 .h5) 與資料標準化工具 (如 .pkl) 
├── arduino/                 # Teensy 4.1 韌體程式碼 (.ino)，包含歷次版本的硬體訊號擷取程式 
├── firmware/
│   └── reademg.py           # 透過 Serial 讀取 Teensy 原始 EMG 訊號的 Python 通訊腳本 
└── src/                     # Python 核心分析與模型原始碼 
    ├── Preprocess/          # 資料前處理模組 
    │   ├── CheckFrame.py           # 檢查/校正擷取影片中的特定幀 
    │   ├── FindFrame.py            # 尋找與定位影片、骨架開始結束的特徵幀 
    │   ├── emg_preprocess.py       # 肌電訊號ms to frame、增加segment 
    │   └── emg_skelton_combined.py # EMG 訊號與骨架節點數據同步整合腳本 
    ├── TestModel/           # 模型測試與推論模組 
    │   └── predict_emg.py          # 載入預訓練模型並進行即時/離線動作預測 
    └── TrainingModel/       # 模型訓練模組 
        └── TrainModel.py           # 定義神經網路架構與執行 EMG 模型訓練

```

## 💻 硬體與腳位設定 (Hardware Setup)
* **微控制器:** Teensy 4.1
* **感測器:** 8 x Grove EMG Sensor
* **通訊協定:** Serial 通訊 (Baud rate 請參照 Arduino 程式碼設定)
* **腳位與資料格式:** 類比訊號腳位設定請參閱 `arduino/` 目錄下的最新程式碼（註：部分訊號使用 Pin 14, 15 等腳位接收，空間座標與張量轉換順序為 `[raw_z, raw_y, raw_x]`，請確保硬體接線與程式定義一致）。

## 🚀 快速開始 (Quick Start)

### 1. 取得專案程式碼
請團隊成員先將專案 Clone 至本地端：
\`\`\`bash
git clone https://github.com/snake0411jerry/sEMG-Teensy-Collector.git
cd sEMG-Teensy-Collector
\`\`\`

### 2. 硬體端 (Teensy 4.1)
使用 Arduino IDE 開啟 `arduino/` 底下最新日期的 `.ino` 檔案，編譯並燒錄至 Teensy 4.1 開發板。

### 3. 軟體端 (Python)
1. 建議使用 conda 或 venv 建立獨立的虛擬環境。
2. 執行 `firmware/reademg.py` 測試電腦是否能順利接收來自 Teensy 的 8 通道訊號。
3. 若需進行模型推論，請執行 `src/TestModel/predict_emg.py`，確保 `Model/` 資料夾內有對應的模型權重檔。

### 使用說明
1. 先使用Opencap錄製影片並生成dataset
2. 下載過後使用 src/Preprocess/FindFrame.py 確認開始與結束偵，若無法確認準確偵可使用 src/Preprocess/CheckFrame.py 確認單獨偵
3. 將opencap dataset 中的 Markerdata 中的trc檔用excel開啟並另存為csv檔
4. 將 Markerdata中的csv 開始偵數從1改為前面所偵測的開始偵與將結束偵後的偵數刪除
5. 使用 src/Preprocess/emg_preprocess.py，將emg訊號切分為不同組次以利之後與Markerdate對齊
6. 使用 src/Preprocess/emg_skelton_combined.py 將Markerdata與emg結合，計算節點速度V與加速度a並輸出訓練模型檔案
7. 使用src/TrainingModel/TrainModel.py 訓練模型

## 👥 開發團隊 (Contributors)
* 林寬泓
* 林士閔
* 吳羽絜
