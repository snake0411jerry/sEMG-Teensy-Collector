import cv2
import os

def extract_video_segment(input_video_path, output_video_path, start_frame, end_frame):
    """
    從指定的影片中擷取特定區間的幀，並輸出為新影片。
    
    :param input_video_path: 來源影片的完整路徑
    :param output_video_path: 輸出影片的完整路徑 (建議附檔名為 .mp4)
    :param start_frame: 起始幀 (整數)
    :param end_frame: 結束幀 (整數)
    """
    # 檢查參數合理性
    if start_frame < 0 or end_frame <= start_frame:
        print("❌ 錯誤：起始幀必須大於等於 0，且結束幀必須大於起始幀！")
        return

    print(f"🎬 開始讀取來源影片: {input_video_path}")
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("❌ 錯誤：無法開啟來源影片，請檢查路徑是否正確。")
        return

    # 取得原始影片的屬性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📊 影片資訊 - FPS: {fps:.2f}, 解析度: {width}x{height}, 總幀數: {total_frames}")

    # 確保結束幀不超過影片總長度
    if end_frame > total_frames:
        print(f"⚠️ 警告：指定的結束幀 ({end_frame}) 超過影片總長度 ({total_frames})，將自動調整至最後一幀。")
        end_frame = total_frames

    # 設定影片編碼器 (四字碼)，這裡使用 mp4v 來輸出 .mp4 檔案
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 建立 VideoWriter 物件
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 🔥 關鍵：讓 OpenCV 直接跳轉到指定的起始幀，省去前面空跑的時間
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    print(f"✂️ 正在擷取影片，從第 {start_frame} 幀 到第 {end_frame} 幀...")

    # 開始逐幀讀取並寫入
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 無法讀取幀或已提早到達影片結尾。")
            break
        
        # 將讀取到的幀寫入新影片
        out.write(frame)
        
        # 每處理 50 幀顯示一次進度，避免畫面洗版
        if current_frame % 50 == 0 or current_frame == end_frame:
            progress = ((current_frame - start_frame) / (end_frame - start_frame + 1e-5)) * 100
            print(f"⏳ 進度: {current_frame}/{end_frame} 幀 ({progress:.1f}%)")

        current_frame += 1

    # 釋放資源
    cap.release()
    out.release()
    print(f"✅ 裁切完成！檔案已儲存至: {output_video_path}")

# ==========================================
# 測試與使用區
# ==========================================
if __name__ == "__main__":
    # 請替換為您實際的路徑與參數
    SOURCE_VIDEO = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0513\opoencap\KUAN\1-20\OpenCapData_c9bccd16-66a2-4039-a5f1-a133ff09e091\Videos\Cam0\InputMedia\general11-20\general11-20_sync.mp4"
    
    # 輸出檔案建議使用 .mp4 格式
    OUTPUT_VIDEO = r"D:\project\NCU\114\Jumior\AI Project\Dataset\Code\Test_video\KUAN_TEST_0513_Segment11-20.mp4"
    
    # 設定您想要擷取的偵數範圍 (例如擷取第 150 幀到第 400 幀)
    START_FRAME = 210
    END_FRAME = 2223

    # 執行函數
    extract_video_segment(SOURCE_VIDEO, OUTPUT_VIDEO, START_FRAME, END_FRAME)