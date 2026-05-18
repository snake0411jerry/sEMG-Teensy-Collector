import cv2
import numpy as np

# ==========================================
# 🔴 請在這裡設定你想測試的 HSV 閾值
# ==========================================
LOWER_RED1 = np.array([0, 70, 80])  
UPPER_RED1 = np.array([5, 170, 255])
LOWER_RED2 = np.array([160, 60, 60])
UPPER_RED2 = np.array([179, 255, 255])


# ==========================================
# 滑鼠點擊回呼函數 (查詢 HSV)
# ==========================================
def mouse_callback(event, x, y, flags, param):
    hsv_frame, scale, original_w = param

    if event == cv2.EVENT_LBUTTONDOWN:
        # 判斷點擊的是左半邊(原始圖)還是右半邊(遮罩圖)
        magnified_w = original_w * scale
        
        if x < magnified_w:
            # 點擊左半邊
            original_roi_x = int(x / scale)
            clicked_side = "左側 (原始圖)"
        else:
            # 點擊右半邊
            original_roi_x = int((x - magnified_w) / scale)
            clicked_side = "右側 (遮罩圖)"

        original_roi_y = int(y / scale)

        # 確保座標在範圍內
        if 0 <= original_roi_x < hsv_frame.shape[1] and 0 <= original_roi_y < hsv_frame.shape[0]:
            h, s, v = hsv_frame[original_roi_y, original_roi_x]
            print("-" * 30)
            print(f"👉 點擊區域: {clicked_side}")
            print(f"✅ 精準 HSV 數值: [H:{h}, S:{s}, V:{v}]")
            print("-" * 30)

# ==========================================
# 主程式
# ==========================================
def test_hsv_thresholds(video_path, frame_index, magnification=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片 {video_path}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("錯誤：無法讀取該幀。")
        return

    print(f"\n[步驟 1] 顯示影片第 {frame_index} 幀。")
    print("請框選包含 LED 的區域，框好後按 'SPACE' 或 'ENTER'。")
    
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    if roi == (0, 0, 0, 0):
        print("未框選，結束程式。")
        return

    x_start, y_start, w, h = roi
    roi_frame = frame[y_start:y_start+h, x_start:x_start+w]
    hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    # 1. 產生遮罩 (Mask)
    mask1 = cv2.inRange(hsv_roi, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv_roi, LOWER_RED2, UPPER_RED2)
    full_mask = cv2.bitwise_or(mask1, mask2)

    # 2. 計算紅色像素數量
    red_pixel_count = cv2.countNonZero(full_mask)
    print(f"\n====================================")
    print(f"🔥 在此 ROI 中找到的紅色像素總數： {red_pixel_count} 像素")
    print(f"====================================\n")

    # 3. 放大影像 (使用 INTER_NEAREST 保持像素顆粒感)
    new_w, new_h = w * magnification, h * magnification
    mag_roi = cv2.resize(roi_frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    mag_mask = cv2.resize(full_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 將單通道的 Mask 轉為三通道 BGR，以便與原始影像並排
    mag_mask_bgr = cv2.cvtColor(mag_mask, cv2.COLOR_GRAY2BGR)

    # 4. 加上文字標示
    cv2.putText(mag_roi, "Original ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(mag_mask_bgr, f"Red Mask: {red_pixel_count} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 5. 並排顯示 (左邊原圖，右邊遮罩)
    combined_view = np.hstack((mag_roi, mag_mask_bgr))

    window_name = "HSV Threshold Debugger (Click to see HSV)"
    cv2.namedWindow(window_name)
    
    # 綁定滑鼠事件
    cv2.setMouseCallback(window_name, mouse_callback, (hsv_roi, magnification, w))

    print("[步驟 2] 已顯示對照圖。")
    print(" - 左圖：原始放大")
    print(" - 右圖：符合閾值的紅色像素 (白色部分)")
    print("👉 你可以點擊左圖中「沒被抓到的紅色像素」，查看終端機顯示的 HSV 值。")
    print("👉 按下鍵盤任意鍵關閉視窗。")

    cv2.imshow(window_name, combined_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==========================================
# 執行區域
# ==========================================
if __name__ == "__main__":
    # 替換成你的影片路徑
    VIDEO_PATH = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0513\opoencap\SHIH\OpenCapData_762f9790-9410-4f0c-827b-329f9a8f73ac_59df4766-83f3-465a-86fc-0d180698ee99\OpenCapData_762f9790-9410-4f0c-827b-329f9a8f73ac\Videos\Cam1\InputMedia\general31-40\general31-40_sync.mp4"
    # ⚠️ 設定你的「兩個」時間區間 (單位：秒)
    # 輸入你想測試的幀數 (例如你圖表中有燈光亮起的那一幀，或是燈光熄滅前的那一幀)
    FRAME_TO_INSPECT = 1855

    
    # 放大倍率 (預設 15倍，格子會很大很好點)
    MAGNIFICATION = 15

    test_hsv_thresholds(VIDEO_PATH, FRAME_TO_INSPECT, MAGNIFICATION)