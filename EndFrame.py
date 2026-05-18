import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_video_range(cap, start_frame, end_frame, roi):
    """專門用來分析特定影格區間內紅色像素數量的輔助函數"""
    x, y, w, h = roi
    red_pixel_counts = []
    frame_indices = []

    # 放寬條件：讓偏白、偏暗的紅色也能被偵測到
    lower_red1 = np.array([0, 70, 80])  
    upper_red1 = np.array([5, 170, 255])
    lower_red2 = np.array([160, 40, 40])
    upper_red2 = np.array([179, 255, 255])

    # 跳轉到區間起始幀
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        roi_frame = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        full_mask = cv2.bitwise_or(mask1, mask2)
        
        count = cv2.countNonZero(full_mask)
        red_pixel_counts.append(count)
        frame_indices.append(current_frame)
        
        current_frame += 1

    return np.array(frame_indices), np.array(red_pixel_counts)


def find_sync_intervals(video_path, start_window, end_window):
    # 1. 讀取影片與基本資訊
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("錯誤：無法開啟影片檔案")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"影片資訊：FPS = {fps:.2f}, 總幀數 = {total_frames}")

    # 計算兩個目標區間的起始與結束幀數
    start_min_f = int(start_window[0] * fps)
    start_max_f = min(int(start_window[1] * fps), total_frames)
    end_min_f = int(end_window[0] * fps)
    end_max_f = min(int(end_window[1] * fps), total_frames)

    print(f"🔎 尋找【開頭亮起】區間：第 {start_min_f} 幀 到 第 {start_max_f} 幀")
    print(f"🔎 尋找【結尾熄滅】區間：第 {end_min_f} 幀 到 第 {end_max_f} 幀")

    # ==========================================
    # 2. 分別框選兩個區間的 ROI
    # ==========================================
    # --- (A) 框選開頭 ROI ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_min_f)
    ret, frame_start = cap.read()
    if not ret:
        print("錯誤：無法讀取開頭起始幀")
        return

    print("\n(1/2) 請在視窗中框選【開頭區間】指示燈範圍...")
    roi_start = cv2.selectROI("Select START LED ROI", frame_start, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select START LED ROI")
    if roi_start == (0, 0, 0, 0):
        print("未框選開頭區域，程式結束。")
        return

    # --- (B) 框選結尾 ROI ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, end_min_f)
    ret, frame_end = cap.read()
    if not ret:
        print("錯誤：無法讀取結尾起始幀")
        return

    print("\n(2/2) 請在視窗中框選【結尾區間】指示燈範圍...")
    roi_end = cv2.selectROI("Select END LED ROI", frame_end, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select END LED ROI")
    if roi_end == (0, 0, 0, 0):
        print("未框選結尾區域，程式結束。")
        return

    # ==========================================
    # 3. 開始分析兩個區間
    # ==========================================
    print("\n開始分析【開頭區間】像素變化...")
    frames_start, counts_start = analyze_video_range(cap, start_min_f, start_max_f, roi_start)
    
    print("開始分析【結尾區間】像素變化...")
    frames_end, counts_end = analyze_video_range(cap, end_min_f, end_max_f, roi_end)
    
    cap.release()

    # ==========================================
    # 4. 偵測突變點
    # ==========================================
    # 🌟 找燈亮起 (開頭區間)：差值最大 (正數最大)
    diffs_start = np.diff(counts_start)
    jump_on_idx = np.argmax(diffs_start)
    absolute_sync_frame_on = frames_start[jump_on_idx + 1]

    # 🌟 找燈熄滅 (結尾區間)：差值最小 (負數最大)
    diffs_end = np.diff(counts_end)
    jump_off_idx = np.argmin(diffs_end)
    absolute_sync_frame_off = frames_end[jump_off_idx + 1]
    
    # 🌟 計算熄滅往前推 180 幀 (確保不小於 0)
    frame_minus_180 = max(0, absolute_sync_frame_off - 180)

    print("\n" + "=" * 55)
    print(f"✅ 成功找到同步點與目標幀！")
    print(f"🟢 [開頭錄製] 燈亮起：第 {absolute_sync_frame_on} 幀 ({absolute_sync_frame_on/fps:.3f} 秒)")
    print(f"🔴 [結束錄製] 燈熄滅：第 {absolute_sync_frame_off} 幀 ({absolute_sync_frame_off/fps:.3f} 秒)")
    print(f"🟡 [往前推算] 熄滅前180幀：第 {frame_minus_180} 幀 ({frame_minus_180/fps:.3f} 秒)")
    print("=" * 55 + "\n")

    # ==========================================
    # 5. 提取並繪製鄰近的幀畫面 (三排)
    # ==========================================
    print("正在擷取鄰近幀畫面以供確認...")
    cap = cv2.VideoCapture(video_path) 
    
    look_back = 5
    look_forward = 5
    
    fig_frames, axes = plt.subplots(3, look_back + look_forward + 1, figsize=(18, 9))
    fig_frames.suptitle("Verification: ON (Top), OFF (Middle), and OFF - 180 Frames (Bottom)", fontsize=16, fontweight='bold')

    # 🌟 重點修改：將每個事件對應到它專屬的 ROI
    sync_points = [
        (absolute_sync_frame_on, "ON", "green", roi_start),
        (absolute_sync_frame_off, "OFF", "red", roi_end),
        (frame_minus_180, "-180F", "orange", roi_end) # -180F 沿用結尾的 ROI
    ]

    for row_idx, (sync_f, label, color, current_roi) in enumerate(sync_points):
        x, y, w, h = current_roi
        extract_start = max(0, sync_f - look_back)
        cap.set(cv2.CAP_PROP_POS_FRAMES, extract_start)
        
        for i in range(look_back + look_forward + 1):
            curr_frame_num = extract_start + i
            ret, frame = cap.read()
            if not ret:
                break
            
            # 使用各自對應的 ROI 進行影像裁切
            roi_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            ax = axes[row_idx, i]
            ax.imshow(roi_rgb)
            ax.axis('off')
            
            if curr_frame_num == sync_f:
                ax.set_title(f"[{curr_frame_num}]\nTarget {label}", color=color, fontweight='bold')
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(4)
                ax.axis('on')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_title(f"{curr_frame_num}")

    cap.release()

    # ==========================================
    # 6. 繪製雙區間的折線圖表 (左右並排)
    # ==========================================
    fig_plot, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig_plot.suptitle("LED Brightness over Time (Split Windows)", fontsize=14)

    # 左圖：開頭亮起區間
    ax1.plot(frames_start, counts_start, color='blue', marker='o', markersize=3, label='Red Pixels')
    ax1.axvline(x=absolute_sync_frame_on, color='green', linestyle='--', linewidth=2, label=f'Sync ON ({absolute_sync_frame_on})')
    ax1.set_title("Start Window (Finding ON)")
    ax1.set_xlabel("Absolute Frame Number")
    ax1.set_ylabel("Number of Red Pixels")
    ax1.legend()
    ax1.grid(True)

    # 右圖：結尾熄滅區間
    ax2.plot(frames_end, counts_end, color='blue', marker='o', markersize=3, label='Red Pixels')
    ax2.axvline(x=absolute_sync_frame_off, color='red', linestyle='--', linewidth=2, label=f'Sync OFF ({absolute_sync_frame_off})')
    
    # 判斷 -180 幀是否落在右圖區間內，如果是的話，也畫上一條橘色虛線
    if end_min_f <= frame_minus_180 <= end_max_f:
        ax2.axvline(x=frame_minus_180, color='orange', linestyle='--', linewidth=2, label=f'-180F ({frame_minus_180})')
    
    ax2.set_title("End Window (Finding OFF)")
    ax2.set_xlabel("Absolute Frame Number")
    ax2.set_ylabel("Number of Red Pixels")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# ==========================================
# 執行區域
# ==========================================
if __name__ == "__main__":
    VIDEO_FILE = r"D:\project\NCU\114\Jumior\AI Project\Dataset\0513\opoencap\SHIH\OpenCapData_762f9790-9410-4f0c-827b-329f9a8f73ac_59df4766-83f3-465a-86fc-0d180698ee99\OpenCapData_762f9790-9410-4f0c-827b-329f9a8f73ac\Videos\Cam1\InputMedia\general41-50\general41-50_sync.mp4"
    # ⚠️ 設定你的「兩個」時間區間 (單位：秒)
    START_WINDOW = (3, 4)   # 尋找亮燈的區間
    END_WINDOW = (27.3, 27.5)    # 尋找熄燈的區間
    
    find_sync_intervals(VIDEO_FILE, START_WINDOW, END_WINDOW)