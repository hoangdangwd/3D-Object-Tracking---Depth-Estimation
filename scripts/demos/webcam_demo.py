import cv2
import torch
import numpy as np
import threading
import time
from cotracker.predictor import CoTrackerOnlinePredictor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")
model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
print("Model loaded.")

# Các biến dùng chung giữa các thread
selected_point = None
point_selected = False
trajectory = []
query_point_tensor = None
is_first_step = True
frame_lock = threading.Lock()
latest_frame = None
running = True
step = 8
FRAME_WIDTH, FRAME_HEIGHT = 320, 240

# === Hàm xử lý model (chạy ở thread phụ) ===
def tracking_loop():
    global is_first_step, trajectory, query_point_tensor
    window_frames = []

    while running:
        time.sleep(0.03)  # tránh dùng 100% CPU
        if not point_selected:
            continue

        with frame_lock:
            if latest_frame is None:
                continue
            frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
            window_frames.append(frame_rgb)

        if len(window_frames) >= step * 2:
            video_chunk_np = np.stack(window_frames[-step * 2:])
            video_tensor = (
                torch.tensor(video_chunk_np).float().permute(0, 3, 1, 2)[None].to(device)
            )

            with torch.no_grad():
                pred_tracks, pred_visibility = model(
                    video_tensor,
                    queries=query_point_tensor if is_first_step else None,
                    is_first_step=is_first_step,
                )
            is_first_step = False

            if pred_tracks is not None:
                last_point = pred_tracks[0, -1, 0].detach().cpu().numpy()
                x, y = int(last_point[0]), int(last_point[1])
                trajectory.append((x, y))
                # Giới hạn chiều dài trajectory nếu muốn mượt hơn
                if len(trajectory) > 100:
                    trajectory.pop(0)

# === Callback chọn điểm ===
def mouse_callback(event, x, y, flags, param):
    global selected_point, point_selected, query_point_tensor, trajectory, is_first_step
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = (x, y)
        point_selected = True
        query_point_tensor = torch.tensor([[0., x, y]], device=device)[None]
        trajectory.clear()
        is_first_step = True
        print(f"🖱️ Chọn điểm: ({x}, {y})")

# === Giao diện chính ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

cv2.namedWindow("CoTracker Realtime")
cv2.setMouseCallback("CoTracker Realtime", mouse_callback)

# Khởi động thread theo dõi
thread = threading.Thread(target=tracking_loop)
thread.start()

print("📌 Click để chọn điểm. Nhấn R để reset. Q để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    with frame_lock:
        latest_frame = frame.copy()

    # Vẽ đường đi
    if trajectory:
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)
        cv2.circle(frame, trajectory[-1], 5, (0, 0, 255), -1)

    # Vẽ điểm đang chọn
    if selected_point and not point_selected:
        cv2.circle(frame, selected_point, 5, (255, 0, 0), -1)

    cv2.imshow("CoTracker Realtime", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("👋 Thoát...")
        running = False
        break
    elif key == ord("r"):
        print("🔄 Reset điểm theo dõi.")
        selected_point = None
        point_selected = False
        query_point_tensor = None
        trajectory.clear()
        is_first_step = True

cap.release()
cv2.destroyAllWindows()
thread.join()
