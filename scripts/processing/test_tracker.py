import os
import torch
import argparse
import numpy as np
import math
import cv2

from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return width

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="./assets/VideoGR1.3.mp4", help="Path to video")
    parser.add_argument("--checkpoint", default=None, help="CoTracker model checkpoint")
    parser.add_argument("--backward_tracking", action="store_true", help="Track both directions")
    parser.add_argument("--use_v2_model", action="store_true", help="Use CoTracker2")
    parser.add_argument("--offline", action="store_true", help="Use offline model")
    args = parser.parse_args()

    # Lấy chiều ngang video để tính toán FOV
    frame_width_pixels = get_video_resolution(args.video_path)

    # Đọc video vào tensor
    video = read_video_from_path(args.video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(DEFAULT_DEVICE)

    # Load mô hình CoTracker
    try:
        if args.checkpoint is not None:
            if args.use_v2_model:
                model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=True)
            else:
                window_len = 60 if args.offline else 16
                model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=False, offline=args.offline, window_len=window_len)
        else:
            model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        exit()

    model = model.to(DEFAULT_DEVICE)

    # Điểm truy vấn [frame_idx, x, y]
    queries = torch.tensor([
        [0., 10., 1250.],
    ], device=video.device)[None]

    # Tracking
    pred_tracks, pred_visibility = model(video, queries=queries, backward_tracking=args.backward_tracking)
    print("Tracking completed")

    track_points = pred_tracks[0].cpu().numpy()

    # === Thông số camera iPhone 12 Pro Max - camera trước ===
    Z = 3.0  # Khoảng cách từ camera đến người (m)
    FOV_deg = 74  # FOV ngang của camera trước
    FOV_rad = math.radians(FOV_deg)
    meters_per_pixel = (2 * Z * math.tan(FOV_rad / 2)) / frame_width_pixels
    print(f"Meters per pixel: {meters_per_pixel:.6f} m/pixel")

    # Visualize và lưu video
    vis = Visualizer(save_dir="../../data/output/videos", linewidth=6, mode="cool", tracks_leave_trace=-1)
    vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename="query_point")

    # Đọc lại video đầu ra để lấy FPS thực tế
    output_video_path = os.path.join("../../data/output/videos", "query_point.mp4")
    cap = cv2.VideoCapture(output_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open rendered video: {output_video_path}")
    output_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"\n[INFO] FPS của video đã render: {output_fps:.2f}")

    # Tính vận tốc theo video đã render
    total_distance = 0.0
    prev_position = None

    for i, point in enumerate(track_points):
        if len(point[0]) == 2:
            current_position = (point[0][0], point[0][1])
            if prev_position is not None:
                distance = calculate_distance(prev_position, current_position)
                total_distance += distance

                time_per_frame = 1 / output_fps
                speed_pixel = distance / time_per_frame
                speed_mps = speed_pixel * meters_per_pixel

                print(f"[OUTPUT VIDEO] Frame {i}: Position: {current_position}, Speed: {speed_mps:.3f} m/s")
            prev_position = current_position
        else:
            print(f"[OUTPUT VIDEO] Frame {i} thiếu dữ liệu")

    total_time = len(track_points) / output_fps
    average_speed_mps = (total_distance * meters_per_pixel) / total_time
    print(f"[OUTPUT VIDEO] Tổng quãng đường: {total_distance:.2f} pixels")
    print(f"[OUTPUT VIDEO] Vận tốc trung bình: {average_speed_mps:.3f} m/s")
