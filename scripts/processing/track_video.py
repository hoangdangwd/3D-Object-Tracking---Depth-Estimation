import os
import torch
import argparse
import numpy as np
import cv2  # Thêm thư viện OpenCV

from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


# Hàm tính khoảng cách Euclidean giữa hai điểm (x1, y1) và (x2, y2)
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# Hàm lấy frame rate (fps) và kích thước của video
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)  # Mở video
    if not cap.isOpened():
        print("Error: Couldn't open video.")
        return None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS)  # Lấy fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Lấy chiều rộng
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Lấy chiều cao
    cap.release()
    return fps, width, height


# Hàm lấy frame rate của video
def get_frame_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open video.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="./assets/VideoGR1.2.2.mp4", help="path to a video")
    parser.add_argument("--checkpoint", default=None, help="CoTracker model parameters")
    parser.add_argument("--backward_tracking", action="store_true", help="Compute tracks in both directions")
    parser.add_argument("--use_v2_model", action="store_true", help="Use CoTracker2 instead of CoTracker++")
    parser.add_argument("--offline", action="store_true", help="Use the offline model")

    args = parser.parse_args()

    # Lấy fps và kích thước video
    fps, width, height = get_video_info(args.video_path)
    if fps is not None:
        print(f"Frame rate of the original video: {fps:.2f} fps")
        print(f"Video resolution: {width}x{height} (Width x Height)")
        print(f"Total pixels per frame: {width * height} pixels")

    # Load video
    video = read_video_from_path(args.video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(DEFAULT_DEVICE)

    # Load model
    if args.checkpoint is not None:
        if args.use_v2_model:
            model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=True)
        else:
            window_len = 60 if args.offline else 16
            model = CoTrackerPredictor(
                checkpoint=args.checkpoint,
                v2=False,
                offline=args.offline,
                window_len=window_len,
            )
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

    model = model.to(DEFAULT_DEVICE)

    # Define custom query points: [frame_index, x, y]
    queries = torch.tensor([
        [0., 100., 800.],
        [0., 50., 750.],
        [0., 150., 650.],
        # Thay đổi tọa độ tại đây nếu cần
    ], device=video.device)[None]

    # Run tracking with queries
    pred_tracks, pred_visibility = model(
        video,
        queries=queries,
        backward_tracking=args.backward_tracking,
    )
    print("Tracking completed")

    # Lấy các tọa độ theo dõi (x, y) từ các track points
    track_points = pred_tracks[0].cpu().numpy()  # [frames, queries, 2]

    # In ra cấu trúc của track_points để kiểm tra
    print("Track points structure:", track_points.shape)

    # Tính quãng đường di chuyển và vận tốc
    total_distance = 0.0
    prev_position = None
    for i, point in enumerate(track_points):
        if len(point[0]) == 2:
            current_position = (point[0][0], point[0][1])
            if prev_position is not None:
                distance = calculate_distance(prev_position, current_position)
                total_distance += distance

                # Sử dụng FPS thực từ video
                time_per_frame = 1 / fps if fps and fps > 0 else 1 / 30  # fallback 30fps
                speed = distance / time_per_frame

                print(f"Frame {i}: Position: {current_position}, Speed: {speed:.2f} pixels/s")

            prev_position = current_position
        else:
            print(f"Frame {i} has insufficient data: {point}")

    print(f"Total distance traveled: {total_distance:.2f} pixels")

    # Visualize and save
    vis = Visualizer(
        save_dir="../../data/output/videos",
        linewidth=6,
        mode="cool",
        tracks_leave_trace=-1
    )
    vis.visualize(
        video=video,
        tracks=pred_tracks,
        visibility=pred_visibility,
        filename="query_point",
    )

    last_frame_index = track_points.shape[0] - 1
    last_frame_points = track_points[last_frame_index]

    print(f"\n📍 Tọa độ các điểm track ở frame cuối ({last_frame_index}):")
    for idx, (x, y) in enumerate(last_frame_points):
        print(f"• Query {idx + 1}: ({x:.2f}, {y:.2f}) pixel")
