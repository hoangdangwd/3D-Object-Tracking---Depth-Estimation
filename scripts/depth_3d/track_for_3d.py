import os
import torch
import argparse
import numpy as np
import cv2
import csv  # Ghi file CSV

from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open video.")
        return None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, width, height

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="../../assets/apple.mp4", help="path to a video")
    parser.add_argument("--checkpoint", default=None, help="CoTracker model parameters")
    parser.add_argument("--backward_tracking", action="store_true", help="Compute tracks in both directions")
    parser.add_argument("--use_v2_model", action="store_true", help="Use CoTracker2 instead of CoTracker++")
    parser.add_argument("--offline", action="store_true", help="Use the offline model")

    args = parser.parse_args()

    fps, width, height = get_video_info(args.video_path)
    if fps is not None:
        print(f"Frame rate: {fps:.2f} fps")
        print(f"Resolution: {width}x{height}")
        print(f"Pixels per frame: {width * height}")

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

    # Define query points: [frame_index, x, y]
    queries = torch.tensor([
        [0., 100., 800.],
        [0., 80., 750.],
        [0., 150., 650.],
        [0., 180., 750.],
        [0., 100., 700.],
        [0., 100., 600.],
        [0., 120., 580.],
        [0., 140., 670.],
    ], device=video.device)[None]

    # Run CoTracker
    pred_tracks, pred_visibility = model(
        video,
        queries=queries,
        backward_tracking=args.backward_tracking,
    )
    print("✅ Tracking completed.")

    track_points = pred_tracks[0].cpu().numpy()  # [frames, queries, 2]
    print("Track points shape:", track_points.shape)

    # ====== 📌 In tất cả tọa độ ======
    print("\n📍 Toạ độ từng điểm ở từng frame:")
    for frame_idx, frame_points in enumerate(track_points):
        print(f"\n🟦 Frame {frame_idx}:")
        for query_idx, (x, y) in enumerate(frame_points):
            print(f"  • Query {query_idx + 1}: ({x:.2f}, {y:.2f})")

    # ====== 💾 Ghi toàn bộ tọa độ vào CSV ======
    output_csv_path = "tracked_points.csv"
    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Query Index", "X (pixel)", "Y (pixel)"])
        for frame_idx, frame_points in enumerate(track_points):
            for query_idx, (x, y) in enumerate(frame_points):
                writer.writerow([frame_idx, query_idx, x, y])

    print(f"\n✅ Đã ghi vào file CSV: {output_csv_path}")

    # ====== Tính tổng quãng đường của điểm đầu tiên (tùy chọn) ======
    total_distance = 0.0
    prev_position = None
    for i, point in enumerate(track_points):
        current_position = (point[0][0], point[0][1])
        if prev_position is not None:
            distance = calculate_distance(prev_position, current_position)
            total_distance += distance
            time_per_frame = 1 / fps
            speed = distance / time_per_frame
            print(f"Frame {i}: Position: {current_position}, Speed: {speed:.2f} pixels/s")
        prev_position = current_position
    print(f"\n📏 Tổng quãng đường của điểm đầu tiên: {total_distance:.2f} pixels")

    # ====== Visualization ======
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

    # ====== In ra frame cuối ======
    last_frame_index = track_points.shape[0] - 1
    last_frame_points = track_points[last_frame_index]
    print(f"\n🎯 Tọa độ các điểm ở frame cuối ({last_frame_index}):")
    for idx, (x, y) in enumerate(last_frame_points):
        print(f"  • Query {idx + 1}: ({x:.2f}, {y:.2f})")
