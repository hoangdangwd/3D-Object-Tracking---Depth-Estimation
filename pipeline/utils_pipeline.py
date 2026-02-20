#!/usr/bin/env python3
"""
Utility functions cho CoTracker Pipeline
Tái sử dụng các hàm chung
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path


def validate_file(filepath, error_msg=None):
    """Kiểm tra file tồn tại"""
    if not os.path.exists(filepath):
        msg = error_msg or f"❌ Không tìm thấy file: {filepath}"
        raise FileNotFoundError(msg)
    return True


def validate_video(video_path):
    """Kiểm tra video có thể mở được"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"❌ Không thể mở video: {video_path}")
    cap.release()
    return True


def get_video_info(video_path):
    """Lấy thông tin video (FPS, width, height, frame_count)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    
    cap.release()
    return info


def compute_rigid_transform(A, B):
    """
    Tính rigid transform (R, t) từ A sang B
    A, B: [N, 3] numpy arrays
    Returns: R (3x3), t (3,)
    """
    assert A.shape == B.shape
    
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    
    AA = A - centroid_A
    BB = B - centroid_B
    
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    
    t = centroid_B - R @ centroid_A
    
    return R, t


def pixel_to_3d(u, v, depth, K):
    """
    Chuyển đổi từ tọa độ pixel sang 3D
    
    Args:
        u, v: tọa độ pixel
        depth: giá trị depth (Z)
        K: ma trận intrinsic (3x3)
    
    Returns:
        (X, Y, Z) trong hệ tọa độ camera
    """
    K_inv = np.linalg.inv(K)
    pixel_homog = np.array([u, v, 1])
    XYZ = depth * (K_inv @ pixel_homog)
    return XYZ


def create_camera_matrix(fx, fy, cx, cy):
    """Tạo ma trận intrinsic camera"""
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K


def calculate_distance_2d(p1, p2):
    """Tính khoảng cách Euclidean 2D"""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_distance_3d(p1, p2):
    """Tính khoảng cách Euclidean 3D"""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_velocity(distance, fps):
    """
    Tính vận tốc
    
    Args:
        distance: khoảng cách di chuyển (pixel hoặc mét)
        fps: frame rate
    
    Returns:
        velocity (pixel/s hoặc m/s)
    """
    time_per_frame = 1.0 / fps if fps > 0 else 1.0 / 30.0
    return distance / time_per_frame


def ensure_dir(directory):
    """Tạo thư mục nếu chưa tồn tại"""
    os.makedirs(directory, exist_ok=True)
    return directory


def get_device():
    """Lấy device tốt nhất có sẵn"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def save_trajectory_plot(trajectory, output_path):
    """
    Lưu plot quỹ đạo 2D
    
    Args:
        trajectory: list of (x, y) tuples
        output_path: đường dẫn lưu ảnh
    """
    import matplotlib.pyplot as plt
    
    if len(trajectory) < 2:
        return
    
    xs, ys = zip(*trajectory)
    
    plt.figure(figsize=(10, 8))
    plt.plot(xs, ys, 'b-', linewidth=2, label='Trajectory')
    plt.plot(xs[0], ys[0], 'go', markersize=10, label='Start')
    plt.plot(xs[-1], ys[-1], 'ro', markersize=10, label='End')
    
    plt.xlabel('X (pixel)')
    plt.ylabel('Y (pixel)')
    plt.title('Point Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # Invert Y axis for image coordinates
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_3d_trajectory_plot(trajectory_3d, output_path):
    """
    Lưu plot quỹ đạo 3D
    
    Args:
        trajectory_3d: list of (x, y, z) tuples
        output_path: đường dẫn lưu ảnh
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    if len(trajectory_3d) < 2:
        return
    
    xs, ys, zs = zip(*trajectory_3d)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(xs, ys, zs, 'b-', linewidth=2, label='Camera Path')
    ax.scatter(xs[0], ys[0], zs[0], c='green', s=100, label='Start')
    ax.scatter(xs[-1], ys[-1], zs[-1], c='red', s=100, label='End')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Camera 3D Trajectory')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_stats(data, name="Data"):
    """In thống kê của dữ liệu"""
    data = np.array(data)
    print(f"\n📊 {name} Statistics:")
    print(f"  Mean: {np.mean(data):.4f}")
    print(f"  Std:  {np.std(data):.4f}")
    print(f"  Min:  {np.min(data):.4f}")
    print(f"  Max:  {np.max(data):.4f}")


class ProgressBar:
    """Simple progress bar wrapper"""
    
    def __init__(self, total, desc="Processing"):
        try:
            from tqdm import tqdm
            self.pbar = tqdm(total=total, desc=desc)
            self.has_tqdm = True
        except ImportError:
            self.has_tqdm = False
            self.total = total
            self.current = 0
            self.desc = desc
            print(f"{desc}: 0/{total}")
    
    def update(self, n=1):
        if self.has_tqdm:
            self.pbar.update(n)
        else:
            self.current += n
            if self.current % max(1, self.total // 10) == 0:
                print(f"{self.desc}: {self.current}/{self.total}")
    
    def close(self):
        if self.has_tqdm:
            self.pbar.close()
        else:
            print(f"{self.desc}: Done!")


if __name__ == "__main__":
    # Test functions
    print("Testing utils...")
    
    # Test camera matrix
    K = create_camera_matrix(800, 800, 320, 240)
    print("Camera matrix:\n", K)
    
    # Test pixel to 3D
    xyz = pixel_to_3d(320, 240, 1.5, K)
    print(f"3D point: {xyz}")
    
    # Test distance
    p1 = (0, 0)
    p2 = (3, 4)
    dist = calculate_distance_2d(p1, p2)
    print(f"Distance 2D: {dist}")
    
    print("✅ Utils OK")
