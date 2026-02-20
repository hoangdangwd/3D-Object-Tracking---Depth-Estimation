import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Add parent directory to path để import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from pipeline.utils_pipeline import compute_rigid_transform

# 1. Đọc dữ liệu từ output_points_3d.csv
INPUT_FILE = "../../data/output/3d/output_points_3d.csv"
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ Không tìm thấy {INPUT_FILE}. Chạy batch_3d_processing.py trước!")

df = pd.read_csv(INPUT_FILE)

# 2. Các điểm pixel được chọn tại frame 0
selected_pixels = [
    [100, 800],
    [80, 750],
    [150, 650],
    [180, 750],
    [100, 700],
    [100, 600],
    [120, 580],
    [140, 670],
]

# 3. Tìm Query Index tương ứng tại frame 0
frame0 = df[df["Frame"] == 0]
selected_indices = []
for (u, v) in selected_pixels:
    match = frame0[(frame0["u"] == u) & (frame0["v"] == v)]
    if not match.empty:
        selected_indices.append(int(match["Query Index"].values[0]))
    else:
        print(f"⚠️ Không tìm thấy điểm ({u},{v}) trong frame 0!")

# 4. Toạ độ thực của các điểm vật thể trong thế giới (gốc tại điểm đầu)
P_world = np.array([
    [-0.009, -0.000, -0.001],
    [-0.015, -0.008, 0.009],
    [-0.000, -0.027, 0.000],
    [0.003, -0.008, 0.009],
    [-0.011, -0.018, 0.008],
    [-0.009, -0.036, -0.002],
    [-0.005, -0.040,-0.000],
    [-0.002, -0.023, 0.003],
])

# 5. Tính vị trí camera qua từng frame
frames = sorted(df["Frame"].unique())
camera_positions = []

for f in frames:
    df_f = df[df["Frame"] == f]
    pts = []
    for idx in selected_indices:
        match = df_f[df_f["Query Index"] == idx]
        if not match.empty:
            x, y, z = match[["X (m)", "Y (m)", "Z (m)"]].values[0]
            pts.append([x, y, z])
    if len(pts) != 8:
        print(f"⚠️ Frame {f} thiếu điểm, bỏ qua")
        continue
    pts_cam = np.array(pts)
    R, t = compute_rigid_transform(P_world, pts_cam)
    cam_pos = -R.T @ t
    camera_positions.append([f, *cam_pos])

# 7. Tính độ dịch chuyển
camera_positions = np.array(camera_positions)
positions = camera_positions[:, 1:]
displacements = np.diff(positions, axis=0)
distances = np.linalg.norm(displacements, axis=1)

# 8. Xuất kết quả
result = pd.DataFrame(camera_positions, columns=["Frame", "Camera_X", "Camera_Y", "Camera_Z"])
result["Displacement_X"] = np.append([[0, 0, 0]], displacements, axis=0)[:, 0]
result["Displacement_Y"] = np.append([[0, 0, 0]], displacements, axis=0)[:, 1]
result["Displacement_Z"] = np.append([[0, 0, 0]], displacements, axis=0)[:, 2]
result["Distance"] = np.append([0], distances)
result["Velocity"] = result["Distance"]

OUTPUT_FILE = "../../data/output/camera/camera_motion.csv"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
result.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Đã lưu kết quả vào: {OUTPUT_FILE}")

total_distance = result["Distance"].sum()
print(f"\n📏 Tổng độ dịch chuyển của camera: {total_distance:.4f} mét")

# 9. Vẽ quỹ đạo 3D của camera
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(result["Camera_X"], result["Camera_Y"], result["Camera_Z"], marker='o', label="Camera Trajectory")
ax.scatter(result["Camera_X"].iloc[0], result["Camera_Y"].iloc[0], result["Camera_Z"].iloc[0], c='red', label="Start")
ax.set_title("📷 Camera 3D Trajectory")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.legend()
plt.tight_layout()
plt.show()
