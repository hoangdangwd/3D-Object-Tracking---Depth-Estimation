"""
Script: compute_3d_coords.py (renamed from toaDo.py)
Description: Tính toán tọa độ 3D của các điểm trong ảnh dựa trên:
             - Ma trận intrinsic camera (K)
             - Depth map từ MiDaS
             - Điểm chuẩn có khoảng cách thực đã biết

Usage:
    python compute_3d_coords.py

Author: CoTracker Extended
Date: 2026-02-03
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ====== 1. Cấu hình ======
IMG_PATH = "../../data/cache/frames/frame_0077.png"

# Ma trận intrinsic camera K (từ calibration)
K = np.array([
    [827.8, 0, 647.73],
    [0, 829.45, 466.9],
    [0, 0, 1]
])
K_inv = np.linalg.inv(K)

# Các điểm cần tính toạ độ 3D
points_uv = np.array([
    [1691., 905.],
    [1646., 850.],
    [1674., 761.]
], dtype=np.float32)

# Điểm chuẩn (reference point) có khoảng cách thực đã biết
reference_u, reference_v = 250, 750
Z_true_m = 0.76  # Khoảng cách thực tại điểm chuẩn (mét)

# ====== 2. Kiểm tra file tồn tại ======
if not os.path.exists(IMG_PATH):
    print(f"❌ Không tìm thấy ảnh: {IMG_PATH}")
    print("💡 Hãy chạy save_frame.py hoặc cập nhật IMG_PATH")
    exit()

# ====== 3. Tải MiDaS model ======
print("🔄 Đang tải MiDaS model...")
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.dpt_transform

# ====== 4. Đọc và xử lý ảnh ======
print(f"📸 Đọc ảnh: {os.path.basename(IMG_PATH)}")
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"❌ Không thể đọc ảnh tại: {IMG_PATH}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]
print(f"📐 Kích thước ảnh: {W}x{H}")

# Transform ảnh cho MiDaS
input_transformed = transform(img_rgb)
input_tensor = input_transformed["image"] if isinstance(input_transformed, dict) else input_transformed
if input_tensor.dim() == 3:
    input_tensor = input_tensor.unsqueeze(0)

# ====== 5. Dự đoán depth map ======
print("🧮 Đang ước lượng depth map...")
with torch.no_grad():
    prediction = midas(input_tensor)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_raw = prediction.cpu().numpy()

# ====== 6. Chuyển đổi và scale depth ======
# MiDaS trả về disparity, đảo ngược để có depth
depth_inv = 1.0 / (depth_raw + 1e-6)

# Lấy depth tại điểm chuẩn
Z_midas_ref = depth_inv[int(reference_v), int(reference_u)]

# Tính hệ số scale để chuyển về đơn vị mét thực
scale = Z_true_m / Z_midas_ref
depth_real = depth_inv * scale

print(f"⚖️ Hệ số scale: {scale:.4f}")
print(f"📏 Depth tại điểm chuẩn ({reference_u}, {reference_v}): {Z_true_m} m")

# ====== 7. Tính toạ độ 3D (tuyệt đối) ======
print("\n" + "="*60)
print("📍 TỌA ĐỘ 3D (hệ tọa độ camera)")
print("="*60)

results_3d = []

for idx, (u, v) in enumerate(points_uv):
    # Lấy độ sâu tại pixel (u, v)
    Z = depth_real[int(v), int(u)]
    
    # Công thức chiếu ngược:
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = Z
    
    # Hoặc dùng ma trận K_inv:
    pixel_homog = np.array([u, v, 1])
    P_3d = Z * (K_inv @ pixel_homog)
    
    X, Y, Z_coord = P_3d
    
    print(f"• Điểm {idx + 1} tại pixel ({int(u)}, {int(v)}):")
    print(f"  X = {X:.3f} m")
    print(f"  Y = {Y:.3f} m")
    print(f"  Z = {Z_coord:.3f} m")
    print(f"  Khoảng cách từ camera: {np.linalg.norm(P_3d):.3f} m\n")
    
    results_3d.append({
        'pixel_u': int(u),
        'pixel_v': int(v),
        'X': X,
        'Y': Y,
        'Z': Z_coord,
        'distance': np.linalg.norm(P_3d)
    })

# ====== 8. Lưu kết quả ======
import pandas as pd
df_output = pd.DataFrame(results_3d)
output_csv = "../../data/output/3d/3d_coords.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df_output.to_csv(output_csv, index=False)
print(f"✅ Đã lưu kết quả vào: {output_csv}")

# ====== 9. Visualize ======
plt.figure(figsize=(15, 5))

# Ảnh gốc với điểm đánh dấu
plt.subplot(1, 3, 1)
plt.title("Ảnh gốc với điểm chuẩn")
plt.imshow(img_rgb)
plt.plot(reference_u, reference_v, 'bo', markersize=10, label=f"Ref ({Z_true_m}m)")
for idx, (u, v) in enumerate(points_uv):
    plt.plot(u, v, 'ro', markersize=8)
    plt.text(u+20, v-20, f"P{idx+1}", color='red', fontsize=10, fontweight='bold')
plt.legend()
plt.axis("off")

# Bản đồ depth
plt.subplot(1, 3, 2)
plt.title("Depth Map (mét)")
plt.imshow(depth_real, cmap='viridis')
plt.colorbar(label="Độ sâu (m)")
plt.plot(reference_u, reference_v, 'bo', markersize=10)
for (u, v) in points_uv:
    plt.plot(u, v, 'ro', markersize=8)
plt.axis("off")

# Biểu đồ 3D scatter
ax = plt.subplot(1, 3, 3, projection='3d')
ax.set_title("Tọa độ 3D (hệ camera)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

for idx, result in enumerate(results_3d):
    ax.scatter(result['X'], result['Y'], result['Z'], 
               c='red', s=100, marker='o', label=f"P{idx+1}")

# Camera tại gốc tọa độ
ax.scatter(0, 0, 0, c='blue', s=200, marker='^', label='Camera')
ax.legend()

plt.tight_layout()
output_img = "../../data/output/3d/3d_coords_visualization.png"
plt.savefig(output_img, dpi=150, bbox_inches='tight')
print(f"✅ Đã lưu visualization: {output_img}")
plt.show()

print("\n" + "="*60)
print("✨ Hoàn tất!")
print("="*60)
