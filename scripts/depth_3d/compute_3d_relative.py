import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ====== 1. File ảnh ======
IMG_PATH = "../../data/cache/frames/frame_0077.png"

if not os.path.exists(IMG_PATH):
    print(f"❌ Không tìm thấy ảnh: {IMG_PATH}")
    print("💡 Hãy chạy extract_frames.py hoặc cập nhật IMG_PATH")
    exit()

K = np.array([
    [827.8, 0, 647.73],
    [0, 829.45, 466.9],
    [0, 0, 1]
])
K_inv = np.linalg.inv(K)

# Các điểm cần trích xuất
points_uv = np.array([
    [1691., 905.],
    [1646., 850.],
    [1674., 761.]
], dtype=np.float32)

# Gốc toạ độ 3D mong muốn (gốc mới)
center_u, center_v = 250, 750
Z_true_m = 0.76  # mét (khoảng cách thật tại điểm gốc)

# ====== 2. Tải MiDaS ======
print("🔄 Đang tải MiDaS...")
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.dpt_transform

# ====== 3. Đọc ảnh & transform ======
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"❌ Không tìm thấy ảnh tại: {IMG_PATH}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_transformed = transform(img_rgb)
input_tensor = input_transformed["image"] if isinstance(input_transformed, dict) else input_transformed
if input_tensor.dim() == 3:
    input_tensor = input_tensor.unsqueeze(0)

# ====== 4. Dự đoán bản đồ độ sâu ======
with torch.no_grad():
    prediction = midas(input_tensor)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_raw = prediction.cpu().numpy()

# ====== 5. Đảo chiều depth & scale ======
depth_inv = 1.0 / (depth_raw + 1e-6)
Z_midas_center = depth_inv[int(center_v), int(center_u)]
scale = Z_true_m / Z_midas_center
depth_real = depth_inv * scale

print(f"⚖️ Hệ số scale: {scale:.4f}")

# ====== 6. Tính toạ độ 3D của gốc (0,0,0) ======
pixel_center = np.array([center_u, center_v, 1])
P_center = depth_real[int(center_v), int(center_u)] * (K_inv @ pixel_center)
print(f"📌 Gốc toạ độ (X0,Y0,Z0): {P_center}")

# ====== 7. Tính toạ độ 3D tương đối của các điểm ======
print("\n📍 Tọa độ 3D (so với gốc (250,750)):")
for idx, (u, v) in enumerate(points_uv):
    Z = depth_real[int(v), int(u)]
    pixel_homog = np.array([u, v, 1])
    P_world = Z * (K_inv @ pixel_homog)

    # Tính toạ độ tương đối (dịch gốc)
    P_relative = P_world - P_center
    Xr, Yr, Zr = P_relative

    print(f"• Điểm {idx + 1} tại pixel ({int(u)}, {int(v)}):")
    print(f"  X' = {Xr:.3f} m, Y' = {Yr:.3f} m, Z' = {Zr:.3f} m\n")

# ====== 8. Hiển thị ảnh & bản đồ độ sâu ======
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Ảnh gốc")
plt.imshow(img_rgb)
plt.plot(center_u, center_v, 'bo', markersize=8, label="Gốc (0,0,0)")
for (u, v) in points_uv:
    plt.plot(u, v, 'go', markersize=8)
plt.legend()
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Bản đồ độ sâu (mét)")
plt.imshow(depth_real, cmap='inferno')
plt.colorbar(label="Độ sâu (m)")
plt.plot(center_u, center_v, 'bo')
for (u, v) in points_uv:
    plt.plot(u, v, 'go')
plt.axis("off")

plt.tight_layout()
plt.show()

