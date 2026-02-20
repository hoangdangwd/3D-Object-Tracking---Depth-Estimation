import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Tải mô hình MiDaS
print("Đang tải mô hình MiDaS...")
try:
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
    midas.eval()
except Exception as e:
    print(f"❌ Lỗi khi tải MiDaS: {e}")
    exit()

# 2. Tải transform phù hợp với DPT_Large
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.dpt_transform

# 3. Đọc ảnh đầu vào
img_path = "../../assets/checkpoints/1.jpg"  # ← Thay ảnh của bạn tại đây

if not os.path.exists(img_path):
    print(f"❌ Không tìm thấy ảnh: {img_path}")
    exit()

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 4. Tiền xử lý ảnh đầu vào
transformed = transform(img)
image_tensor = transformed['image'] if isinstance(transformed, dict) else transformed

if image_tensor.dim() == 3:
    input_tensor = image_tensor.unsqueeze(0)  # [C,H,W] → [1,C,H,W]
elif image_tensor.dim() == 4:
    input_tensor = image_tensor
else:
    raise ValueError(f"Sai kích thước tensor đầu vào: {image_tensor.shape}")

# 5. Dự đoán bản đồ độ sâu
with torch.no_grad():
    prediction = midas(input_tensor)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

# 6. Chuyển sang NumPy
depth_map = prediction.cpu().numpy()

# 7. Hiển thị ảnh và bản đồ độ sâu
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Ảnh gốc")
plt.imshow(img)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title("Bản đồ độ sâu (MiDaS)")
plt.imshow(depth_map, cmap='inferno')
plt.axis('off')
plt.tight_layout()
plt.show()

# 8. Lưu bản đồ độ sâu dạng ảnh xám
depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
depth_uint8 = (255 * depth_norm).astype("uint8")
cv2.imwrite("depth_midas.png", cv2.cvtColor(depth_uint8, cv2.COLOR_RGB2BGR))

# 9. Trích xuất độ sâu tại pixel cụ thể (u, v)
u, v = 450, 700  # ← Điểm ảnh cần lấy tọa độ
if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
    Z = depth_map[v, u]
    print(f"Độ sâu tại pixel ({u}, {v}) là: {Z:.4f}")
else:
    print("Toạ độ pixel nằm ngoài kích thước ảnh.")
    exit()

# 10. MA TRẬN NỘI TẠI CAMERA (K) - thay bằng K của bạn nếu khác
K = np.array([
    [2822.683, 0, 421.945],  # fx, 0, cx
    [0, 2813.273, 715.515],  # 0, fy, cy
    [0, 0, 1]
])

# 11. Suy ra toạ độ 3D từ (u, v, Z)
fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy

print(f"Tọa độ 3D trong hệ camera tại pixel ({u}, {v}):")
print(f"X = {X:.4f}, Y = {Y:.4f}, Z = {Z:.4f}")

