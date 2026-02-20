import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# ========== 1. Đọc dữ liệu ==========
CSV_PATH = "../../data/output/tracking/tracked_points.csv"
IMG_FOLDER = "../../data/cache/frames"

# Kiểm tra file tồn tại
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy file {CSV_PATH}. Chạy track_for_3d.py trước!")

if not os.path.exists(IMG_FOLDER):
    raise FileNotFoundError(f"❌ Không tìm thấy thư mục {IMG_FOLDER}. Chạy extract_frames.py trước!")

K = np.array([
    [3440.17076, 0, 957.163265],
    [0, 3437.76682, 543.944954],
    [0, 0, 1]
])
K_inv = np.linalg.inv(K)

df = pd.read_csv(CSV_PATH)
frames = df["Frame"].unique()

# ========== 2. Tải MiDaS ==========
print("🔄 Đang tải MiDaS...")
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.dpt_transform

# ========== 3. Danh sách điểm chuẩn ==========
reference_points = [
    (100, 800, 0.62),
    (80, 750, 0.63),
    (150, 650, 0.65),
    (180, 750, 0.64),
    (100, 700, 0.65),
    (100, 600, 0.63),
    (120, 580, 0.66),
    (140, 670, 0.63),
]

output_data = []
scale_factor = None

# ========== 4. Xử lý từng frame ==========
for frame_id in tqdm(frames, desc="📸 Xử lý ảnh"):
    # --- Đọc ảnh ---
    img_name = f"frame_{int(frame_id):04d}.png"
    img_path = os.path.join(IMG_FOLDER, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Không tìm thấy ảnh {img_path}")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Depth với MiDaS ---
    input_transformed = transform(img_rgb)
    # Xử lý kiểu dữ liệu transform
    if isinstance(input_transformed, dict):
        input_tensor = input_transformed["image"]
    else:
        input_tensor = input_transformed
    
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_raw = 1.0 / prediction.cpu().numpy()  # Chuyển về Z ~ khoảng cách thực

    # --- Tính scale trung bình tại frame đầu ---
    if scale_factor is None and frame_id == frames[0]:
        scale_list = []
        print("🔍 Tính scale từ 8 điểm chuẩn:")
        for (u, v, z_real) in reference_points:
            if 0 <= v < depth_raw.shape[0] and 0 <= u < depth_raw.shape[1]:
                z_pred = depth_raw[v, u]
                scale = z_real / z_pred
                scale_list.append(scale)
                print(f"   📌 Điểm ({u},{v}): Z_thực={z_real}, Z_pred={z_pred:.4f}, scale={scale:.4f}")
            else:
                print(f"⚠️ Điểm ({u},{v}) nằm ngoài ảnh!")

        if scale_list:
            scale_factor = np.mean(scale_list)
            print(f"✅ Scale factor trung bình: {scale_factor:.6f}")
        else:
            print("❌ Không có điểm hợp lệ để tính scale!")

    # --- Scale lại depth map ---
    depth_scaled = depth_raw * scale_factor if scale_factor else depth_raw

    # --- Tính toạ độ 3D ---
    df_frame = df[df["Frame"] == frame_id]
    for _, row in df_frame.iterrows():
        u = int(row["X (pixel)"])
        v = int(row["Y (pixel)"])
        query_idx = int(row["Query Index"])

        if 0 <= v < depth_scaled.shape[0] and 0 <= u < depth_scaled.shape[1]:
            Z = depth_scaled[v, u]
            pixel = np.array([u, v, 1])
            XYZ = Z * (K_inv @ pixel)
            X, Y, Z = XYZ
            output_data.append([frame_id, query_idx, u, v, X, Y, Z])
        else:
            print(f"⚠️ Điểm ({u}, {v}) nằm ngoài ảnh!")

# ========== 5. Xuất kết quả ==========
out_df = pd.DataFrame(output_data, columns=["Frame", "Query Index", "u", "v", "X (m)", "Y (m)", "Z (m)"])
# Lưu kết quả
OUTPUT_FILE = "../../data/output/3d/output_points_3d.csv"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
out_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Đã lưu kết quả ra '{OUTPUT_FILE}'")
