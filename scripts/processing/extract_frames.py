import cv2
import os
import argparse

# 🔧 Đường dẫn video và thư mục lưu ảnh
parser = argparse.ArgumentParser()
parser.add_argument("--video_path", default="../../assets/VideoGR1.2.2.mp4", help="Đường dẫn video")
parser.add_argument("--output_dir", default="../../data/cache/frames", help="Thư mục lưu frame")
args = parser.parse_args()

video_path = args.video_path
output_dir = args.output_dir

# Tạo thư mục nếu chưa có
os.makedirs(output_dir, exist_ok=True)

# Kiểm tra file tồn tại
if not os.path.exists(video_path):
    print(f"❌ Không tìm thấy video: {video_path}")
    exit()

# Mở video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Không mở được video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"✅ Tổng số frame: {total_frames}")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Đã lưu xong tất cả frame.")
        break

    # Đặt tên file ảnh
    filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")

    # Lưu ảnh
    cv2.imwrite(filename, frame)
    print(f"💾 Đã lưu: {filename}")

    frame_idx += 1

# Giải phóng bộ nhớ
cap.release()
print(f"🎉 Tất cả frame đã được lưu vào thư mục: {output_dir}")
