# CoTracker3: Point Tracking in Video

[![License](https://img.shields.io/badge/License-CC--BY--NC%204.0-blue)](LICENSE.md)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

**CoTracker3** là mô hình AI theo dõi điểm (point tracking) trong video, phát triển bởi Meta AI Research và University of Oxford.

---

## 🎯 Tính Năng

- **Point Tracking**: Theo dõi bất kỳ điểm nào trong video (offline/online modes)
- **Dense Tracking**: Theo dõi grid lên đến 265×265 điểm đồng thời
- **3D Reconstruction**: Tính toán tọa độ 3D từ tracking + depth estimation
- **Camera Motion**: Ước lượng quỹ đạo di chuyển của camera
- **Real-time**: Hỗ trợ webcam tracking

---

## 📦 Cài Đặt

```bash
# Clone repository
git clone https://github.com/facebookresearch/co-tracker.git
cd co-tracker

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt package
pip install -e .
```

**Requirements:**
- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA (optional, khuyến nghị cho GPU acceleration)

---

## 🚀 Pipeline Sử Dụng Nhanh

### Mode 1: Full Pipeline (Tracking → 3D → Camera)
```bash
python pipeline/pipeline.py --video_path assets/apple.mp4 --mode full
```

### Mode 2: Tracking Only
```bash
python pipeline/pipeline.py --video_path video.mp4 --mode tracking --grid_size 20
```

### Mode 3: Webcam Real-time
```bash
python pipeline/pipeline.py --mode webcam
```

**Output:**
- `data/output/tracking/tracked_points.csv` - Tọa độ pixel tracked points
- `data/output/3d/points_3d.csv` - Tọa độ 3D (m) của các điểm
- `data/output/velocity_3d.csv` - Vận tốc thực (m/s) của các điểm
- `data/output/camera/camera_motion.csv` - Quỹ đạo camera
- `data/output/videos/tracked_video.mp4` - Video visualization

---

## 📂 Cấu Trúc Dự Án

```
co-tracker-main/
├── cotracker/              # Core package (models, datasets, evaluation)
├── pipeline/               # Integrated pipeline
│   ├── pipeline.py         # Main orchestrator
│   ├── config.yaml         # Configuration
│   └── utils_pipeline.py   # Utilities
├── scripts/
│   ├── demos/              # Demo scripts
│   ├── processing/         # Video processing
│   ├── depth_3d/           # 3D reconstruction & depth
│   ├── calibration/        # Camera calibration
│   └── training/           # Model training
├── assets/                 # Sample videos & images
├── models/                 # Model checkpoints
├── data/                   # Input/Output data
├── tools/                  # Quick start scripts
└── docs/                   # Documentation
```

---

## 🔧 Configuration

### Camera Calibration
Chỉnh sửa `pipeline/config.yaml`:

```yaml
camera_matrix:
  - [fx,  0, cx]
  - [ 0, fy, cy]
  - [ 0,  0,  1]

reference_points:  # (u, v, depth_meters)
  - [100, 800, 0.62]
  - [80, 750, 0.63]
```

**Lấy camera matrix:**
```bash
python scripts/calibration/calibrate_camera.py
```

### Kiểm Tra Depth Accuracy
**Test depth estimation với reference points:**
```bash
python scripts/depth_3d/test_depth_accuracy.py --image path/to/image.jpg
```

**Output:**
- Scale factor với confidence interval (mean ± std)
- MAE/RMSE tại reference points
- Outlier detection status
- Visualization: `data/output/depth_accuracy_test.png`

**Tiêu chí chấp nhận:**
- ✅ Scale factor std < 10% mean: Calibration tốt
- ✅ MAE < 5cm: Độ chính xác cao
- ✅ ≥3 valid points: Đáng tin cậy
- ⚠️ Std > 20%: Cần kiểm tra lại reference points
- ❌ <3 points: Không thể scale chính xác

---

## 📊 Models

### CoTracker Models

| Model | Checkpoint | Window | Description |
|-------|-----------|--------|-------------|
| CoTracker3 Offline | `models/cotracker.pth` | 60 frames | Chính xác cao, xử lý offline |
| CoTracker3 Online | `models/cotracker_stride_4_wind_8.pth` | 16 frames | Real-time tracking |
| Scaled Offline | `models/scaled_offline.pth` | 60 frames | High-resolution tracking |

### Depth Model
- **MiDaS DPT_Large** (Intel ISL): Depth estimation từ monocular images

---

## 🎨 Examples

### 1. Track Specific Points
```python
from cotracker.predictor import CoTrackerPredictor
import torch

model = CoTrackerPredictor(checkpoint="models/cotracker.pth")
video = torch.randn(1, 10, 3, 480, 640)  # [B, T, C, H, W]
queries = torch.tensor([[[0, 100, 200]]])  # [B, N, 3] (frame_idx, x, y)

pred_tracks, pred_visibility = model(video, queries=queries)
```

### 2. Dense Grid Tracking
```bash
python scripts/demos/demo.py --video_path video.mp4 --grid_size 50
```

### 3. 3D Coordinates from Single Image
```bash
python scripts/depth_3d/compute_3d_coords.py
```

### 4. Full Workflow with Custom Config
```bash
python pipeline/pipeline.py \
    --video_path video.mp4 \
    --mode full \
    --grid_size 20 \
    --output_dir results
```

---

## 🧪 Scripts

### Demos (`scripts/demos/`)
- `demo.py` - Basic tracking demo
- `online_demo.py` - Online tracking mode
- `webcam_demo.py` - Webcam real-time tracking
- `demo_pipeline.py` - Interactive menu

### Processing (`scripts/processing/`)
- `extract_frames.py` - Extract frames từ video
- `track_video.py` - Track video với velocity calculation
- `test_tracker.py` - Test tracking accuracy

### Depth & 3D (`scripts/depth_3d/`)
- `compute_3d_coords.py` - Tính tọa độ 3D từ pixel + depth
- `compute_3d_relative.py` - Tọa độ 3D tương đối
- `compute_velocity_3d.py` - **Tính vận tốc thực (m/s) trong không gian 3D**
- `estimate_depth.py` - Depth estimation với MiDaS
- `test_depth_accuracy.py` - **Kiểm tra độ chính xác depth estimation**
- `batch_3d_processing.py` - Xử lý batch nhiều frames
- `estimate_camera_motion.py` - Ước lượng camera pose
- `track_for_3d.py` - Track video để chuẩn bị cho 3D processing

### Calibration (`scripts/calibration/`)
- `calibrate_camera.py` - Camera calibration với checkerboard

### Training (`scripts/training/`)
- `train_kubric.py` - Train trên Kubric dataset
- `train_on_real_data.py` - Fine-tune trên real videos

---

## 📐 Mathematical Formulas

### 1. Pixel to 3D Projection
```
Given: pixel (u, v), depth Z, camera matrix K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

X = (u - cx) × Z / fx
Y = (v - cy) × Z / fy
Z = Z
```

### 2. Depth Scaling
```
# MiDaS returns disparity (inverse depth)
disparity = MiDaS(image)
depth_inv = 1 / (disparity + ε)  # ε = 1e-6 để tránh chia cho 0

# Scale từ reference points với outlier removal
scales = [Z_real / depth_inv[v, u] for (u, v, Z_real) in reference_points]
median = median(scales)
MAD = median(|scales - median|)
scales_filtered = [s for s in scales if |s - median| < 2*MAD]
scale_factor = mean(scales_filtered)

depth_real = depth_inv × scale_factor
```

**Critical Notes:**
- ✅ MiDaS trả về **disparity**, không phải depth trực tiếp
- ✅ Phải dùng `1/(disparity + ε)` để convert sang metric depth
- ✅ Scale factor cần ≥3 reference points hợp lệ
- ✅ Outlier removal dùng MAD (Median Absolute Deviation)
- ✅ Sub-pixel depth dùng bilinear interpolation

### 3. Rigid Transform (Camera Motion)
```
Given: 3D points A (frame t), B (frame t+1)

Centroid: c_A = mean(A), c_B = mean(B)
Centered: A' = A - c_A, B' = B - c_B
Covariance: H = A'^T × B'
SVD: U, S, V^T = svd(H)
Rotation: R = V × U^T
Translation: t = c_B - R × c_A
```

---

## 🔬 Technical Details

### CoTracker Architecture
- **Backbone**: Vision Transformer (ViT)
- **Temporal Context**: Sliding window (60 frames offline, 16 frames online)
- **Output**: Tracks shape `[B, T, N, 2]`, Visibility `[B, T, N]`

### Pipeline Flow
```
Video (T×H×W×3) 
  → CoTracker → Tracks (T×N×2) pixel coords
  → MiDaS → Disparity (T×H×W) → depth_inv = 1/(disparity+ε)
  → Scale with reference points → Depth_real (T×H×W) meters
  → Bilinear interpolation → Points_3D (T×N×3) meters
  → Velocity Calculation → Velocity (T×N) m/s
  → Rigid Transform → Camera_pose (T×3)
```

**Depth Estimation Quality Checks:**
- ✅ Scale factor std < 10% của mean → Good calibration
- ✅ MAE < 5cm at reference points → Accurate
- ✅ ≥3 valid reference points → Reliable
- ⚠️ Scale factor std > 20% → Check reference points
- ❌ <3 valid points → Cannot scale reliably

---

## 📊 Performance

| Model | TAP-Vid-DAVIS J&F | FPS (GPU) | Memory |
|-------|-------------------|-----------|--------|
| CoTracker3 Offline | 77.3 | 25 | 11 GB |
| CoTracker3 Online | 71.2 | 60 | 6 GB |

**Hardware**: NVIDIA RTX 3090, Video 480×640

---

## 🛠️ Quick Start Tools

```bash
# Windows
cd tools
quick_start.bat

# Linux/Mac
cd tools
./quick_start.sh
```

---

## 📝 Data Format

### Tracked Points CSV
```csv
Frame,Query,X,Y,Visibility
0,0,320.5,240.2,1.0
0,1,450.1,180.7,1.0
1,0,322.1,241.0,1.0
```

### 3D Points CSV
```csv
Frame,Query,X_m,Y_m,Z_m
0,0,0.125,-0.083,0.750
0,1

### Velocity 3D CSV
```csv
Query Index,Frame,X (m),Y (m),Z (m),Distance (m),Velocity (m/s),Velocity_X (m/s),Velocity_Y (m/s),Velocity_Z (m/s)
0,1,0.126,-0.082,0.751,0.0025,0.075,0.003,-0.001,0.001
0,2,0.128,-0.081,0.753,0.0032,0.096,0.006,0.003,0.006
```,0.201,-0.142,0.755
```

### Camera Motion CSV
```csv
Frame,Camera_X,Camera_Y,Camera_Z,Distance_m
0,0.000,0.000,0.000,0.000
1,0.005,-0.002,0.010,0.011
```

---

## 🔗 References

### Papers
- **CoTracker3** (2024): "CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos"
- **CoTracker** (2023): "CoTracker: It is Better to Track Together"

### Links
- GitHub: https://github.com/facebookresearch/co-tracker
- Project Page: https://co-tracker.github.io
- PyTorch Hub: `torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")`

---

## 📄 License

CC-BY-NC 4.0 License. Xem [LICENSE.md](LICENSE.md) để biết chi tiết.

**Developed by:**
- Meta AI Research (FAIR)
- University of Oxford - Visual Geometry Group

---

## 🙏 Acknowledgments

- **MiDaS**: Intel ISL depth estimation model
- **TAPNet**: TAP-Vid benchmark datasets
- **Kubric**: Synthetic video generation

---

## 📮 Contact & Support

- **Issues**: https://github.com/facebookresearch/co-tracker/issues
- **Discussions**: https://github.com/facebookresearch/co-tracker/discussions
- **Citation**:
```bibtex
@article{karaev2024cotracker3,
  title={CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos},
  author={Karaev, Nikita and Rocco, Ignacio and Graham, Benjamin and Neverova, Natalia and Vedaldi, Andrea and Rupprecht, Christian},
  journal={arXiv preprint arXiv:2410.11831},
  year={2024}
}
```
