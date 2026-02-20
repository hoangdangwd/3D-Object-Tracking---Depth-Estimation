"""
CoTracker Pipeline - Tích hợp đầy đủ từ Video → Tracking → 3D → Camera Motion

Usage:
    python pipeline.py --video_path video.mp4 --mode full
    python pipeline.py --video_path video.mp4 --mode tracking_only
    python pipeline.py --mode webcam
"""

import os
import sys
import argparse
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import các module cần thiết
from cotracker.predictor import CoTrackerPredictor, CoTrackerOnlinePredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path


class CoTrackerPipeline:
    """Pipeline tích hợp đầy đủ cho CoTracker"""
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Sử dụng device: {self.device}")
        
        # Models
        self.tracker_model = None
        self.midas_model = None
        self.midas_transform = None
        
    @staticmethod
    def get_default_config():
        """Cấu hình mặc định"""
        return {
            # Paths
            'output_dir': '../data/output',
            'frames_dir': '../data/cache/frames',
            'videos_dir': '../data/output/videos',
            'data_dir': '../data/output',
            
            # CoTracker settings
            'grid_size': 10,
            'offline_mode': True,
            'window_len': 60,
            'checkpoint': None,
            
            # Camera intrinsic (iPhone 12 Pro Max example)
            'camera_matrix': np.array([
                [827.8, 0, 647.73],
                [0, 829.45, 466.9],
                [0, 0, 1]
            ]),
            
            # Depth estimation
            'use_depth': True,
            'midas_model_type': 'DPT_Large',
            
            # Reference points for depth scaling (pixel coords + real depth in meters)
            'reference_points': [
                (100, 800, 0.62),
                (80, 750, 0.63),
                (150, 650, 0.65),
                (180, 750, 0.64),
                (100, 700, 0.65),
                (100, 600, 0.63),
                (120, 580, 0.66),
                (140, 670, 0.63),
            ],
            
            # Query points (frame_idx, x, y)
            'query_points': [
                [0., 100., 800.],
                [0., 80., 750.],
                [0., 150., 650.],
                [0., 180., 750.],
                [0., 100., 700.],
                [0., 100., 600.],
                [0., 120., 580.],
                [0., 140., 670.],
            ],
            
            # Visualization
            'visualize': True,
            'save_frames': False,
        }
    
    def setup_directories(self):
        """Tạo các thư mục cần thiết"""
        for key in ['output_dir', 'frames_dir', 'videos_dir', 'data_dir']:
            path = self.config[key]
            os.makedirs(path, exist_ok=True)
        print(f"✅ Đã tạo thư mục: {self.config['output_dir']}")
    
    def load_tracker_model(self):
        """Load CoTracker model"""
        if self.tracker_model is not None:
            return
        
        print("🔄 Đang load CoTracker model...")
        try:
            if self.config['checkpoint']:
                self.tracker_model = CoTrackerPredictor(
                    checkpoint=self.config['checkpoint'],
                    offline=self.config['offline_mode'],
                    window_len=self.config['window_len']
                )
            else:
                model_name = "cotracker3_offline" if self.config['offline_mode'] else "cotracker3_online"
                self.tracker_model = torch.hub.load("facebookresearch/co-tracker", model_name)
            
            self.tracker_model = self.tracker_model.to(self.device)
            print("✅ CoTracker model loaded")
        except Exception as e:
            print(f"❌ Lỗi khi load CoTracker: {e}")
            raise
    
    def load_midas_model(self):
        """Load MiDaS depth estimation model"""
        if not self.config['use_depth']:
            return
        
        if self.midas_model is not None:
            return
        
        print("🔄 Đang load MiDaS model...")
        try:
            self.midas_model = torch.hub.load("intel-isl/MiDaS", self.config['midas_model_type'], trust_repo=True)
            self.midas_model.eval()
            self.midas_model = self.midas_model.to(self.device)
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.midas_transform = midas_transforms.dpt_transform
            
            print("✅ MiDaS model loaded")
        except Exception as e:
            print(f"❌ Lỗi khi load MiDaS: {e}")
            raise
    
    def extract_frames(self, video_path):
        """Trích xuất frames từ video"""
        if not self.config['save_frames']:
            return None
        
        print(f"📹 Trích xuất frames từ {video_path}...")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise IOError(f"Không thể mở video: {video_path}")
        
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for _ in tqdm(range(total_frames), desc="Extracting frames"):
            ret, frame = cap.read()
            if not ret:
                break
            
            filename = os.path.join(self.config['frames_dir'], f"frame_{frame_idx:04d}.png")
            cv2.imwrite(filename, frame)
            frame_idx += 1
        
        cap.release()
        print(f"✅ Đã lưu {frame_idx} frames")
        return frame_idx
    
    def run_tracking(self, video_path, queries=None, grid_size=None):
        """Chạy tracking trên video"""
        print(f"🎯 Bắt đầu tracking video: {video_path}")
        
        # Load model
        self.load_tracker_model()
        
        # Load video
        video = read_video_from_path(video_path)
        if video is None:
            raise IOError(f"Không thể đọc video: {video_path}")
        
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(self.device)
        
        # Prepare queries
        if queries is None and grid_size is None:
            queries = torch.tensor(self.config['query_points'], device=self.device)[None]
        elif queries is not None:
            queries = torch.tensor(queries, device=self.device)[None]
        
        # Run tracking
        try:
            if queries is not None:
                pred_tracks, pred_visibility = self.tracker_model(
                    video_tensor,
                    queries=queries
                )
            else:
                pred_tracks, pred_visibility = self.tracker_model(
                    video_tensor,
                    grid_size=grid_size or self.config['grid_size']
                )
            
            print("✅ Tracking hoàn tất")
            
            # Save visualization
            if self.config['visualize']:
                self.visualize_tracks(video_tensor, pred_tracks, pred_visibility, 
                                    os.path.basename(video_path))
            
            # Save tracked points to CSV
            csv_path = os.path.join(self.config['data_dir'], 'tracked_points.csv')
            self.save_tracks_to_csv(pred_tracks, csv_path)
            
            return pred_tracks, pred_visibility, video_tensor
            
        except Exception as e:
            print(f"❌ Lỗi khi tracking: {e}")
            raise
    
    def save_tracks_to_csv(self, pred_tracks, csv_path):
        """Lưu tracked points vào CSV"""
        track_points = pred_tracks[0].cpu().numpy()  # [frames, queries, 2]
        
        data = []
        for frame_idx, frame_points in enumerate(track_points):
            for query_idx, (x, y) in enumerate(frame_points):
                data.append([frame_idx, query_idx, x, y])
        
        df = pd.DataFrame(data, columns=["Frame", "Query Index", "X (pixel)", "Y (pixel)"])
        df.to_csv(csv_path, index=False)
        print(f"💾 Đã lưu tracked points: {csv_path}")
        return csv_path
    
    def visualize_tracks(self, video, tracks, visibility, filename):
        """Visualize và lưu video với tracks"""
        vis = Visualizer(
            save_dir=self.config['videos_dir'],
            linewidth=3,
            mode="cool",
            tracks_leave_trace=-1
        )
        
        output_name = f"tracked_{filename.replace('.mp4', '')}"
        vis.visualize(video=video, tracks=tracks, visibility=visibility, filename=output_name)
        print(f"🎬 Đã lưu video: {self.config['videos_dir']}/{output_name}.mp4")
    
    def estimate_depth_and_3d(self, video_path, csv_path):
        """Ước lượng depth và tính toạ độ 3D"""
        if not self.config['use_depth']:
            print("⚠️ Depth estimation bị tắt")
            return None
        
        print("🔍 Bắt đầu ước lượng depth và tính toạ độ 3D...")
        
        # Load MiDaS
        self.load_midas_model()
        
        # Load tracked points
        df = pd.read_csv(csv_path)
        frames = df["Frame"].unique()
        
        K = self.config['camera_matrix']
        K_inv = np.linalg.inv(K)
        
        output_data = []
        scale_factor = None
        
        # Process each frame
        for frame_id in tqdm(frames, desc="Estimating depth"):
            # Read frame
            if self.config['save_frames']:
                img_name = f"frame_{int(frame_id):04d}.png"
                img_path = os.path.join(self.config['frames_dir'], img_name)
                img = cv2.imread(img_path)
            else:
                # Extract from video
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
                ret, img = cap.read()
                cap.release()
                
                if not ret:
                    continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Depth estimation with MiDaS
            input_transformed = self.midas_transform(img_rgb)
            if isinstance(input_transformed, dict):
                input_tensor = input_transformed["image"]
            else:
                input_tensor = input_transformed
            
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                prediction = self.midas_model(input_tensor)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # MiDaS returns inverse depth (disparity), convert to depth
            depth_map = prediction.cpu().numpy()
            depth_raw = depth_map  # Keep raw disparity for now
            depth_inv = 1.0 / (depth_raw + 1e-6)  # Convert to metric depth with epsilon
            
            # Calculate scale factor from reference points (first frame only)
            if scale_factor is None and frame_id == frames[0]:
                scale_list = []
                print("\n🔍 Tính scale factor từ reference points:")
                for (u, v, z_real) in self.config['reference_points']:
                    if 0 <= v < depth_inv.shape[0] and 0 <= u < depth_inv.shape[1]:
                        z_pred = depth_inv[v, u]
                        if z_pred > 1e-6:  # Validate depth is not too small
                            scale = z_real / z_pred
                            scale_list.append(scale)
                            print(f"  📌 ({u},{v}): Z_real={z_real:.3f}m, Z_pred={z_pred:.3f}, scale={scale:.3f}")
                
                if len(scale_list) >= 3:  # Need at least 3 valid points
                    # Remove outliers using median ± 2*MAD
                    scale_median = np.median(scale_list)
                    mad = np.median(np.abs(np.array(scale_list) - scale_median))
                    scale_filtered = [s for s in scale_list if abs(s - scale_median) < 2 * mad]
                    
                    if scale_filtered:
                        scale_factor = np.mean(scale_filtered)
                        scale_std = np.std(scale_filtered)
                        print(f"\n✅ Scale factor: {scale_factor:.6f} ± {scale_std:.6f}")
                        print(f"   Valid points: {len(scale_filtered)}/{len(scale_list)}")
                    else:
                        print("\n⚠️ Tất cả scale đều là outliers, dùng median")
                        scale_factor = scale_median
                else:
                    print(f"\n❌ Chỉ có {len(scale_list)} điểm hợp lệ (cần ≥3), không thể tin cậy!")
                    raise ValueError(f"Không đủ reference points hợp lệ để scale depth. Kiểm tra lại config.")
            
            # Scale depth map to metric depth
            depth_scaled = depth_inv * scale_factor
            
            # Calculate 3D coordinates with bilinear interpolation
            df_frame = df[df["Frame"] == frame_id]
            for _, row in df_frame.iterrows():
                u_raw = row["X (pixel)"]
                v_raw = row["Y (pixel)"]
                query_idx = int(row["Query Index"])
                
                # Bilinear interpolation for sub-pixel accuracy
                u_floor = int(np.floor(u_raw))
                v_floor = int(np.floor(v_raw))
                u_frac = u_raw - u_floor
                v_frac = v_raw - v_floor
                
                # Check bounds (need 2x2 region for interpolation)
                if 0 <= v_floor < depth_scaled.shape[0]-1 and 0 <= u_floor < depth_scaled.shape[1]-1:
                    # Get 2x2 depth patch
                    Z00 = depth_scaled[v_floor, u_floor]
                    Z01 = depth_scaled[v_floor, u_floor+1]
                    Z10 = depth_scaled[v_floor+1, u_floor]
                    Z11 = depth_scaled[v_floor+1, u_floor+1]
                    
                    # Bilinear interpolation
                    Z = (Z00 * (1-u_frac) * (1-v_frac) +
                         Z01 * u_frac * (1-v_frac) +
                         Z10 * (1-u_frac) * v_frac +
                         Z11 * u_frac * v_frac)
                    
                    # Validate depth
                    if Z > 0.01:  # Depth must be at least 1cm
                        pixel = np.array([u_raw, v_raw, 1])
                        XYZ = Z * (K_inv @ pixel)
                        X, Y, Z_final = XYZ
                        output_data.append([frame_id, query_idx, u_raw, v_raw, X, Y, Z_final])
        
        # Save 3D points
        output_csv = os.path.join(self.config['data_dir'], 'points_3d.csv')
        out_df = pd.DataFrame(output_data, columns=["Frame", "Query Index", "u", "v", "X (m)", "Y (m)", "Z (m)"])
        out_df.to_csv(output_csv, index=False)
        print(f"💾 Đã lưu tọa độ 3D: {output_csv}")
        
        return output_csv
    
    def estimate_camera_motion(self, points_3d_csv):
        """Ước lượng chuyển động camera"""
        print("📷 Tính toán camera motion...")
        
        # Load 3D points
        df = pd.read_csv(points_3d_csv)
        
        # World coordinates of fixed points (relative to first point)
        P_world = np.array([
            [-0.009, -0.000, -0.001],
            [-0.015, -0.008, 0.009],
            [-0.000, -0.027, 0.000],
            [0.003, -0.008, 0.009],
            [-0.011, -0.018, 0.008],
            [-0.009, -0.036, -0.002],
            [-0.005, -0.040, -0.000],
            [-0.002, -0.023, 0.003],
        ])
        
        # Find query indices
        frame0 = df[df["Frame"] == 0]
        selected_indices = list(range(min(8, len(frame0))))
        
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
            
            if len(pts) != len(selected_indices):
                continue
            
            pts_cam = np.array(pts)
            R, t = self.compute_rigid_transform(P_world[:len(pts)], pts_cam)
            cam_pos = -R.T @ t
            camera_positions.append([f, *cam_pos])
        
        # Calculate displacement
        camera_positions = np.array(camera_positions)
        positions = camera_positions[:, 1:]
        displacements = np.diff(positions, axis=0)
        distances = np.linalg.norm(displacements, axis=1)
        
        # Save results
        result = pd.DataFrame(camera_positions, columns=["Frame", "Camera_X", "Camera_Y", "Camera_Z"])
        result["Displacement_X"] = np.append([[0, 0, 0]], displacements, axis=0)[:, 0]
        result["Displacement_Y"] = np.append([[0, 0, 0]], displacements, axis=0)[:, 1]
        result["Displacement_Z"] = np.append([[0, 0, 0]], displacements, axis=0)[:, 2]
        result["Distance"] = np.append([0], distances)
        
        output_csv = os.path.join(self.config['data_dir'], 'camera_motion.csv')
        result.to_csv(output_csv, index=False)
        
        total_distance = result["Distance"].sum()
        print(f"📏 Tổng độ dịch chuyển camera: {total_distance:.4f} mét")
        print(f"💾 Đã lưu camera motion: {output_csv}")
        
        return output_csv
    
    def compute_point_velocity(self, points_3d_csv, fps=30):
        """Tính vận tốc thực của các điểm trong không gian 3D"""
        print("🏃 Tính vận tốc 3D của các điểm...")
        
        # Load 3D points
        df = pd.read_csv(points_3d_csv)
        
        query_indices = sorted(df["Query Index"].unique())
        time_per_frame = 1.0 / fps
        
        results = []
        
        for query_idx in query_indices:
            df_query = df[df["Query Index"] == query_idx].sort_values("Frame")
            
            if len(df_query) < 2:
                continue
            
            frames = df_query["Frame"].values
            positions = df_query[["X (m)", "Y (m)", "Z (m)"]].values
            
            # Tính displacement và velocity
            displacements = np.diff(positions, axis=0)
            distances = np.linalg.norm(displacements, axis=1)
            frame_gaps = np.diff(frames)
            time_gaps = frame_gaps * time_per_frame
            velocities = distances / time_gaps
            
            velocity_x = displacements[:, 0] / time_gaps
            velocity_y = displacements[:, 1] / time_gaps
            velocity_z = displacements[:, 2] / time_gaps
            
            for i in range(len(velocities)):
                frame_current = frames[i + 1]
                results.append({
                    'Query Index': query_idx,
                    'Frame': frame_current,
                    'X (m)': positions[i + 1, 0],
                    'Y (m)': positions[i + 1, 1],
                    'Z (m)': positions[i + 1, 2],
                    'Distance (m)': distances[i],
                    'Velocity (m/s)': velocities[i],
                    'Velocity_X (m/s)': velocity_x[i],
                    'Velocity_Y (m/s)': velocity_y[i],
                    'Velocity_Z (m/s)': velocity_z[i],
                })
            
            avg_velocity = np.mean(velocities)
            max_velocity = np.max(velocities)
            total_distance = np.sum(distances)
            print(f"  Query {query_idx}: avg={avg_velocity:.4f} m/s, max={max_velocity:.4f} m/s, dist={total_distance:.4f} m")
        
        # Save results
        result_df = pd.DataFrame(results)
        output_csv = os.path.join(self.config['data_dir'], 'velocity_3d.csv')
        result_df.to_csv(output_csv, index=False)
        print(f"💾 Đã lưu vận tốc 3D: {output_csv}")
        
        return output_csv
    
    @staticmethod
    def compute_rigid_transform(A, B):
        """Tính rigid transform từ A sang B"""
        centroid_A = A.mean(axis=0)
        centroid_B = B.mean(axis=0)
        AA = A - centroid_A
        BB = B - centroid_B
        H = AA.T @ BB
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        t = centroid_B - R @ centroid_A
        return R, t
    
    def run_full_pipeline(self, video_path):
        """Chạy pipeline đầy đủ"""
        print("=" * 60)
        print("🚀 CHẠY PIPELINE ĐẦY ĐỦ")
        print("=" * 60)
        
        self.setup_directories()
        
        # Step 1: Extract frames (optional)
        if self.config['save_frames']:
            self.extract_frames(video_path)
        
        # Step 2: Run tracking
        pred_tracks, pred_visibility, video = self.run_tracking(video_path)
        csv_tracking = os.path.join(self.config['data_dir'], 'tracked_points.csv')
        
        # Step 3: Estimate depth and 3D coordinates
        if self.config['use_depth']:
            csv_3d = self.estimate_depth_and_3d(video_path, csv_tracking)
            
            # Step 4: Compute point velocity in 3D
            csv_velocity = self.compute_point_velocity(csv_3d, fps=30)
            
            # Step 5: Estimate camera motion
            csv_camera = self.estimate_camera_motion(csv_3d)
        
        print("=" * 60)
        print("✅ PIPELINE HOÀN TẤT!")
        print(f"📁 Kết quả lưu tại: {self.config['output_dir']}")
        print("=" * 60)
        
        return {
            'tracking_csv': csv_tracking,
            '3d_csv': csv_3d if self.config['use_depth'] else None,
            'velocity_csv': csv_velocity if self.config['use_depth'] else None,
            'camera_csv': csv_camera if self.config['use_depth'] else None,
        }
    
    def run_webcam_mode(self):
        """Chạy tracking real-time từ webcam"""
        print("📹 Khởi động webcam tracking...")
        
        # Load online model
        print("🔄 Đang load CoTracker Online model...")
        try:
            model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(self.device)
        except Exception as e:
            print(f"❌ Lỗi khi load model: {e}")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Không thể mở webcam")
            return
        
        # Settings
        FRAME_WIDTH = 320
        FRAME_HEIGHT = 240
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        # State
        selected_point = None
        point_selected = False
        query_point_tensor = None
        trajectory = []
        window_frames = []
        is_first_step = True
        step = 8
        
        prev_time = cv2.getTickCount()
        fps = 0
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_point, point_selected, query_point_tensor, trajectory, is_first_step
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_point = (x, y)
                point_selected = True
                query_point_tensor = torch.tensor([[0., x, y]], device=self.device)[None]
                trajectory.clear()
                is_first_step = True
                print(f"🖱️ Đã chọn điểm: ({x}, {y})")
        
        cv2.namedWindow("CoTracker Webcam - Click to Track")
        cv2.setMouseCallback("CoTracker Webcam - Click to Track", mouse_callback)
        
        print("📌 Bấm chuột trái để chọn điểm. Nhấn R để reset. Nhấn Q để thoát.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            
            # Calculate FPS
            curr_time = cv2.getTickCount()
            time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
            fps = 0.9 * fps + 0.1 * (1 / time_diff) if time_diff > 0 else fps
            prev_time = curr_time
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            window_frames.append(frame_rgb)
            
            # Limit buffer size
            if len(window_frames) > step * 4:
                window_frames = window_frames[-step * 4:]
            
            if point_selected and len(window_frames) >= step * 2:
                video_chunk = np.stack(window_frames[-step * 2:])
                video_tensor = torch.tensor(video_chunk).float().permute(0, 3, 1, 2)[None].to(self.device)
                
                with torch.no_grad():
                    pred_tracks, pred_visibility = model(
                        video_tensor,
                        queries=query_point_tensor if is_first_step else None,
                        is_first_step=is_first_step,
                    )
                is_first_step = False
                
                if pred_tracks is not None:
                    last_point = pred_tracks[0, -1, 0].detach().cpu().numpy()
                    x, y = int(last_point[0]), int(last_point[1])
                    trajectory.append((x, y))
                    
                    # Draw trajectory
                    for i in range(1, len(trajectory)):
                        cv2.line(frame, trajectory[i-1], trajectory[i], (0, 255, 0), 2)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            
            # Draw UI
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "Q: Quit | R: Reset | Click: Track", (10, FRAME_HEIGHT - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("CoTracker Webcam - Click to Track", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                selected_point = None
                point_selected = False
                query_point_tensor = None
                trajectory.clear()
                is_first_step = True
                print("🔄 Reset")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Đã thoát webcam mode")


def main():
    parser = argparse.ArgumentParser(description="CoTracker Pipeline - Tích hợp đầy đủ")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="full", 
                       choices=["full", "tracking", "depth", "camera", "webcam"],
                       help="Chế độ chạy: full (tất cả), tracking (chỉ tracking), depth (tracking+3D), camera (tracking+3D+camera), webcam (real-time)")
    
    # Input
    parser.add_argument("--video_path", type=str, default="../assets/apple.mp4",
                       help="Đường dẫn video")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="pipeline_output",
                       help="Thư mục output")
    
    # Tracking settings
    parser.add_argument("--grid_size", type=int, default=0,
                       help="Kích thước grid (0 = dùng query points)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Đường dẫn checkpoint model")
    
    # Processing options
    parser.add_argument("--save_frames", action="store_true",
                       help="Lưu frames từ video")
    parser.add_argument("--no_depth", action="store_true",
                       help="Tắt depth estimation")
    parser.add_argument("--no_visualize", action="store_true",
                       help="Tắt visualization")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = CoTrackerPipeline()
    pipeline.config['output_dir'] = args.output_dir
    pipeline.config['frames_dir'] = os.path.join(args.output_dir, 'frames')
    pipeline.config['videos_dir'] = os.path.join(args.output_dir, 'videos')
    pipeline.config['data_dir'] = os.path.join(args.output_dir, 'data')
    pipeline.config['save_frames'] = args.save_frames
    pipeline.config['use_depth'] = not args.no_depth
    pipeline.config['visualize'] = not args.no_visualize
    pipeline.config['checkpoint'] = args.checkpoint
    
    # Run based on mode
    if args.mode == "webcam":
        pipeline.run_webcam_mode()
    
    elif args.mode == "full":
        if not os.path.exists(args.video_path):
            print(f"❌ Không tìm thấy video: {args.video_path}")
            return
        pipeline.run_full_pipeline(args.video_path)
    
    elif args.mode == "tracking":
        if not os.path.exists(args.video_path):
            print(f"❌ Không tìm thấy video: {args.video_path}")
            return
        pipeline.setup_directories()
        pipeline.run_tracking(args.video_path, grid_size=args.grid_size if args.grid_size > 0 else None)
    
    elif args.mode == "depth":
        if not os.path.exists(args.video_path):
            print(f"❌ Không tìm thấy video: {args.video_path}")
            return
        pipeline.setup_directories()
        pipeline.run_tracking(args.video_path)
        csv_tracking = os.path.join(pipeline.config['data_dir'], 'tracked_points.csv')
        pipeline.estimate_depth_and_3d(args.video_path, csv_tracking)
    
    elif args.mode == "camera":
        if not os.path.exists(args.video_path):
            print(f"❌ Không tìm thấy video: {args.video_path}")
            return
        pipeline.setup_directories()
        pipeline.run_tracking(args.video_path)
        csv_tracking = os.path.join(pipeline.config['data_dir'], 'tracked_points.csv')
        csv_3d = pipeline.estimate_depth_and_3d(args.video_path, csv_tracking)
        pipeline.estimate_camera_motion(csv_3d)


if __name__ == "__main__":
    main()
