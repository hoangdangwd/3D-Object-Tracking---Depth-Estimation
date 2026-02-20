"""
Script: compute_velocity_3d.py
Description: Tính vận tốc thực trong không gian 3D cho các điểm được track

Usage:
    python compute_velocity_3d.py --input points_3d.csv --fps 30

Output:
    - velocity_3d.csv: Vận tốc của từng điểm (m/s)
    - Visualization: Plot trajectory và velocity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

def compute_3d_velocity(points_3d_csv, fps=30, output_dir="../../data/output/velocity"):
    """
    Tính vận tốc 3D từ file CSV chứa tọa độ 3D
    
    Args:
        points_3d_csv: Path to CSV file with 3D coordinates
        fps: Video frame rate (frames per second)
        output_dir: Output directory
    
    Returns:
        DataFrame with velocity information
    """
    
    # Đọc dữ liệu
    if not os.path.exists(points_3d_csv):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {points_3d_csv}")
    
    df = pd.read_csv(points_3d_csv)
    print(f"📊 Đã load {len(df)} rows từ {points_3d_csv}")
    
    # Kiểm tra columns
    required_cols = ["Frame", "Query Index", "X (m)", "Y (m)", "Z (m)"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"❌ CSV phải có columns: {required_cols}")
    
    # Tính time per frame
    time_per_frame = 1.0 / fps  # seconds
    
    # Nhóm theo Query Index
    query_indices = sorted(df["Query Index"].unique())
    print(f"🎯 Tìm thấy {len(query_indices)} điểm được track")
    
    results = []
    
    for query_idx in query_indices:
        df_query = df[df["Query Index"] == query_idx].sort_values("Frame")
        
        if len(df_query) < 2:
            print(f"⚠️ Query {query_idx}: Không đủ dữ liệu (chỉ có {len(df_query)} frames)")
            continue
        
        frames = df_query["Frame"].values
        positions = df_query[["X (m)", "Y (m)", "Z (m)"]].values
        
        # Tính displacement giữa các frames liên tiếp
        displacements = np.diff(positions, axis=0)  # [N-1, 3]
        
        # Tính khoảng cách Euclidean
        distances = np.linalg.norm(displacements, axis=1)  # [N-1]
        
        # Tính vận tốc (m/s)
        # Giả định frames liên tiếp, nếu không liên tiếp cần điều chỉnh
        frame_gaps = np.diff(frames)
        time_gaps = frame_gaps * time_per_frame
        velocities = distances / time_gaps
        
        # Tính vận tốc theo từng trục
        velocity_x = displacements[:, 0] / time_gaps
        velocity_y = displacements[:, 1] / time_gaps
        velocity_z = displacements[:, 2] / time_gaps
        
        # Lưu kết quả cho từng frame (frame thứ 2 trở đi)
        for i in range(len(velocities)):
            frame_current = frames[i + 1]
            results.append({
                'Query Index': query_idx,
                'Frame': frame_current,
                'X (m)': positions[i + 1, 0],
                'Y (m)': positions[i + 1, 1],
                'Z (m)': positions[i + 1, 2],
                'Distance (m)': distances[i],
                'Time_gap (s)': time_gaps[i],
                'Velocity (m/s)': velocities[i],
                'Velocity_X (m/s)': velocity_x[i],
                'Velocity_Y (m/s)': velocity_y[i],
                'Velocity_Z (m/s)': velocity_z[i],
            })
        
        # Thống kê
        avg_velocity = np.mean(velocities)
        max_velocity = np.max(velocities)
        total_distance = np.sum(distances)
        
        print(f"\n📌 Query {query_idx}:")
        print(f"   - Số frames: {len(df_query)}")
        print(f"   - Quãng đường: {total_distance:.4f} m")
        print(f"   - Vận tốc trung bình: {avg_velocity:.4f} m/s")
        print(f"   - Vận tốc tối đa: {max_velocity:.4f} m/s")
    
    # Tạo DataFrame
    result_df = pd.DataFrame(results)
    
    # Lưu kết quả
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "velocity_3d.csv")
    result_df.to_csv(output_csv, index=False)
    print(f"\n✅ Đã lưu kết quả vào: {output_csv}")
    
    # Visualization
    visualize_velocity(result_df, df, output_dir)
    
    return result_df


def visualize_velocity(velocity_df, position_df, output_dir):
    """Visualize trajectory và velocity"""
    
    query_indices = sorted(velocity_df["Query Index"].unique())
    
    # 1. Trajectory 3D với màu theo vận tốc
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: 3D Trajectory colored by velocity
    ax1 = fig.add_subplot(131, projection='3d')
    for query_idx in query_indices:
        df_q = velocity_df[velocity_df["Query Index"] == query_idx]
        scatter = ax1.scatter(
            df_q["X (m)"], 
            df_q["Y (m)"], 
            df_q["Z (m)"],
            c=df_q["Velocity (m/s)"],
            cmap='plasma',
            s=50,
            label=f"Query {query_idx}"
        )
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D Trajectory (colored by velocity)")
    plt.colorbar(scatter, ax=ax1, label="Velocity (m/s)")
    
    # Plot 2: Velocity over time
    ax2 = fig.add_subplot(132)
    for query_idx in query_indices:
        df_q = velocity_df[velocity_df["Query Index"] == query_idx]
        ax2.plot(df_q["Frame"], df_q["Velocity (m/s)"], marker='o', label=f"Query {query_idx}")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.set_title("Velocity over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Velocity components
    ax3 = fig.add_subplot(133)
    if len(query_indices) > 0:
        df_q = velocity_df[velocity_df["Query Index"] == query_indices[0]]
        ax3.plot(df_q["Frame"], df_q["Velocity_X (m/s)"], marker='o', label="V_x")
        ax3.plot(df_q["Frame"], df_q["Velocity_Y (m/s)"], marker='s', label="V_y")
        ax3.plot(df_q["Frame"], df_q["Velocity_Z (m/s)"], marker='^', label="V_z")
        ax3.plot(df_q["Frame"], df_q["Velocity (m/s)"], 'k--', linewidth=2, label="V_total")
        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Velocity (m/s)")
        ax3.set_title(f"Velocity Components (Query {query_indices[0]})")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_img = os.path.join(output_dir, "velocity_visualization.png")
    plt.savefig(output_img, dpi=150, bbox_inches='tight')
    print(f"✅ Đã lưu visualization: {output_img}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tính vận tốc 3D từ tracked points")
    parser.add_argument(
        "--input",
        type=str,
        default="../../data/output/points_3d.csv",
        help="Path to 3D points CSV"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video frame rate (FPS)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../data/output/velocity",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  🚀 TÍNH VẬN TỐC 3D")
    print("=" * 70)
    print(f"📄 Input: {args.input}")
    print(f"🎬 FPS: {args.fps}")
    print(f"📁 Output: {args.output_dir}")
    print()
    
    try:
        result_df = compute_3d_velocity(args.input, args.fps, args.output_dir)
        
        print("\n" + "=" * 70)
        print("✨ HOÀN TẤT!")
        print("=" * 70)
        print(f"📊 Đã tính vận tốc cho {len(result_df)} data points")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
