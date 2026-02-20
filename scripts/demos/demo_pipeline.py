"""
Demo nhanh Pipeline
Chạy file này để test pipeline
"""

import os
import sys

# Add parent directories to path để import pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def print_banner():
    print("=" * 70)
    print("  🚀 COTRACKER PIPELINE - DEMO NHANH")
    print("=" * 70)
    print()

def check_requirements():
    print("📋 Kiểm tra requirements...")
    
    # Check Python
    import sys
    print(f"  ✓ Python {sys.version.split()[0]}")
    
    # Check packages
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except:
        print("  ✗ PyTorch chưa cài đặt")
        return False
    
    try:
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__}")
    except:
        print("  ✗ OpenCV chưa cài đặt")
        return False
    
    try:
        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")
    except:
        print("  ✗ NumPy chưa cài đặt")
        return False
    
    try:
        import pandas
        print(f"  ✓ Pandas {pandas.__version__}")
    except:
        print("  ✗ Pandas chưa cài đặt")
        return False
    
    print()
    return True

def show_menu():
    print("🎯 CHỌN CHẾ ĐỘ:")
    print()
    print("  1. Tracking Only (nhanh nhất)")
    print("     └─ Chỉ tracking điểm trong video")
    print()
    print("  2. Full Pipeline (đầy đủ)")
    print("     └─ Tracking + Depth + 3D + Camera Motion")
    print()
    print("  3. Webcam Real-time")
    print("     └─ Tracking từ webcam")
    print()
    print("  4. Xem hướng dẫn")
    print()
    print("  5. Thoát")
    print()

def run_tracking():
    print("\n🎬 TRACKING MODE")
    print("-" * 70)
    
    video_path = input("📹 Nhập đường dẫn video (Enter = demo): ").strip()
    if not video_path:
        video_path = "assets/apple.mp4"
    
    grid_size = input("📐 Grid size (Enter = 10): ").strip()
    if not grid_size:
        grid_size = "10"
    
    cmd = f'python ../../pipeline/pipeline.py --video_path "{video_path}" --mode tracking --grid_size {grid_size}'
    print(f"\n💻 Chạy: {cmd}\n")
    os.system(cmd)

def run_full():
    print("\n🎬 FULL PIPELINE MODE")
    print("-" * 70)
    
    video_path = input("📹 Nhập đường dẫn video (Enter = demo): ").strip()
    if not video_path:
        video_path = "../../assets/apple.mp4"
    
    cmd = f'python ../../pipeline/pipeline.py --video_path "{video_path}" --mode full'
    print(f"\n💻 Chạy: {cmd}\n")
    os.system(cmd)

def run_webcam():
    print("\n🎬 WEBCAM MODE")
    print("-" * 70)
    print("📌 Controls:")
    print("  - Click chuột: Chọn điểm cần track")
    print("  - R: Reset")
    print("  - Q: Thoát")
    print()
    input("Nhấn Enter để bắt đầu...")
    
    cmd = 'python ../../pipeline/pipeline.py --mode webcam'
    print(f"\n💻 Chạy: {cmd}\n")
    os.system(cmd)

def show_guide():
    print("\n📚 HƯỚNG DẪN NHANH")
    print("-" * 70)
    print()
    print("1. TRACKING ONLY:")
    print("   python pipeline.py --video_path video.mp4 --mode tracking")
    print()
    print("2. FULL PIPELINE:")
    print("   python pipeline.py --video_path video.mp4 --mode full")
    print()
    print("3. WEBCAM:")
    print("   python pipeline.py --mode webcam")
    print()
    print("4. CÁC TÙY CHỌN:")
    print("   --grid_size 20          # Grid 20x20")
    print("   --no_depth              # Tắt depth estimation")
    print("   --save_frames           # Lưu frames")
    print("   --output_dir results    # Custom output folder")
    print()
    print("📖 Chi tiết: Xem PIPELINE_GUIDE.md")
    print()
    input("Nhấn Enter để tiếp tục...")

def main():
    print_banner()
    
    if not check_requirements():
        print("\n❌ Thiếu dependencies!")
        print("📦 Chạy: pip install -r requirements.txt")
        return
    
    while True:
        show_menu()
        choice = input("👉 Chọn (1-5): ").strip()
        
        if choice == "1":
            run_tracking()
        elif choice == "2":
            run_full()
        elif choice == "3":
            run_webcam()
        elif choice == "4":
            show_guide()
        elif choice == "5":
            print("\n👋 Tạm biệt!")
            break
        else:
            print("\n❌ Lựa chọn không hợp lệ!")
        
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Đã dừng!")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
