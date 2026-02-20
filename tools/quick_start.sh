#!/bin/bash
# Quick start script for CoTracker Pipeline (Linux/Mac)

echo "============================================"
echo "  CoTracker Pipeline - Quick Start"
echo "============================================"
echo ""

function show_menu {
    echo "Chọn mode:"
    echo "  1. Full Pipeline (Tracking + 3D + Camera)"
    echo "  2. Tracking Only"
    echo "  3. Webcam Real-time"
    echo "  4. Custom"
    echo "  5. Exit"
    echo ""
    read -p "Nhập lựa chọn (1-5): " choice
    
    case $choice in
        1) run_full ;;
        2) run_tracking ;;
        3) run_webcam ;;
        4) run_custom ;;
        5) exit 0 ;;
        *) echo "Lựa chọn không hợp lệ!"; show_menu ;;
    esac
}

function run_full {
    echo ""
    read -p "Nhập đường dẫn video (VD: ../assets/apple.mp4): " video
    echo ""
    echo "Chạy Full Pipeline..."
    python ../pipeline/pipeline.py --video_path "$video" --mode full
    done_message
}

function run_tracking {
    echo ""
    read -p "Nhập đường dẫn video: " video
    read -p "Nhập grid size (mặc định 10): " grid
    grid=${grid:-10}
    echo ""
    echo "Chạy Tracking..."
    python ../pipeline/pipeline.py --video_path "$video" --mode tracking --grid_size $grid
    done_message
}

function run_webcam {
    echo ""
    echo "Khởi động webcam..."
    echo "(Click chuột để chọn điểm, R=reset, Q=thoát)"
    python ../pipeline/pipeline.py --mode webcam
    done_message
}

function run_custom {
    echo ""
    read -p "Video path: " video
    read -p "Mode (full/tracking/depth/camera): " mode
    read -p "Output dir (Enter = mặc định): " output
    
    if [ -z "$output" ]; then
        python ../pipeline/pipeline.py --video_path "$video" --mode "$mode"
    else
        python ../pipeline/pipeline.py --video_path "$video" --mode "$mode" --output_dir "$output"
    fi
    done_message
}

function done_message {
    echo ""
    echo "============================================"
    echo "  Hoàn tất!"
    echo "============================================"
    echo ""
    read -p "Nhấn Enter để tiếp tục..."
    show_menu
}

# Main
show_menu
