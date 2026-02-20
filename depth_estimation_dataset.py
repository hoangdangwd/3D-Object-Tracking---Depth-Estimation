import torch
import cv2
import numpy as np
from pathlib import Path
import time
import pandas as pd
from collections import Counter
import re
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from tqdm import tqdm
import random


def load_midas_model():
    """Load MiDaS model một lần duy nhất"""
    print("🔄 Đang load MiDaS model...")
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDAS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    
    print(f"✅ Model đã load xong (Device: {device})")
    return midas, transform, device


def extract_ground_truth_distance(filename):
    """
    Trích xuất khoảng cách ground truth từ tên file
    Ví dụ: motorblue_10m_down_001.JPG -> 10.0
    """
    match = re.search(r'_(\d+(?:\.\d+)?)m_', filename)
    if match:
        return float(match.group(1))
    return None


def get_class_from_filename(filename):
    """
    Xác định class từ tên file
    motorblue_... -> motorblue
    motorwhite_... -> motorwhite
    Các file khác -> dựa vào folder
    """
    if filename.startswith('motorblue'):
        return 'motorblue'
    elif filename.startswith('motorwhite'):
        return 'motorwhite'
    return None


def load_detection_label(label_path):
    """
    Load detection label
    Format: class_id x_center y_center width height
    """
    boxes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x_center, y_center, w, h = map(float, parts)
                    boxes.append({
                        'class': int(cls),
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': w,
                        'height': h
                    })
    except:
        pass
    return boxes


def load_segmentation_label(label_path):
    """
    Load segmentation label
    Format: class_id x1 y1 x2 y2 x3 y3 ...
    """
    polygons = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:  # class + at least 3 points
                    cls = int(float(parts[0]))
                    points = []
                    for i in range(1, len(parts), 2):
                        x = float(parts[i])
                        y = float(parts[i+1])
                        points.append([x, y])
                    polygons.append({
                        'class': cls,
                        'points': points
                    })
    except:
        pass
    return polygons


def estimate_depth_detection(image, boxes, midas, transform, device):
    """
    Estimate depth sử dụng detection labels (tâm bounding box)
    Trả về: depth value và inference time
    """
    if not boxes:
        return None, 0.0
    
    # Chỉ lấy box đầu tiên
    box = boxes[0]
    
    h, w = image.shape[:2]
    x_center_px = int(box['x_center'] * w)
    y_center_px = int(box['y_center'] * h)
    
    # Transform image
    input_batch = transform(image).to(device)
    
    # Inference với timing
    start_time = time.time()
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    inference_time = time.time() - start_time
    
    depth_map = prediction.cpu().numpy()
    
    # Lấy depth value tại tâm box
    depth_value = depth_map[y_center_px, x_center_px]
    
    return depth_value, inference_time


def estimate_depth_segmentation(image, polygons, midas, transform, device):
    """
    Estimate depth sử dụng segmentation labels (tất cả điểm trong polygon)
    Trả về: depth value (mode hoặc mean) và inference time
    """
    if not polygons:
        return None, 0.0
    
    # Chỉ lấy polygon đầu tiên
    polygon = polygons[0]
    points = polygon['points']
    
    h, w = image.shape[:2]
    
    # Convert normalized points to pixel coordinates
    points_px = np.array([[int(p[0] * w), int(p[1] * h)] for p in points], dtype=np.int32)
    
    # Transform image
    input_batch = transform(image).to(device)
    
    # Inference với timing
    start_time = time.time()
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    inference_time = time.time() - start_time
    
    depth_map = prediction.cpu().numpy()
    
    # Tạo mask từ polygon
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points_px], 1)
    
    # Lấy tất cả depth values trong mask
    depth_values = depth_map[mask == 1]
    
    if len(depth_values) == 0:
        return None, inference_time
    
    # Làm tròn đến 2 chữ số thập phân
    depth_values_rounded = np.round(depth_values, 2)
    
    # Tính mode (giá trị xuất hiện nhiều nhất)
    counter = Counter(depth_values_rounded)
    most_common = counter.most_common(2)
    
    # Nếu tất cả giá trị chỉ xuất hiện 1 lần, lấy mean
    if most_common[0][1] == 1:
        depth_value = np.mean(depth_values_rounded)
    else:
        depth_value = most_common[0][0]
    
    return depth_value, inference_time


def depth_to_distance_inverse(depth_value, a, b=0):
    """
    Convert depth value sang distance sử dụng inverse relationship
    distance = a / depth + b
    """
    if depth_value == 0:
        return float('inf')
    return a / depth_value + b


def calibrate_depth_to_distance(depth_values, ground_truth_distances):
    """
    Calibrate để tìm hệ số chuyển đổi từ relative depth sang actual distance
    Sử dụng inverse model: distance = a / depth + b
    
    Returns: (a, b, r2_score)
    """
    if len(depth_values) < 2:
        print("  ⚠️  Không đủ dữ liệu để calibrate")
        return None, None, 0.0
    
    depth_values = np.array(depth_values)
    ground_truth_distances = np.array(ground_truth_distances)
    
    # Loại bỏ các giá trị invalid
    valid_mask = (depth_values > 0) & np.isfinite(depth_values) & np.isfinite(ground_truth_distances)
    depth_values = depth_values[valid_mask]
    ground_truth_distances = ground_truth_distances[valid_mask]
    
    if len(depth_values) < 2:
        return None, None, 0.0
    
    try:
        # Fit inverse model: y = a/x + b
        def inverse_model(x, a, b):
            return a / x + b
        
        # Initial guess
        p0 = [np.mean(depth_values * ground_truth_distances), 0]
        
        # Curve fitting
        params, _ = curve_fit(inverse_model, depth_values, ground_truth_distances, p0=p0, maxfev=10000)
        a, b = params
        
        # Calculate R² score
        predictions = inverse_model(depth_values, a, b)
        r2 = r2_score(ground_truth_distances, predictions)
        
        return a, b, r2
    except Exception as e:
        print(f"  ❌ Calibration failed: {e}")
        return None, None, 0.0


def process_dataset(base_path, output_dir):
    """
    Xử lý toàn bộ dataset và tạo 4 file CSV
    Thực hiện 2 passes:
    - Pass 1: Thu thập dữ liệu và calibrate
    - Pass 2: Tính toán MAE với calibrated depth
    """
    base_path = Path(base_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model một lần duy nhất
    midas, transform, device = load_midas_model()
    
    # Dictionary để lưu kết quả cho 4 classes
    results = {
        'motorblue': [],
        'motorwhite': [],
        'person': [],
        'pot': []
    }
    
    # Dictionary để lưu calibration data
    calibration_data = {
        'motorblue': {'depth_detect': [], 'depth_segment': [], 'gt': []},
        'motorwhite': {'depth_detect': [], 'depth_segment': [], 'gt': []},
        'person': {'depth_detect': [], 'depth_segment': [], 'gt': []},
        'pot': {'depth_detect': [], 'depth_segment': [], 'gt': []}
    }
    
    # Mapping folder names to class
    folder_class_map = {
        'motorcycle': ['motorblue', 'motorwhite'],  # Cần check filename
        'person': ['person'],
        'pot': ['pot']
    }
    
    # YOLO class IDs (COCO dataset)
    yolo_class_ids = {
        'motorcycle': 3,
        'person': 0,
        'pot': 58  # potted plant
    }
    
    print("\n" + "="*80)
    print("PASS 1: THU THẬP DỮ LIỆU VÀ CALIBRATION")
    print("="*80)
    print("💡 Sử dụng sampling: 100 ảnh/class để calibration (nhanh hơn)")
    
    # PASS 1: Thu thập tất cả depth values và ground truth
    for folder_name, possible_classes in folder_class_map.items():
        image_folder = base_path / "image" / folder_name
        detect_folder = base_path / "label_detection" / folder_name
        segment_folder = base_path / "label_segmentation" / folder_name
        
        if not image_folder.exists():
            print(f"⚠️  Không tìm thấy folder: {image_folder}")
            continue
        
        print(f"\n📁 Đang thu thập dữ liệu từ folder: {folder_name}")
        
        all_images = sorted(list(image_folder.glob("*.JPG")) + list(image_folder.glob("*.jpg")))
        
        # Sampling: chỉ lấy tối đa 100 ảnh mỗi class cho calibration
        if len(all_images) > 100:
            import random
            random.seed(42)
            images = random.sample(all_images, 100)
            print(f"  📊 Sampling: {len(images)}/{len(all_images)} ảnh")
        else:
            images = all_images
            print(f"  📊 Sử dụng tất cả: {len(images)} ảnh")
        
            print(f"  📊 Sử dụng tất cả: {len(images)} ảnh")
        
        processed = 0
        skipped = 0
        
        for img_path in tqdm(images, desc=f"  {folder_name}"):
            try:
                # Xác định class
                if folder_name == 'motorcycle':
                    class_name = get_class_from_filename(img_path.name)
                    if class_name is None:
                        skipped += 1
                        continue
                else:
                    class_name = possible_classes[0]
                
                # Extract ground truth
                gt_distance = extract_ground_truth_distance(img_path.name)
                if gt_distance is None:
                    skipped += 1
                    continue
                
                # Load labels
                label_name = img_path.stem + ".txt"
                detect_label_path = detect_folder / label_name
                segment_label_path = segment_folder / label_name
                
                if not detect_label_path.exists() or not segment_label_path.exists():
                    skipped += 1
                    continue
                
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    skipped += 1
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Load labels
                boxes = load_detection_label(detect_label_path)
                polygons = load_segmentation_label(segment_label_path)
                
                if not boxes or not polygons:
                    skipped += 1
                    continue
                
                # Estimate depth
                depth_detect, _ = estimate_depth_detection(image_rgb, boxes, midas, transform, device)
                depth_segment, _ = estimate_depth_segmentation(image_rgb, polygons, midas, transform, device)
                
                if depth_detect is not None and depth_segment is not None:
                    calibration_data[class_name]['depth_detect'].append(depth_detect)
                    calibration_data[class_name]['depth_segment'].append(depth_segment)
                    calibration_data[class_name]['gt'].append(gt_distance)
                    processed += 1
                else:
                    skipped += 1
                    
            except Exception as e:
                skipped += 1
                continue
        
        print(f"  ✅ Thu thập: {processed} ảnh | Bỏ qua: {skipped} ảnh")
    
    # Calibration cho từng class
    print("\n" + "="*80)
    print("THỰC HIỆN CALIBRATION")
    print("="*80)
    
    calibration_params = {}
    
    for class_name in ['motorblue', 'motorwhite', 'person', 'pot']:
        data = calibration_data[class_name]
        
        if len(data['gt']) < 2:
            print(f"\n⚠️  Class {class_name}: Không đủ dữ liệu để calibrate")
            calibration_params[class_name] = {
                'detect': (None, None, 0.0),
                'segment': (None, None, 0.0)
            }
            continue
        
        print(f"\n📊 Class: {class_name.upper()}")
        print(f"  Số mẫu: {len(data['gt'])}")
        
        # Calibrate cho Detection
        a_det, b_det, r2_det = calibrate_depth_to_distance(data['depth_detect'], data['gt'])
        if a_det is not None:
            print(f"  ✅ Detection: distance = {a_det:.2f} / depth + {b_det:.2f}")
            print(f"     R² score: {r2_det:.4f}")
        else:
            print(f"  ❌ Detection calibration failed")
        
        # Calibrate cho Segmentation
        a_seg, b_seg, r2_seg = calibrate_depth_to_distance(data['depth_segment'], data['gt'])
        if a_seg is not None:
            print(f"  ✅ Segmentation: distance = {a_seg:.2f} / depth + {b_seg:.2f}")
            print(f"     R² score: {r2_seg:.4f}")
        else:
            print(f"  ❌ Segmentation calibration failed")
        
        calibration_params[class_name] = {
            'detect': (a_det, b_det, r2_det),
            'segment': (a_seg, b_seg, r2_seg)
        }
    
    # PASS 2: Tính toán MAE với calibrated depth
    print("\n" + "="*80)
    print("PASS 2: TÍNH TOÁN MAE VỚI CALIBRATED DEPTH")
    print("="*80)
    
    total_processed = 0
    
    for folder_name, possible_classes in folder_class_map.items():
        image_folder = base_path / "image" / folder_name
        detect_folder = base_path / "label_detection" / folder_name
        segment_folder = base_path / "label_segmentation" / folder_name
        
        if not image_folder.exists():
            continue
        
        print(f"\n📁 Đang xử lý folder: {folder_name}")
        
        images = sorted(list(image_folder.glob("*.JPG")) + list(image_folder.glob("*.jpg")))
        
        for img_path in images:
            # Xác định class
            if folder_name == 'motorcycle':
                class_name = get_class_from_filename(img_path.name)
                if class_name is None:
                    continue
            else:
                class_name = possible_classes[0]
            
            # Extract ground truth
            gt_distance = extract_ground_truth_distance(img_path.name)
            if gt_distance is None:
                print(f"  ⚠️  Không thể extract ground truth từ: {img_path.name}")
                continue
            
            # Load labels
            label_name = img_path.stem + ".txt"
            detect_label_path = detect_folder / label_name
            segment_label_path = segment_folder / label_name
            
            if not detect_label_path.exists() or not segment_label_path.exists():
                print(f"  ⚠️  Thiếu labels cho: {img_path.name}")
                continue
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  ❌ Không thể load ảnh: {img_path.name}")
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load labels
            boxes = load_detection_label(detect_label_path)
            polygons = load_segmentation_label(segment_label_path)
            
            if not boxes or not polygons:
                print(f"  ⚠️  Labels rỗng cho: {img_path.name}")
                continue
            
            # Estimate depth với Detection
            depth_detect, time_detect = estimate_depth_detection(
                image_rgb, boxes, midas, transform, device
            )
            
            # Estimate depth với Segmentation
            depth_segment, time_segment = estimate_depth_segmentation(
                image_rgb, polygons, midas, transform, device
            )
            
            if depth_detect is None or depth_segment is None:
                print(f"  ❌ Không thể estimate depth cho: {img_path.name}")
                continue
            
            # Get calibration parameters
            a_det, b_det, r2_det = calibration_params[class_name]['detect']
            a_seg, b_seg, r2_seg = calibration_params[class_name]['segment']
            
            # Convert depth to distance sử dụng calibrated parameters
            if a_det is not None:
                dist_detect = depth_to_distance_inverse(depth_detect, a_det, b_det)
            else:
                dist_detect = depth_detect  # Fallback to raw depth
            
            if a_seg is not None:
                dist_segment = depth_to_distance_inverse(depth_segment, a_seg, b_seg)
            else:
                dist_segment = depth_segment  # Fallback to raw depth
            
            # Calculate MAE
            mae_detect = abs(dist_detect - gt_distance)
            mae_segment = abs(dist_segment - gt_distance)
            
            # Lưu kết quả
            result = {
                'sample_name': img_path.name,
                'MAE_Detection': round(mae_detect, 4),
                'MAE_Segmentation': round(mae_segment, 4),
                'Time_Detection_ms': round(time_detect * 1000, 2),
                'Time_Segmentation_ms': round(time_segment * 1000, 2)
            }
            
            results[class_name].append(result)
            total_processed += 1
            
            # In kết quả
            print(f"  ✅ {img_path.name}")
            print(f"     GT: {gt_distance}m | Det: {dist_detect:.2f}m (MAE: {mae_detect:.4f}) | "
                  f"Seg: {dist_segment:.2f}m (MAE: {mae_segment:.4f})")
            print(f"     Time - Det: {time_detect*1000:.2f}ms | Seg: {time_segment*1000:.2f}ms")
    
    print(f"\n✅ Đã xử lý {total_processed} ảnh")
    
    # Lưu kết quả ra CSV
    print("\n" + "="*80)
    print("LƯU KẾT QUẢ VÀO CSV FILES")
    print("="*80)
    
    for class_name, data in results.items():
        if not data:
            print(f"  ⚠️  Không có dữ liệu cho class: {class_name}")
            continue
        
        df = pd.DataFrame(data)
        csv_path = output_dir / f"depth_results_{class_name}.csv"
        df.to_csv(csv_path, index=False)
        
        # Get calibration info
        a_det, b_det, r2_det = calibration_params[class_name]['detect']
        a_seg, b_seg, r2_seg = calibration_params[class_name]['segment']
        
        print(f"\n📊 Class: {class_name.upper()}")
        print(f"  • Số mẫu: {len(data)}")
        if a_det is not None:
            print(f"  • Calibration Detection: dist = {a_det:.2f} / depth + {b_det:.2f} (R²={r2_det:.4f})")
        if a_seg is not None:
            print(f"  • Calibration Segmentation: dist = {a_seg:.2f} / depth + {b_seg:.2f} (R²={r2_seg:.4f})")
        print(f"  • MAE Detection - Mean: {df['MAE_Detection'].mean():.4f} ± {df['MAE_Detection'].std():.4f}")
        print(f"  • MAE Segmentation - Mean: {df['MAE_Segmentation'].mean():.4f} ± {df['MAE_Segmentation'].std():.4f}")
        print(f"  • Time Detection - Mean: {df['Time_Detection_ms'].mean():.2f}ms")
        print(f"  • Time Segmentation - Mean: {df['Time_Segmentation_ms'].mean():.2f}ms")
        print(f"  • Đã lưu: {csv_path}")
        
        # In preview bảng
        print("\n  Preview (5 mẫu đầu tiên):")
        print(df.head().to_string(index=False))
    
    print("\n" + "="*80)
    print("HOÀN TẤT")
    print("="*80)


def main():
    base_path = "data/input/GNSS"
    output_dir = "depth_estimation_results"
    
    process_dataset(base_path, output_dir)


if __name__ == "__main__":
    main()
