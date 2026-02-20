"""
Depth Estimation Dataset - Scale Factor Approach V2
Improvements:
1. Stronger MAD threshold (2.5 instead of 2.0)
2. Per-distance-range calibration (separate scales for 1m, 3m, 6m, 10m, 15m)
3. Keep all samples for complete data comparison
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import time
import pandas as pd
import re
from tqdm import tqdm
import random


# ============================================================================
# CONFIGURATION
# ============================================================================
N_CALIB = 100  # Per distance range
MAD_THRESHOLD = 2.5  # Increased from 2.0 for stronger outlier removal

# Paths
BASE_PATH = Path("data/input/GNSS")
IMAGE_PATH = BASE_PATH / "image"
LABEL_DETECT_PATH = BASE_PATH / "label_detection"
LABEL_SEGMENT_PATH = BASE_PATH / "label_segmentation"
OUTPUT_PATH = Path("depth_estimation_results")

CLASS_FOLDERS = ["motorcycle", "person", "pot"]

# Distance ranges for separate calibration
DISTANCE_RANGES = [1.0, 3.0, 6.0, 10.0, 15.0]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_midas_model():
    """Load MiDaS DPT_Large model once"""
    print("🔄 Loading MiDaS DPT_Large...")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.dpt_transform
    
    print(f"✅ MiDaS loaded on {device}")
    return midas, transform, device


def extract_ground_truth_distance(filename):
    """Extract distance from filename: motorblue_10m_down_001.JPG -> 10.0"""
    match = re.search(r'_(\d+(?:\.\d+)?)m_', filename)
    return float(match.group(1)) if match else None


def extract_viewing_angle(filename):
    """Extract viewing angle: motorblue_10m_down_001.JPG -> 'down'"""
    match = re.search(r'_\d+m_(up|down|left|right|front|back)_', filename)
    return match.group(1) if match else None


def get_distance_range(distance):
    """Get the closest distance range"""
    return min(DISTANCE_RANGES, key=lambda x: abs(x - distance))


def load_yolo_detection(label_path, img_width, img_height):
    """Load YOLO detection bbox: [x_min, y_min, x_max, y_max] in pixels"""
    boxes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x_c, y_c, w, h = map(float, parts)
                    
                    x_center = x_c * img_width
                    y_center = y_c * img_height
                    width = w * img_width
                    height = h * img_height
                    
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    
                    boxes.append([x_min, y_min, x_max, y_max])
    except:
        pass
    return boxes


def load_yolo_segmentation(label_path, img_width, img_height):
    """Load YOLO segmentation polygon: [[x1,y1], [x2,y2], ...] in pixels"""
    polygons = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    points = []
                    for i in range(1, len(parts), 2):
                        x = float(parts[i]) * img_width
                        y = float(parts[i+1]) * img_height
                        points.append([x, y])
                    polygons.append(points)
    except:
        pass
    return polygons


def get_depth_at_point(depth_map, u, v):
    """Get depth value at (u, v) with bilinear interpolation"""
    h, w = depth_map.shape
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    
    u0, v0 = int(np.floor(u)), int(np.floor(v))
    u1, v1 = min(u0 + 1, w - 1), min(v0 + 1, h - 1)
    
    du = u - u0
    dv = v - v0
    
    depth = (depth_map[v0, u0] * (1 - du) * (1 - dv) +
             depth_map[v0, u1] * du * (1 - dv) +
             depth_map[v1, u0] * (1 - du) * dv +
             depth_map[v1, u1] * du * dv)
    
    return depth


# ============================================================================
# DEPTH ESTIMATION
# ============================================================================

def estimate_disparity_map(image_path, midas, transform, device):
    """Estimate disparity map from image using MiDaS"""
    image = cv2.imread(str(image_path))
    if image is None:
        return None, 0
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    input_batch = transform(image_rgb)
    if isinstance(input_batch, dict):
        input_batch = input_batch["image"]
    
    if input_batch.dim() == 3:
        input_batch = input_batch.unsqueeze(0)
    
    input_batch = input_batch.to(device)
    
    start_time = time.time()
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    disparity_map = prediction.cpu().numpy()
    inference_time = (time.time() - start_time) * 1000
    
    return disparity_map, inference_time


def disparity_to_inverse_depth(disparity_map):
    """Convert MiDaS disparity to inverse depth (1/Z)"""
    return 1.0 / (disparity_map + 1e-6)


def estimate_depth_detection(inv_depth_map, boxes):
    """Estimate depth at bbox center"""
    if len(boxes) == 0:
        return None
    
    box = boxes[0]
    x_min, y_min, x_max, y_max = box
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    
    inv_depth = get_depth_at_point(inv_depth_map, cx, cy)
    return inv_depth


def estimate_depth_segmentation(inv_depth_map, polygons):
    """Estimate depth for polygon points"""
    if len(polygons) == 0:
        return None
    
    polygon = polygons[0]
    inv_depths = []
    
    for x, y in polygon:
        inv_depth = get_depth_at_point(inv_depth_map, x, y)
        if inv_depth > 0 and np.isfinite(inv_depth):
            inv_depths.append(inv_depth)
    
    if len(inv_depths) == 0:
        return None
    
    return np.median(inv_depths)


# ============================================================================
# CALIBRATION
# ============================================================================

def compute_scale_factor_mad(inv_depth_values, ground_truth_distances, verbose=True):
    """Compute scale factor with MAD outlier removal (stronger threshold)"""
    inv_depth_values = np.array(inv_depth_values)
    ground_truth_distances = np.array(ground_truth_distances)
    
    scales = []
    for z_pred, z_real in zip(inv_depth_values, ground_truth_distances):
        if z_pred > 1e-6:
            scale = z_real / z_pred
            scales.append(scale)
    
    if len(scales) < 3:
        if verbose:
            print(f"  ❌ Only {len(scales)} valid samples (need ≥3)")
        return None, None
    
    scales = np.array(scales)
    
    # MAD outlier removal with stronger threshold
    scale_median = np.median(scales)
    mad = np.median(np.abs(scales - scale_median))
    
    mask = np.abs(scales - scale_median) < MAD_THRESHOLD * mad
    scale_filtered = scales[mask]
    
    if len(scale_filtered) == 0:
        if verbose:
            print("  ⚠️  All scales are outliers, using median")
        scale_filtered = [scale_median]
    
    scale_factor = np.mean(scale_filtered)
    scale_std = np.std(scale_filtered)
    
    stats = {
        'mean': scale_factor,
        'std': scale_std,
        'median': scale_median,
        'mad': mad,
        'n_total': len(scales),
        'n_filtered': len(scale_filtered),
        'outliers_removed': len(scales) - len(scale_filtered)
    }
    
    if verbose:
        print(f"  ✅ Scale: {scale_factor:.6f} ± {scale_std:.6f}")
        print(f"     Valid: {stats['n_filtered']}/{stats['n_total']} (removed {stats['outliers_removed']})")
    
    return scale_factor, stats


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_dataset():
    """Main processing with per-distance-range calibration"""
    
    midas, transform, device = load_midas_model()
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # ====================
    # PASS 1: CALIBRATION PER DISTANCE RANGE
    # ====================
    print("\n" + "="*60)
    print("🔬 PASS 1: PER-DISTANCE CALIBRATION (MAD threshold=2.5)")
    print("="*60)
    
    calibration_params = {}
    
    for class_folder in CLASS_FOLDERS:
        print(f"\n📊 Calibrating class: {class_folder}")
        
        image_folder = IMAGE_PATH / class_folder
        label_detect_folder = LABEL_DETECT_PATH / class_folder
        label_segment_folder = LABEL_SEGMENT_PATH / class_folder
        
        if not image_folder.exists():
            print(f"  ⚠️  Folder not found: {image_folder}")
            continue
        
        # Organize by distance range
        all_images = sorted(list(image_folder.glob("*.JPG")) + list(image_folder.glob("*.jpg")))
        
        images_by_distance = {dist: [] for dist in DISTANCE_RANGES}
        
        for img_path in all_images:
            gt_dist = extract_ground_truth_distance(img_path.name)
            
            if gt_dist is not None:
                dist_range = get_distance_range(gt_dist)
                images_by_distance[dist_range].append(img_path)
        
        # Calibrate separately for each distance range
        calibration_params[class_folder] = {}
        
        for dist_range in DISTANCE_RANGES:
            images = images_by_distance[dist_range]
            
            if len(images) < 3:
                print(f"  ⚠️  Distance {dist_range}m: insufficient data ({len(images)} images)")
                calibration_params[class_folder][dist_range] = {
                    'detect': (None, None),
                    'segment': (None, None)
                }
                continue
            
            # Sample
            if len(images) > N_CALIB:
                sampled = random.sample(images, N_CALIB)
            else:
                sampled = images
            
            print(f"\n  📏 Distance {dist_range}m: {len(sampled)} samples")
            
            # Collect data
            inv_depth_det_list = []
            inv_depth_seg_list = []
            gt_distances = []
            
            for img_path in tqdm(sampled, desc=f"    {dist_range}m", leave=False):
                gt_distance = extract_ground_truth_distance(img_path.name)
                if gt_distance is None:
                    continue
                
                label_detect_path = label_detect_folder / f"{img_path.stem}.txt"
                label_segment_path = label_segment_folder / f"{img_path.stem}.txt"
                
                if not label_detect_path.exists() or not label_segment_path.exists():
                    continue
                
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                h, w = image.shape[:2]
                
                boxes = load_yolo_detection(label_detect_path, w, h)
                polygons = load_yolo_segmentation(label_segment_path, w, h)
                
                if len(boxes) == 0 or len(polygons) == 0:
                    continue
                
                disparity_map, _ = estimate_disparity_map(img_path, midas, transform, device)
                if disparity_map is None:
                    continue
                
                inv_depth_map = disparity_to_inverse_depth(disparity_map)
                
                inv_depth_det = estimate_depth_detection(inv_depth_map, boxes)
                inv_depth_seg = estimate_depth_segmentation(inv_depth_map, polygons)
                
                if inv_depth_det is not None and inv_depth_seg is not None:
                    if inv_depth_det > 0 and inv_depth_seg > 0:
                        inv_depth_det_list.append(inv_depth_det)
                        inv_depth_seg_list.append(inv_depth_seg)
                        gt_distances.append(gt_distance)
            
            if len(inv_depth_det_list) < 3:
                print(f"    ❌ Insufficient data after filtering")
                calibration_params[class_folder][dist_range] = {
                    'detect': (None, None),
                    'segment': (None, None)
                }
                continue
            
            print(f"    🔧 Detection ({len(inv_depth_det_list)} samples):")
            scale_det, stats_det = compute_scale_factor_mad(inv_depth_det_list, gt_distances, verbose=True)
            
            print(f"    🔧 Segmentation ({len(inv_depth_seg_list)} samples):")
            scale_seg, stats_seg = compute_scale_factor_mad(inv_depth_seg_list, gt_distances, verbose=True)
            
            calibration_params[class_folder][dist_range] = {
                'detect': (scale_det, stats_det),
                'segment': (scale_seg, stats_seg)
            }
    
    # ====================
    # PASS 2: EVALUATION
    # ====================
    print("\n" + "="*60)
    print("📈 PASS 2: FULL EVALUATION")
    print("="*60)
    
    output_classes = {
        "motorblue": "motorcycle",
        "motorwhite": "motorcycle",
        "person": "person",
        "pot": "pot"
    }
    
    results = {cls: [] for cls in output_classes.keys()}
    
    for output_class, parent_class in output_classes.items():
        print(f"\n📂 Processing: {output_class}")
        
        if parent_class not in calibration_params:
            print(f"  ❌ No calibration for {parent_class}")
            continue
        
        image_folder = IMAGE_PATH / parent_class
        if not image_folder.exists():
            continue
        
        all_images = sorted(list(image_folder.glob("*.JPG")) + list(image_folder.glob("*.jpg")))
        
        if output_class in ["motorblue", "motorwhite"]:
            all_images = [img for img in all_images if img.name.startswith(output_class)]
        
        print(f"  📸 Processing {len(all_images)} images...")
        
        label_detect_folder = LABEL_DETECT_PATH / parent_class
        label_segment_folder = LABEL_SEGMENT_PATH / parent_class
        
        for img_path in tqdm(all_images, desc=f"  {output_class}"):
            gt_distance = extract_ground_truth_distance(img_path.name)
            if gt_distance is None:
                continue
            
            # Get calibration for this distance range
            dist_range = get_distance_range(gt_distance)
            
            if dist_range not in calibration_params[parent_class]:
                continue
            
            scale_det, stats_det = calibration_params[parent_class][dist_range]['detect']
            scale_seg, stats_seg = calibration_params[parent_class][dist_range]['segment']
            
            if scale_det is None or scale_seg is None:
                continue
            
            label_detect_path = label_detect_folder / f"{img_path.stem}.txt"
            label_segment_path = label_segment_folder / f"{img_path.stem}.txt"
            
            if not label_detect_path.exists() or not label_segment_path.exists():
                continue
            
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            h, w = image.shape[:2]
            
            boxes = load_yolo_detection(label_detect_path, w, h)
            polygons = load_yolo_segmentation(label_segment_path, w, h)
            
            if len(boxes) == 0 or len(polygons) == 0:
                continue
            
            disparity_map, time_infer = estimate_disparity_map(img_path, midas, transform, device)
            if disparity_map is None:
                continue
            
            inv_depth_map = disparity_to_inverse_depth(disparity_map)
            
            inv_depth_det = estimate_depth_detection(inv_depth_map, boxes)
            if inv_depth_det is None or inv_depth_det <= 0 or not np.isfinite(inv_depth_det):
                continue
            
            inv_depth_seg = estimate_depth_segmentation(inv_depth_map, polygons)
            if inv_depth_seg is None or inv_depth_seg <= 0 or not np.isfinite(inv_depth_seg):
                continue
            
            dist_det = inv_depth_det * scale_det
            dist_seg = inv_depth_seg * scale_seg
            
            dist_det = np.clip(dist_det, 0.1, 100.0)
            dist_seg = np.clip(dist_seg, 0.1, 100.0)
            
            if not np.isfinite(dist_det) or not np.isfinite(dist_seg):
                continue
            
            mae_det = abs(dist_det - gt_distance)
            mae_seg = abs(dist_seg - gt_distance)
            
            results[output_class].append({
                'sample_name': img_path.name,
                'MAE_Detection': round(mae_det, 2),
                'MAE_Segmentation': round(mae_seg, 2),
                'Time_Detection_ms': round(time_infer, 2),
                'Time_Segmentation_ms': round(time_infer, 2)
            })
    
    # ====================
    # SAVE RESULTS
    # ====================
    print("\n" + "="*60)
    print("💾 SAVING RESULTS")
    print("="*60)
    
    for class_name, data in results.items():
        if len(data) == 0:
            print(f"  ⚠️  No results for {class_name}")
            continue
        
        df = pd.DataFrame(data)
        output_file = OUTPUT_PATH / f"depth_results_{class_name}_v2.csv"
        df.to_csv(output_file, index=False)
        
        print(f"  ✅ {class_name}: {len(data)} samples")
        print(f"     MAE_Det: {df['MAE_Detection'].mean():.2f}m (median={df['MAE_Detection'].median():.2f}m)")
        print(f"     MAE_Seg: {df['MAE_Segmentation'].mean():.2f}m (median={df['MAE_Segmentation'].median():.2f}m)")
    
    print("\n✅ DONE!")


if __name__ == "__main__":
    process_dataset()
