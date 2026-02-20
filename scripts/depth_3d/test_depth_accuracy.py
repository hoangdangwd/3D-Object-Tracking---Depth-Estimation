"""
Test Depth Estimation Accuracy

Script này kiểm tra độ chính xác của depth estimation bằng cách:
1. So sánh depth dự đoán với ground truth
2. Tính toán các metrics: MAE, RMSE, relative error
3. Visualize error distribution
4. Kiểm tra scale factor stability

Usage:
    python scripts/depth_3d/test_depth_accuracy.py --image path/to/image.jpg
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def load_midas_model(model_type='DPT_Large'):
    """Load MiDaS model"""
    print(f"🔄 Loading MiDaS {model_type}...")
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    midas.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.dpt_transform
    
    print("✅ Model loaded")
    return midas, transform

def estimate_depth(image, midas, transform):
    """Estimate depth map from image"""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Transform input
    input_transformed = transform(img_rgb)
    if isinstance(input_transformed, dict):
        input_tensor = input_transformed["image"]
    else:
        input_tensor = input_transformed
    
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Convert to depth (correct way)
    depth_map = prediction.cpu().numpy()
    depth_inv = 1.0 / (depth_map + 1e-6)
    
    return depth_inv, depth_map

def compute_scale_factor(depth_inv, reference_points, verbose=True):
    """
    Compute scale factor from reference points
    
    Args:
        depth_inv: Inverse depth map (H, W)
        reference_points: List of (u, v, z_real) tuples
        verbose: Print detailed info
    
    Returns:
        scale_factor: Computed scale
        scale_stats: Dictionary with statistics
    """
    scale_list = []
    
    if verbose:
        print("\n🔍 Computing scale factor:")
    
    for i, (u, v, z_real) in enumerate(reference_points):
        if 0 <= v < depth_inv.shape[0] and 0 <= u < depth_inv.shape[1]:
            z_pred = depth_inv[v, u]
            
            if z_pred > 1e-6:
                scale = z_real / z_pred
                scale_list.append(scale)
                
                if verbose:
                    print(f"  Point {i+1} ({u},{v}): Z_real={z_real:.3f}m, Z_pred={z_pred:.3f}, scale={scale:.3f}")
    
    if len(scale_list) < 3:
        raise ValueError(f"❌ Only {len(scale_list)} valid points (need ≥3)")
    
    # Remove outliers using MAD (Median Absolute Deviation)
    scale_median = np.median(scale_list)
    mad = np.median(np.abs(np.array(scale_list) - scale_median))
    scale_filtered = [s for s in scale_list if abs(s - scale_median) < 2 * mad]
    
    if not scale_filtered:
        print("⚠️ All scales are outliers, using median")
        scale_filtered = [scale_median]
    
    scale_factor = np.mean(scale_filtered)
    scale_std = np.std(scale_filtered)
    
    stats = {
        'mean': scale_factor,
        'std': scale_std,
        'median': scale_median,
        'mad': mad,
        'n_total': len(scale_list),
        'n_filtered': len(scale_filtered),
        'outliers_removed': len(scale_list) - len(scale_filtered)
    }
    
    if verbose:
        print(f"\n✅ Scale factor: {scale_factor:.6f} ± {scale_std:.6f}")
        print(f"   Median: {scale_median:.6f}")
        print(f"   Valid points: {stats['n_filtered']}/{stats['n_total']}")
        print(f"   Outliers removed: {stats['outliers_removed']}")
    
    return scale_factor, stats

def compute_accuracy_metrics(depth_pred, depth_gt, mask=None):
    """
    Compute depth accuracy metrics
    
    Args:
        depth_pred: Predicted depth (H, W)
        depth_gt: Ground truth depth (H, W)
        mask: Valid region mask (H, W) - optional
    
    Returns:
        metrics: Dictionary with MAE, RMSE, relative error, etc.
    """
    if mask is None:
        mask = np.ones_like(depth_gt, dtype=bool)
    
    # Filter valid regions
    valid = mask & (depth_gt > 0.01) & (depth_pred > 0.01)
    
    if valid.sum() == 0:
        return None
    
    pred = depth_pred[valid]
    gt = depth_gt[valid]
    
    # Absolute errors
    abs_error = np.abs(pred - gt)
    mae = np.mean(abs_error)
    rmse = np.sqrt(np.mean((pred - gt)**2))
    
    # Relative error
    rel_error = abs_error / gt
    mean_rel_error = np.mean(rel_error)
    median_rel_error = np.median(rel_error)
    
    # Threshold accuracy (δ < 1.25)
    max_ratio = np.maximum(pred / gt, gt / pred)
    delta_1 = (max_ratio < 1.25).mean()
    delta_2 = (max_ratio < 1.25**2).mean()
    delta_3 = (max_ratio < 1.25**3).mean()
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'Mean_Rel_Error': mean_rel_error,
        'Median_Rel_Error': median_rel_error,
        'Delta_1.25': delta_1,
        'Delta_1.25^2': delta_2,
        'Delta_1.25^3': delta_3,
        'n_valid_pixels': valid.sum()
    }
    
    return metrics

def visualize_depth_accuracy(image, depth_pred, reference_points, depth_gt=None):
    """
    Visualize depth estimation results
    
    Args:
        image: Input image (H, W, 3)
        depth_pred: Predicted depth (H, W)
        reference_points: List of (u, v, z_real)
        depth_gt: Ground truth depth (optional)
    """
    if depth_gt is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Row 1: Image, Depth prediction, Reference points
    if depth_gt is not None:
        ax_img, ax_depth, ax_ref = axes[0]
    else:
        ax_img, ax_depth, ax_ref = axes
    
    # Original image
    ax_img.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax_img.set_title("Original Image")
    ax_img.axis('off')
    
    # Predicted depth
    im_depth = ax_depth.imshow(depth_pred, cmap='turbo')
    ax_depth.set_title(f"Predicted Depth\n(mean={depth_pred.mean():.2f}m)")
    ax_depth.axis('off')
    plt.colorbar(im_depth, ax=ax_depth, label='Depth (m)')
    
    # Reference points overlay
    ax_ref.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for i, (u, v, z_real) in enumerate(reference_points):
        ax_ref.plot(u, v, 'r+', markersize=15, markeredgewidth=2)
        ax_ref.text(u+10, v-10, f"{z_real:.2f}m", color='red', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_ref.set_title(f"Reference Points (n={len(reference_points)})")
    ax_ref.axis('off')
    
    if depth_gt is not None:
        ax_gt, ax_error, ax_hist = axes[1]
        
        # Ground truth depth
        im_gt = ax_gt.imshow(depth_gt, cmap='turbo')
        ax_gt.set_title("Ground Truth Depth")
        ax_gt.axis('off')
        plt.colorbar(im_gt, ax=ax_gt, label='Depth (m)')
        
        # Error map
        error = np.abs(depth_pred - depth_gt)
        im_error = ax_error.imshow(error, cmap='hot')
        ax_error.set_title(f"Absolute Error\n(MAE={error.mean():.3f}m)")
        ax_error.axis('off')
        plt.colorbar(im_error, ax=ax_error, label='Error (m)')
        
        # Error histogram
        ax_hist.hist(error.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
        ax_hist.set_xlabel('Absolute Error (m)')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Error Distribution')
        ax_hist.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../data/output/depth_accuracy_test.png', dpi=150, bbox_inches='tight')
    print("💾 Saved visualization: data/output/depth_accuracy_test.png")
    plt.show()

def test_bilinear_interpolation():
    """Test bilinear interpolation accuracy"""
    print("\n" + "="*60)
    print("🧪 Testing Bilinear Interpolation")
    print("="*60)
    
    # Create synthetic depth map
    depth_map = np.array([
        [1.0, 1.2, 1.4],
        [1.1, 1.3, 1.5],
        [1.2, 1.4, 1.6]
    ])
    
    # Test points
    test_points = [
        (0.0, 0.0, 1.0),      # Corner - should be exact
        (1.0, 1.0, 1.3),      # Center - should be exact
        (0.5, 0.5, 1.15),     # Mid-point - interpolated
        (1.5, 1.5, 1.45),     # Mid-point - interpolated
    ]
    
    print("\nTest cases:")
    for u, v, expected in test_points:
        u_floor = int(np.floor(u))
        v_floor = int(np.floor(v))
        u_frac = u - u_floor
        v_frac = v - v_floor
        
        if 0 <= v_floor < depth_map.shape[0]-1 and 0 <= u_floor < depth_map.shape[1]-1:
            Z00 = depth_map[v_floor, u_floor]
            Z01 = depth_map[v_floor, u_floor+1]
            Z10 = depth_map[v_floor+1, u_floor]
            Z11 = depth_map[v_floor+1, u_floor+1]
            
            Z = (Z00 * (1-u_frac) * (1-v_frac) +
                 Z01 * u_frac * (1-v_frac) +
                 Z10 * (1-u_frac) * v_frac +
                 Z11 * u_frac * v_frac)
            
            error = abs(Z - expected)
            status = "✅" if error < 0.01 else "❌"
            print(f"  {status} ({u:.1f}, {v:.1f}): expected={expected:.3f}, got={Z:.3f}, error={error:.4f}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test depth estimation accuracy')
    parser.add_argument('--image', type=str, default='../../assets/checkpoints/1.jpg',
                       help='Path to test image')
    parser.add_argument('--model', type=str, default='DPT_Large',
                       choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'],
                       help='MiDaS model type')
    args = parser.parse_args()
    
    print("="*60)
    print("🔬 DEPTH ESTIMATION ACCURACY TEST")
    print("="*60)
    
    # Test bilinear interpolation
    test_bilinear_interpolation()
    
    # Load image
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return
    
    print(f"\n📸 Loading image: {args.image}")
    image = cv2.imread(args.image)
    H, W = image.shape[:2]
    print(f"   Size: {W}x{H}")
    
    # Load model
    midas, transform = load_midas_model(args.model)
    
    # Estimate depth
    print("\n🧮 Estimating depth...")
    depth_inv, depth_disparity = estimate_depth(image, midas, transform)
    
    print(f"   Disparity range: [{depth_disparity.min():.3f}, {depth_disparity.max():.3f}]")
    print(f"   Inverse depth range: [{depth_inv.min():.3f}, {depth_inv.max():.3f}]")
    
    # Reference points (update these with your calibration data)
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
    
    # Compute scale factor
    try:
        scale_factor, scale_stats = compute_scale_factor(depth_inv, reference_points)
        
        # Scale depth
        depth_scaled = depth_inv * scale_factor
        
        print(f"\n📊 Scaled depth statistics:")
        print(f"   Mean: {depth_scaled.mean():.3f}m")
        print(f"   Std: {depth_scaled.std():.3f}m")
        print(f"   Range: [{depth_scaled.min():.3f}, {depth_scaled.max():.3f}]m")
        
        # Verify reference points accuracy
        print("\n✅ Verification - Reference points accuracy:")
        errors = []
        for u, v, z_real in reference_points:
            if 0 <= v < depth_scaled.shape[0] and 0 <= u < depth_scaled.shape[1]:
                z_pred = depth_scaled[v, u]
                error = abs(z_pred - z_real)
                rel_error = error / z_real * 100
                errors.append(error)
                print(f"   ({u},{v}): Z_real={z_real:.3f}m, Z_pred={z_pred:.3f}m, "
                      f"error={error:.4f}m ({rel_error:.1f}%)")
        
        if errors:
            print(f"\n📈 Error statistics:")
            print(f"   MAE: {np.mean(errors):.4f}m")
            print(f"   Max error: {np.max(errors):.4f}m")
            print(f"   Mean relative error: {np.mean([e/r[2]*100 for e, r in zip(errors, reference_points)]):.2f}%")
        
        # Visualize
        visualize_depth_accuracy(image, depth_scaled, reference_points)
        
    except ValueError as e:
        print(f"\n{e}")
        print("⚠️ Cannot compute accurate scale factor. Check reference points configuration.")

if __name__ == "__main__":
    main()
