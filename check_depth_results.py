import pandas as pd
from pathlib import Path
import numpy as np


def check_csv_structure(csv_path):
    """
    Kiểm tra cấu trúc và nội dung của CSV file
    """
    try:
        df = pd.read_csv(csv_path)
        
        issues = []
        warnings = []
        
        # Kiểm tra columns
        required_cols = ['sample_name', 'MAE_Detection', 'MAE_Segmentation', 
                        'Time_Detection_ms', 'Time_Segmentation_ms']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            issues.append(f"❌ Thiếu cột: {', '.join(missing_cols)}")
            return None, issues, warnings
        
        # Kiểm tra số lượng records
        if len(df) == 0:
            issues.append("❌ File rỗng - không có dữ liệu")
            return None, issues, warnings
        
        # Kiểm tra missing values
        for col in required_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                warnings.append(f"⚠️  Cột '{col}': {missing_count} giá trị missing")
        
        # Kiểm tra giá trị âm
        if (df['MAE_Detection'] < 0).any():
            issues.append("❌ MAE_Detection có giá trị âm")
        if (df['MAE_Segmentation'] < 0).any():
            issues.append("❌ MAE_Segmentation có giá trị âm")
        if (df['Time_Detection_ms'] < 0).any():
            issues.append("❌ Time_Detection_ms có giá trị âm")
        if (df['Time_Segmentation_ms'] < 0).any():
            issues.append("❌ Time_Segmentation_ms có giá trị âm")
        
        # Kiểm tra giá trị vô lý
        if (df['MAE_Detection'] > 1000).any():
            count = (df['MAE_Detection'] > 1000).sum()
            warnings.append(f"⚠️  MAE_Detection: {count} giá trị > 1000m (có thể sai)")
        
        if (df['MAE_Segmentation'] > 1000).any():
            count = (df['MAE_Segmentation'] > 1000).sum()
            warnings.append(f"⚠️  MAE_Segmentation: {count} giá trị > 1000m (có thể sai)")
        
        if (df['Time_Detection_ms'] > 10000).any():
            count = (df['Time_Detection_ms'] > 10000).sum()
            warnings.append(f"⚠️  Time_Detection_ms: {count} giá trị > 10s (quá chậm)")
        
        # Kiểm tra inf values
        for col in ['MAE_Detection', 'MAE_Segmentation']:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                issues.append(f"❌ {col}: {inf_count} giá trị infinity")
        
        return df, issues, warnings
        
    except Exception as e:
        return None, [f"❌ Lỗi đọc file: {str(e)}"], []


def analyze_depth_results(result_dir):
    """
    Phân tích tất cả 4 file CSV results
    """
    result_dir = Path(result_dir)
    
    classes = ['motorblue', 'motorwhite', 'person', 'pot']
    
    print("\n" + "="*80)
    print("KIỂM TRA KẾT QUẢ DEPTH ESTIMATION")
    print("="*80)
    
    all_results = {}
    
    for cls in classes:
        csv_path = result_dir / f"depth_results_{cls}.csv"
        
        print(f"\n{'='*80}")
        print(f"📊 CLASS: {cls.upper()}")
        print(f"{'='*80}")
        print(f"File: {csv_path.name}")
        
        if not csv_path.exists():
            print(f"❌ File không tồn tại: {csv_path}")
            continue
        
        df, issues, warnings = check_csv_structure(csv_path)
        
        if df is None:
            print("\n❌ KIỂM TRA CẤU TRÚC:")
            for issue in issues:
                print(f"  {issue}")
            continue
        
        # In issues và warnings
        if issues:
            print("\n❌ LỖI:")
            for issue in issues:
                print(f"  {issue}")
        
        if warnings:
            print("\n⚠️  CẢNH BÁO:")
            for warning in warnings:
                print(f"  {warning}")
        
        if not issues and not warnings:
            print("\n✅ CẤU TRÚC: OK")
        
        # Thống kê
        print("\n📈 THỐNG KÊ:")
        print(f"  • Số mẫu: {len(df)}")
        
        print(f"\n  📏 MAE DETECTION (meters):")
        print(f"    • Mean: {df['MAE_Detection'].mean():.4f}m")
        print(f"    • Median: {df['MAE_Detection'].median():.4f}m")
        print(f"    • Std: {df['MAE_Detection'].std():.4f}m")
        print(f"    • Min: {df['MAE_Detection'].min():.4f}m")
        print(f"    • Max: {df['MAE_Detection'].max():.4f}m")
        print(f"    • 25%: {df['MAE_Detection'].quantile(0.25):.4f}m")
        print(f"    • 75%: {df['MAE_Detection'].quantile(0.75):.4f}m")
        
        print(f"\n  🎨 MAE SEGMENTATION (meters):")
        print(f"    • Mean: {df['MAE_Segmentation'].mean():.4f}m")
        print(f"    • Median: {df['MAE_Segmentation'].median():.4f}m")
        print(f"    • Std: {df['MAE_Segmentation'].std():.4f}m")
        print(f"    • Min: {df['MAE_Segmentation'].min():.4f}m")
        print(f"    • Max: {df['MAE_Segmentation'].max():.4f}m")
        print(f"    • 25%: {df['MAE_Segmentation'].quantile(0.25):.4f}m")
        print(f"    • 75%: {df['MAE_Segmentation'].quantile(0.75):.4f}m")
        
        print(f"\n  ⏱️  THỜI GIAN DETECTION (ms):")
        print(f"    • Mean: {df['Time_Detection_ms'].mean():.2f}ms")
        print(f"    • Median: {df['Time_Detection_ms'].median():.2f}ms")
        print(f"    • Min: {df['Time_Detection_ms'].min():.2f}ms")
        print(f"    • Max: {df['Time_Detection_ms'].max():.2f}ms")
        
        print(f"\n  ⏱️  THỜI GIAN SEGMENTATION (ms):")
        print(f"    • Mean: {df['Time_Segmentation_ms'].mean():.2f}ms")
        print(f"    • Median: {df['Time_Segmentation_ms'].median():.2f}ms")
        print(f"    • Min: {df['Time_Segmentation_ms'].min():.2f}ms")
        print(f"    • Max: {df['Time_Segmentation_ms'].max():.2f}ms")
        
        # So sánh Detection vs Segmentation
        better_detection = (df['MAE_Detection'] < df['MAE_Segmentation']).sum()
        better_segmentation = (df['MAE_Segmentation'] < df['MAE_Detection']).sum()
        equal = (df['MAE_Detection'] == df['MAE_Segmentation']).sum()
        
        print(f"\n  🆚 SO SÁNH DETECTION vs SEGMENTATION:")
        print(f"    • Detection tốt hơn: {better_detection} mẫu ({better_detection/len(df)*100:.1f}%)")
        print(f"    • Segmentation tốt hơn: {better_segmentation} mẫu ({better_segmentation/len(df)*100:.1f}%)")
        print(f"    • Bằng nhau: {equal} mẫu ({equal/len(df)*100:.1f}%)")
        
        # Top 5 mẫu tốt nhất và tệ nhất
        print(f"\n  🏆 TOP 5 MẪU TỐT NHẤT (MAE Detection):")
        best_samples = df.nsmallest(5, 'MAE_Detection')[['sample_name', 'MAE_Detection', 'MAE_Segmentation']]
        for idx, row in best_samples.iterrows():
            print(f"    • {row['sample_name']}: Det={row['MAE_Detection']:.4f}m, Seg={row['MAE_Segmentation']:.4f}m")
        
        print(f"\n  ⚠️  TOP 5 MẪU TỆ NHẤT (MAE Detection):")
        worst_samples = df.nlargest(5, 'MAE_Detection')[['sample_name', 'MAE_Detection', 'MAE_Segmentation']]
        for idx, row in worst_samples.iterrows():
            print(f"    • {row['sample_name']}: Det={row['MAE_Detection']:.4f}m, Seg={row['MAE_Segmentation']:.4f}m")
        
        # Lưu vào dict
        all_results[cls] = df
    
    # So sánh giữa các classes
    print("\n" + "="*80)
    print("SO SÁNH GIỮA CÁC CLASSES")
    print("="*80)
    
    if all_results:
        comparison_data = []
        for cls, df in all_results.items():
            comparison_data.append({
                'Class': cls,
                'Samples': len(df),
                'MAE_Det_Mean': df['MAE_Detection'].mean(),
                'MAE_Seg_Mean': df['MAE_Segmentation'].mean(),
                'Time_Det_Mean': df['Time_Detection_ms'].mean(),
                'Time_Seg_Mean': df['Time_Segmentation_ms'].mean()
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n📊 Bảng so sánh:")
        print(comparison_df.to_string(index=False))
        
        # Tìm class tốt nhất
        best_det_cls = comparison_df.loc[comparison_df['MAE_Det_Mean'].idxmin(), 'Class']
        best_seg_cls = comparison_df.loc[comparison_df['MAE_Seg_Mean'].idxmin(), 'Class']
        fastest_det_cls = comparison_df.loc[comparison_df['Time_Det_Mean'].idxmin(), 'Class']
        
        print(f"\n🏆 KẾT QUẢ TỐT NHẤT:")
        print(f"  • MAE Detection thấp nhất: {best_det_cls} ({comparison_df[comparison_df['Class']==best_det_cls]['MAE_Det_Mean'].values[0]:.4f}m)")
        print(f"  • MAE Segmentation thấp nhất: {best_seg_cls} ({comparison_df[comparison_df['Class']==best_seg_cls]['MAE_Seg_Mean'].values[0]:.4f}m)")
        print(f"  • Xử lý nhanh nhất: {fastest_det_cls} ({comparison_df[comparison_df['Class']==fastest_det_cls]['Time_Det_Mean'].values[0]:.2f}ms)")
    
    # Đánh giá tổng quan
    print("\n" + "="*80)
    print("ĐÁNH GIÁ TỔNG QUAN")
    print("="*80)
    
    if all_results:
        # Tính MAE trung bình của tất cả
        all_mae_det = []
        all_mae_seg = []
        for df in all_results.values():
            all_mae_det.extend(df['MAE_Detection'].tolist())
            all_mae_seg.extend(df['MAE_Segmentation'].tolist())
        
        avg_mae_det = np.mean(all_mae_det)
        avg_mae_seg = np.mean(all_mae_seg)
        
        print(f"\n📏 MAE TRUNG BÌNH TOÀN BỘ DATASET:")
        print(f"  • Detection: {avg_mae_det:.4f}m")
        print(f"  • Segmentation: {avg_mae_seg:.4f}m")
        
        # Đánh giá chất lượng
        print(f"\n💯 ĐÁNH GIÁ CHẤT LƯỢNG:")
        
        if avg_mae_det < 1.0:
            print(f"  ✅ Detection: XUẤT SẮC (MAE < 1m)")
        elif avg_mae_det < 2.0:
            print(f"  ✅ Detection: TỐT (MAE 1-2m)")
        elif avg_mae_det < 5.0:
            print(f"  ⚠️  Detection: TRUNG BÌNH (MAE 2-5m)")
        else:
            print(f"  ❌ Detection: KÉM (MAE > 5m)")
        
        if avg_mae_seg < 1.0:
            print(f"  ✅ Segmentation: XUẤT SẮC (MAE < 1m)")
        elif avg_mae_seg < 2.0:
            print(f"  ✅ Segmentation: TỐT (MAE 1-2m)")
        elif avg_mae_seg < 5.0:
            print(f"  ⚠️  Segmentation: TRUNG BÌNH (MAE 2-5m)")
        else:
            print(f"  ❌ Segmentation: KÉM (MAE > 5m)")
        
        # Khuyến nghị
        print(f"\n💡 KHUYẾN NGHỊ:")
        if avg_mae_seg < avg_mae_det:
            print(f"  • Segmentation cho kết quả tốt hơn Detection")
            print(f"  • Nên sử dụng Segmentation labels cho depth estimation")
        else:
            print(f"  • Detection cho kết quả tốt hơn Segmentation")
            print(f"  • Detection nhanh hơn, phù hợp cho real-time")
        
        print(f"\n  • Nếu MAE cao: Xem xét cải thiện calibration hoặc thử model depth khác")
        print(f"  • Kiểm tra các mẫu tệ nhất để hiểu nguyên nhân lỗi")
    
    print("\n" + "="*80)


def main():
    result_dir = Path("data/output/depth_estimation_results")
    
    if not result_dir.exists():
        print(f"❌ Không tìm thấy thư mục: {result_dir}")
        print(f"   Vui lòng chạy depth_estimation_dataset.py trước!")
        return
    
    analyze_depth_results(result_dir)


if __name__ == "__main__":
    main()
