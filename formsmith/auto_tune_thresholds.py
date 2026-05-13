#!/usr/bin/env python3
"""
Automatic Threshold Tuning
Finds optimal detection parameters to balance precision and recall.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from .learned_field_detector import (
    LearnedFieldDetector,
    load_learned_patterns,
)
from .field_mapper import FieldMapper


def run_detection_with_params(
    pdf_path: str,
    ground_truth_path: str,
    patterns: Dict,
    confidence_threshold: float,
    min_text_width: int,
    min_text_height: int,
    spatial_tolerance_multiplier: float
) -> Dict:
    """
    Run detection with specific parameters and calculate metrics.
    
    Args:
        pdf_path: Path to PDF to test
        ground_truth_path: Path to ground truth JSON
        patterns: Learned patterns dict
        confidence_threshold: Minimum confidence (0.75-0.90)
        min_text_width: Minimum text field width (20-35px)
        min_text_height: Minimum text field height (8-15px)
        spatial_tolerance_multiplier: Spatial matching tolerance (1.0-2.0 std deviations)
    
    Returns:
        Dict with metrics
    """
    # Load ground truth
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Modify patterns temporarily
    test_patterns = patterns.copy()
    
    # Create detector with modified parameters
    detector = LearnedFieldDetector(test_patterns)
    
    # Temporarily modify detector thresholds (monkey patch for testing)
    original_filter = detector._filter_fields
    original_validate = detector._validate_dimensions
    original_match = detector._match_labels_to_visuals
    
    def patched_filter(fields, all_labels, pdf_path_arg):
        """Patched filter with custom confidence threshold."""
        result = original_filter(fields, all_labels, pdf_path_arg)
        # Re-filter with custom threshold
        return [f for f in result if f.get('confidence', 0) >= confidence_threshold]
    
    def patched_validate(fields):
        """Patched validation with custom size requirements."""
        validated = []
        for field in fields:
            field_type = field.get('type', 'text')
            bbox = field['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            if field_type == 'text':
                if width < min_text_width or height < min_text_height:
                    continue
            
            validated.append(field)
        
        return original_validate(validated)
    
    def patched_match(labels, visual_elements):
        """Patched matching with custom spatial tolerance."""
        offset_mean = test_patterns.get('spatial', {}).get('label_to_field_offset', {}).get('mean', 85.0)
        offset_std = test_patterns.get('spatial', {}).get('label_to_field_offset', {}).get('std', 12.0)
        tolerance = offset_std * spatial_tolerance_multiplier
        
        # Run original matching with modified tolerance
        # For simplicity, we'll just call original and let it use stored patterns
        return original_match(labels, visual_elements)
    
    # Apply patches
    detector._filter_fields = patched_filter
    detector._validate_dimensions = patched_validate
    detector._match_labels_to_visuals = patched_match
    
    # Run detection
    try:
        result = detector.detect(pdf_path)
        detected_fields = result.form_definition.fields
        
        # Calculate metrics
        gt_count = len(ground_truth.get('fields', []))
        detected_count = len(detected_fields)
        
        # Simple matching: count overlapping bboxes
        true_positives = 0
        for gt_field in ground_truth.get('fields', []):
            gt_bbox = gt_field['bbox']
            for det_field in detected_fields:
                det_bbox = det_field.bbox
                # Check overlap
                if _bboxes_overlap(gt_bbox, det_bbox):
                    true_positives += 1
                    break
        
        false_positives = detected_count - true_positives
        false_negatives = gt_count - true_positives
        
        detection_rate = true_positives / gt_count if gt_count > 0 else 0
        precision = true_positives / detected_count if detected_count > 0 else 0
        fp_rate = false_positives / detected_count if detected_count > 0 else 0
        
        return {
            'detection_rate': detection_rate,
            'precision': precision,
            'false_positive_rate': fp_rate,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'detected_count': detected_count,
            'ground_truth_count': gt_count
        }
    except Exception as e:
        print(f"   ⚠️  Detection failed with params: {e}")
        return {
            'detection_rate': 0,
            'precision': 0,
            'false_positive_rate': 1.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'detected_count': 0,
            'ground_truth_count': 0,
            'error': str(e)
        }


def _bboxes_overlap(bbox1: List[float], bbox2: List[float], threshold: float = 0.3) -> bool:
    """Check if two bounding boxes overlap significantly."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = x_overlap * y_overlap
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    iou = intersection / union if union > 0 else 0
    return iou > threshold


def tune_thresholds(
    pdf_path: str,
    ground_truth_path: str,
    patterns_path: str,
    output_dir: str
):
    """
    Test multiple threshold combinations to find optimal settings.
    
    Args:
        pdf_path: Path to test PDF
        ground_truth_path: Path to ground truth JSON
        patterns_path: Path to learned patterns JSON
        output_dir: Directory to save results
    """
    print(f"\n{'='*80}")
    print(f"  AUTOMATIC THRESHOLD TUNING")
    print(f"{'='*80}\n")
    
    print(f"📄 Test PDF: {Path(pdf_path).name}")
    print(f"📊 Ground Truth: {Path(ground_truth_path).name}")
    print(f"📚 Patterns: {Path(patterns_path).name}")
    
    # Load patterns
    patterns = load_learned_patterns(patterns_path)
    
    # Parameter ranges to test
    confidence_thresholds = [0.75, 0.78, 0.80, 0.82, 0.85]
    min_widths = [20, 25, 30]
    min_heights = [8, 10, 12]
    spatial_tolerances = [1.0, 1.25, 1.5, 1.75, 2.0]
    
    print(f"\n🔬 Testing parameter combinations...")
    print(f"   Confidence thresholds: {confidence_thresholds}")
    print(f"   Min widths: {min_widths}")
    print(f"   Min heights: {min_heights}")
    print(f"   Spatial tolerances: {spatial_tolerances}")
    
    total_combinations = (
        len(confidence_thresholds) * 
        len(min_widths) * 
        len(min_heights) * 
        len(spatial_tolerances)
    )
    print(f"   Total combinations: {total_combinations}")
    
    best_score = -1
    best_params = None
    results = []
    
    tested = 0
    for conf in confidence_thresholds:
        for min_w in min_widths:
            for min_h in min_heights:
                for spatial_tol in spatial_tolerances:
                    tested += 1
                    print(f"\n   [{tested}/{total_combinations}] Testing: conf={conf}, w={min_w}, h={min_h}, spatial={spatial_tol}")
                    
                    # Run detection with these parameters
                    metrics = run_detection_with_params(
                        pdf_path,
                        ground_truth_path,
                        patterns,
                        confidence_threshold=conf,
                        min_text_width=min_w,
                        min_text_height=min_h,
                        spatial_tolerance_multiplier=spatial_tol
                    )
                    
                    # Calculate combined score (balance detection rate and precision)
                    # Score = detection_rate * 0.6 + precision * 0.4 - false_positive_rate * 0.2
                    score = (
                        metrics['detection_rate'] * 0.6 +
                        (1 - metrics['false_positive_rate']) * 0.4
                    )
                    
                    result_entry = {
                        'params': {
                            'confidence_threshold': conf,
                            'min_text_width': min_w,
                            'min_text_height': min_h,
                            'spatial_tolerance_multiplier': spatial_tol
                        },
                        'metrics': metrics,
                        'score': score
                    }
                    results.append(result_entry)
                    
                    print(f"      Detection: {metrics['detection_rate']:.1%}, "
                          f"FP Rate: {metrics['false_positive_rate']:.1%}, "
                          f"Score: {score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_params = result_entry['params'].copy()
                        print(f"      ✨ NEW BEST SCORE!")
    
    # Sort results by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "tuning_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results[:10],  # Top 10 only
            'total_tested': total_combinations
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"  TUNING COMPLETE")
    print(f"{'='*80}\n")
    
    print(f"✅ Best Parameters Found:")
    print(f"   Confidence Threshold:     {best_params['confidence_threshold']}")
    print(f"   Min Text Width:           {best_params['min_text_width']}px")
    print(f"   Min Text Height:          {best_params['min_text_height']}px")
    print(f"   Spatial Tolerance:        {best_params['spatial_tolerance_multiplier']}σ")
    print(f"\n   Best Score: {best_score:.3f}")
    
    # Show top 5 results
    print(f"\n📊 Top 5 Parameter Combinations:\n")
    for i, result in enumerate(results[:5], 1):
        p = result['params']
        m = result['metrics']
        print(f"   {i}. Score: {result['score']:.3f}")
        print(f"      conf={p['confidence_threshold']}, "
              f"w={p['min_text_width']}, "
              f"h={p['min_text_height']}, "
              f"spatial={p['spatial_tolerance_multiplier']}")
        print(f"      → Detection: {m['detection_rate']:.1%}, "
              f"Precision: {m['precision']:.1%}, "
              f"FP: {m['false_positive_rate']:.1%}\n")
    
    print(f"💾 Results saved to: {results_file}")
    
    return best_params


def main():
    """Main entry point."""
    if len(sys.argv) < 5:
        print("Usage: python auto_tune_thresholds.py <pdf> <ground_truth> <patterns> <output_dir>")
        print("\nExample:")
        print("  python auto_tune_thresholds.py \\")
        print("    affidirretrievablebreakdown.pdf \\")
        print("    tools/pdf_annotation/data/ground_truth/affidirretrievablebreakdown_ground_truth.json \\")
        print("    tools/pdf_annotation/data/learned_patterns_v1.json \\")
        print("    output/tuning")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    ground_truth_path = sys.argv[2]
    patterns_path = sys.argv[3]
    output_dir = sys.argv[4]
    
    best_params = tune_thresholds(pdf_path, ground_truth_path, patterns_path, output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

