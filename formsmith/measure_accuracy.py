#!/usr/bin/env python3
"""
Phase 1.2: Accuracy Measurement Script

Compares detected fields against ground truth to calculate precision metrics.
This is the foundation for measuring improvements over time.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import math


def load_json(path: str) -> Dict:
    """Load JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def calculate_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Euclidean distance between two bounding boxes (centers)
    
    Args:
        bbox1: [x, y, width, height] or [x0, y0, x1, y1]
        bbox2: [x, y, width, height] or [x0, y0, x1, y1]
    
    Returns:
        Distance in pixels
    """
    # Convert to [x, y, width, height] format if needed
    if len(bbox1) == 4:
        # Check if it's [x0, y0, x1, y1] (x1 > x0 for valid bbox)
        if bbox1[2] > bbox1[0]:  # [x0, y0, x1, y1] format
            x1 = bbox1[0] + (bbox1[2] - bbox1[0]) / 2
            y1 = bbox1[1] + (bbox1[3] - bbox1[1]) / 2
        else:  # [x, y, width, height] format
            x1 = bbox1[0] + bbox1[2] / 2
            y1 = bbox1[1] + bbox1[3] / 2
    else:
        x1, y1 = bbox1[0], bbox1[1]
    
    if len(bbox2) == 4:
        if bbox2[2] > bbox2[0]:  # [x0, y0, x1, y1] format
            x2 = bbox2[0] + (bbox2[2] - bbox2[0]) / 2
            y2 = bbox2[1] + (bbox2[3] - bbox2[1]) / 2
        else:  # [x, y, width, height] format
            x2 = bbox2[0] + bbox2[2] / 2
            y2 = bbox2[1] + bbox2[3] / 2
    else:
        x2, y2 = bbox2[0], bbox2[1]
    
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def find_best_match(detected_field: Dict, ground_truth_fields: List[Dict], max_distance: float = 50) -> Tuple[Dict, float]:
    """
    Find the best matching ground truth field for a detected field
    
    Args:
        detected_field: Detected field dict
        ground_truth_fields: List of ground truth fields
        max_distance: Maximum distance to consider a match (pixels)
    
    Returns:
        (best_match, distance) or (None, inf) if no match
    """
    best_match = None
    best_distance = float('inf')
    
    # Get detected field position
    if 'x' in detected_field and 'y_from_top' in detected_field:
        detected_bbox = [
            detected_field['x'],
            detected_field['y_from_top'],
            detected_field.get('width', 0),
            detected_field.get('height', 0)
        ]
    elif 'bbox' in detected_field:
        detected_bbox = detected_field['bbox']
    else:
        return None, float('inf')
    
    # Find closest ground truth field
    for gt_field in ground_truth_fields:
        if 'bbox' in gt_field:
            gt_bbox = gt_field['bbox']
        elif 'x' in gt_field and 'y_from_top' in gt_field:
            gt_bbox = [
                gt_field['x'],
                gt_field['y_from_top'],
                gt_field.get('width', 0),
                gt_field.get('height', 0)
            ]
        else:
            continue
        
        distance = calculate_distance(detected_bbox, gt_bbox)
        
        if distance < best_distance and distance <= max_distance:
            best_distance = distance
            best_match = gt_field
    
    return best_match, best_distance


def calculate_accuracy_metrics(detected_fields: List[Dict], ground_truth_fields: List[Dict]) -> Dict:
    """
    Calculate comprehensive accuracy metrics
    
    Args:
        detected_fields: List of detected field dicts
        ground_truth_fields: List of ground truth field dicts
    
    Returns:
        Dictionary of metrics
    """
    print("\n" + "="*80)
    print("  CALCULATING ACCURACY METRICS")
    print("="*80 + "\n")
    
    # Match detected fields to ground truth
    matched_fields = []
    unmatched_detected = []
    position_errors = []
    size_errors = []
    name_matches = 0
    
    for detected in detected_fields:
        match, distance = find_best_match(detected, ground_truth_fields)
        
        if match:
            matched_fields.append({
                'detected': detected,
                'ground_truth': match,
                'distance': distance
            })
            position_errors.append(distance)
            
            # Check size error
            d_width = abs(detected.get('width', 0))
            d_height = abs(detected.get('height', 0))
            gt_width = match.get('width', 0) if 'width' in match else (match['bbox'][2] - match['bbox'][0])
            gt_height = match.get('height', 0) if 'height' in match else (match['bbox'][3] - match['bbox'][1])
            
            size_error = abs(d_width - gt_width) + abs(d_height - gt_height)
            size_errors.append(size_error)
            
            # Check name match
            if detected.get('name') == match.get('name'):
                name_matches += 1
        else:
            unmatched_detected.append(detected)
    
    # Find missed fields (in ground truth but not detected)
    matched_gt_names = [m['ground_truth'].get('name') for m in matched_fields]
    missed_fields = [gt for gt in ground_truth_fields if gt.get('name') not in matched_gt_names]
    
    # Calculate metrics
    total_ground_truth = len(ground_truth_fields)
    total_detected = len(detected_fields)
    true_positives = len(matched_fields)
    false_positives = len(unmatched_detected)
    false_negatives = len(missed_fields)
    
    detection_rate = (true_positives / total_ground_truth * 100) if total_ground_truth > 0 else 0
    precision = (true_positives / total_detected * 100) if total_detected > 0 else 0
    false_positive_rate = (false_positives / total_detected * 100) if total_detected > 0 else 0
    
    avg_position_error = sum(position_errors) / len(position_errors) if position_errors else 0
    avg_size_error = sum(size_errors) / len(size_errors) if size_errors else 0
    naming_accuracy = (name_matches / true_positives * 100) if true_positives > 0 else 0
    
    # Count fields with issues
    negative_width_count = sum(1 for f in detected_fields if f.get('width', 0) < 0)
    tiny_fields = sum(1 for f in detected_fields if abs(f.get('width', 0)) < 10 and abs(f.get('height', 0)) < 10)
    
    metrics = {
        'detection_rate': detection_rate,
        'precision': precision,
        'false_positive_rate': false_positive_rate,
        'avg_position_error': avg_position_error,
        'avg_size_error': avg_size_error,
        'naming_accuracy': naming_accuracy,
        'counts': {
            'ground_truth_total': total_ground_truth,
            'detected_total': total_detected,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'name_matches': name_matches,
            'negative_width_fields': negative_width_count,
            'tiny_fields': tiny_fields
        },
        'matched_fields': matched_fields,
        'unmatched_detected': unmatched_detected,
        'missed_fields': missed_fields
    }
    
    return metrics


def print_metrics_report(metrics: Dict):
    """Print formatted metrics report"""
    
    print("\n" + "="*80)
    print("  ACCURACY METRICS REPORT")
    print("="*80 + "\n")
    
    # Overall metrics
    print("DETECTION PERFORMANCE:")
    print(f"  Detection Rate:       {metrics['detection_rate']:6.2f}% (fields found / ground truth)")
    print(f"  Precision:            {metrics['precision']:6.2f}% (correct / detected)")
    print(f"  False Positive Rate:  {metrics['false_positive_rate']:6.2f}% (wrong fields detected)")
    
    print("\nPOSITIONING ACCURACY:")
    print(f"  Avg Position Error:   {metrics['avg_position_error']:6.2f} pixels")
    print(f"  Avg Size Error:       {metrics['avg_size_error']:6.2f} pixels")
    
    print("\nFIELD NAMING:")
    print(f"  Naming Accuracy:      {metrics['naming_accuracy']:6.2f}% (correct names)")
    
    print("\nFIELD COUNTS:")
    counts = metrics['counts']
    print(f"  Ground Truth Total:   {counts['ground_truth_total']:3d} fields")
    print(f"  Detected Total:       {counts['detected_total']:3d} fields")
    print(f"  True Positives:       {counts['true_positives']:3d} fields (✅ correctly detected)")
    print(f"  False Positives:      {counts['false_positives']:3d} fields (❌ wrongly detected)")
    print(f"  False Negatives:      {counts['false_negatives']:3d} fields (❌ missed)")
    
    print("\nFIELD QUALITY ISSUES:")
    print(f"  Negative Width:       {counts['negative_width_fields']:3d} fields (🐛 BUG!)")
    print(f"  Too Small (<10x10):   {counts['tiny_fields']:3d} fields (⚠️  suspicious)")
    
    # Grade the system
    print("\n" + "="*80)
    print("  OVERALL GRADE")
    print("="*80 + "\n")
    
    detection = metrics['detection_rate']
    position_error = metrics['avg_position_error']
    
    if detection >= 95 and position_error < 3:
        grade = "A+ (Professional Grade)"
        emoji = "🏆"
    elif detection >= 90 and position_error < 5:
        grade = "A (Excellent)"
        emoji = "⭐"
    elif detection >= 80 and position_error < 10:
        grade = "B (Good)"
        emoji = "👍"
    elif detection >= 70 and position_error < 15:
        grade = "C (Acceptable)"
        emoji = "👌"
    elif detection >= 50 and position_error < 25:
        grade = "D (Needs Improvement)"
        emoji = "⚠️"
    else:
        grade = "F (Failing)"
        emoji = "❌"
    
    print(f"  {emoji} Grade: {grade}")
    print(f"     Detection: {detection:.1f}% | Position Error: {position_error:.1f}px")
    
    # Show what's needed for each grade
    print("\n  Grade Requirements:")
    print("    A+ (Pro):  95%+ detection, <3px error")
    print("    A:         90%+ detection, <5px error")
    print("    B:         80%+ detection, <10px error")
    print("    C:         70%+ detection, <15px error")
    print("    D:         50%+ detection, <25px error")
    print("    F:         Below D requirements")
    
    # Recommendations
    print("\n" + "="*80)
    print("  RECOMMENDATIONS")
    print("="*80 + "\n")
    
    if counts['negative_width_fields'] > 0:
        print("  🐛 CRITICAL: Fix negative width bug in field positioning logic")
    
    if metrics['detection_rate'] < 50:
        print("  ❌ CRITICAL: Detection rate too low - improve pattern recognition")
    
    if metrics['false_positive_rate'] > 20:
        print("  ⚠️  HIGH: Too many false positives - add validation/filtering")
    
    if metrics['avg_position_error'] > 15:
        print("  ⚠️  HIGH: Position error too large - improve positioning algorithms")
    
    if metrics['naming_accuracy'] < 70:
        print("  ⚠️  MEDIUM: Poor field naming - improve name generation")
    
    if counts['false_negatives'] > counts['true_positives']:
        print("  ⚠️  HIGH: Missing more fields than finding - expand detection patterns")


def measure_accuracy(detected_file: str, ground_truth_file: str, output_file: str = None):
    """
    Main function to measure accuracy
    
    Args:
        detected_file: Path to detected fields JSON
        ground_truth_file: Path to ground truth fields JSON
        output_file: Optional path to save metrics JSON
    """
    print("="*80)
    print("  ACCURACY MEASUREMENT")
    print("="*80)
    print(f"\nDetected Fields:   {detected_file}")
    print(f"Ground Truth:      {ground_truth_file}")
    
    # Load data
    try:
        detected_data = load_json(detected_file)
        detected_fields = detected_data.get('fields', detected_data.get('refinements', []))
    except Exception as e:
        print(f"\n❌ Error loading detected fields: {e}")
        return
    
    try:
        ground_truth_data = load_json(ground_truth_file)
        ground_truth_fields = ground_truth_data.get('fields', [])
    except Exception as e:
        print(f"\n❌ Error loading ground truth: {e}")
        return
    
    if not ground_truth_fields:
        print("\n❌ No ground truth fields found! Please create ground truth dataset first.")
        return
    
    # Calculate metrics
    metrics = calculate_accuracy_metrics(detected_fields, ground_truth_fields)
    
    # Print report
    print_metrics_report(metrics)
    
    # Save metrics
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Don't save the full matched_fields (too verbose)
        save_metrics = metrics.copy()
        save_metrics['matched_fields_summary'] = [
            {
                'detected_name': m['detected'].get('name'),
                'ground_truth_name': m['ground_truth'].get('name'),
                'distance': m['distance']
            }
            for m in metrics['matched_fields']
        ]
        del save_metrics['matched_fields']
        
        with open(output_path, 'w') as f:
            json.dump(save_metrics, f, indent=2)
        
        print(f"\n✅ Metrics saved to: {output_file}")
    
    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python measure_accuracy.py <detected_fields.json> <ground_truth.json> [output_metrics.json]")
        print("\nExample:")
        print("  python measure_accuracy.py \\")
        print("    data/baseline_test/latest/02_detected_fields.json \\")
        print("    ../validation/fixtures/ground_truth/divorce_form_fields.json \\")
        print("    data/baseline_test/latest/accuracy_metrics.json")
        sys.exit(1)
    
    detected_file = sys.argv[1]
    ground_truth_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    measure_accuracy(detected_file, ground_truth_file, output_file)

