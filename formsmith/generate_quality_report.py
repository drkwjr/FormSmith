#!/usr/bin/env python3
"""
Quality Report Generator
Creates comprehensive quality reports with actionable insights.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


def generate_quality_report(
    detected_fields_path: str,
    ground_truth_path: str,
    output_path: str
) -> Dict:
    """
    Generate detailed quality report with actionable insights.
    
    Args:
        detected_fields_path: Path to detected fields JSON
        ground_truth_path: Path to ground truth JSON
        output_path: Path to save report
    
    Returns:
        Report dict
    """
    # Load data
    with open(detected_fields_path, 'r') as f:
        detected_data = json.load(f)
    
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    detected_fields = detected_data.get('fields', [])
    gt_fields = ground_truth.get('fields', [])
    
    # Match fields
    matched_fields = []
    false_positives = []
    false_negatives = []
    
    for gt_field in gt_fields:
        gt_bbox = gt_field['bbox']
        matched = False
        
        for det_field in detected_fields:
            det_bbox = det_field['bbox']
            if _bboxes_overlap(gt_bbox, det_bbox):
                # Calculate position error
                gt_center_x = (gt_bbox[0] + gt_bbox[2]) / 2
                gt_center_y = (gt_bbox[1] + gt_bbox[3]) / 2
                det_center_x = (det_bbox[0] + det_bbox[2]) / 2
                det_center_y = (det_bbox[1] + det_bbox[3]) / 2
                
                position_error = ((gt_center_x - det_center_x)**2 + 
                                 (gt_center_y - det_center_y)**2)**0.5
                
                matched_fields.append({
                    'gt_name': gt_field.get('name', 'unknown'),
                    'det_name': det_field.get('interview_variable', 'unknown'),
                    'position_error': position_error,
                    'confidence': det_field.get('confidence', 0),
                    'status': 'correct',
                    'bbox': det_bbox
                })
                matched = True
                break
        
        if not matched:
            false_negatives.append({
                'name': gt_field.get('name', 'unknown'),
                'status': 'missed',
                'bbox': gt_bbox,
                'recommendations': [
                    'Check if visual detection missed this element',
                    'Verify label text is extractable',
                    'Check if field is in page margins'
                ]
            })
    
    # Find false positives
    for det_field in detected_fields:
        det_bbox = det_field['bbox']
        is_false_positive = True
        
        for gt_field in gt_fields:
            gt_bbox = gt_field['bbox']
            if _bboxes_overlap(gt_bbox, det_bbox):
                is_false_positive = False
                break
        
        if is_false_positive:
            false_positives.append({
                'name': det_field.get('interview_variable', 'unknown'),
                'status': 'false_positive',
                'confidence': det_field.get('confidence', 0),
                'bbox': det_bbox,
                'recommendations': [
                    'Likely decorative element or table border',
                    'Consider stricter size/aspect ratio validation',
                    'May need better context filtering'
                ]
            })
    
    # Calculate metrics
    total_gt = len(gt_fields)
    total_detected = len(detected_fields)
    true_positives = len(matched_fields)
    fp_count = len(false_positives)
    fn_count = len(false_negatives)
    
    detection_rate = true_positives / total_gt if total_gt > 0 else 0
    precision = true_positives / total_detected if total_detected > 0 else 0
    fp_rate = fp_count / total_detected if total_detected > 0 else 0
    
    avg_position_error = sum(f['position_error'] for f in matched_fields) / len(matched_fields) if matched_fields else 0
    
    # Grade
    if detection_rate >= 0.95 and avg_position_error < 3:
        grade = 'A+'
    elif detection_rate >= 0.90 and avg_position_error < 5:
        grade = 'A'
    elif detection_rate >= 0.80 and avg_position_error < 10:
        grade = 'B'
    elif detection_rate >= 0.70 and avg_position_error < 15:
        grade = 'C'
    elif detection_rate >= 0.50 and avg_position_error < 25:
        grade = 'D'
    else:
        grade = 'F'
    
    # Build report
    report = {
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'grade': grade,
            'detection_rate': f"{detection_rate:.1%}",
            'precision': f"{precision:.1%}",
            'false_positive_rate': f"{fp_rate:.1%}",
            'avg_position_error': f"{avg_position_error:.2f}px",
            'total_ground_truth': total_gt,
            'total_detected': total_detected,
            'true_positives': true_positives,
            'false_positives': fp_count,
            'false_negatives': fn_count
        },
        'field_details': {
            'correct_matches': matched_fields,
            'false_positives': false_positives,
            'missed_fields': false_negatives
        },
        'recommendations': []
    }
    
    # Generate recommendations
    if fp_rate > 0.10:
        report['recommendations'].append({
            'priority': 'HIGH',
            'issue': f'High false positive rate ({fp_rate:.1%})',
            'actions': [
                'Increase confidence threshold',
                'Add stricter size validation',
                'Improve spatial context filtering',
                'Review detected fields manually'
            ]
        })
    
    if avg_position_error > 10:
        report['recommendations'].append({
            'priority': 'HIGH',
            'issue': f'Position error too large ({avg_position_error:.1f}px)',
            'actions': [
                'Tighten spatial matching tolerance',
                'Verify label extraction accuracy',
                'Check coordinate system consistency'
            ]
        })
    
    if fn_count > 0:
        report['recommendations'].append({
            'priority': 'MEDIUM',
            'issue': f'{fn_count} fields missed',
            'actions': [
                'Check if visual detection is working',
                'Verify text extraction finds all labels',
                'Review margin filtering thresholds'
            ]
        })
    
    if fn_count == 0 and fp_count <= 1 and detection_rate >= 0.95:
        report['recommendations'].append({
            'priority': 'INFO',
            'issue': 'Excellent performance!',
            'actions': [
                'System is ready for production use',
                'Test on additional forms to verify consistency',
                'Document optimal parameters for future use'
            ]
        })
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


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


def print_report(report: Dict):
    """Print report to console in readable format."""
    print(f"\n{'='*80}")
    print(f"  QUALITY REPORT")
    print(f"{'='*80}\n")
    
    summary = report['summary']
    
    print(f"📊 OVERALL GRADE: {summary['grade']}\n")
    
    print(f"Detection Metrics:")
    print(f"  Detection Rate:       {summary['detection_rate']}")
    print(f"  Precision:            {summary['precision']}")
    print(f"  False Positive Rate:  {summary['false_positive_rate']}")
    print(f"  Avg Position Error:   {summary['avg_position_error']}")
    
    print(f"\nField Counts:")
    print(f"  Ground Truth:     {summary['total_ground_truth']} fields")
    print(f"  Detected:         {summary['total_detected']} fields")
    print(f"  ✅ Correct:       {summary['true_positives']} fields")
    print(f"  ❌ False Pos:     {summary['false_positives']} fields")
    print(f"  ❌ Missed:        {summary['false_negatives']} fields")
    
    if report['recommendations']:
        print(f"\n💡 Recommendations:\n")
        for rec in report['recommendations']:
            print(f"  {rec['priority']}: {rec['issue']}")
            for action in rec['actions']:
                print(f"    - {action}")
            print()
    
    print(f"Generated: {report['generated_at']}")


def main():
    """Main entry point."""
    if len(sys.argv) < 4:
        print("Usage: python generate_quality_report.py <detected_fields> <ground_truth> <output>")
        print("\nExample:")
        print("  python generate_quality_report.py \\")
        print("    output/test/form_da_ready.json \\")
        print("    tools/pdf_annotation/data/ground_truth/form_ground_truth.json \\")
        print("    output/test/quality_report.json")
        sys.exit(1)
    
    detected_path = sys.argv[1]
    ground_truth_path = sys.argv[2]
    output_path = sys.argv[3]
    
    report = generate_quality_report(detected_path, ground_truth_path, output_path)
    print_report(report)
    
    print(f"\n💾 Report saved to: {output_path}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

