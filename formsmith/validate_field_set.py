#!/usr/bin/env python3
"""
Field Set Validation
Pre-flight validation before outputting fields.
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict


def validate_field_set(fields_json_path: str) -> Tuple[bool, List[str], Dict]:
    """
    Validate field set before output.
    
    Checks:
    - No negative dimensions
    - No duplicate names
    - All required types present
    - Reasonable field count (not too many/few)
    - Valid confidence scores
    - Valid bounding boxes
    
    Args:
        fields_json_path: Path to fields JSON (DA-ready format)
    
    Returns:
        Tuple of (is_valid, issues_list, stats_dict)
    """
    # Load fields
    with open(fields_json_path, 'r') as f:
        data = json.load(f)
    
    fields = data.get('fields', [])
    issues = []
    stats = {
        'total_fields': len(fields),
        'field_types': {},
        'avg_confidence': 0,
        'min_confidence': 1.0,
        'max_confidence': 0,
        'negative_dimensions': 0,
        'duplicate_names': 0,
        'invalid_bboxes': 0,
        'too_small': 0,
        'too_large': 0
    }
    
    if not fields:
        issues.append("CRITICAL: No fields detected")
        return False, issues, stats
    
    # Check dimensions
    for field in fields:
        bbox = field.get('bbox', [0, 0, 0, 0])
        
        # Check valid bbox format
        if len(bbox) != 4:
            issues.append(f"Invalid bbox format for {field.get('interview_variable', 'unknown')}")
            stats['invalid_bboxes'] += 1
            continue
        
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Check for negative dimensions
        if width <= 0 or height <= 0:
            issues.append(f"CRITICAL: Negative dimensions for {field.get('interview_variable', 'unknown')}: "
                        f"width={width:.1f}, height={height:.1f}")
            stats['negative_dimensions'] += 1
        
        # Check for suspiciously small fields
        if width < 5 or height < 5:
            issues.append(f"WARNING: Very small field {field.get('interview_variable', 'unknown')}: "
                        f"{width:.1f}x{height:.1f}px")
            stats['too_small'] += 1
        
        # Check for suspiciously large fields
        if width > 500 or height > 100:
            issues.append(f"WARNING: Very large field {field.get('interview_variable', 'unknown')}: "
                        f"{width:.1f}x{height:.1f}px")
            stats['too_large'] += 1
    
    # Check for duplicate names
    names = [f.get('interview_variable', f.get('pdf_name', 'unknown')) for f in fields]
    unique_names = set(names)
    
    if len(names) != len(unique_names):
        duplicate_count = len(names) - len(unique_names)
        issues.append(f"CRITICAL: {duplicate_count} duplicate field names detected")
        stats['duplicate_names'] = duplicate_count
        
        # Find which names are duplicated
        from collections import Counter
        name_counts = Counter(names)
        duplicates = [name for name, count in name_counts.items() if count > 1]
        for dup_name in duplicates[:5]:  # Show first 5
            issues.append(f"  → Duplicate: {dup_name} (appears {name_counts[dup_name]} times)")
    
    # Check field count heuristics
    total_fields = len(fields)
    
    if total_fields > 200:
        issues.append(f"WARNING: Very high field count ({total_fields}), "
                     "likely has false positives")
    elif total_fields < 5:
        issues.append(f"WARNING: Very low field count ({total_fields}), "
                     "detection may have failed")
    
    # Count field types
    for field in fields:
        field_type = field.get('pdf_type', field.get('interview_type', 'unknown'))
        stats['field_types'][field_type] = stats['field_types'].get(field_type, 0) + 1
    
    # Check confidence scores
    confidences = [f.get('confidence', 0) for f in fields]
    if confidences:
        stats['avg_confidence'] = sum(confidences) / len(confidences)
        stats['min_confidence'] = min(confidences)
        stats['max_confidence'] = max(confidences)
        
        low_confidence_count = sum(1 for c in confidences if c < 0.75)
        if low_confidence_count > 0:
            issues.append(f"WARNING: {low_confidence_count} fields with low confidence (<0.75)")
    
    # Check for required field types (for legal forms)
    has_text = 'text' in stats['field_types']
    if not has_text:
        issues.append("WARNING: No text fields detected, may be incorrect")
    
    # Determine if valid
    critical_issues = [i for i in issues if i.startswith('CRITICAL')]
    is_valid = len(critical_issues) == 0
    
    return is_valid, issues, stats


def print_validation_result(is_valid: bool, issues: List[str], stats: Dict):
    """Print validation results in readable format."""
    print(f"\n{'='*80}")
    print(f"  FIELD SET VALIDATION")
    print(f"{'='*80}\n")
    
    # Print overall status
    if is_valid:
        print(f"✅ VALIDATION PASSED\n")
    else:
        print(f"❌ VALIDATION FAILED\n")
    
    # Print stats
    print(f"📊 Field Statistics:")
    print(f"   Total Fields:     {stats['total_fields']}")
    print(f"   Avg Confidence:   {stats['avg_confidence']:.2f}")
    print(f"   Min Confidence:   {stats['min_confidence']:.2f}")
    print(f"   Max Confidence:   {stats['max_confidence']:.2f}")
    
    print(f"\n   Field Types:")
    for field_type, count in stats['field_types'].items():
        print(f"     {field_type}: {count}")
    
    print(f"\n   Quality Issues:")
    print(f"     Negative Dimensions:  {stats['negative_dimensions']}")
    print(f"     Duplicate Names:      {stats['duplicate_names']}")
    print(f"     Invalid Bboxes:       {stats['invalid_bboxes']}")
    print(f"     Too Small:            {stats['too_small']}")
    print(f"     Too Large:            {stats['too_large']}")
    
    # Print issues
    if issues:
        print(f"\n⚠️  Issues Found:\n")
        for issue in issues:
            if issue.startswith('CRITICAL'):
                print(f"   🔴 {issue}")
            elif issue.startswith('WARNING'):
                print(f"   🟡 {issue}")
            else:
                print(f"   ℹ️  {issue}")
    else:
        print(f"\n✅ No issues found - field set looks good!")
    
    print()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python validate_field_set.py <fields_json>")
        print("\nExample:")
        print("  python validate_field_set.py output/test/form_da_ready.json")
        sys.exit(1)
    
    fields_json_path = sys.argv[1]
    
    is_valid, issues, stats = validate_field_set(fields_json_path)
    print_validation_result(is_valid, issues, stats)
    
    # Exit with non-zero if validation failed
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    sys.exit(main())

