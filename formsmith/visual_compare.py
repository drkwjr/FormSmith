#!/usr/bin/env python3
"""
Phase 2.2: Visual Comparison Tool

Creates side-by-side visual comparison of detected fields vs. ground truth.
Color-codes accuracy for easy identification of issues.

Color Legend:
  - Green:  Correct (within 2px)
  - Yellow: Close (within 10px)
  - Red:    Wrong (>10px off)
  - Purple: False positive (detected but not in ground truth)
  - Blue:   False negative (in ground truth but not detected)
"""

import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont


def load_json(path: str) -> Dict:
    """Load JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def calculate_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate distance between bbox centers"""
    # Convert to center points
    x1 = bbox1[0] + (abs(bbox1[2]) / 2 if len(bbox1) > 2 else 0)
    y1 = bbox1[1] + (abs(bbox1[3]) / 2 if len(bbox1) > 3 else 0)
    
    x2 = bbox2[0] + (abs(bbox2[2]) / 2 if len(bbox2) > 2 else 0)
    y2 = bbox2[1] + (abs(bbox2[3]) / 2 if len(bbox2) > 3 else 0)
    
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def match_fields(detected: List[Dict], ground_truth: List[Dict], max_distance: float = 50) -> Tuple[List, List, List]:
    """
    Match detected fields to ground truth
    
    Returns:
        (matched, false_positives, false_negatives)
    """
    matched = []
    false_positives = []
    used_gt = set()
    
    # Try to match each detected field
    for d_field in detected:
        # Get bbox
        if 'bbox' in d_field:
            d_bbox = d_field['bbox']
        elif 'x' in d_field:
            d_bbox = [
                d_field['x'],
                d_field.get('y_from_top', d_field.get('y', 0)),
                abs(d_field.get('width', 0)),
                abs(d_field.get('height', 0))
            ]
        else:
            continue
        
        # Find closest ground truth
        best_match = None
        best_distance = float('inf')
        best_idx = None
        
        for idx, gt_field in enumerate(ground_truth):
            if idx in used_gt:
                continue
            
            if 'bbox' in gt_field:
                gt_bbox = gt_field['bbox']
            elif 'x' in gt_field:
                gt_bbox = [
                    gt_field['x'],
                    gt_field.get('y_from_top', gt_field.get('y', 0)),
                    gt_field.get('width', 0),
                    gt_field.get('height', 0)
                ]
            else:
                continue
            
            distance = calculate_distance(d_bbox, gt_bbox)
            
            if distance < best_distance and distance <= max_distance:
                best_distance = distance
                best_match = gt_field
                best_idx = idx
        
        if best_match:
            matched.append({
                'detected': d_field,
                'ground_truth': best_match,
                'distance': best_distance
            })
            used_gt.add(best_idx)
        else:
            false_positives.append(d_field)
    
    # Find false negatives (ground truth not matched)
    false_negatives = [gt for idx, gt in enumerate(ground_truth) if idx not in used_gt]
    
    return matched, false_positives, false_negatives


def render_comparison(
    pdf_path: str,
    detected: List[Dict],
    ground_truth: List[Dict],
    output_path: str,
    dpi: int = 150
):
    """
    Render side-by-side comparison image
    
    Args:
        pdf_path: Path to PDF
        detected: List of detected fields
        ground_truth: List of ground truth fields
        output_path: Where to save comparison image
        dpi: Resolution for rendering
    """
    print(f"\n{'='*80}")
    print("  VISUAL COMPARISON")
    print(f"{'='*80}\n")
    
    # Match fields
    matched, false_positives, false_negatives = match_fields(detected, ground_truth)
    
    print(f"Matched fields:      {len(matched)}")
    print(f"False positives:     {len(false_positives)}")
    print(f"False negatives:     {len(false_negatives)}")
    
    # Open PDF and render
    doc = fitz.open(pdf_path)
    page = doc[0]
    
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=matrix)
    
    scale_x = pix.width / page.rect.width
    scale_y = pix.height / page.rect.height
    
    # Create two images (detected and ground truth)
    base_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    detected_img = base_img.copy()
    gt_img = base_img.copy()
    
    draw_detected = ImageDraw.Draw(detected_img, "RGBA")
    draw_gt = ImageDraw.Draw(gt_img, "RGBA")
    
    # Try to load font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except:
        font = ImageFont.load_default()
        label_font = font
    
    # Draw ground truth (blue) on GT image
    print("\nDrawing ground truth fields...")
    for gt_field in ground_truth:
        if 'bbox' in gt_field:
            x, y, w, h = gt_field['bbox'][0], gt_field['bbox'][1], gt_field['bbox'][2] - gt_field['bbox'][0], gt_field['bbox'][3] - gt_field['bbox'][1]
        else:
            x = gt_field.get('x', 0)
            y = gt_field.get('y_from_top', gt_field.get('y', 0))
            w = gt_field.get('width', 0)
            h = gt_field.get('height', 0)
        
        left = x * scale_x
        top = y * scale_y
        right = (x + w) * scale_x
        bottom = (y + h) * scale_y
        
        # Blue for ground truth
        color = (0, 100, 255, 120)
        border = (0, 50, 200)
        
        draw_gt.rectangle([left, top, right, bottom], fill=color, outline=border, width=2)
        
        # Label
        name = gt_field.get('name', 'unnamed')
        draw_gt.text((left + 2, top + 2), name, fill=(255, 255, 255), font=label_font)
    
    # Draw detected fields on detected image with color coding
    print("Drawing detected fields...")
    for match in matched:
        d_field = match['detected']
        distance = match['distance']
        
        # Get bbox
        if 'bbox' in d_field:
            x, y = d_field['bbox'][0], d_field['bbox'][1]
            w, h = d_field['bbox'][2] - d_field['bbox'][0], d_field['bbox'][3] - d_field['bbox'][1]
        else:
            x = d_field.get('x', 0)
            y = d_field.get('y_from_top', d_field.get('y', 0))
            w = abs(d_field.get('width', 0))
            h = abs(d_field.get('height', 0))
        
        left = x * scale_x
        top = y * scale_y
        right = (x + w) * scale_x
        bottom = (y + h) * scale_y
        
        # Color based on accuracy
        if distance < 2:
            color = (0, 255, 0, 100)  # Green - perfect
            border = (0, 200, 0)
            label_text = "✓"
        elif distance < 10:
            color = (255, 255, 0, 100)  # Yellow - close
            border = (200, 200, 0)
            label_text = f"~{distance:.0f}px"
        else:
            color = (255, 0, 0, 100)  # Red - wrong
            border = (200, 0, 0)
            label_text = f"✗{distance:.0f}px"
        
        if left < right and top < bottom:  # Valid dimensions
            draw_detected.rectangle([left, top, right, bottom], fill=color, outline=border, width=2)
            
            # Label
            name = d_field.get('name', 'unnamed')
            draw_detected.text((left + 2, top + 2), f"{name} {label_text}", fill=(0, 0, 0), font=label_font)
    
    # Draw false positives (purple) on detected image
    print("Drawing false positives...")
    for fp_field in false_positives:
        if 'bbox' in fp_field:
            x, y = fp_field['bbox'][0], fp_field['bbox'][1]
            w, h = fp_field['bbox'][2] - fp_field['bbox'][0], fp_field['bbox'][3] - fp_field['bbox'][1]
        else:
            x = fp_field.get('x', 0)
            y = fp_field.get('y_from_top', fp_field.get('y', 0))
            w = abs(fp_field.get('width', 0))
            h = abs(fp_field.get('height', 0))
        
        left = x * scale_x
        top = y * scale_y
        right = (x + w) * scale_x
        bottom = (y + h) * scale_y
        
        color = (255, 0, 255, 120)  # Purple - false positive
        border = (200, 0, 200)
        
        if left < right and top < bottom:
            draw_detected.rectangle([left, top, right, bottom], fill=color, outline=border, width=2)
            name = fp_field.get('name', 'unnamed')
            draw_detected.text((left + 2, top + 2), f"{name} [FP]", fill=(255, 255, 255), font=label_font)
    
    # Draw false negatives (cyan) on ground truth image
    print("Drawing false negatives...")
    for fn_field in false_negatives:
        if 'bbox' in fn_field:
            x, y = fn_field['bbox'][0], fn_field['bbox'][1]
            w, h = fn_field['bbox'][2] - fn_field['bbox'][0], fn_field['bbox'][3] - fn_field['bbox'][1]
        else:
            x = fn_field.get('x', 0)
            y = fn_field.get('y_from_top', fn_field.get('y', 0))
            w = fn_field.get('width', 0)
            h = fn_field.get('height', 0)
        
        left = x * scale_x
        top = y * scale_y
        right = (x + w) * scale_x
        bottom = (y + h) * scale_y
        
        color = (0, 255, 255, 150)  # Cyan - missed
        border = (0, 200, 200)
        
        draw_gt.rectangle([left, top, right, bottom], fill=color, outline=border, width=3)
        name = fn_field.get('name', 'unnamed')
        draw_gt.text((left + 2, top + 2), f"{name} [MISSED]", fill=(0, 0, 0), font=label_font)
    
    # Create side-by-side image
    print("Creating side-by-side comparison...")
    combined_width = detected_img.width + gt_img.width + 40  # 20px margin on each side
    combined_height = max(detected_img.height, gt_img.height) + 100  # Space for legend
    
    combined = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
    
    # Paste images
    combined.paste(detected_img, (20, 50))
    combined.paste(gt_img, (detected_img.width + 20, 50))
    
    # Add labels and legend
    draw_combined = ImageDraw.Draw(combined)
    
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica-Bold.ttc", 18)
    except:
        title_font = font
    
    # Titles
    draw_combined.text((detected_img.width // 2, 10), "DETECTED FIELDS", fill=(0, 0, 0), font=title_font, anchor="mm")
    draw_combined.text((detected_img.width + 20 + gt_img.width // 2, 10), "GROUND TRUTH", fill=(0, 0, 0), font=title_font, anchor="mm")
    
    # Legend
    legend_y = detected_img.height + 60
    legend_items = [
        ("Green", "Correct (<2px)"),
        ("Yellow", "Close (<10px)"),
        ("Red", "Wrong (>10px)"),
        ("Purple", "False Positive"),
        ("Cyan", "False Negative (missed)")
    ]
    
    legend_colors = {
        "Green": (0, 200, 0),
        "Yellow": (200, 200, 0),
        "Red": (200, 0, 0),
        "Purple": (200, 0, 200),
        "Cyan": (0, 200, 200)
    }
    
    x_offset = 40
    for color_name, label in legend_items:
        color = legend_colors[color_name]
        draw_combined.rectangle([x_offset, legend_y, x_offset + 20, legend_y + 20], fill=color)
        draw_combined.text((x_offset + 30, legend_y + 10), f"{color_name}: {label}", fill=(0, 0, 0), font=label_font, anchor="lm")
        x_offset += 200
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path)
    
    print(f"\n✅ Comparison saved to: {output_path}")
    print(f"   Image size: {combined.width}x{combined.height}")
    
    doc.close()
    
    return matched, false_positives, false_negatives


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visual Field Comparison Tool")
    parser.add_argument('pdf', help='PDF file')
    parser.add_argument('detected', help='Detected fields JSON')
    parser.add_argument('ground_truth', help='Ground truth fields JSON')
    parser.add_argument('--output', '-o', default='comparison.png', help='Output image path')
    parser.add_argument('--dpi', type=int, default=150, help='Rendering DPI')
    
    args = parser.parse_args()
    
    # Load data
    detected_data = load_json(args.detected)
    gt_data = load_json(args.ground_truth)
    
    detected_fields = detected_data.get('fields', detected_data.get('refinements', []))
    gt_fields = gt_data.get('fields', [])
    
    # Render comparison
    render_comparison(
        args.pdf,
        detected_fields,
        gt_fields,
        args.output,
        args.dpi
    )
    
    print("\nOpen the comparison image to visually inspect accuracy!")

