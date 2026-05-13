#!/usr/bin/env python3
"""
Improved Field Mapper - Phase 3.1 Implementation

Fixes critical bugs and adds multi-modal field detection:
1. Text-based detection (improved)
2. Visual element detection (NEW - OpenCV)
3. Spatial reasoning (NEW - label → field relationship)

This replaces the broken intelligent_field_mapper.py
"""

import sys
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

# Try to import OpenCV (optional for visual detection)
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("⚠️  Warning: OpenCV not installed. Visual element detection disabled.")
    print("   Install with: pip install opencv-python")

# Try to import pdfminer (for text extraction)
try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LTTextBox, LTChar
    HAS_PDFMINER = True
except ImportError:
    HAS_PDFMINER = False
    print("⚠️  Warning: pdfminer.six not installed. Advanced text extraction disabled.")
    print("   Install with: pip install pdfminer.six")


class ImprovedFieldMapper:
    """
    Multi-modal field detection with critical bug fixes
    """
    
    def __init__(self):
        """Initialize without PDF - will be passed to detect_all()"""
        self.pdf_path = None
        self.doc = None
        self.page = None
        self.page_height = None
        self.page_width = None
        self.text_dict = None
        self.visual_elements = []
        self.existing_fields = []
    
    def detect_all(self, pdf_path: str) -> List[Dict]:
        """
        Detect all visual elements on PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of detected elements in standardized format:
            [
                {
                    'bbox': [x0, y0, x1, y1],
                    'type': 'text' or 'checkbox',
                    'page': int,
                    'confidence': float,
                    'method': 'hough_line_detection' or 'contour_detection' or 'text_pattern'
                },
                ...
            ]
        """
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.page = self.doc[0]  # Focus on first page for now
        self.page_height = self.page.rect.height
        self.page_width = self.page.rect.width
        
        try:
            # Run detection
            fields = self.detect_fields()
            return fields
        finally:
            # Always close document
            if self.doc:
                self.doc.close()
        
    def detect_fields(self) -> List[Dict]:
        """
        Main detection method - combines all approaches
        """
        print("\n" + "="*80)
        print("  IMPROVED FIELD DETECTION")
        print("="*80 + "\n")
        
        # Extract all data first
        self._extract_text()
        self._extract_visual_elements()
        self._extract_existing_fields()
        
        # Run all detection methods
        detected_fields = []
        
        # Method 1: Existing fields (if PDF has them)
        if self.existing_fields:
            print(f"✅ Found {len(self.existing_fields)} existing fields")
            detected_fields.extend(self.existing_fields)
        
        # Method 2: Text-based detection (improved)
        text_fields = self._detect_from_text()
        print(f"✅ Text-based detection found {len(text_fields)} potential fields")
        detected_fields.extend(text_fields)
        
        # Method 3: Visual element detection (NEW!)
        if HAS_OPENCV:
            visual_fields = self._detect_from_visual()
            print(f"✅ Visual detection found {len(visual_fields)} potential fields")
            detected_fields.extend(visual_fields)
        
        # Merge nearby detections (same field detected multiple ways)
        merged_fields = self._merge_nearby_detections(detected_fields)
        print(f"✅ Merged to {len(merged_fields)} unique fields")
        
        # Calculate confidence scores
        for field in merged_fields:
            field['confidence'] = self._calculate_confidence(field, merged_fields)
        
        # Sort by confidence
        merged_fields.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Filter low confidence
        filtered_fields = [f for f in merged_fields if f['confidence'] >= 0.4]
        print(f"✅ Filtered to {len(filtered_fields)} high-confidence fields")
        
        return filtered_fields
    
    def _extract_text(self):
        """Extract text with precise positions"""
        self.text_dict = self.page.get_text("dict")
    
    def _extract_visual_elements(self):
        """Extract visual elements (lines, rectangles)"""
        # Try vector drawings first
        drawings = self.page.get_drawings()
        self.visual_elements = []
        
        for drawing in drawings:
            if drawing.get('type') == 'line':
                self.visual_elements.append({
                    'type': 'line',
                    'bbox': drawing.get('rect', [0, 0, 0, 0]),
                    'method': 'vector_drawing'
                })
            elif drawing.get('type') == 'rect':
                self.visual_elements.append({
                    'type': 'rectangle',
                    'bbox': drawing.get('rect', [0, 0, 0, 0]),
                    'method': 'vector_drawing'
                })
    
    def _extract_existing_fields(self):
        """Extract existing form fields if any"""
        self.existing_fields = []
        
        for widget in self.page.widgets():
            rect = widget.rect
            
            # Map widget types
            type_map = {
                7: "text",
                2: "checkbox",
                6: "signature"
            }
            
            self.existing_fields.append({
                'name': widget.field_name or f"field_{len(self.existing_fields)}",
                'type': type_map.get(widget.field_type, 'text'),
                'bbox': [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                'x': float(rect.x0),
                'y_from_top': float(rect.y0),
                'width': float(rect.x1 - rect.x0),
                'height': float(rect.y1 - rect.y0),
                'confidence': 1.0,
                'method': 'existing_field',
                'source': 'existing'
            })
    
    def _detect_from_text(self) -> List[Dict]:
        """
        Improved text-based detection with spatial reasoning
        """
        fields = []
        
        if not self.text_dict:
            return fields
        
        # Find all text labels (ending with ":")
        for block in self.text_dict.get("blocks", []):
            if block.get("type") != 0:  # Not text
                continue
            
            for line in block.get("lines", []):
                # Get full line text
                line_text = "".join([span.get("text", "") for span in line.get("spans", [])])
                line_bbox = line.get("bbox", [0, 0, 0, 0])
                
                # Check if it's a label (ends with ":")
                if line_text.strip().endswith(":"):
                    # This is a label - find field AFTER it using spatial reasoning
                    field = self._find_field_after_label(line_text, line_bbox)
                    
                    if field:
                        field['label'] = line_text.strip()
                        field['method'] = 'text_with_spatial_reasoning'
                        fields.append(field)
                
                # Check for other indicators
                elif "___" in line_text or "____" in line_text:
                    # Underscores indicate a text field
                    fields.append({
                        'name': self._generate_field_name(line_text),
                        'type': 'text',
                        'bbox': line_bbox,
                        'x': line_bbox[0],
                        'y_from_top': line_bbox[1],
                        'width': max(0, line_bbox[2] - line_bbox[0]),  # FIX: Ensure non-negative!
                        'height': max(0, line_bbox[3] - line_bbox[1]),  # FIX: Ensure non-negative!
                        'confidence': 0.7,
                        'method': 'underscore_detection',
                        'label': line_text.strip()
                    })
        
        return fields
    
    def _find_field_after_label(self, label_text: str, label_bbox: List[float]) -> Optional[Dict]:
        """
        KEY INNOVATION: Find actual field location after label using spatial reasoning
        """
        label_x1 = label_bbox[2]  # Right edge of label
        label_y_center = (label_bbox[1] + label_bbox[3]) / 2
        
        # Look for blank space after label
        # Scan horizontally from label end to find next text element
        next_element_x = None
        min_distance = float('inf')
        
        for block in self.text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            
            for line in block.get("lines", []):
                line_bbox = line.get("bbox", [0, 0, 0, 0])
                line_x0 = line_bbox[0]
                line_y_center = (line_bbox[1] + line_bbox[3]) / 2
                
                # Is this element on same line (same vertical position)?
                if abs(line_y_center - label_y_center) < 5:
                    # Is it to the right of label?
                    if line_x0 > label_x1:
                        distance = line_x0 - label_x1
                        if distance < min_distance:
                            min_distance = distance
                            next_element_x = line_x0
        
        # Calculate field dimensions
        if next_element_x and next_element_x > label_x1 + 10:
            # Field spans from label to next element
            field_width = next_element_x - label_x1 - 10  # 5px margin on each side
        else:
            # No next element found, use default width
            field_width = min(150, self.page_width - label_x1 - 20)
        
        # CRITICAL FIX: Validate dimensions!
        if field_width <= 0:
            field_width = 100  # Fallback to reasonable default
        
        field_height = 15  # Standard height
        
        field_x = label_x1 + 5  # 5px gap after label
        field_y = label_y_center - (field_height / 2)  # Vertically centered
        
        # Validate field is within page bounds
        if field_x + field_width > self.page_width:
            field_width = self.page_width - field_x - 10
        
        if field_width <= 0:
            return None  # Can't place field
        
        return {
            'name': self._generate_field_name(label_text),
            'type': 'text',
            'bbox': [field_x, field_y, field_x + field_width, field_y + field_height],
            'x': field_x,
            'y_from_top': field_y,
            'width': field_width,
            'height': field_height,
            'confidence': 0.75
        }
    
    def _detect_from_visual(self) -> List[Dict]:
        """
        NEW: Visual element detection using OpenCV
        Detects underscores and checkboxes from pixel analysis
        """
        if not HAS_OPENCV:
            return []
        
        fields = []
        
        try:
            # Render page to image
            pix = self.page.get_pixmap(dpi=150)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Scale factors for coordinate conversion
            scale_x = self.page_width / pix.width
            scale_y = self.page_height / pix.height
            
            # Detect horizontal lines (underscores) using Hough transform
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                                   minLineLength=30, maxLineGap=5)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Is it horizontal? (y difference small)
                    if abs(y2 - y1) < 5 and (x2 - x1) > 30:
                        # Convert image coords to PDF coords
                        pdf_x1 = x1 * scale_x
                        pdf_y1 = y1 * scale_y
                        pdf_x2 = x2 * scale_x
                        pdf_y2 = y2 * scale_y
                        
                        # Create field at this location
                        field_height = 15
                        fields.append({
                            'name': f"text_field_line_{len(fields)}",
                            'type': 'text',
                            'bbox': [pdf_x1, pdf_y1 - 5, pdf_x2, pdf_y1 + field_height - 5],
                            'x': pdf_x1,
                            'y_from_top': pdf_y1 - 5,
                            'width': pdf_x2 - pdf_x1,
                            'height': field_height,
                            'confidence': 0.9,  # High confidence - actual visual element!
                            'method': 'hough_line_detection'
                        })
            
            # Detect small rectangles (checkboxes) using contour detection
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Is it checkbox-sized and square-ish?
                aspect_ratio = w / h if h > 0 else 0
                if 0.7 < aspect_ratio < 1.3 and 8 < w < 25 and 8 < h < 25:
                    # Convert to PDF coords
                    pdf_x = x * scale_x
                    pdf_y = y * scale_y
                    pdf_w = w * scale_x
                    pdf_h = h * scale_y
                    
                    fields.append({
                        'name': f"checkbox_{len(fields)}",
                        'type': 'checkbox',
                        'bbox': [pdf_x, pdf_y, pdf_x + pdf_w, pdf_y + pdf_h],
                        'x': pdf_x,
                        'y_from_top': pdf_y,
                        'width': pdf_w,
                        'height': pdf_h,
                        'confidence': 0.95,  # Very high - actual visual box!
                        'method': 'contour_detection'
                    })
        
        except Exception as e:
            print(f"⚠️  Visual detection error: {e}")
        
        return fields
    
    def _merge_nearby_detections(self, fields: List[Dict], distance_threshold: float = 15) -> List[Dict]:
        """
        Merge fields detected by multiple methods
        """
        if not fields:
            return []
        
        merged = []
        used = set()
        
        for i, field1 in enumerate(fields):
            if i in used:
                continue
            
            # Find all fields near this one
            nearby = [field1]
            bbox1 = field1['bbox']
            center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
            
            for j, field2 in enumerate(fields):
                if i == j or j in used:
                    continue
                
                bbox2 = field2['bbox']
                center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
                
                # Calculate distance between centers
                distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                if distance < distance_threshold:
                    nearby.append(field2)
                    used.add(j)
            
            # Merge nearby fields (use highest confidence one as base)
            nearby.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            best = nearby[0].copy()
            
            # Track which methods detected this field
            methods = [f.get('method', 'unknown') for f in nearby]
            best['detection_methods'] = methods
            best['detection_count'] = len(nearby)
            
            merged.append(best)
            used.add(i)
        
        return merged
    
    def _calculate_confidence(self, field: Dict, all_fields: List[Dict]) -> float:
        """
        Calculate confidence score for a field
        """
        confidence = field.get('confidence', 0.5)
        
        # Boost: Multiple detection methods (cross-validation!)
        detection_count = field.get('detection_count', 1)
        if detection_count >= 2:
            confidence += 0.2
        elif detection_count >= 3:
            confidence += 0.3
        
        # Boost: Reasonable dimensions
        width = field.get('width', 0)
        height = field.get('height', 0)
        if 20 < width < 400 and 8 < height < 30:
            confidence += 0.1
        
        # Penalty: Too small or too large
        if width < 10 or height < 5:
            confidence -= 0.3
        if width > 500 or height > 100:
            confidence -= 0.3
        
        # Penalty: Near page edge (likely not a field)
        x = field.get('x', 0)
        y = field.get('y_from_top', 0)
        if x < 20 or x > self.page_width - 20:
            confidence -= 0.2
        if y < 30 or y > self.page_height - 30:
            confidence -= 0.2
        
        # Penalty: Overlaps significantly with another field
        bbox1 = field['bbox']
        for other in all_fields:
            if other is field:
                continue
            
            bbox2 = other['bbox']
            iou = self._intersection_over_union(bbox1, bbox2)
            
            if iou > 0.5:
                confidence -= 0.4
                break
        
        return max(0, min(1, confidence))
    
    def _intersection_over_union(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Intersection
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)
        
        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            return 0.0
        
        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _generate_field_name(self, label_text: str) -> str:
        """
        Generate field name from label text
        """
        import re
        
        # Clean text
        text = label_text.lower().strip()
        text = text.replace(":", "").strip()
        
        # Replace spaces with underscores
        text = re.sub(r'\s+', '_', text)
        
        # Remove special characters
        text = re.sub(r'[^\w_]', '', text)
        
        # Common mappings
        mappings = {
            'petitioner_s_name': 'petitioner_name',
            'defendant_s_name': 'defendant_name',
            'date_of_birth': 'date_of_birth',
            'docket_no': 'docket_number',
            'docket_number': 'docket_number',
        }
        
        return mappings.get(text, text) if text else 'unnamed_field'
    
    def close(self):
        """Close PDF document"""
        if self.doc:
            self.doc.close()


def main():
    """Test the improved field mapper"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python improved_field_mapper.py <pdf_file> [output_json]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else "improved_detected_fields.json"
    
    # Run detection
    mapper = ImprovedFieldMapper()
    fields = mapper.detect_all(pdf_path)
    
    # Save results
    output = {
        'pdf': Path(pdf_path).name,
        'method': 'improved_field_mapper',
        'field_count': len(fields),
        'fields': fields
    }
    
    with open(output_json, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Detected {len(fields)} fields")
    print(f"✅ Saved to: {output_json}")
    
    # Show summary
    by_method = {}
    by_type = {}
    
    for field in fields:
        methods = field.get('detection_methods', [field.get('method', 'unknown')])
        for method in methods:
            by_method[method] = by_method.get(method, 0) + 1
        
        ftype = field.get('type', 'unknown')
        by_type[ftype] = by_type.get(ftype, 0) + 1
    
    print(f"\nDetection Methods:")
    for method, count in sorted(by_method.items()):
        print(f"  • {method}: {count}")
    
    print(f"\nField Types:")
    for ftype, count in sorted(by_type.items()):
        print(f"  • {ftype}: {count}")
    
    # Show high-confidence fields
    high_conf = [f for f in fields if f.get('confidence', 0) >= 0.8]
    print(f"\nHigh-confidence fields (≥0.8): {len(high_conf)}")
    for field in high_conf[:10]:
        print(f"  • {field.get('name', 'unnamed')}: {field.get('confidence', 0):.2f} ({field.get('method', 'unknown')})")


if __name__ == "__main__":
    main()

