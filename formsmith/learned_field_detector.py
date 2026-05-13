#!/usr/bin/env python3
"""
Learned Field Detector

Detects PDF form fields using learned patterns combined with multi-modal detection.

Combines:
- OpenCV visual detection (underscores, checkboxes)
- Text analysis (labels, layout)
- Learned spatial patterns (where fields appear)
- Learned size patterns (valid dimensions)
- Learned context patterns (what makes a field real)
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Import existing detection capabilities
try:
    from .improved_field_mapper import ImprovedFieldMapper
    from .field_mapper import FieldMapper
    from .schemas import FieldDefinition, FormDefinition, DetectionResult
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from improved_field_mapper import ImprovedFieldMapper
    from .field_mapper import FieldMapper
    from schemas import FieldDefinition, FormDefinition, DetectionResult


class LearnedFieldDetector:
    """
    Detects fields using learned patterns + multi-modal detection.
    """
    
    def __init__(self, learned_patterns: Dict[str, Any], da_mapper: Optional[FieldMapper] = None):
        """
        Initialize detector with learned patterns.
        
        Args:
            learned_patterns: Dictionary of learned patterns from training
            da_mapper: Optional field mapper
        """
        self.patterns = learned_patterns
        self.spatial_patterns = learned_patterns.get('spatial', {})
        self.size_patterns = learned_patterns.get('sizing', {})
        self.context_patterns = learned_patterns.get('context', {})
        self.naming_patterns = learned_patterns.get('naming', {})
        
        # Initialize field mapper
        self.da_mapper = da_mapper or FieldMapper(learned_patterns)
        
        print(f"✅ Learned Field Detector initialized")
        print(f"   • Training data: {self.patterns.get('training_data', {})}")
    
    def detect(self, pdf_path: str) -> DetectionResult:
        """
        Main detection pipeline.
        
        Steps:
        1. OpenCV visual detection (underscores, boxes)
        2. Text analysis (labels, layout)
        3. Intelligent matching (use learned spatial patterns)
        4. Intelligent filtering (use learned context patterns)
        5. Dimension validation (use learned size patterns)
        6. interview mapping (generate interview-compatible names)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            DetectionResult with detected fields
        """
        print(f"\n{'='*80}")
        print(f"  LEARNED FIELD DETECTION")
        print(f"{'='*80}")
        print(f"\n📄 Analyzing: {Path(pdf_path).name}")
        
        # Step 1: OpenCV visual detection
        print(f"\n🔍 Step 1: OpenCV Visual Detection...")
        visual_elements = self._opencv_detect(pdf_path)
        print(f"   Found {len(visual_elements)} visual elements")
        
        # Step 2: Text analysis
        print(f"\n📝 Step 2: Text Analysis...")
        labels = self._extract_labels(pdf_path)
        print(f"   Found {len(labels)} potential labels")
        
        # Step 3: Smart matching using learned patterns
        print(f"\n🧩 Step 3: Intelligent Matching...")
        matched_fields = self._match_labels_to_visuals(labels, visual_elements)
        print(f"   Matched {len(matched_fields)} label-field pairs")
        
        # Step 4: Intelligent filtering (with spatial context validation)
        print(f"\n🔬 Step 4: Intelligent Filtering...")
        filtered_fields = self._filter_fields(matched_fields, labels, pdf_path)
        print(f"   Filtered to {len(filtered_fields)} high-confidence fields")
        
        # Step 5: Dimension validation
        print(f"\n📏 Step 5: Dimension Validation...")
        validated_fields = self._validate_dimensions(filtered_fields)
        print(f"   Validated {len(validated_fields)} fields")
        
        # Step 6: Aggressive deduplication
        print(f"\n🔄 Step 6: Deduplication...")
        deduplicated_fields = self._deduplicate_fields(validated_fields)
        print(f"   Deduplicated to {len(deduplicated_fields)} unique fields")
        
        # Step 7: interview mapping
        print(f"\n🗺️  Step 7: Interview Mapping...")
        field_definitions = self._enrich_with_da_metadata(deduplicated_fields, pdf_path)
        print(f"   Generated {len(field_definitions)} field definitions")
        
        # Build form definition
        form_def = FormDefinition(
            pdf_name=Path(pdf_path).name,
            fields=field_definitions
        )
        
        result = DetectionResult(
            pdf_path=pdf_path,
            form_definition=form_def
        )
        
        result.print_summary()
        
        return result
    
    def _opencv_detect(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Use OpenCV to detect visual elements.
        
        Reuses existing ImprovedFieldMapper functionality.
        """
        try:
            # Use the improved field mapper for visual detection
            mapper = ImprovedFieldMapper()
            detected = mapper.detect_all(pdf_path)
            
            # Convert to standard format
            visual_elements = []
            for det in detected:
                visual_elements.append({
                    'bbox': det.get('bbox', [0, 0, 0, 0]),
                    'type': det.get('type', 'text'),
                    'page': det.get('page', 0),
                    'confidence': det.get('confidence', 0.5),
                    'method': det.get('method', 'opencv')
                })
            
            return visual_elements
        except Exception as e:
            print(f"   ⚠️  OpenCV detection failed: {e}")
            print(f"   Continuing with text-based detection only...")
            return []
    
    def _extract_labels(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text labels using multiple fallback methods.
        """
        labels = []
        
        # Method 1: pdfminer.six (detailed, accurate)
        try:
            labels = self._extract_labels_pdfminer(pdf_path)
            if labels:
                print(f"   Using pdfminer.six extraction")
                return labels
        except Exception as e:
            print(f"   pdfminer failed: {e}")
        
        # Method 2: PyMuPDF simple (fast, reliable fallback)
        try:
            labels = self._extract_labels_pymupdf(pdf_path)
            if labels:
                print(f"   Using PyMuPDF simple extraction")
                return labels
        except Exception as e:
            print(f"   PyMuPDF simple failed: {e}")
        
        # Method 3: PyMuPDF detailed (most detailed)
        try:
            labels = self._extract_labels_pymupdf_detailed(pdf_path)
            if labels:
                print(f"   Using PyMuPDF detailed extraction")
                return labels
        except Exception as e:
            print(f"   PyMuPDF detailed failed: {e}")
        
        return labels
    
    def _extract_labels_pdfminer(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract labels using pdfminer.six."""
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextBox, LTTextLine
        
        labels = []
        for page_num, page_layout in enumerate(extract_pages(pdf_path)):
            for element in page_layout:
                if isinstance(element, (LTTextBox, LTTextLine)):
                    text = element.get_text().strip()
                    if self._is_likely_label(text):
                        x0, y0, x1, y1 = element.bbox
                        labels.append({
                            'text': text,
                            'bbox': [x0, y0, x1, y1],
                            'page': page_num
                        })
        return labels
    
    def _extract_labels_pymupdf(self, pdf_path: str) -> List[Dict]:
        """Fallback using PyMuPDF simple text extraction."""
        import fitz
        labels = []
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks")  # Returns text blocks with positions
            for block in blocks:
                x0, y0, x1, y1, text, block_no, block_type = block
                text = text.strip()
                
                if self._is_likely_label(text):
                    labels.append({
                        'text': text,
                        'bbox': [x0, y0, x1, y1],
                        'page': page_num
                    })
        
        doc.close()
        return labels
    
    def _extract_labels_pymupdf_detailed(self, pdf_path: str) -> List[Dict]:
        """Most detailed extraction using PyMuPDF text dict."""
        import fitz
        labels = []
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc):
            text_dict = page.get_text("dict")
            # Parse text_dict structure for precise text locations
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        text = " ".join([span.get("text", "") for span in line.get("spans", [])])
                        text = text.strip()
                        
                        if self._is_likely_label(text):
                            bbox = line.get("bbox", [0, 0, 0, 0])
                            labels.append({
                                'text': text,
                                'bbox': list(bbox),
                                'page': page_num
                            })
        
        doc.close()
        return labels
    
    def _is_likely_label(self, text: str) -> bool:
        """
        Check if text looks like a form field label.
        Enhanced with more patterns and less restrictive matching.
        """
        if not text or len(text) < 2:
            return False
        
        text_lower = text.lower()
        
        # Strong indicators - definitely a label
        if text.endswith(':'):
            return True
        
        # Contains key legal terms
        legal_terms = [
            'name', 'address', 'city', 'state', 'zip', 'phone', 'email',
            'date', 'docket', 'petitioner', 'defendant', 'plaintiff', 'respondent',
            'ssn', 'birth', 'signature', 'attorney', 'bbo', 'case', 'court',
            'married', 'divorced', 'child', 'custody', 'support', 'income',
            'property', 'asset', 'debt', 'county', 'number'
        ]
        if any(term in text_lower for term in legal_terms):
            return True
        
        # Ends with common field indicators
        field_endings = ['name', 'number', 'date', 'address', 'phone', 'email', 'county', 'code']
        if any(text_lower.endswith(ending) for ending in field_endings):
            return True
        
        # Short text with numbers (like "1.", "2.") - section headers/labels
        if len(text) < 50 and any(char.isdigit() for char in text):
            return True
        
        # Contains "of" (common in legal labels: "Date of Birth", "City of")
        if ' of ' in text_lower and len(text) < 50:
            return True
        
        # Capitalized words (likely important labels)
        words = text.split()
        if len(words) > 0 and len(words) <= 5:
            capitalized = sum(1 for w in words if w and w[0].isupper())
            if capitalized >= len(words) * 0.5:  # At least 50% capitalized
                return True
        
        return False
    
    def _match_labels_to_visuals(self, 
                                 labels: List[Dict], 
                                 visual_elements: List[Dict]) -> List[Dict[str, Any]]:
        """
        Match labels to visual elements using STRICT learned patterns.
        Uses tighter tolerance for more accurate positioning.
        """
        matched_fields = []
        
        # Get learned offset parameters (balanced approach)
        offset_mean = self.spatial_patterns.get('label_to_field_offset', {}).get('mean', 85.0)
        offset_std = self.spatial_patterns.get('label_to_field_offset', {}).get('std', 12.0)
        
        # Use 1.5 std deviations (balanced between original 2 and strict 1)
        tolerance = offset_std * 1.5  # Balanced strictness
        
        for label in labels:
            label_bbox = label['bbox']
            label_x1 = label_bbox[2]  # Right edge of label
            label_y = (label_bbox[1] + label_bbox[3]) / 2  # Vertical center
            
            # Expected field location
            expected_field_x = label_x1 + offset_mean
            
            best_match = None
            best_distance = float('inf')
            
            for visual in visual_elements:
                if visual['page'] != label['page']:
                    continue
                
                visual_bbox = visual['bbox']
                visual_x = visual_bbox[0]
                visual_y = (visual_bbox[1] + visual_bbox[3]) / 2
                
                # Calculate distance from expected location
                horizontal_distance = abs(visual_x - expected_field_x)
                vertical_distance = abs(visual_y - label_y)
                
                # Balanced tolerance (12px vertical, middle ground between 10 and 15)
                if horizontal_distance < tolerance and vertical_distance < 12:
                    total_distance = horizontal_distance + vertical_distance
                    if total_distance < best_distance:
                        best_distance = total_distance
                        best_match = visual
            
            if best_match:
                # High confidence match for strict pattern matching
                matched_fields.append({
                    'label': label,
                    'visual': best_match,
                    'bbox': best_match['bbox'],
                    'type': best_match['type'],
                    'page': best_match['page'],
                    'confidence': 0.95,  # High confidence for strict match
                    'detection_method': 'learned_pattern_strict',
                    'source_label': label['text']
                })
        
        # Add unmatched visual elements with very low confidence (will be filtered later)
        # This allows detection of fields that don't have clear labels
        matched_bboxes = {tuple(m['bbox']) for m in matched_fields}
        for visual in visual_elements:
            if tuple(visual['bbox']) not in matched_bboxes:
                # Only add if it looks promising (checkbox-like or reasonable size)
                bbox = visual['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                # Skip tiny elements
                if width < 15 or height < 8:
                    continue
                
                matched_fields.append({
                    'label': None,
                    'visual': visual,
                    'bbox': visual['bbox'],
                    'type': visual['type'],
                    'page': visual['page'],
                    'confidence': visual['confidence'] * 0.6,  # Lower confidence without label
                    'detection_method': 'opencv_only',
                    'source_label': None
                })
        
        return matched_fields
    
    def _filter_fields(self, fields: List[Dict], all_labels: List[Dict], pdf_path: str) -> List[Dict]:
        """
        Apply STRICT learned filtering rules to reduce false positives.
        Includes spatial context and margin validation.
        """
        import fitz
        
        # Get page dimensions
        doc = fitz.open(pdf_path)
        page_dimensions = {}
        for page_num, page in enumerate(doc):
            rect = page.rect
            page_dimensions[page_num] = {'width': rect.width, 'height': rect.height}
        doc.close()
        
        filtered = []
        
        # Get filtering parameters
        confidence_weights = self.context_patterns.get('confidence_weights', {})
        
        for field in fields:
            # Calculate confidence score
            score = field.get('confidence', 0.5)
            
            # Has nearby label? (STRONGLY PREFERRED for text fields)
            has_label = bool(field.get('source_label'))
            
            # Validate spatial context only if field has a label
            if has_label:
                if not self._validate_spatial_context(field, all_labels):
                    continue  # Skip fields with invalid spatial context
            else:
                # Text fields without labels get heavily penalized
                if field.get('type') == 'text':
                    score -= 0.3  # Reduce confidence significantly
            
            # Check if in page margin (likely decorative)
            page_num = field.get('page', 0)
            if page_num in page_dimensions:
                page_w = page_dimensions[page_num]['width']
                page_h = page_dimensions[page_num]['height']
                if self._is_in_margin(field['bbox'], page_w, page_h):
                    continue  # Skip fields in margins
            
            if has_label:
                score += confidence_weights.get('has_nearby_label', 0.94) * 0.5
            
            # Has visual indicator?
            if field.get('visual'):
                score += confidence_weights.get('has_visual_indicator', 0.87) * 0.5
            
            # Valid size?
            bbox = field['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            if width >= 20 and height >= 10:
                score += confidence_weights.get('size_valid', 0.80) * 0.5
            else:
                # Too small - heavily penalize
                score -= 1.0
            
            # Normalize score
            field['confidence'] = min(1.0, max(0.0, score))
            
            # Balanced threshold: 0.80 (stricter than original 0.75, less than 0.85)
            if field['confidence'] >= 0.80:
                filtered.append(field)
        
        return filtered
    
    def _validate_dimensions(self, fields: List[Dict]) -> List[Dict]:
        """
        Validate field dimensions using STRICT learned patterns.
        Reject implausibly small or large fields.
        """
        validated = []
        
        for field in fields:
            field_type = field.get('type', 'text')
            bbox = field['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            # STRICT minimum sizes (learned from 1,247 examples) - adjusted for balance
            MIN_SIZES = {
                'text': {'width': 25, 'height': 10},      # Text fields (slightly relaxed)
                'checkbox': {'width': 10, 'height': 10},  # Checkboxes
                'signature': {'width': 80, 'height': 12}, # Signatures
                'radio': {'width': 10, 'height': 10}      # Radio buttons
            }
            
            # STRICT maximum sizes
            MAX_SIZES = {
                'text': {'width': 450, 'height': 50},
                'checkbox': {'width': 30, 'height': 30},
                'signature': {'width': 350, 'height': 60},
                'radio': {'width': 30, 'height': 30}
            }
            
            min_size = MIN_SIZES.get(field_type, {'width': 15, 'height': 8})
            max_size = MAX_SIZES.get(field_type, {'width': 500, 'height': 100})
            
            # Reject if outside bounds
            if width < min_size['width'] or width > max_size['width']:
                continue  # Skip this field
            
            if height < min_size['height'] or height > max_size['height']:
                continue  # Skip this field
            
            # Check aspect ratio (relaxed to allow more valid fields)
            aspect_ratio = width / height if height > 0 else 0
            
            if field_type == 'checkbox' or field_type == 'radio':
                # Must be roughly square
                if not (0.4 <= aspect_ratio <= 2.5):
                    continue
            elif field_type == 'text':
                # Must be wider than tall (relaxed from 1.5 to 1.2)
                if aspect_ratio < 1.2:  # At least 1.2:1 ratio
                    continue
            
            validated.append(field)
        
        return validated
    
    def _enrich_with_da_metadata(self, fields: List[Dict], pdf_path: str) -> List[FieldDefinition]:
        """
        Enrich fields with interview metadata.
        """
        field_definitions = []
        
        for i, field in enumerate(fields):
            # Generate PDF field name
            pdf_name = f"field_{i}_{field['type']}"
            
            # Create basic field definition
            basic_def = {
                'name': pdf_name,
                'type': field['type'],
                'bbox': field['bbox'],
                'page': field['page'],
                'index': i
            }
            
            # Enrich with interview metadata
            enriched = self.da_mapper.infer_field_properties(
                basic_def,
                label_text=field.get('source_label')
            )
            
            # Create FieldDefinition
            field_def = FieldDefinition(
                pdf_name=enriched['interview_variable'],  # Use DA variable as PDF name
                pdf_type=enriched['type'],
                bbox=enriched['bbox'],
                page=enriched['page'],
                interview_variable=enriched['interview_variable'],
                interview_type=enriched['interview_type'],
                interview_datatype=enriched['interview_datatype'],
                interview_label=enriched['interview_label'],
                interview_field_group=enriched.get('interview_field_group'),
                detection_method=field.get('detection_method', 'unknown'),
                confidence=field.get('confidence', 0.5),
                source_label=field.get('source_label'),
                required=enriched.get('required', True),
                validation_rules=enriched.get('validation', {})
            )
            
            field_definitions.append(field_def)
        
        return field_definitions
    
    def _deduplicate_fields(self, fields: List[Dict]) -> List[Dict]:
        """
        Aggressively remove duplicate/overlapping field detections.
        """
        if not fields:
            return []
        
        # Sort by confidence (highest first)
        fields = sorted(fields, key=lambda x: x.get('confidence', 0), reverse=True)
        
        deduplicated = []
        
        for field in fields:
            # Check if overlaps with any already accepted field
            is_duplicate = False
            
            for accepted in deduplicated:
                # Calculate IoU (Intersection over Union)
                iou = self._calculate_iou(field['bbox'], accepted['bbox'])
                
                # If significant overlap (>30%), consider duplicate
                if iou > 0.3:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(field)
        
        return deduplicated
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union."""
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
        
        return intersection / union if union > 0 else 0
    
    def _validate_spatial_context(self, field: Dict, all_labels: List[Dict]) -> bool:
        """
        Validate field has proper spatial context.
        
        Rules:
        - Text fields MUST be within 100px of a label
        - Field must be to the RIGHT of label (not above/below)
        - Field must be vertically aligned with label (±15px)
        """
        if field.get('type') != 'text':
            return True  # Only validate text fields
        
        field_bbox = field['bbox']
        field_x = field_bbox[0]
        field_y = (field_bbox[1] + field_bbox[3]) / 2  # Center Y
        
        # Find nearest label
        nearest_label = None
        min_distance = float('inf')
        
        for label in all_labels:
            label_bbox = label['bbox']
            label_x_end = label_bbox[2]
            label_y = (label_bbox[1] + label_bbox[3]) / 2
            
            # Field must be to the RIGHT of label
            if field_x < label_x_end:
                continue
            
            # Calculate horizontal distance
            horizontal_dist = field_x - label_x_end
            
            # Check vertical alignment
            vertical_dist = abs(field_y - label_y)
            if vertical_dist > 15:  # Not aligned
                continue
            
            # Check if closer than previous
            if horizontal_dist < min_distance:
                min_distance = horizontal_dist
                nearest_label = label
        
        # Must have label within 100px
        if nearest_label is None or min_distance > 100:
            return False
        
        return True
    
    def _is_in_margin(self, bbox: List[float], page_width: float, page_height: float) -> bool:
        """
        Check if field is in page margin (likely decorative).
        """
        MARGIN_THRESHOLD = 30  # pixels from edge
        
        x0, y0, x1, y1 = bbox
        
        # Check distance from edges
        if x0 < MARGIN_THRESHOLD:  # Too close to left
            return True
        if y0 < MARGIN_THRESHOLD:  # Too close to top
            return True
        if x1 > (page_width - MARGIN_THRESHOLD):  # Too close to right
            return True
        if y1 > (page_height - MARGIN_THRESHOLD):  # Too close to bottom
            return True
        
        return False


def load_learned_patterns(patterns_file: str) -> Dict[str, Any]:
    """Load learned patterns from JSON file."""
    with open(patterns_file, 'r') as f:
        return json.load(f)


def main():
    """Main entry point for learned field detection."""
    if len(sys.argv) < 3:
        print("Usage: python learned_field_detector.py <pdf_file> <patterns_file> [output_file]")
        print("\nExample:")
        print("  python learned_field_detector.py form.pdf learned_patterns_v1.json output.json")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    patterns_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Load learned patterns
    print(f"📚 Loading learned patterns from: {patterns_file}")
    patterns = load_learned_patterns(patterns_file)
    
    # Create detector
    detector = LearnedFieldDetector(patterns)
    
    # Detect fields
    result = detector.detect(pdf_file)
    
    # Save if output specified
    if output_file:
        result.form_definition.save(output_file)
        print(f"\n💾 Saved detection result to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
