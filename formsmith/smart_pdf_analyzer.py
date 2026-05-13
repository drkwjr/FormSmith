#!/usr/bin/env python3
"""
Smart PDF Analysis Engine for Intelligent Field Placement
"""

import pikepdf
import fitz  # pymupdf
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTLine, LTRect
import json
from collections import defaultdict
import math
import os
from typing import List, Dict

from PIL import Image, ImageDraw, ImageFont

class SmartPDFAnalyzer:
    """Analyzes PDF structure for intelligent field placement"""
    
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.analysis = None
        
    def analyze(self):
        """Perform comprehensive PDF analysis"""
        print(f"🔍 Analyzing PDF: {self.pdf_path}")
        
        # 1. Text Analysis
        text_elements = self._extract_text_elements()
        
        # 2. Visual Element Analysis
        visual_elements = self._extract_visual_elements()
        
        # 3. Layout Pattern Analysis
        layout_patterns = self._analyze_layout_patterns(text_elements, visual_elements)
        
        # 4. Design Rules Extraction
        design_rules = self._extract_design_rules(text_elements, visual_elements)
        
        # 5. Checkbox Detection
        checkbox_indicators = self._detect_checkbox_indicators(text_elements, visual_elements)
        
        self.analysis = {
            'text_elements': text_elements,
            'visual_elements': visual_elements,
            'layout_patterns': layout_patterns,
            'design_rules': design_rules,
            'checkbox_indicators': checkbox_indicators
        }
        
        print(f"✅ Analysis complete: {len(text_elements)} text elements, {len(visual_elements)} visual elements")
        return self.analysis
    
    def _extract_text_elements(self):
        """Extract all text with precise positioning and styling"""
        text_elements = []
        
        for page_num, page in enumerate(extract_pages(self.pdf_path)):
            for element in page:
                if isinstance(element, LTTextContainer):
                    # Extract word-level information
                    text = element.get_text().strip()
                    if text:
                        text_elements.append({
                            'text': text,
                            'x': element.x0,
                            'y': element.y0,
                            'width': element.width,
                            'height': element.height,
                            'bbox': (element.x0, element.y0, element.x1, element.y1),
                            'page': page_num,
                            'font_size': self._get_avg_font_size(element),
                            'font_name': self._get_font_name(element)
                        })
                        
                        # For checkbox detection, also extract character-level data
                        if self._contains_checkbox_symbols(text):
                            char_elements = self._extract_character_details(element)
                            text_elements.extend(char_elements)
        
        return text_elements
    
    def _extract_visual_elements(self):
        """Detect lines, boxes, and other visual elements"""
        visual_elements = []
        
        doc = fitz.open(self.pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Detect lines and rectangles
            drawings = page.get_drawings()
            for drawing in drawings:
                if drawing['type'] == 'line':
                    visual_elements.append({
                        'type': 'line',
                        'start': drawing['from'],
                        'end': drawing['to'],
                        'thickness': drawing['width'],
                        'color': drawing['color'],
                        'page': page_num
                    })
                elif drawing['type'] == 'rect':
                    visual_elements.append({
                        'type': 'rectangle',
                        'bbox': drawing['rect'],
                        'thickness': drawing['width'],
                        'color': drawing['color'],
                        'page': page_num
                    })
            
            # Detect existing form fields
            widgets = page.widgets()
            for widget in widgets:
                visual_elements.append({
                    'type': 'existing_field',
                    'field_type': widget.field_type,
                    'bbox': widget.rect,
                    'field_name': widget.field_name,
                    'page': page_num
                })
        
        doc.close()
        return visual_elements

    def export_existing_fields(self, output_json: str, overlay_image: str = None, dpi: int = 150,
                                page_index: int = 0) -> List[Dict]:
        """Export existing PDF form fields to JSON and optional overlay image"""
        doc = fitz.open(self.pdf_path)

        if page_index >= len(doc):
            raise ValueError(f"PDF only has {len(doc)} pages; page_index {page_index} is invalid")

        page = doc[page_index]
        page_rect = page.rect

        widgets = page.widgets()
        fields = []

        # Prepare overlay image if requested
        pix = None
        draw = None
        scale_x = scale_y = None
        font = None

        if overlay_image:
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=matrix)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            draw = ImageDraw.Draw(image, "RGBA")
            scale_x = pix.width / page_rect.width
            scale_y = pix.height / page_rect.height

            try:
                font = ImageFont.truetype("Helvetica", 14)
            except Exception:
                font = ImageFont.load_default()

        type_map = {
            getattr(fitz, "PDF_WIDGET_TYPE_TEXT", None): "text",
            getattr(fitz, "PDF_WIDGET_TYPE_CHECKBOX", None): "checkbox",
            getattr(fitz, "PDF_WIDGET_TYPE_COMBOBOX", None): "combobox",
            getattr(fitz, "PDF_WIDGET_TYPE_LISTBOX", None): "listbox",
            getattr(fitz, "PDF_WIDGET_TYPE_SIGNATURE", None): "signature",
            getattr(fitz, "PDF_WIDGET_TYPE_BUTTON", None): "button"
        }

        for widget in widgets:
            rect = widget.rect
            field_type_raw = widget.field_type
            field_type = type_map.get(field_type_raw, str(field_type_raw))
            name = widget.field_name or "unnamed"

            bbox = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
            field_entry = {
                "name": name,
                "type": field_type,
                "bbox": bbox,
                "page": page_index,
                "source": "existing_field"
            }
            fields.append(field_entry)

            if draw is not None:
                left = rect.x0 * scale_x
                right = rect.x1 * scale_x
                top = pix.height - (rect.y1 * scale_y)
                bottom = pix.height - (rect.y0 * scale_y)

                color = (255, 0, 0, 120) if field_type == "checkbox" else (0, 128, 255, 120)
                draw.rectangle([left, top, right, bottom], outline=color, width=3)
                draw.rectangle([left, top, right, bottom], fill=(color[0], color[1], color[2], 60))

                label = f"{name}\n{field_type}"
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_pos = (left, max(0, top - text_height - 2))

                draw.rectangle(
                    [text_pos[0], text_pos[1], text_pos[0] + text_width + 4, text_pos[1] + text_height + 4],
                    fill=(0, 0, 0, 160)
                )
                draw.text((text_pos[0] + 2, text_pos[1] + 2), label, fill=(255, 255, 255), font=font)

        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump({
                "pdf": os.path.basename(self.pdf_path),
                "page_width": float(page_rect.width),
                "page_height": float(page_rect.height),
                "fields": fields
            }, f, indent=2)

        if overlay_image and pix is not None:
            os.makedirs(os.path.dirname(overlay_image), exist_ok=True)
            image.save(overlay_image)

        doc.close()
        return fields
    
    def _analyze_layout_patterns(self, text_elements, visual_elements):
        """Analyze overall layout structure"""
        
        # Group elements by sections
        sections = self._identify_form_sections(text_elements)
        
        # Analyze alignment patterns
        alignment_patterns = self._analyze_text_alignment(text_elements)
        
        # Detect field indicators
        field_indicators = self._detect_field_indicators(text_elements, visual_elements)
        
        # Map visual hierarchy
        visual_hierarchy = self._map_visual_hierarchy(text_elements)
        
        return {
            'sections': sections,
            'alignment_patterns': alignment_patterns,
            'field_indicators': field_indicators,
            'visual_hierarchy': visual_hierarchy
        }
    
    def _extract_design_rules(self, text_elements, visual_elements):
        """Extract deterministic design rules"""
        
        # Analyze spacing patterns
        spacing_rules = self._analyze_spacing_patterns(text_elements)
        
        # Analyze alignment patterns
        alignment_rules = self._analyze_alignment_patterns(text_elements)
        
        # Analyze typography patterns
        typography_rules = self._analyze_typography_patterns(text_elements)
        
        # Analyze visual element patterns
        visual_rules = self._analyze_visual_element_patterns(visual_elements)
        
        return {
            'spacing_rules': spacing_rules,
            'alignment_rules': alignment_rules,
            'typography_rules': typography_rules,
            'visual_rules': visual_rules
        }
    
    def _detect_checkbox_indicators(self, text_elements, visual_elements):
        """Multi-modal checkbox detection"""
        
        checkbox_indicators = []
        
        # 1. Text symbols
        text_checkboxes = self._find_checkbox_symbols_in_text(text_elements)
        checkbox_indicators.extend(text_checkboxes)
        
        # 2. Visual elements that look like checkboxes
        visual_checkboxes = self._find_checkbox_like_visuals(visual_elements)
        checkbox_indicators.extend(visual_checkboxes)
        
        # 3. Existing checkbox fields
        existing_checkboxes = [elem for elem in visual_elements if elem['type'] == 'existing_field' and 'checkbox' in str(elem['field_type']).lower()]
        checkbox_indicators.extend(existing_checkboxes)
        
        return checkbox_indicators
    
    def _contains_checkbox_symbols(self, text):
        """Check if text contains checkbox symbols"""
        checkbox_symbols = ['□', '☐', '☑', '☒', '☑', '☐', '☑', '☒']
        return any(symbol in text for symbol in checkbox_symbols)
    
    def _extract_character_details(self, text_container):
        """Extract character-level details for checkbox symbols"""
        char_elements = []
        
        for char in text_container:
            if isinstance(char, LTChar):
                if self._contains_checkbox_symbols(char.get_text()):
                    char_elements.append({
                        'text': char.get_text(),
                        'x': char.x0,
                        'y': char.y0,
                        'width': char.width,
                        'height': char.height,
                        'bbox': (char.x0, char.y0, char.x1, char.y1),
                        'font_size': char.size,
                        'font_name': char.fontname,
                        'type': 'checkbox_symbol'
                    })
        
        return char_elements
    
    def _get_avg_font_size(self, text_container):
        """Get average font size from text container"""
        sizes = []
        for char in text_container:
            if isinstance(char, LTChar):
                sizes.append(char.size)
        return sum(sizes) / len(sizes) if sizes else 12
    
    def _get_font_name(self, text_container):
        """Get font name from text container"""
        for char in text_container:
            if isinstance(char, LTChar):
                return char.fontname
        return 'Unknown'
    
    def _identify_form_sections(self, text_elements):
        """Identify different sections of the form"""
        sections = []
        
        # Look for section headers (typically larger font or bold text)
        for element in text_elements:
            if element['font_size'] > 14 or element['text'].isupper():
                sections.append({
                    'type': 'header',
                    'text': element['text'],
                    'bbox': element['bbox'],
                    'font_size': element['font_size']
                })
        
        return sections
    
    def _analyze_text_alignment(self, text_elements):
        """Analyze text alignment patterns"""
        alignments = defaultdict(list)
        
        for element in text_elements:
            # Group by y-coordinate (same line)
            y_key = round(element['y'], 1)
            alignments[y_key].append(element)
        
        alignment_patterns = {}
        for y, elements in alignments.items():
            if len(elements) > 1:
                # Analyze alignment of elements on same line
                x_positions = [elem['x'] for elem in elements]
                alignment_patterns[y] = {
                    'elements': elements,
                    'alignment': self._determine_alignment(x_positions)
                }
        
        return alignment_patterns
    
    def _determine_alignment(self, x_positions):
        """Determine alignment type from x positions"""
        if len(x_positions) < 2:
            return 'single'
        
        # Check for left alignment (increasing x positions)
        if all(x_positions[i] <= x_positions[i+1] for i in range(len(x_positions)-1)):
            return 'left_aligned'
        
        # Check for center alignment (similar x positions)
        x_range = max(x_positions) - min(x_positions)
        if x_range < 50:  # Elements are close together
            return 'center_aligned'
        
        return 'mixed'
    
    def _detect_field_indicators(self, text_elements, visual_elements):
        """Detect visual indicators for form fields"""
        field_indicators = []
        
        # Look for underlines near text
        for text_elem in text_elements:
            nearby_lines = self._find_nearby_lines(text_elem, visual_elements)
            if nearby_lines:
                field_indicators.append({
                    'type': 'underline',
                    'text_element': text_elem,
                    'lines': nearby_lines
                })
        
        # Look for boxes around text
        for text_elem in text_elements:
            nearby_rects = self._find_nearby_rectangles(text_elem, visual_elements)
            if nearby_rects:
                field_indicators.append({
                    'type': 'box',
                    'text_element': text_elem,
                    'rectangles': nearby_rects
                })
        
        return field_indicators
    
    def _find_nearby_lines(self, text_elem, visual_elements, max_distance=10):
        """Find lines near a text element"""
        nearby_lines = []
        
        for elem in visual_elements:
            if elem['type'] == 'line':
                # Check if line is near the text element
                line_distance = self._calculate_distance_to_line(text_elem, elem)
                if line_distance < max_distance:
                    nearby_lines.append(elem)
        
        return nearby_lines
    
    def _find_nearby_rectangles(self, text_elem, visual_elements, max_distance=10):
        """Find rectangles near a text element"""
        nearby_rects = []
        
        for elem in visual_elements:
            if elem['type'] == 'rectangle':
                # Check if rectangle is near the text element
                rect_distance = self._calculate_distance_to_rectangle(text_elem, elem)
                if rect_distance < max_distance:
                    nearby_rects.append(elem)
        
        return nearby_rects
    
    def _calculate_distance_to_line(self, text_elem, line_elem):
        """Calculate distance from text element to line"""
        # Simplified distance calculation
        text_bbox = text_elem['bbox']
        line_start = line_elem['start']
        line_end = line_elem['end']
        
        # Calculate minimum distance from text bbox to line
        min_distance = float('inf')
        
        # Check distance to line endpoints
        for point in [line_start, line_end]:
            distance = math.sqrt((text_bbox[0] - point[0])**2 + (text_bbox[1] - point[1])**2)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _calculate_distance_to_rectangle(self, text_elem, rect_elem):
        """Calculate distance from text element to rectangle"""
        text_bbox = text_elem['bbox']
        rect_bbox = rect_elem['bbox']
        
        # Calculate minimum distance between bounding boxes
        dx = max(0, max(text_bbox[0] - rect_bbox[2], rect_bbox[0] - text_bbox[2]))
        dy = max(0, max(text_bbox[1] - rect_bbox[3], rect_bbox[1] - text_bbox[3]))
        
        return math.sqrt(dx*dx + dy*dy)
    
    def _map_visual_hierarchy(self, text_elements):
        """Map visual hierarchy of the form"""
        hierarchy = []
        
        # Sort by font size and position
        sorted_elements = sorted(text_elements, key=lambda x: (-x['font_size'], x['y']))
        
        for element in sorted_elements:
            hierarchy.append({
                'level': self._determine_hierarchy_level(element, text_elements),
                'element': element
            })
        
        return hierarchy
    
    def _determine_hierarchy_level(self, element, all_elements):
        """Determine hierarchy level based on font size and position"""
        font_size = element['font_size']
        
        if font_size > 16:
            return 1  # Main heading
        elif font_size > 14:
            return 2  # Section heading
        elif font_size > 12:
            return 3  # Subsection
        else:
            return 4  # Regular text
    
    def _analyze_spacing_patterns(self, text_elements):
        """Analyze spacing patterns between text elements"""
        spacing_patterns = {}
        
        # Group by x-coordinate to find columns
        columns = defaultdict(list)
        for elem in text_elements:
            x_key = round(elem['x'], 10)  # Group by x position
            columns[x_key].append(elem)
        
        # Analyze spacing within columns
        for x_pos, column_elements in columns.items():
            if len(column_elements) > 1:
                # Sort by y position
                column_elements.sort(key=lambda x: x['y'])
                
                # Calculate spacing between consecutive elements
                spacings = []
                for i in range(len(column_elements) - 1):
                    spacing = column_elements[i+1]['y'] - column_elements[i]['y'] - column_elements[i]['height']
                    spacings.append(spacing)
                
                if spacings:
                    spacing_patterns[x_pos] = {
                        'mean_spacing': sum(spacings) / len(spacings),
                        'min_spacing': min(spacings),
                        'max_spacing': max(spacings),
                        'spacings': spacings
                    }
        
        return spacing_patterns
    
    def _analyze_alignment_patterns(self, text_elements):
        """Analyze text alignment patterns"""
        alignment_patterns = {}
        
        # Group by y-coordinate (same horizontal line)
        lines = defaultdict(list)
        for elem in text_elements:
            y_key = round(elem['y'], 1)
            lines[y_key].append(elem)
        
        for y_pos, line_elements in lines.items():
            if len(line_elements) > 1:
                # Analyze alignment of elements on same line
                x_positions = [elem['x'] for elem in line_elements]
                alignment_patterns[y_pos] = {
                    'alignment': self._determine_alignment(x_positions),
                    'elements': line_elements
                }
        
        return alignment_patterns
    
    def _analyze_typography_patterns(self, text_elements):
        """Analyze typography patterns"""
        font_sizes = [elem['font_size'] for elem in text_elements]
        font_names = [elem['font_name'] for elem in text_elements]
        
        return {
            'common_font_sizes': self._get_common_values(font_sizes),
            'common_font_names': self._get_common_values(font_names),
            'font_size_range': (min(font_sizes), max(font_sizes)) if font_sizes else (12, 12)
        }
    
    def _analyze_visual_element_patterns(self, visual_elements):
        """Analyze patterns in visual elements"""
        line_thicknesses = [elem['thickness'] for elem in visual_elements if elem['type'] == 'line']
        rect_sizes = []
        
        for elem in visual_elements:
            if elem['type'] == 'rectangle':
                bbox = elem['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                rect_sizes.append((width, height))
        
        return {
            'common_line_thicknesses': self._get_common_values(line_thicknesses),
            'common_rect_sizes': self._get_common_values(rect_sizes),
            'line_thickness_range': (min(line_thicknesses), max(line_thicknesses)) if line_thicknesses else (1, 1)
        }
    
    def _get_common_values(self, values, top_n=5):
        """Get most common values from a list"""
        from collections import Counter
        counter = Counter(values)
        return counter.most_common(top_n)
    
    def _find_checkbox_symbols_in_text(self, text_elements):
        """Find checkbox symbols in text elements"""
        checkbox_symbols = ['□', '☐', '☑', '☒', '☑', '☐', '☑', '☒']
        found_checkboxes = []
        
        for element in text_elements:
            if element['text'] in checkbox_symbols:
                found_checkboxes.append({
                    'type': 'text_symbol',
                    'symbol': element['text'],
                    'bbox': element['bbox'],
                    'font_size': element['font_size'],
                    'font_name': element['font_name']
                })
        
        return found_checkboxes
    
    def _find_checkbox_like_visuals(self, visual_elements):
        """Find visual elements that look like checkboxes"""
        checkbox_like = []
        
        for element in visual_elements:
            if element['type'] == 'rectangle':
                bbox = element['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                # Check if it's checkbox-sized (typically 10-20 points)
                if 8 <= width <= 25 and 8 <= height <= 25:
                    checkbox_like.append({
                        'type': 'visual_checkbox',
                        'bbox': bbox,
                        'width': width,
                        'height': height,
                        'thickness': element.get('thickness', 1)
                    })
        
        return checkbox_like
    
    def save_analysis(self, output_path):
        """Save analysis results to JSON file"""
        if self.analysis:
            with open(output_path, 'w') as f:
                json.dump(self.analysis, f, indent=2, default=str)
            print(f"💾 Analysis saved to: {output_path}")
        else:
            print("❌ No analysis to save. Run analyze() first.")
    
    def find_text_coordinates(self, search_text):
        """Find coordinates of specific text"""
        if not self.analysis:
            return None
        
        for element in self.analysis['text_elements']:
            if search_text.lower() in element['text'].lower():
                return element
        
        return None
    
    def find_nearby_checkboxes(self, target_coords, max_distance=50):
        """Find checkboxes near target coordinates"""
        if not self.analysis:
            return []
        
        nearby_checkboxes = []
        
        for checkbox in self.analysis['checkbox_indicators']:
            if 'bbox' in checkbox:
                distance = self._calculate_distance_to_bbox(target_coords, checkbox['bbox'])
                if distance < max_distance:
                    nearby_checkboxes.append(checkbox)
        
        return nearby_checkboxes

def main():
    """Test the analyzer"""
    pdf_path = "jud-tc-Petition-to-Deem-Satisfied.pdf"
    
    analyzer = SmartPDFAnalyzer(pdf_path)
    analysis = analyzer.analyze()
    
    # Save analysis
    analyzer.save_analysis("pdf_analysis.json")
    
    # Test finding specific text
    docket_coords = analyzer.find_text_coordinates("DOCKET NO")
    if docket_coords:
        print(f"📍 Found 'DOCKET NO' at: {docket_coords['bbox']}")
    
    # Test checkbox detection
    checkboxes = analyzer.find_nearby_checkboxes((100, 650))
    print(f"☑️ Found {len(checkboxes)} checkboxes near (100, 650)")

if __name__ == "__main__":
    main()
