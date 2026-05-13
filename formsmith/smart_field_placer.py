#!/usr/bin/env python3
"""
Smart Field Placement Engine for Deterministic Field Positioning
"""

import math
from .smart_pdf_analyzer import SmartPDFAnalyzer
import json

class SmartFieldPlacer:
    """Places form fields intelligently based on PDF analysis"""
    
    def __init__(self, pdf_analysis):
        self.analysis = pdf_analysis
        self.field_positions = {}
        
    def calculate_optimal_positions(self, field_requirements):
        """Calculate optimal positions for all required fields"""
        
        print("🎯 Calculating optimal field positions...")
        
        for field_req in field_requirements:
            field_name = field_req['name']
            field_type = field_req['type']
            label_text = field_req.get('label_text', '')
            
            print(f"  📍 Placing {field_type} field: {field_name}")
            
            if field_type == 'text':
                position = self._place_text_field_smartly(field_req)
            elif field_type == 'checkbox':
                position = self._place_checkbox_smartly(field_req)
            elif field_type == 'signature':
                position = self._place_signature_field_smartly(field_req)
            else:
                print(f"    ⚠️ Unknown field type: {field_type}")
                continue
            
            if position:
                self.field_positions[field_name] = position
                print(f"    ✅ Position: {position}")
            else:
                print(f"    ❌ Could not place field: {field_name}")
        
        return self.field_positions
    
    def _place_text_field_smartly(self, field_req):
        """Place text field using intelligent analysis"""
        
        label_text = field_req['label_text']
        field_width = field_req.get('width', 100)
        field_height = field_req.get('height', 15)
        
        # Find the label text
        label_coords = self._find_text_coordinates(label_text)
        if not label_coords:
            print(f"    ⚠️ Could not find label text: {label_text}")
            return None
        
        # Analyze visual context around the label
        context = self._analyze_visual_context(label_coords)
        
        # Calculate position based on context
        if context['has_underline']:
            # Place field to align with existing underline
            position = self._align_with_underline(label_coords, context['underline'], field_width, field_height)
        elif context['has_box']:
            # Place field to align with existing box
            position = self._align_with_box(label_coords, context['box'], field_width, field_height)
        else:
            # Calculate position based on spacing rules
            position = self._calculate_position_from_spacing_rules(label_coords, field_width, field_height)
        
        return position
    
    def _place_checkbox_smartly(self, field_req):
        """Place checkbox using intelligent analysis"""
        
        label_text = field_req['label_text']
        
        # Find the label text
        label_coords = self._find_text_coordinates(label_text)
        if not label_coords:
            print(f"    ⚠️ Could not find label text: {label_text}")
            return None
        
        # Find existing checkboxes near the label
        nearby_checkboxes = self._find_nearby_checkboxes(label_coords)
        
        # Determine checkbox size
        if nearby_checkboxes:
            # Use size from existing checkbox
            reference_checkbox = nearby_checkboxes[0]
            checkbox_size = self._extract_checkbox_size(reference_checkbox)
        else:
            # Use default size based on form analysis
            checkbox_size = self._get_default_checkbox_size()
        
        # Calculate position
        if nearby_checkboxes:
            # Use existing checkbox as reference
            position = self._calculate_relative_position(label_coords, nearby_checkboxes[0], checkbox_size)
        else:
            # Calculate position based on spacing rules
            position = self._calculate_checkbox_position_from_rules(label_coords, checkbox_size)
        
        return position
    
    def _place_signature_field_smartly(self, field_req):
        """Place signature field using intelligent analysis"""
        
        label_text = field_req['label_text']
        field_width = field_req.get('width', 150)
        field_height = field_req.get('height', 15)
        
        # Find the label text
        label_coords = self._find_text_coordinates(label_text)
        if not label_coords:
            print(f"    ⚠️ Could not find label text: {label_text}")
            return None
        
        # Signature fields typically go to the right of the label
        position = self._calculate_signature_position(label_coords, field_width, field_height)
        
        return position
    
    def _find_text_coordinates(self, search_text):
        """Find coordinates of specific text"""
        
        # Search in text elements
        for element in self.analysis['text_elements']:
            if search_text.lower() in element['text'].lower():
                return element
        
        # Try partial matches
        for element in self.analysis['text_elements']:
            if self._text_similarity(search_text.lower(), element['text'].lower()) > 0.8:
                return element
        
        return None
    
    def _text_similarity(self, text1, text2):
        """Calculate text similarity score"""
        if not text1 or not text2:
            return 0
        
        # Simple similarity based on common words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0
    
    def _analyze_visual_context(self, label_coords):
        """Analyze visual context around label coordinates"""
        
        context = {
            'has_underline': False,
            'has_box': False,
            'underline': None,
            'box': None,
            'nearby_elements': []
        }
        
        # Find nearby visual elements
        for element in self.analysis['visual_elements']:
            distance = self._calculate_distance_to_element(label_coords, element)
            
            if distance < 30:  # Within 30 points
                context['nearby_elements'].append(element)
                
                if element['type'] == 'line':
                    # Check if line is horizontal and below the text (underline)
                    if self._is_horizontal_line(element) and self._is_below_text(label_coords, element):
                        context['has_underline'] = True
                        context['underline'] = element
                
                elif element['type'] == 'rectangle':
                    # Check if rectangle is around the text (box)
                    if self._is_around_text(label_coords, element):
                        context['has_box'] = True
                        context['box'] = element
        
        return context
    
    def _calculate_distance_to_element(self, label_coords, element):
        """Calculate distance from label to visual element"""
        
        if 'bbox' in element:
            return self._calculate_distance_to_bbox(label_coords['bbox'], element['bbox'])
        elif element['type'] == 'line':
            return self._calculate_distance_to_line(label_coords['bbox'], element)
        else:
            return float('inf')
    
    def _calculate_distance_to_bbox(self, bbox1, bbox2):
        """Calculate distance between two bounding boxes"""
        
        # Calculate minimum distance between bounding boxes
        dx = max(0, max(bbox1[0] - bbox2[2], bbox2[0] - bbox1[2]))
        dy = max(0, max(bbox1[1] - bbox2[3], bbox2[1] - bbox1[3]))
        
        return math.sqrt(dx*dx + dy*dy)
    
    def _calculate_distance_to_line(self, bbox, line_element):
        """Calculate distance from bounding box to line"""
        
        line_start = line_element['start']
        line_end = line_element['end']
        
        # Calculate minimum distance from bbox to line
        min_distance = float('inf')
        
        # Check distance to line endpoints
        for point in [line_start, line_end]:
            distance = math.sqrt((bbox[0] - point[0])**2 + (bbox[1] - point[1])**2)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _is_horizontal_line(self, line_element):
        """Check if line is approximately horizontal"""
        
        start = line_element['start']
        end = line_element['end']
        
        # Calculate angle
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        if dx == 0:
            return False  # Vertical line
        
        angle = math.atan(abs(dy / dx))
        return angle < math.pi / 6  # Less than 30 degrees from horizontal
    
    def _is_below_text(self, label_coords, line_element):
        """Check if line is below the text"""
        
        line_y = (line_element['start'][1] + line_element['end'][1]) / 2
        text_y = label_coords['bbox'][1]
        
        return line_y < text_y  # Line is below text
    
    def _is_around_text(self, label_coords, rect_element):
        """Check if rectangle is around the text"""
        
        text_bbox = label_coords['bbox']
        rect_bbox = rect_element['bbox']
        
        # Check if rectangle contains or is very close to text
        margin = 10  # Allow some margin
        
        return (rect_bbox[0] <= text_bbox[0] - margin and
                rect_bbox[1] <= text_bbox[1] - margin and
                rect_bbox[2] >= text_bbox[2] + margin and
                rect_bbox[3] >= text_bbox[3] + margin)
    
    def _align_with_underline(self, label_coords, underline, field_width, field_height):
        """Align text field with existing underline"""
        
        # Position field above the underline
        y_position = underline['start'][1] - field_height - 2  # 2 points above underline
        
        # Center field horizontally relative to underline
        line_start_x = min(underline['start'][0], underline['end'][0])
        line_end_x = max(underline['start'][0], underline['end'][0])
        line_center_x = (line_start_x + line_end_x) / 2
        
        x_position = line_center_x - field_width / 2
        
        return {
            'x': x_position,
            'y': y_position,
            'width': field_width,
            'height': field_height,
            'placement_method': 'underline_alignment'
        }
    
    def _align_with_box(self, label_coords, box, field_width, field_height):
        """Align text field with existing box"""
        
        # Position field inside the box
        box_bbox = box['bbox']
        
        # Center field within the box
        x_position = box_bbox[0] + (box_bbox[2] - box_bbox[0] - field_width) / 2
        y_position = box_bbox[1] + (box_bbox[3] - box_bbox[1] - field_height) / 2
        
        return {
            'x': x_position,
            'y': y_position,
            'width': field_width,
            'height': field_height,
            'placement_method': 'box_alignment'
        }
    
    def _calculate_position_from_spacing_rules(self, label_coords, field_width, field_height):
        """Calculate position based on spacing rules"""
        
        # Get spacing rules for this area
        spacing_rules = self.analysis['design_rules']['spacing_rules']
        
        # Find the closest spacing rule
        label_x = label_coords['x']
        closest_rule = None
        min_distance = float('inf')
        
        for x_pos, rule in spacing_rules.items():
            distance = abs(x_pos - label_x)
            if distance < min_distance:
                min_distance = distance
                closest_rule = rule
        
        if closest_rule:
            # Use spacing rules to position field
            x_position = label_x + closest_rule['mean_spacing']
            y_position = label_coords['y']
        else:
            # Default positioning
            x_position = label_coords['x'] + 100  # 100 points to the right
            y_position = label_coords['y']
        
        return {
            'x': x_position,
            'y': y_position,
            'width': field_width,
            'height': field_height,
            'placement_method': 'spacing_rules'
        }
    
    def _find_nearby_checkboxes(self, label_coords, max_distance=50):
        """Find checkboxes near label coordinates"""
        
        nearby_checkboxes = []
        
        for checkbox in self.analysis['checkbox_indicators']:
            if 'bbox' in checkbox:
                distance = self._calculate_distance_to_bbox(label_coords['bbox'], checkbox['bbox'])
                if distance < max_distance:
                    nearby_checkboxes.append(checkbox)
        
        return nearby_checkboxes
    
    def _extract_checkbox_size(self, checkbox):
        """Extract size from existing checkbox"""
        
        if 'bbox' in checkbox:
            bbox = checkbox['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            return {'width': width, 'height': height}
        elif 'width' in checkbox and 'height' in checkbox:
            return {'width': checkbox['width'], 'height': checkbox['height']}
        else:
            return self._get_default_checkbox_size()
    
    def _get_default_checkbox_size(self):
        """Get default checkbox size based on form analysis"""
        
        # Analyze existing checkboxes to determine typical size
        checkbox_sizes = []
        
        for checkbox in self.analysis['checkbox_indicators']:
            if 'bbox' in checkbox:
                bbox = checkbox['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                checkbox_sizes.append((width, height))
        
        if checkbox_sizes:
            # Use average size
            avg_width = sum(size[0] for size in checkbox_sizes) / len(checkbox_sizes)
            avg_height = sum(size[1] for size in checkbox_sizes) / len(checkbox_sizes)
            return {'width': avg_width, 'height': avg_height}
        else:
            # Default size
            return {'width': 15, 'height': 15}
    
    def _calculate_relative_position(self, label_coords, reference_checkbox, checkbox_size):
        """Calculate position relative to existing checkbox"""
        
        ref_bbox = reference_checkbox['bbox']
        
        # Position checkbox to the right of the label
        x_position = label_coords['x'] + 200  # 200 points to the right
        y_position = label_coords['y']
        
        return {
            'x': x_position,
            'y': y_position,
            'width': checkbox_size['width'],
            'height': checkbox_size['height'],
            'placement_method': 'relative_to_existing'
        }
    
    def _calculate_checkbox_position_from_rules(self, label_coords, checkbox_size):
        """Calculate checkbox position based on spacing rules"""
        
        # Position checkbox to the right of the label
        x_position = label_coords['x'] + 200  # 200 points to the right
        y_position = label_coords['y']
        
        return {
            'x': x_position,
            'y': y_position,
            'width': checkbox_size['width'],
            'height': checkbox_size['height'],
            'placement_method': 'spacing_rules'
        }
    
    def _calculate_signature_position(self, label_coords, field_width, field_height):
        """Calculate signature field position"""
        
        # Position signature field to the right of the label
        x_position = label_coords['x'] + 200  # 200 points to the right
        y_position = label_coords['y']
        
        return {
            'x': x_position,
            'y': y_position,
            'width': field_width,
            'height': field_height,
            'placement_method': 'signature_positioning'
        }
    
    def save_positions(self, output_path):
        """Save calculated positions to JSON file"""
        
        with open(output_path, 'w') as f:
            json.dump(self.field_positions, f, indent=2)
        
        print(f"💾 Field positions saved to: {output_path}")

def main():
    """Test the field placer"""
    
    # Load analysis
    with open('pdf_analysis.json', 'r') as f:
        analysis = json.load(f)
    
    # Define field requirements
    field_requirements = [
        {
            'name': 'docket_number',
            'type': 'text',
            'label_text': 'DOCKET NO',
            'width': 100,
            'height': 15
        },
        {
            'name': 'trial_court_department',
            'type': 'text',
            'label_text': 'COURT DEPARTMENT',
            'width': 95,
            'height': 15
        },
        {
            'name': 'service_to_plaintiff',
            'type': 'checkbox',
            'label_text': 'the plaintiff'
        },
        {
            'name': 'service_to_plaintiff_attorney',
            'type': 'checkbox',
            'label_text': "plaintiff's lawyer"
        }
    ]
    
    # Create field placer
    placer = SmartFieldPlacer(analysis)
    
    # Calculate positions
    positions = placer.calculate_optimal_positions(field_requirements)
    
    # Save positions
    placer.save_positions('field_positions.json')
    
    print(f"✅ Calculated positions for {len(positions)} fields")

if __name__ == "__main__":
    main()
