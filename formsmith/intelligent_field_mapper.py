#!/usr/bin/env python3
"""
Intelligent Field Mapper - Analyzes PDF layout and calculates precise field positions
"""

from pdfminer.layout import LAParams, LTTextBox, LTChar, LTAnno, LTTextLine
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
import json
import re

def analyze_pdf_layout(pdf_path):
    """Extract detailed layout information with text blocks and lines"""
    
    layout_data = {
        'pages': [],
        'text_blocks': [],
        'field_candidates': []
    }
    
    with open(pdf_path, 'rb') as fp:
        rsrcmgr = PDFResourceManager()
        laparams = LAParams(
            word_margin=0.1,
            char_margin=2.0,
            line_margin=0.5,
            boxes_flow=0.5
        )
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        
        for page_num, page in enumerate(PDFPage.get_pages(fp)):
            interpreter.process_page(page)
            layout = device.get_result()
            
            page_height = page.mediabox[3]
            page_data = {
                'page_num': page_num,
                'height': page_height,
                'text_blocks': [],
                'lines': []
            }
            
            # Extract text blocks and lines
            for element in layout:
                if isinstance(element, LTTextBox):
                    # Text block
                    x0, y0, x1, y1 = element.bbox
                    text = element.get_text().strip()
                    
                    if text:
                        block = {
                            'text': text,
                            'x0': round(x0, 2),
                            'y0': round(y0, 2),
                            'x1': round(x1, 2),
                            'y1': round(y1, 2),
                            'y_from_top': round(page_height - y1, 2),
                            'width': round(x1 - x0, 2),
                            'height': round(y1 - y0, 2),
                            'lines': []
                        }
                        
                        # Extract individual lines within the block
                        for line in element:
                            if isinstance(line, LTTextLine):
                                lx0, ly0, lx1, ly1 = line.bbox
                                line_text = line.get_text().strip()
                                if line_text:
                                    line_data = {
                                        'text': line_text,
                                        'x0': round(lx0, 2),
                                        'y0': round(ly0, 2),
                                        'x1': round(lx1, 2),
                                        'y1': round(ly1, 2),
                                        'y_from_top': round(page_height - ly1, 2),
                                        'width': round(lx1 - lx0, 2),
                                        'height': round(ly1 - ly0, 2)
                                    }
                                    block['lines'].append(line_data)
                                    page_data['lines'].append(line_data)
                        
                        page_data['text_blocks'].append(block)
                        layout_data['text_blocks'].append(block)
            
            layout_data['pages'].append(page_data)
    
    return layout_data

def identify_field_locations(layout_data):
    """Identify where form fields should be placed based on text analysis"""
    
    field_locations = []
    
    # Define patterns for different field types
    field_patterns = {
        'text_field': {
            'indicators': [
                r'.*:\s*$',  # Ends with colon
                r'.*_\s*$',  # Ends with underscore
                r'.*____.*', # Contains underscores (fill-in-the-blank)
            ],
            'field_type': 'text'
        },
        'checkbox': {
            'indicators': [
                r'check one',
                r'check all that apply',
                r'☐',  # Empty checkbox symbol
                r'□',  # Alternative checkbox symbol
            ],
            'field_type': 'checkbox'
        },
        'date_field': {
            'indicators': [
                r'.*date.*:?\s*$',
                r'__/__/____',  # Date pattern
            ],
            'field_type': 'text'
        },
        'signature_field': {
            'indicators': [
                r'signature.*:?\s*$',
                r'name.*signature',
            ],
            'field_type': 'signature'
        }
    }
    
    # Analyze each text block
    for block in layout_data['text_blocks']:
        text = block['text'].lower()
        
        # Check for field indicators
        for pattern_name, pattern_info in field_patterns.items():
            for indicator in pattern_info['indicators']:
                if re.search(indicator, text, re.IGNORECASE):
                    field_location = {
                        'pattern': pattern_name,
                        'field_type': pattern_info['field_type'],
                        'label_text': block['text'],
                        'label_x0': block['x0'],
                        'label_x1': block['x1'],
                        'label_y_from_top': block['y_from_top'],
                        'label_width': block['width'],
                        'label_height': block['height'],
                        'page_num': 0  # Assuming single page for now
                    }
                    field_locations.append(field_location)
                    break
    
    return field_locations

def calculate_field_positions(field_locations):
    """Calculate precise field positions based on label positions"""
    
    field_definitions = []
    
    for location in field_locations:
        # Calculate field position based on pattern type
        if location['pattern'] == 'text_field':
            # Text field goes after the label
            field_x = location['label_x1'] + 5  # 5 units after label
            field_y = location['label_y_from_top']
            field_width = 200  # Default width, will be refined
            field_height = 15
            
        elif location['pattern'] == 'checkbox':
            # Checkbox goes before the label
            field_x = location['label_x0'] - 20  # 20 units before label
            field_y = location['label_y_from_top']
            field_width = 15
            field_height = 15
            
        elif location['pattern'] == 'signature_field':
            # Signature field below the label
            field_x = location['label_x0']
            field_y = location['label_y_from_top'] + 20  # 20 units below
            field_width = 150
            field_height = 15
            
        else:
            # Default text field
            field_x = location['label_x1'] + 5
            field_y = location['label_y_from_top']
            field_width = 200
            field_height = 15
        
        # Generate field name based on label text
        field_name = generate_field_name(location['label_text'])
        
        field_def = {
            'name': field_name,
            'x': round(field_x, 2),
            'y_from_top': round(field_y, 2),
            'width': field_width,
            'height': field_height,
            'type': location['field_type'],
            'label_text': location['label_text'],
            'pattern': location['pattern']
        }
        
        field_definitions.append(field_def)
    
    return field_definitions

def generate_field_name(label_text):
    """Generate field name from label text"""
    
    # Clean the text
    text = label_text.lower().strip()
    
    # Remove common words
    text = re.sub(r'\b(the|a|an|to|of|in|on|at|for|with|by)\b', '', text)
    
    # Replace spaces and special chars with underscores
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', '_', text)
    
    # Remove leading/trailing underscores
    text = text.strip('_')
    
    # Handle specific cases
    name_mapping = {
        'docket_no': 'docket_number',
        'court_department': 'trial_court_department',
        'court_divisioncounty': 'trial_court_division',
        'plaintiff': 'plaintiff_name',
        'defendant': 'defendant_name',
        'petitioner_name': 'petitioner_name',
        'petitioner_signature': 'petitioner_signature',
        'petitioner_signature_date': 'petitioner_signature_date',
        'petitioner_address': 'petitioner_address',
        'mobile_phone_number': 'petitioner_phone',
        'email_address': 'petitioner_email',
    }
    
    # Check if we have a mapping
    for key, value in name_mapping.items():
        if key in text:
            return value
    
    # Default: use cleaned text
    return text if text else 'unknown_field'

def refine_field_positions(field_definitions, layout_data):
    """Refine field positions based on actual PDF layout"""
    
    refined_fields = []
    
    for field in field_definitions:
        # Find the best position based on surrounding text
        best_position = find_optimal_position(field, layout_data)
        
        if best_position:
            field.update(best_position)
        
        refined_fields.append(field)
    
    return refined_fields

def find_optimal_position(field, layout_data):
    """Find the optimal position for a field based on layout analysis"""
    
    # Look for nearby text elements that might indicate field boundaries
    nearby_elements = []
    
    for block in layout_data['text_blocks']:
        # Check if this block is near our field's label
        y_diff = abs(block['y_from_top'] - field['y_from_top'])
        if y_diff < 30:  # Within 30 units vertically
            nearby_elements.append(block)
    
    # For text fields, try to find the end of the line or next text element
    if field['type'] == 'text':
        # Look for elements on the same line
        same_line_elements = [
            elem for elem in nearby_elements
            if abs(elem['y_from_top'] - field['y_from_top']) < 5
        ]
        
        if same_line_elements:
            # Find the rightmost element to determine field width
            rightmost_x = max(elem['x1'] for elem in same_line_elements)
            field['width'] = rightmost_x - field['x'] - 10  # 10 unit margin
    
    return field

if __name__ == "__main__":
    pdf_path = "jud-tc-Petition-to-Deem-Satisfied.pdf"
    
    print("Analyzing PDF layout for intelligent field placement...")
    print("=" * 70)
    
    # Step 1: Analyze layout
    layout_data = analyze_pdf_layout(pdf_path)
    print(f"Found {len(layout_data['text_blocks'])} text blocks")
    
    # Step 2: Identify field locations
    field_locations = identify_field_locations(layout_data)
    print(f"Identified {len(field_locations)} field locations")
    
    # Step 3: Calculate positions
    field_definitions = calculate_field_positions(field_locations)
    print(f"Calculated {len(field_definitions)} field definitions")
    
    # Step 4: Refine positions
    refined_fields = refine_field_positions(field_definitions, layout_data)
    
    # Save results
    output = {
        'layout_data': layout_data,
        'field_definitions': refined_fields
    }
    
    with open('intelligent_field_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nField definitions:")
    print("-" * 70)
    for field in refined_fields:
        print(f"{field['name']:25} | {field['type']:8} | ({field['x']:6.1f}, {field['y_from_top']:6.1f}) | {field['width']:4.0f}x{field['height']}")
    
    print(f"\n✅ Analysis saved to: intelligent_field_analysis.json")

