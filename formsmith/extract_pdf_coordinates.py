#!/usr/bin/env python3
"""
Extract text with precise coordinates from PDF
"""

from pdfminer.layout import LAParams, LTTextBox, LTChar, LTAnno
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
import json

def extract_text_with_coordinates(pdf_path):
    """Extract all text elements with their coordinates"""
    
    elements = []
    
    with open(pdf_path, 'rb') as fp:
        # Create a PDF resource manager
        rsrcmgr = PDFResourceManager()
        
        # Set parameters for analysis
        laparams = LAParams()
        
        # Create a PDF page aggregator
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        
        # Create a PDF interpreter
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        
        # Process each page
        for page_num, page in enumerate(PDFPage.get_pages(fp)):
            interpreter.process_page(page)
            layout = device.get_result()
            
            # Get page dimensions
            page_height = page.mediabox[3]
            
            # Extract text boxes
            for element in layout:
                if isinstance(element, LTTextBox):
                    x0, y0, x1, y1 = element.bbox
                    text = element.get_text().strip()
                    
                    if text:
                        elements.append({
                            'text': text,
                            'x0': round(x0, 2),
                            'y0': round(y0, 2),
                            'x1': round(x1, 2),
                            'y1': round(y1, 2),
                            'page': page_num,
                            'page_height': page_height,
                            # Convert to standard coords (0,0 at top-left)
                            'y_from_top': round(page_height - y1, 2)
                        })
    
    return elements

def identify_field_patterns(elements):
    """Identify patterns that indicate form fields"""
    
    field_indicators = []
    
    for elem in elements:
        text = elem['text']
        
        # Look for common field indicators
        indicators = {
            'has_colon': ':' in text,
            'has_underscores': '___' in text or '____' in text,
            'has_checkbox': '☐' in text or '□' in text,
            'has_date_pattern': '__/__/____' in text or '________________' in text,
            'ends_with_colon': text.strip().endswith(':'),
            'is_label': any(word in text.upper() for word in [
                'NAME', 'ADDRESS', 'DATE', 'SIGNATURE', 'EMAIL', 'PHONE',
                'DOCKET', 'COURT', 'NUMBER', 'PLAINTIFF', 'DEFENDANT'
            ])
        }
        
        if any(indicators.values()):
            elem['indicators'] = indicators
            field_indicators.append(elem)
    
    return field_indicators

if __name__ == "__main__":
    pdf_path = "jud-tc-Petition-to-Deem-Satisfied.pdf"
    
    print(f"Extracting coordinates from: {pdf_path}")
    print("=" * 70)
    
    elements = extract_text_with_coordinates(pdf_path)
    field_indicators = identify_field_patterns(elements)
    
    print(f"\n📍 Total text elements: {len(elements)}")
    print(f"📍 Field indicators found: {len(field_indicators)}")
    
    print(f"\n🔍 Field Indicators (showing position and text):")
    print("-" * 70)
    
    for i, field in enumerate(field_indicators[:20], 1):  # Show first 20
        print(f"\n{i}. Text: \"{field['text'][:60]}...\" " if len(field['text']) > 60 else f"\n{i}. Text: \"{field['text']}\"")
        print(f"   Position: x={field['x0']}, y_from_top={field['y_from_top']}")
        print(f"   Indicators: {[k for k,v in field['indicators'].items() if v]}")
    
    # Save all data
    output = {
        'all_elements': elements,
        'field_indicators': field_indicators
    }
    
    with open('pdf_coordinates.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\n✅ Full coordinate data saved to: pdf_coordinates.json")

