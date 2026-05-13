#!/usr/bin/env python3
"""
Extract Ground Truth from Filled PDF

Takes a filled PDF with existing form fields and extracts them as ground truth.
This is actually BETTER than manual placement because the coordinates are proven to work!
"""

import fitz
import json
from pathlib import Path


def extract_ground_truth(filled_pdf: str, output_json: str, pdf_name: str = None):
    """
    Extract form fields from filled PDF as ground truth
    
    Args:
        filled_pdf: Path to filled PDF with form fields
        output_json: Where to save ground truth JSON
        pdf_name: Original blank PDF name (for reference)
    """
    print(f"\n{'='*80}")
    print("  EXTRACTING GROUND TRUTH FROM FILLED PDF")
    print(f"{'='*80}\n")
    
    doc = fitz.open(filled_pdf)
    page = doc[0]
    
    # Extract all widgets (form fields)
    widgets = list(page.widgets())
    
    print(f"Found {len(widgets)} form fields")
    
    # Map widget types
    type_map = {
        7: "text",  # PDF_WIDGET_TYPE_TEXT
        2: "checkbox",  # PDF_WIDGET_TYPE_CHECKBOX
        6: "signature",  # PDF_WIDGET_TYPE_SIGNATURE
        4: "combobox",  # PDF_WIDGET_TYPE_COMBOBOX
    }
    
    # Extract fields
    fields = []
    type_counts = {}
    
    for i, widget in enumerate(widgets):
        rect = widget.rect
        field_type_raw = widget.field_type
        field_type = type_map.get(field_type_raw, "text")
        name = widget.field_name or f"field_{i}"
        value = widget.field_value or ""
        
        # Create ground truth entry
        field = {
            "name": name,
            "type": field_type,
            "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
            "x": float(rect.x0),
            "y_from_top": float(rect.y0),
            "width": float(rect.x1 - rect.x0),
            "height": float(rect.y1 - rect.y0),
            "page": 0,
            "filled_value": value if value else None,
            "source": "extracted_from_filled_pdf"
        }
        
        fields.append(field)
        type_counts[field_type] = type_counts.get(field_type, 0) + 1
    
    # Create ground truth
    ground_truth = {
        "pdf": pdf_name or Path(filled_pdf).name,
        "description": "Ground truth extracted from filled PDF (proven working fields)",
        "source": filled_pdf,
        "extraction_method": "PyMuPDF widget extraction",
        "field_count": len(fields),
        "fields": fields
    }
    
    # Save
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"✅ Ground Truth Created")
    print(f"   Fields: {len(fields)}")
    print(f"   Saved to: {output_path}")
    
    print(f"\n   Field Types:")
    for ftype, count in sorted(type_counts.items()):
        print(f"     • {ftype}: {count}")
    
    # Show statistics
    widths = [f['width'] for f in fields]
    heights = [f['height'] for f in fields]
    
    print(f"\n   Field Dimensions:")
    print(f"     Width:  min={min(widths):.1f}, max={max(widths):.1f}, avg={sum(widths)/len(widths):.1f}")
    print(f"     Height: min={min(heights):.1f}, max={max(heights):.1f}, avg={sum(heights)/len(heights):.1f}")
    
    # Warn about generic names
    generic_names = [f['name'] for f in fields if f['name'].startswith('Textbox') or f['name'].startswith('field_')]
    if generic_names:
        print(f"\n   ⚠️  Note: {len(generic_names)} fields have generic names (Textbox6, etc.)")
        print(f"      This is fine for position ground truth!")
        print(f"      We'll need to map these to semantic names later.")
    
    doc.close()
    
    return ground_truth


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_ground_truth_from_filled.py <filled_pdf> [output_json]")
        print("\nExample:")
        print("  python extract_ground_truth_from_filled.py CJD101A-FILLEDOUT.pdf \\")
        print("    tools/validation/fixtures/ground_truth/divorce_form_fields.json")
        sys.exit(1)
    
    filled_pdf = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else "ground_truth.json"
    
    extract_ground_truth(filled_pdf, output_json)
    
    print("\n" + "="*80)
    print("  NEXT STEPS")
    print("="*80)
    print("\n1. Review the extracted ground truth")
    print("2. Run visual comparison to see current detection vs. ground truth")
    print("3. Start Phase 3 algorithm improvements!")

