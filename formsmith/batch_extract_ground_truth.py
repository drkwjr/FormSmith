#!/usr/bin/env python3
"""
Batch extract ground truth from multiple filled PDFs.

Extracts form field positions from filled PDFs and saves as ground truth datasets.
"""

import sys
import json
from pathlib import Path
import fitz  # PyMuPDF

def extract_fields_from_filled_pdf(pdf_path: str) -> dict:
    """
    Extract all form fields from a filled PDF.
    
    Args:
        pdf_path: Path to filled PDF
        
    Returns:
        Dictionary with form metadata and field list
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    
    fields = []
    total_pages = len(doc)
    
    print(f"\n📄 Analyzing: {pdf_path.name}")
    print(f"   Pages: {total_pages}")
    
    for page_num in range(total_pages):
        page = doc[page_num]
        
        # Get all widgets (form fields) on this page
        widgets = page.widgets()
        page_fields = list(widgets)
        
        if page_fields:
            print(f"   Page {page_num + 1}: {len(page_fields)} fields")
        
        for widget in page_fields:
            # Extract field properties
            field_info = {
                "name": widget.field_name or f"unnamed_field_{len(fields)}",
                "type": _widget_type_name(widget.field_type),
                "page": page_num,
                "bbox": list(widget.rect),  # [x0, y0, x1, y1]
                "value": widget.field_value or "",
                "flags": widget.field_flags
            }
            
            fields.append(field_info)
    
    doc.close()
    
    # Build result
    result = {
        "source_pdf": str(pdf_path.name),
        "source_path": str(pdf_path.absolute()),
        "total_pages": total_pages,
        "total_fields": len(fields),
        "extraction_method": "existing_form_fields",
        "confidence": 1.0,
        "fields": fields
    }
    
    # Field type summary
    field_types = {}
    for field in fields:
        ftype = field["type"]
        field_types[ftype] = field_types.get(ftype, 0) + 1
    
    result["field_types"] = field_types
    
    print(f"   ✅ Extracted {len(fields)} fields")
    print(f"   Field types: {field_types}")
    
    return result


def _widget_type_name(field_type: int) -> str:
    """Convert PyMuPDF field type constant to readable name."""
    type_map = {
        fitz.PDF_WIDGET_TYPE_TEXT: "text",
        fitz.PDF_WIDGET_TYPE_CHECKBOX: "checkbox",
        fitz.PDF_WIDGET_TYPE_RADIOBUTTON: "radio",
        fitz.PDF_WIDGET_TYPE_COMBOBOX: "combobox",
        fitz.PDF_WIDGET_TYPE_LISTBOX: "listbox",
        fitz.PDF_WIDGET_TYPE_SIGNATURE: "signature",
    }
    
    if hasattr(fitz, 'PDF_WIDGET_TYPE_PUSHBUTTON'):
        type_map[fitz.PDF_WIDGET_TYPE_PUSHBUTTON] = "button"
    
    return type_map.get(field_type, f"unknown_{field_type}")


def main():
    """Extract ground truth from all filled PDFs in root directory."""
    
    print("="*80)
    print("  BATCH GROUND TRUTH EXTRACTION")
    print("="*80)
    
    # Find all filled PDFs
    root_dir = Path(__file__).parent.parent.parent
    filled_pdfs = sorted(root_dir.glob("*-FILLEDOUT*.pdf"))
    
    if not filled_pdfs:
        print("❌ No filled PDFs found in root directory")
        print(f"   Looked in: {root_dir}")
        print("   Pattern: *-FILLEDOUT*.pdf")
        return 1
    
    print(f"\n📁 Found {len(filled_pdfs)} filled PDFs:")
    for pdf in filled_pdfs:
        print(f"   • {pdf.name}")
    
    # Create output directory
    output_dir = Path(__file__).parent / "data" / "ground_truth"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Output directory: {output_dir}")
    
    # Extract from each PDF
    results = []
    total_fields = 0
    
    for pdf_path in filled_pdfs:
        try:
            result = extract_fields_from_filled_pdf(pdf_path)
            results.append(result)
            total_fields += result["total_fields"]
            
            # Save individual ground truth file
            # Convert filename: "CJD101A-FILLEDOUT.pdf" -> "CJD101A_ground_truth.json"
            base_name = pdf_path.stem.replace("-FILLEDOUT", "").replace(" ", "_")
            output_path = output_dir / f"{base_name}_ground_truth.json"
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"   💾 Saved: {output_path.name}")
            
        except Exception as e:
            print(f"   ❌ Error processing {pdf_path.name}: {e}")
            continue
    
    # Create summary report
    summary = {
        "total_forms": len(results),
        "total_fields": total_fields,
        "forms": [
            {
                "name": r["source_pdf"],
                "fields": r["total_fields"],
                "pages": r["total_pages"],
                "types": r["field_types"]
            }
            for r in results
        ]
    }
    
    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("  EXTRACTION SUMMARY")
    print("="*80)
    print(f"\n✅ Processed {len(results)} forms")
    print(f"✅ Extracted {total_fields} total fields")
    print(f"✅ Saved to: {output_dir}")
    
    print("\n📊 Form Details:")
    for form in summary["forms"]:
        print(f"\n   {form['name']}")
        print(f"      Fields: {form['fields']}")
        print(f"      Pages: {form['pages']}")
        print(f"      Types: {form['types']}")
    
    print(f"\n💾 Summary saved: {summary_path}")
    print("\n" + "="*80)
    print("  NEXT STEPS")
    print("="*80)
    print("\n1. Review extracted fields in:", output_dir)
    print("2. Build template matching system")
    print("3. Test on blank PDFs")
    print("4. Compare accuracy: template vs OpenCV")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

