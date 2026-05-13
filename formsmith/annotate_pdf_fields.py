#!/usr/bin/env python3
"""
PDF Field Annotation Tool

Generates visual documentation of PDF form fields by:
1. Extracting all field names, types, and positions
2. Creating an annotated PDF with field names overlaid at their positions
3. Generating a mapping table sorted by vertical position (top to bottom)
4. Creating a visual reference image showing field locations

Usage:
    python -m formsmith.annotate_pdf_fields <input.pdf> [output_dir]

Example:
    python -m formsmith.annotate_pdf_fields \\
        examples/sample_form.pdf \\
        out/sample_analysis
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import pikepdf
except ImportError:
    print("ERROR: pikepdf not installed. Install with: pip3 install pikepdf")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Install with: pip3 install Pillow PyMuPDF")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def extract_field_info(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract all form field information from a PDF.
    
    Returns a list of dicts with keys:
    - name: Field name
    - type: Field type (/Tx, /Btn, /Sig)
    - x: X coordinate (left edge)
    - y: Y coordinate (bottom edge, PDF coordinate system)
    - width: Field width
    - height: Field height
    - page: Page number (0-indexed)
    """
    logger.info(f"Opening PDF: {pdf_path}")
    pdf = pikepdf.open(pdf_path)
    
    if not hasattr(pdf.Root, 'AcroForm') or not hasattr(pdf.Root.AcroForm, 'Fields'):
        logger.error("PDF has no form fields")
        return []
    
    fields = []
    for idx, field in enumerate(pdf.Root.AcroForm.Fields):
        field_name = str(field.T)
        field_type = str(field.FT) if hasattr(field, 'FT') else 'unknown'
        
        # Get widget annotation (contains position info)
        if hasattr(field, 'Kids') and field.Kids:
            widget = field.Kids[0]
            if hasattr(widget, 'Rect'):
                rect = widget.Rect
                # PDF coordinates: [x_min, y_min, x_max, y_max]
                # Origin is bottom-left
                x = float(rect[0])
                y = float(rect[1])
                width = float(rect[2]) - x
                height = float(rect[3]) - y
                
                # Get page number
                page_num = 0  # Default to first page
                if hasattr(widget, 'P'):
                    # Find which page this widget is on
                    for i, page in enumerate(pdf.pages):
                        if page.obj == widget.P:
                            page_num = i
                            break
                
                fields.append({
                    'index': idx + 1,
                    'name': field_name,
                    'type': field_type,
                    'type_label': {'/Tx': 'Text', '/Btn': 'Checkbox', '/Sig': 'Signature'}.get(field_type, field_type),
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'page': page_num,
                })
    
    logger.info(f"Extracted {len(fields)} fields")
    return fields


def create_field_mapping_table(fields: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Create a markdown table of all fields, sorted by position (top to bottom).
    
    PDF coordinates have origin at bottom-left, so higher Y = lower on page.
    We sort by descending Y to get top-to-bottom order.
    """
    # Sort by page, then by Y descending (top to bottom)
    sorted_fields = sorted(fields, key=lambda f: (f['page'], -f['y']))
    
    lines = [
        "# CJD-101A PDF Field Mapping",
        "",
        f"**Total Fields**: {len(fields)}",
        f"**Generated**: {Path(__file__).name}",
        "",
        "## Field List (Top to Bottom)",
        "",
        "| # | Field Name | Type | Page | X | Y | Width | Height | Visual Location |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    
    for field in sorted_fields:
        # Determine general location
        page_width = 612  # Standard US Letter width in points
        page_height = 792  # Standard US Letter height in points
        
        # Horizontal position
        if field['x'] < page_width / 3:
            h_pos = "Left"
        elif field['x'] < 2 * page_width / 3:
            h_pos = "Center"
        else:
            h_pos = "Right"
        
        # Vertical position (remember: higher Y = lower on page in PDF coords)
        if field['y'] > 2 * page_height / 3:
            v_pos = "Bottom"
        elif field['y'] > page_height / 3:
            v_pos = "Middle"
        else:
            v_pos = "Top"
        
        location = f"{v_pos}-{h_pos}"
        
        lines.append(
            f"| {field['index']} | `{field['name']}` | {field['type_label']} | "
            f"{field['page'] + 1} | {field['x']:.0f} | {field['y']:.0f} | "
            f"{field['width']:.0f} | {field['height']:.0f} | {location} |"
        )
    
    lines.extend([
        "",
        "## Notes",
        "",
        "- **Coordinates**: PDF coordinate system has origin at bottom-left",
        "- **Y-axis**: Higher Y values = closer to bottom of page",
        "- **X-axis**: Higher X values = closer to right edge",
        "- **Visual Location**: Approximate position on the page",
        "",
        "## Next Steps",
        "",
        "1. Open `CJD101A-FILLEDOUT.pdf` in a PDF viewer",
        "2. For each field in this table, note what label is next to it on the form",
        "3. Update the mapping table with the correct YAML variable",
        "",
        "## Mapping Template",
        "",
        "```yaml",
        "fields:",
        "  # Update these with actual field purposes based on visual inspection",
    ])
    
    for field in sorted_fields[:5]:  # Show first 5 as examples
        lines.append(f'  - "{field["name"]}": ${{ FIXME_REPLACE_WITH_YAML_VARIABLE }}')
    
    lines.append("  # ... continue for all fields")
    lines.append("```")
    
    output_path.write_text('\n'.join(lines))
    logger.info(f"Created mapping table: {output_path}")


def create_annotated_image(pdf_path: Path, fields: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Create an image of the PDF with field names and bounding boxes overlaid.
    """
    logger.info("Generating annotated image...")
    
    # Open PDF with PyMuPDF for rendering
    doc = fitz.open(str(pdf_path))
    
    # For now, just annotate the first page
    page = doc[0]
    
    # Render page to image at high DPI
    zoom = 2  # 2x zoom for clarity
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    # Convert to PIL Image
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data))
    
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    # PDF page dimensions
    page_height = page.rect.height
    
    # Draw each field
    for field in fields:
        if field['page'] != 0:
            continue  # Only annotate first page for now
        
        # Convert PDF coordinates to image coordinates
        # PDF: origin bottom-left, Y increases upward
        # Image: origin top-left, Y increases downward
        x = field['x'] * zoom
        y = (page_height - field['y'] - field['height']) * zoom
        width = field['width'] * zoom
        height = field['height'] * zoom
        
        # Draw bounding box
        color = {
            'Text': (0, 128, 255),      # Blue
            'Checkbox': (255, 128, 0),  # Orange
            'Signature': (255, 0, 128),  # Pink
        }.get(field['type_label'], (128, 128, 128))
        
        draw.rectangle(
            [x, y, x + width, y + height],
            outline=color,
            width=2
        )
        
        # Draw field name above the box
        text = f"{field['index']}: {field['name']}"
        
        # Place text above box if there's room, otherwise below
        text_y = y - 15 if y > 20 else y + height + 5
        
        # Draw text background for readability
        text_bbox = draw.textbbox((x, text_y), text, font=font_small)
        draw.rectangle(text_bbox, fill=(255, 255, 255, 200))
        draw.text((x, text_y), text, fill=color, font=font_small)
    
    # Add legend
    legend_x = 10
    legend_y = 10
    legend_items = [
        ("Text Fields", (0, 128, 255)),
        ("Checkboxes", (255, 128, 0)),
        ("Signatures", (255, 0, 128)),
    ]
    
    for label, color in legend_items:
        draw.rectangle(
            [legend_x, legend_y, legend_x + 20, legend_y + 12],
            outline=color,
            width=2
        )
        draw.text((legend_x + 25, legend_y), label, fill=(0, 0, 0), font=font)
        legend_y += 20
    
    # Save image
    img.save(output_path, "PNG")
    logger.info(f"Created annotated image: {output_path}")
    
    doc.close()


def create_json_export(fields: List[Dict[str, Any]], output_path: Path) -> None:
    """Export field data as JSON for programmatic use."""
    data = {
        'total_fields': len(fields),
        'fields': fields,
    }
    
    output_path.write_text(json.dumps(data, indent=2))
    logger.info(f"Created JSON export: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate PDF form fields with visual overlays and mapping tables"
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to input PDF with form fields"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs='?',
        default=None,
        help="Output directory for generated files (default: data/exports/<pdf_name>)"
    )
    
    args = parser.parse_args()
    
    if not args.pdf_path.exists():
        logger.error(f"PDF not found: {args.pdf_path}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        pdf_name = args.pdf_path.stem
        output_dir = Path("data/exports") / f"{pdf_name}_analysis"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Extract field information
    fields = extract_field_info(args.pdf_path)
    
    if not fields:
        logger.error("No fields found in PDF")
        sys.exit(1)
    
    # Generate outputs
    create_field_mapping_table(
        fields,
        output_dir / "FIELD_MAPPING_TABLE.md"
    )
    
    create_json_export(
        fields,
        output_dir / "fields.json"
    )
    
    try:
        create_annotated_image(
            args.pdf_path,
            fields,
            output_dir / "annotated_page1.png"
        )
    except Exception as e:
        logger.warning(f"Could not create annotated image: {e}")
        logger.info("Image generation skipped (optional)")
    
    logger.info("=" * 60)
    logger.info("✅ Annotation complete!")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info("")
    logger.info("Generated files:")
    logger.info(f"  - FIELD_MAPPING_TABLE.md  (field list sorted top-to-bottom)")
    logger.info(f"  - fields.json              (machine-readable field data)")
    logger.info(f"  - annotated_page1.png      (visual field overlay)")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Open FIELD_MAPPING_TABLE.md to see all fields")
    logger.info("  2. Open CJD101A-FILLEDOUT.pdf side-by-side")
    logger.info("  3. For each field, note what it represents on the form")
    logger.info("  4. Create correct mapping for v0.60")


if __name__ == "__main__":
    # Add missing import
    import io
    main()

