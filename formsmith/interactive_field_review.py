#!/usr/bin/env python3
"""
Interactive Field Review Tool
Allows quick visual review and cleanup of detected fields.

Shows each detected field on the form and lets you:
- Accept (keep it)
- Reject (mark as false positive)
- Adjust position/size if needed

Output: Cleaned field list ready for production.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import io


def visualize_field(pdf_path: str, field: Dict, page_num: int) -> Image.Image:
    """
    Create visualization of a single field on the PDF page.
    
    Args:
        pdf_path: Path to PDF
        field: Field definition
        page_num: Page number
    
    Returns:
        PIL Image with field highlighted
    """
    # Open PDF and render page to image
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Render at 2x resolution for clarity
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    
    # Convert to PIL Image
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data))
    
    # Draw field bbox
    draw = ImageDraw.Draw(img)
    
    bbox = field['bbox']
    # Scale bbox for 2x resolution
    x0, y0, x1, y1 = [coord * 2 for coord in bbox]
    
    # Draw field rectangle (green for review)
    draw.rectangle([x0, y0, x1, y1], outline='green', width=3)
    
    # Draw label if present
    if field.get('source_label'):
        label_text = field['source_label'][:30]  # Truncate long labels
        # Try to get a font, fall back to default
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw label above field
        draw.text((x0, y0 - 25), label_text, fill='green', font=font)
    
    # Draw field info
    field_info = f"{field.get('interview_variable', 'unknown')} (conf: {field.get('confidence', 0):.2f})"
    draw.text((x0, y1 + 5), field_info, fill='blue', font=font if 'font' in locals() else None)
    
    doc.close()
    
    return img


def generate_review_html(
    pdf_path: str,
    detected_fields_path: str,
    output_html: str
) -> str:
    """
    Generate interactive HTML review interface.
    
    Shows all detected fields with Accept/Reject buttons.
    """
    # Load detected fields
    with open(detected_fields_path, 'r') as f:
        data = json.load(f)
    
    fields = data.get('fields', [])
    
    # Group by page
    fields_by_page = {}
    for field in fields:
        page = field.get('page', 0)
        if page not in fields_by_page:
            fields_by_page[page] = []
        fields_by_page[page].append(field)
    
    # Generate HTML
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Field Review - {pdf_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .field-card {{
            border: 2px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            background: white;
            transition: all 0.3s;
        }}
        .field-card.accepted {{
            border-color: #4CAF50;
            background: #f1f8f4;
        }}
        .field-card.rejected {{
            border-color: #f44336;
            background: #fef1f0;
            opacity: 0.6;
        }}
        .field-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .field-name {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }}
        .field-info {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 10px;
            margin: 10px 0;
            font-size: 14px;
        }}
        .info-item {{
            padding: 5px;
            background: #f9f9f9;
            border-radius: 4px;
        }}
        .info-label {{
            font-weight: bold;
            color: #666;
        }}
        .buttons {{
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }}
        button {{
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.2s;
        }}
        .accept-btn {{
            background: #4CAF50;
            color: white;
        }}
        .accept-btn:hover {{
            background: #45a049;
        }}
        .reject-btn {{
            background: #f44336;
            color: white;
        }}
        .reject-btn:hover {{
            background: #da190b;
        }}
        .undo-btn {{
            background: #ff9800;
            color: white;
        }}
        .undo-btn:hover {{
            background: #e68900;
        }}
        .export-btn {{
            background: #2196F3;
            color: white;
            padding: 15px 30px;
            font-size: 16px;
            position: fixed;
            bottom: 20px;
            right: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .export-btn:hover {{
            background: #0b7dda;
        }}
        .confidence {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .conf-high {{ background: #4CAF50; color: white; }}
        .conf-med {{ background: #ff9800; color: white; }}
        .conf-low {{ background: #f44336; color: white; }}
        .page-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin: 30px 0 20px 0;
            font-size: 20px;
            font-weight: bold;
        }}
        .stats {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            min-width: 200px;
        }}
        .stat-item {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            font-size: 14px;
        }}
        .stat-value {{
            font-weight: bold;
            color: #4CAF50;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Interactive Field Review</h1>
        
        <div class="summary">
            <strong>PDF:</strong> {pdf_name}<br>
            <strong>Total Fields Detected:</strong> <span id="total-fields">{total_fields}</span><br>
            <strong>Instructions:</strong> Review each field and click "Accept" to keep it or "Reject" to mark as false positive.
            Click "Export Clean Results" when done.
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <span>Total:</span>
                <span class="stat-value" id="stat-total">{total_fields}</span>
            </div>
            <div class="stat-item">
                <span>Accepted:</span>
                <span class="stat-value" id="stat-accepted" style="color: #4CAF50;">0</span>
            </div>
            <div class="stat-item">
                <span>Rejected:</span>
                <span class="stat-value" id="stat-rejected" style="color: #f44336;">0</span>
            </div>
            <div class="stat-item">
                <span>Pending:</span>
                <span class="stat-value" id="stat-pending" style="color: #ff9800;">{total_fields}</span>
            </div>
        </div>
        
        {fields_html}
        
        <button class="export-btn" onclick="exportResults()">
            📥 Export Clean Results ({total_fields} fields)
        </button>
    </div>
    
    <script>
        let fieldStates = {{}};
        let totalFields = {total_fields};
        
        function updateStats() {{
            let accepted = Object.values(fieldStates).filter(s => s === 'accepted').length;
            let rejected = Object.values(fieldStates).filter(s => s === 'rejected').length;
            let pending = totalFields - accepted - rejected;
            
            document.getElementById('stat-accepted').textContent = accepted;
            document.getElementById('stat-rejected').textContent = rejected;
            document.getElementById('stat-pending').textContent = pending;
            
            document.querySelector('.export-btn').textContent = 
                `📥 Export Clean Results (${{accepted}} fields)`;
        }}
        
        function acceptField(fieldId) {{
            fieldStates[fieldId] = 'accepted';
            document.getElementById('field-' + fieldId).classList.add('accepted');
            document.getElementById('field-' + fieldId).classList.remove('rejected');
            updateStats();
        }}
        
        function rejectField(fieldId) {{
            fieldStates[fieldId] = 'rejected';
            document.getElementById('field-' + fieldId).classList.add('rejected');
            document.getElementById('field-' + fieldId).classList.remove('accepted');
            updateStats();
        }}
        
        function undoField(fieldId) {{
            delete fieldStates[fieldId];
            document.getElementById('field-' + fieldId).classList.remove('accepted', 'rejected');
            updateStats();
        }}
        
        function exportResults() {{
            // Collect accepted field IDs
            let acceptedIds = Object.keys(fieldStates).filter(id => fieldStates[id] === 'accepted');
            
            // Create JSON blob
            let result = {{
                accepted_field_ids: acceptedIds,
                total_reviewed: totalFields,
                total_accepted: acceptedIds.length,
                total_rejected: Object.values(fieldStates).filter(s => s === 'rejected').length,
                timestamp: new Date().toISOString()
            }};
            
            // Download as JSON
            let blob = new Blob([JSON.stringify(result, null, 2)], {{type: 'application/json'}});
            let url = URL.createObjectURL(blob);
            let a = document.createElement('a');
            a.href = url;
            a.download = 'field_review_results.json';
            a.click();
            
            alert(`Exported ${{acceptedIds.length}} accepted fields!`);
        }}
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'a' || e.key === 'A') {{
                // Find first pending field and accept it
                let pending = document.querySelector('.field-card:not(.accepted):not(.rejected)');
                if (pending) {{
                    let fieldId = pending.id.replace('field-', '');
                    acceptField(fieldId);
                    pending.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                }}
            }} else if (e.key === 'r' || e.key === 'R') {{
                // Find first pending field and reject it
                let pending = document.querySelector('.field-card:not(.accepted):not(.rejected)');
                if (pending) {{
                    let fieldId = pending.id.replace('field-', '');
                    rejectField(fieldId);
                    pending.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                }}
            }}
        }});
        
        updateStats();
    </script>
</body>
</html>
    """.format(
        pdf_name=Path(pdf_path).name,
        total_fields=len(fields),
        fields_html=_generate_fields_html(fields)
    )
    
    # Save HTML
    with open(output_html, 'w') as f:
        f.write(html)
    
    return output_html


def _generate_fields_html(fields: List[Dict]) -> str:
    """Generate HTML for all field cards."""
    html_parts = []
    
    # Group by page
    fields_by_page = {}
    for i, field in enumerate(fields):
        page = field.get('page', 0)
        if page not in fields_by_page:
            fields_by_page[page] = []
        fields_by_page[page].append((i, field))
    
    # Generate HTML for each page
    for page_num in sorted(fields_by_page.keys()):
        html_parts.append(f'<div class="page-header">📄 Page {page_num + 1}</div>')
        
        for field_id, field in fields_by_page[page_num]:
            confidence = field.get('confidence', 0)
            
            # Confidence class
            if confidence >= 0.85:
                conf_class = 'conf-high'
            elif confidence >= 0.75:
                conf_class = 'conf-med'
            else:
                conf_class = 'conf-low'
            
            bbox = field.get('bbox', [0, 0, 0, 0])
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            html_parts.append(f"""
                <div class="field-card" id="field-{field_id}">
                    <div class="field-header">
                        <div class="field-name">{field.get('interview_variable', 'unknown')}</div>
                        <span class="confidence {conf_class}">
                            {confidence:.1%}
                        </span>
                    </div>
                    
                    <div class="field-info">
                        <div class="info-item">
                            <span class="info-label">Type:</span> {field.get('pdf_type', 'unknown')}
                        </div>
                        <div class="info-item">
                            <span class="info-label">Size:</span> {width:.0f}×{height:.0f}px
                        </div>
                        <div class="info-item">
                            <span class="info-label">Method:</span> {field.get('detection_method', 'unknown')}
                        </div>
                    </div>
                    
                    {f'<div style="margin: 10px 0; color: #666;"><strong>Label:</strong> {field.get("source_label", "N/A")}</div>' if field.get('source_label') else ''}
                    
                    <div class="field-info">
                        <div class="info-item">
                            <span class="info-label">Position:</span> ({bbox[0]:.0f}, {bbox[1]:.0f})
                        </div>
                        <div class="info-item">
                            <span class="info-label">DA Type:</span> {field.get('interview_type', 'unknown')}
                        </div>
                        <div class="info-item">
                            <span class="info-label">DA Datatype:</span> {field.get('interview_datatype', 'text')}
                        </div>
                    </div>
                    
                    <div class="buttons">
                        <button class="accept-btn" onclick="acceptField({field_id})">
                            ✓ Accept (A)
                        </button>
                        <button class="reject-btn" onclick="rejectField({field_id})">
                            ✗ Reject (R)
                        </button>
                        <button class="undo-btn" onclick="undoField({field_id})">
                            ↶ Undo
                        </button>
                    </div>
                </div>
            """)
    
    return '\n'.join(html_parts)


def apply_review_results(
    original_fields_path: str,
    review_results_path: str,
    output_path: str
) -> Dict:
    """
    Apply review results to original fields and export cleaned version.
    
    Args:
        original_fields_path: Path to original detected fields JSON
        review_results_path: Path to review results JSON
        output_path: Path to save cleaned fields
    
    Returns:
        Stats dict
    """
    # Load original fields
    with open(original_fields_path, 'r') as f:
        original_data = json.load(f)
    
    original_fields = original_data.get('fields', [])
    
    # Load review results
    with open(review_results_path, 'r') as f:
        review = json.load(f)
    
    accepted_ids = review.get('accepted_field_ids', [])
    accepted_ids_int = [int(id) for id in accepted_ids]
    
    # Filter to accepted fields only
    cleaned_fields = [
        field for i, field in enumerate(original_fields)
        if i in accepted_ids_int
    ]
    
    # Create cleaned output
    cleaned_data = original_data.copy()
    cleaned_data['fields'] = cleaned_fields
    cleaned_data['review_metadata'] = {
        'original_count': len(original_fields),
        'accepted_count': len(cleaned_fields),
        'rejected_count': len(original_fields) - len(cleaned_fields),
        'false_positive_rate': (len(original_fields) - len(cleaned_fields)) / len(original_fields) if original_fields else 0,
        'review_timestamp': review.get('timestamp')
    }
    
    # Save cleaned fields
    with open(output_path, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    return cleaned_data['review_metadata']


def main():
    """Main entry point."""
    if len(sys.argv) < 4:
        print("Usage: python interactive_field_review.py <pdf> <detected_fields_json> <output_html>")
        print("\nExample:")
        print("  python interactive_field_review.py \\")
        print("    form.pdf \\")
        print("    output/test/form_da_ready.json \\")
        print("    output/test/field_review.html")
        print("\nThen:")
        print("  1. Open field_review.html in browser")
        print("  2. Review each field (Accept/Reject)")
        print("  3. Export results (downloads field_review_results.json)")
        print("  4. Apply results:")
        print("     python interactive_field_review.py apply \\")
        print("       output/test/form_da_ready.json \\")
        print("       field_review_results.json \\")
        print("       output/test/form_da_ready_cleaned.json")
        sys.exit(1)
    
    if sys.argv[1] == 'apply':
        # Apply mode
        original_path = sys.argv[2]
        review_path = sys.argv[3]
        output_path = sys.argv[4]
        
        print(f"\n{'='*80}")
        print(f"  APPLYING REVIEW RESULTS")
        print(f"{'='*80}\n")
        
        stats = apply_review_results(original_path, review_path, output_path)
        
        print(f"✅ Review applied successfully!\n")
        print(f"Original fields:  {stats['original_count']}")
        print(f"Accepted fields:  {stats['accepted_count']}")
        print(f"Rejected fields:  {stats['rejected_count']}")
        print(f"FP rate removed: {stats['false_positive_rate']:.1%}")
        print(f"\n💾 Cleaned fields saved to: {output_path}\n")
    else:
        # Generate mode
        pdf_path = sys.argv[1]
        fields_path = sys.argv[2]
        output_html = sys.argv[3]
        
        print(f"\n{'='*80}")
        print(f"  GENERATING INTERACTIVE REVIEW")
        print(f"{'='*80}\n")
        
        html_path = generate_review_html(pdf_path, fields_path, output_html)
        
        print(f"✅ Review interface generated!\n")
        print(f"📄 Open in browser: {html_path}")
        print(f"\nKeyboard shortcuts:")
        print(f"  A = Accept current field")
        print(f"  R = Reject current field")
        print(f"\nWhen done, click 'Export Clean Results' button.\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

