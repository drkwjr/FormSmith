#!/usr/bin/env python3
"""
Phase 1.1: Baseline Testing of Automated Field Detection System

This script runs the complete existing detection pipeline and generates
a comprehensive report showing exactly what it produces.
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime
import traceback

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent))

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def run_baseline_test(pdf_path: str, output_dir: str = None):
    """
    Run complete baseline test of automated field detection
    
    Args:
        pdf_path: Path to PDF to analyze
        output_dir: Directory to store outputs (default: data/baseline_test)
    """
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "data" / "baseline_test" / timestamp
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = Path(pdf_path).resolve()
    pdf_name = pdf_path.name
    
    print_section(f"BASELINE TEST: {pdf_name}")
    print(f"PDF: {pdf_path}")
    print(f"Output Directory: {output_dir}")
    
    # Store all results
    results = {
        "pdf": str(pdf_path),
        "pdf_name": pdf_name,
        "timestamp": datetime.now().isoformat(),
        "stages": {}
    }
    
    # ========================================================================
    # STAGE 1: PDF Analysis (smart_pdf_analyzer.py)
    # ========================================================================
    print_section("STAGE 1: PDF Analysis (smart_pdf_analyzer.py)")
    
    try:
        from smart_pdf_analyzer import SmartPDFAnalyzer
        
        analyzer = SmartPDFAnalyzer(str(pdf_path))
        analysis = analyzer.analyze()
        
        # Save analysis
        analysis_file = output_dir / "01_pdf_analysis.json"
        analyzer.save_analysis(str(analysis_file))
        
        # Report
        text_count = len(analysis.get('text_elements', []))
        visual_count = len(analysis.get('visual_elements', []))
        checkbox_count = len(analysis.get('checkbox_indicators', []))
        
        print(f"✅ PDF Analysis Complete")
        print(f"   - Text elements: {text_count}")
        print(f"   - Visual elements: {visual_count}")
        print(f"   - Checkbox indicators: {checkbox_count}")
        print(f"   - Saved to: {analysis_file.name}")
        
        results["stages"]["pdf_analysis"] = {
            "status": "success",
            "text_elements": text_count,
            "visual_elements": visual_count,
            "checkbox_indicators": checkbox_count,
            "output_file": str(analysis_file)
        }
        
        # Try to export existing fields if any
        try:
            existing_fields_file = output_dir / "01_existing_fields.json"
            existing_overlay_file = output_dir / "01_existing_fields_overlay.png"
            existing_fields = analyzer.export_existing_fields(
                str(existing_fields_file),
                str(existing_overlay_file)
            )
            if existing_fields:
                print(f"   - Existing fields found: {len(existing_fields)}")
                print(f"   - Overlay saved: {existing_overlay_file.name}")
        except Exception as e:
            print(f"   - No existing fields or error exporting: {e}")
        
    except Exception as e:
        print(f"❌ PDF Analysis Failed: {e}")
        traceback.print_exc()
        results["stages"]["pdf_analysis"] = {
            "status": "failed",
            "error": str(e)
        }
        analysis = None
    
    # ========================================================================
    # STAGE 2: Field Detection (intelligent_field_mapper.py)
    # ========================================================================
    print_section("STAGE 2: Field Detection (intelligent_field_mapper.py)")
    
    try:
        from intelligent_field_mapper import (
            analyze_pdf_layout,
            identify_field_locations,
            calculate_field_positions,
            refine_field_positions
        )
        
        # Step 1: Analyze layout
        print("Analyzing PDF layout...")
        layout_data = analyze_pdf_layout(str(pdf_path))
        
        layout_file = output_dir / "02_layout_data.json"
        with open(layout_file, 'w') as f:
            json.dump(layout_data, f, indent=2)
        
        text_blocks = len(layout_data.get('text_blocks', []))
        print(f"✅ Layout Analysis Complete")
        print(f"   - Text blocks: {text_blocks}")
        print(f"   - Saved to: {layout_file.name}")
        
        # Step 2: Identify field locations
        print("\nIdentifying field locations...")
        field_locations = identify_field_locations(layout_data)
        
        print(f"✅ Field Detection Complete")
        print(f"   - Fields detected: {len(field_locations)}")
        
        # Show detected fields by type
        field_types = {}
        for fl in field_locations:
            ftype = fl.get('field_type', 'unknown')
            field_types[ftype] = field_types.get(ftype, 0) + 1
        
        print(f"   - By type:")
        for ftype, count in field_types.items():
            print(f"     • {ftype}: {count}")
        
        # Step 3: Calculate positions
        print("\nCalculating field positions...")
        field_definitions = calculate_field_positions(field_locations)
        
        # Step 4: Refine positions
        print("Refining field positions...")
        refined_fields = refine_field_positions(field_definitions, layout_data)
        
        # Save detected fields
        detected_file = output_dir / "02_detected_fields.json"
        detected_data = {
            "pdf": pdf_name,
            "fields": refined_fields,
            "field_count": len(refined_fields),
            "field_types": field_types
        }
        
        with open(detected_file, 'w') as f:
            json.dump(detected_data, f, indent=2)
        
        print(f"✅ Field positions calculated")
        print(f"   - Saved to: {detected_file.name}")
        
        # Show sample fields
        print(f"\n   Sample detected fields:")
        for i, field in enumerate(refined_fields[:10], 1):
            name = field.get('name', 'unnamed')
            ftype = field.get('type', 'unknown')
            x = field.get('x', 0)
            y = field.get('y_from_top', 0)
            w = field.get('width', 0)
            h = field.get('height', 0)
            print(f"     {i:2d}. {name:30s} ({ftype:10s}) @ ({x:6.1f}, {y:6.1f}) {w:4.0f}x{h:4.0f}")
        
        if len(refined_fields) > 10:
            print(f"     ... and {len(refined_fields) - 10} more")
        
        results["stages"]["field_detection"] = {
            "status": "success",
            "fields_detected": len(refined_fields),
            "field_types": field_types,
            "output_file": str(detected_file)
        }
        
    except Exception as e:
        print(f"❌ Field Detection Failed: {e}")
        traceback.print_exc()
        results["stages"]["field_detection"] = {
            "status": "failed",
            "error": str(e)
        }
        refined_fields = []
    
    # ========================================================================
    # STAGE 3: Visual Overlay Generation
    # ========================================================================
    print_section("STAGE 3: Visual Overlay Generation")
    
    try:
        import fitz
        from PIL import Image, ImageDraw, ImageFont
        
        doc = fitz.open(str(pdf_path))
        page = doc[0]
        
        # Render page to image
        matrix = fitz.Matrix(2.0, 2.0)  # 2x scale for better quality
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        draw = ImageDraw.Draw(img, "RGBA")
        
        # Scale factor for coordinates
        scale_x = pix.width / page.rect.width
        scale_y = pix.height / page.rect.height
        
        # Draw detected fields
        for field in refined_fields:
            x = field.get('x', 0)
            y_from_top = field.get('y_from_top', 0)
            w = field.get('width', 0)
            h = field.get('height', 0)
            ftype = field.get('type', 'text')
            
            # Convert coordinates
            left = x * scale_x
            top = y_from_top * scale_y
            right = (x + w) * scale_x
            bottom = (y_from_top + h) * scale_y
            
            # Color by type
            if ftype == 'checkbox':
                color = (255, 165, 0, 150)  # Orange
                border_color = (255, 140, 0)
            elif ftype == 'signature':
                color = (0, 255, 0, 100)  # Green
                border_color = (0, 200, 0)
            else:
                color = (0, 150, 255, 100)  # Blue
                border_color = (0, 100, 255)
            
            # Draw filled rectangle
            draw.rectangle([left, top, right, bottom], fill=color, outline=border_color, width=3)
            
            # Draw field name label
            name = field.get('name', 'unnamed')
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            except:
                font = ImageFont.load_default()
            
            # Draw label background
            bbox = draw.textbbox((0, 0), name, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            label_x = left
            label_y = max(0, top - text_height - 4)
            
            draw.rectangle(
                [label_x, label_y, label_x + text_width + 4, label_y + text_height + 4],
                fill=(0, 0, 0, 200)
            )
            draw.text((label_x + 2, label_y + 2), name, fill=(255, 255, 255), font=font)
        
        # Save overlay
        overlay_file = output_dir / "03_detected_fields_overlay.png"
        img.save(overlay_file)
        
        print(f"✅ Visual Overlay Created")
        print(f"   - Image size: {img.width}x{img.height}")
        print(f"   - Fields rendered: {len(refined_fields)}")
        print(f"   - Saved to: {overlay_file.name}")
        print(f"   - Color legend:")
        print(f"     • Blue = Text fields")
        print(f"     • Orange = Checkboxes")
        print(f"     • Green = Signature fields")
        
        results["stages"]["visual_overlay"] = {
            "status": "success",
            "output_file": str(overlay_file)
        }
        
        doc.close()
        
    except Exception as e:
        print(f"❌ Visual Overlay Failed: {e}")
        traceback.print_exc()
        results["stages"]["visual_overlay"] = {
            "status": "failed",
            "error": str(e)
        }
    
    # ========================================================================
    # STAGE 4: Analysis Summary
    # ========================================================================
    print_section("ANALYSIS SUMMARY")
    
    # Count field names
    if refined_fields:
        unnamed_count = sum(1 for f in refined_fields if f.get('name', 'unnamed') == 'unnamed')
        named_count = len(refined_fields) - unnamed_count
        
        print(f"Field Naming Quality:")
        print(f"   - Named fields: {named_count} ({named_count/len(refined_fields)*100:.1f}%)")
        print(f"   - Unnamed fields: {unnamed_count} ({unnamed_count/len(refined_fields)*100:.1f}%)")
        
        # Check for reasonable field sizes
        tiny_fields = sum(1 for f in refined_fields if f.get('width', 0) < 10 or f.get('height', 0) < 5)
        huge_fields = sum(1 for f in refined_fields if f.get('width', 0) > 500 or f.get('height', 0) > 100)
        
        print(f"\nField Size Analysis:")
        print(f"   - Reasonable size: {len(refined_fields) - tiny_fields - huge_fields}")
        print(f"   - Too small (<10x5): {tiny_fields} (possible false positives)")
        print(f"   - Too large (>500x100): {huge_fields} (possible errors)")
        
        results["summary"] = {
            "total_fields": len(refined_fields),
            "named_fields": named_count,
            "unnamed_fields": unnamed_count,
            "naming_percentage": named_count/len(refined_fields)*100 if refined_fields else 0,
            "tiny_fields": tiny_fields,
            "huge_fields": huge_fields,
            "suspicious_fields": tiny_fields + huge_fields
        }
    else:
        print("No fields detected!")
        results["summary"] = {
            "total_fields": 0,
            "error": "No fields detected"
        }
    
    # ========================================================================
    # Save Complete Results
    # ========================================================================
    results_file = output_dir / "00_BASELINE_TEST_RESULTS.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print_section("BASELINE TEST COMPLETE")
    print(f"All outputs saved to: {output_dir}")
    print(f"\nKey files:")
    print(f"  - Summary: {results_file.name}")
    print(f"  - PDF Analysis: 01_pdf_analysis.json")
    print(f"  - Detected Fields: 02_detected_fields.json")
    print(f"  - Visual Overlay: 03_detected_fields_overlay.png")
    print(f"\nOpen the overlay image to see detected field positions!")
    
    return results, output_dir

if __name__ == "__main__":
    # Default: test on the divorce form
    divorce_form = Path(__file__).parent.parent.parent / "reference" / "forms" / "Joint Petition for Divorce Under M G L Ch 208 Sec 1A (CJ-D 101A)_10-14-2025_1406.pdf"
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = divorce_form
    
    if not Path(pdf_path).exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)
    
    results, output_dir = run_baseline_test(str(pdf_path))
    
    # Print next steps
    print("\n" + "="*80)
    print("  NEXT STEPS")
    print("="*80)
    print("\n1. Review the visual overlay to see detected field locations")
    print("2. Compare against what you expect (MA divorce form should have ~48 fields)")
    print("3. Note any false positives (fields where they shouldn't be)")
    print("4. Note any false negatives (missed fields)")
    print("\n5. This baseline data will be used to measure improvements!")

