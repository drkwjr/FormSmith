#!/usr/bin/env python3
"""
Create fillable PDF using Visual Field System with gpt-5-mini precision placement
"""

import pikepdf
from pikepdf import Array, Dictionary, Name, String
from .visual_field_system import VisualFieldSystem
import os

def create_text_field(pdf, page, field_name, bbox, page_height):
    """Create a visible text field WITHOUT borders"""
    x, y, x2, y2 = bbox
    width = x2 - x
    height = y2 - y
    
    # Flip Y coordinate
    y_flipped = page_height - y2
    
    # Create field object
    field = Dictionary(
        FT=Name("/Tx"),
        T=String(field_name),
        V=String(""),
        Rect=Array([x, y_flipped, x2, y_flipped + height]),
        F=4,  # Print flag
        Ff=0
    )
    
    # Create widget annotation (NO BORDER)
    widget = Dictionary(
        Type=Name("/Annot"),
        Subtype=Name("/Widget"),
        Rect=Array([x, y_flipped, x2, y_flipped + height]),
        F=4
    )
    widget.P = page.obj
    
    # Create appearance stream for text field
    appearance = f"""
    /Tx BMC
    BT
    /Helv 10 Tf
    0 0 0 rg
    2 2 Td
    ET
    EMC
    """.strip()
    
    appearance_stream = pikepdf.Stream(pdf, appearance.encode())
    appearance_stream.Type = Name("/XObject")
    appearance_stream.Subtype = Name("/Form")
    appearance_stream.BBox = Array([0, 0, width, height])
    appearance_stream.Resources = Dictionary(
        ProcSet=Array([Name("/PDF"), Name("/Text")]),
        Font=Dictionary(Helv=Dictionary(
            Type=Name("/Font"),
            Subtype=Name("/Type1"),
            BaseFont=Name("/Helvetica")
        ))
    )
    
    widget.AP = Dictionary(N=appearance_stream)
    widget.Parent = field
    field.Kids = Array([widget])
    
    return field, widget

def create_checkbox_field(pdf, page, field_name, bbox, page_height):
    """Create a visible checkbox field WITHOUT borders"""
    x, y, x2, y2 = bbox
    width = x2 - x
    height = y2 - y
    
    # Flip Y coordinate
    y_flipped = page_height - y2
    
    # Create field object
    field = Dictionary(
        FT=Name("/Btn"),
        T=String(field_name),
        V=Name("/Off"),
        Rect=Array([x, y_flipped, x2, y_flipped + height]),
        F=4,
        Ff=0
    )
    
    # Create widget annotation
    widget = Dictionary(
        Type=Name("/Annot"),
        Subtype=Name("/Widget"),
        Rect=Array([x, y_flipped, x2, y_flipped + height]),
        F=4,
        H=Name("/P"),
        MK=Dictionary(CA=String("✓"))
    )
    widget.P = page.obj
    
    # Create appearance streams for checked/unchecked
    off_appearance = f"""
    0.8 0.8 0.8 rg
    0 0 {width} {height} re f
    """.strip()
    
    yes_appearance = f"""
    0.8 0.8 0.8 rg
    0 0 {width} {height} re f
    0 0 0 rg
    BT
    /ZapfDingbats {height*0.8} Tf
    {width*0.1} {height*0.1} Td
    (4) Tj
    ET
    """.strip()
    
    off_stream = pikepdf.Stream(pdf, off_appearance.encode())
    off_stream.Type = Name("/XObject")
    off_stream.Subtype = Name("/Form")
    off_stream.BBox = Array([0, 0, width, height])
    
    yes_stream = pikepdf.Stream(pdf, yes_appearance.encode())
    yes_stream.Type = Name("/XObject")
    yes_stream.Subtype = Name("/Form")
    yes_stream.BBox = Array([0, 0, width, height])
    yes_stream.Resources = Dictionary(
        ProcSet=Array([Name("/PDF"), Name("/Text")]),
        Font=Dictionary(ZapfDingbats=Dictionary(
            Type=Name("/Font"),
            Subtype=Name("/Type1"),
            BaseFont=Name("/ZapfDingbats")
        ))
    )
    
    widget.AP = Dictionary(N=Dictionary(Off=off_stream, Yes=yes_stream))
    widget.AS = Name("/Off")
    widget.Parent = field
    field.Kids = Array([widget])
    
    return field, widget

def create_signature_field(pdf, page, field_name, bbox, page_height):
    """Create a signature field (text field with larger area)"""
    return create_text_field(pdf, page, field_name, bbox, page_height)

def create_fillable_petition_pdf(input_pdf, output_pdf):
    """Create fillable PDF using visual precision system"""
    
    print("=" * 60)
    print("PDF FIELD CREATION WITH VISUAL PRECISION SYSTEM")
    print("=" * 60)
    
    # Initialize visual field system
    system = VisualFieldSystem(input_pdf)
    
    # Step 1: Detect all fields
    detected_fields = system.detect_fields()
    system.token_log["fields_detected"] = len(detected_fields)
    
    print(f"\n📋 Detected {len(detected_fields)} fields to process")
    
    # Open PDF for editing
    pdf = pikepdf.open(input_pdf)
    page = pdf.pages[0]
    mediabox = page.MediaBox
    page_height = float(mediabox[3])
    
    # Initialize AcroForm
    if "/AcroForm" not in pdf.Root:
        pdf.Root.AcroForm = Dictionary(Fields=Array([]))
    
    if "/Annots" not in page:
        page.Annots = Array([])
    
    # Step 2: For each field, iteratively refine placement
    approved_fields = []
    rejected_fields = []
    
    for field_info in detected_fields:
        field_name = field_info["field_name"]
        field_type = field_info["field_type"]
        initial_bbox = tuple(field_info["bbox"])
        
        system.token_log["fields_processed"] += 1
        
        # Iterative refinement loop
        current_bbox = initial_bbox
        iteration = 1
        max_iterations = 10
        
        while True:
            # Create temporary PDF with this field to show AI
            temp_field, temp_widget = None, None
            
            if field_type in ["text", "date", "email", "phone", "address"]:
                temp_field, temp_widget = create_text_field(pdf, page, field_name, current_bbox, page_height)
            elif field_type == "checkbox":
                temp_field, temp_widget = create_checkbox_field(pdf, page, field_name, current_bbox, page_height)
            elif field_type == "signature":
                temp_field, temp_widget = create_signature_field(pdf, page, field_name, current_bbox, page_height)
            else:
                print(f"  ⚠ Unknown field type '{field_type}' for {field_name}, skipping")
                rejected_fields.append(field_name)
                break
            
            # Temporarily add field to PDF so AI can see it
            pdf.Root.AcroForm.Fields.append(temp_field)
            page.Annots.append(temp_widget)
            
            # Save temporary PDF for AI to analyze
            temp_pdf_path = input_pdf.replace(".pdf", "_TEMP_FOR_AI.pdf")
            pdf.save(temp_pdf_path)
            
            # Update system's PDF path to point to temp
            original_path = system.pdf_path
            system.pdf_path = temp_pdf_path
            
            # Ask AI to validate/refine
            new_bbox, status, quality_scores = system.refine_field_placement(
                field_info, current_bbox, iteration, max_iterations
            )
            
            # Restore original path
            system.pdf_path = original_path
            
            # Remove temporary field
            pdf.Root.AcroForm.Fields = Array([f for f in pdf.Root.AcroForm.Fields if f != temp_field])
            page.Annots = Array([a for a in page.Annots if a != temp_widget])
            
            # Clean up temp file
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
            
            # Handle status
            if status == "APPROVED":
                # Create final field with approved bbox
                final_field, final_widget = None, None
                if field_type in ["text", "date", "email", "phone", "address"]:
                    final_field, final_widget = create_text_field(pdf, page, field_name, current_bbox, page_height)
                elif field_type == "checkbox":
                    final_field, final_widget = create_checkbox_field(pdf, page, field_name, current_bbox, page_height)
                elif field_type == "signature":
                    final_field, final_widget = create_signature_field(pdf, page, field_name, current_bbox, page_height)
                
                pdf.Root.AcroForm.Fields.append(final_field)
                page.Annots.append(final_widget)
                approved_fields.append(field_name)
                break
                
            elif status == "REJECT":
                rejected_fields.append(field_name)
                break
                
            elif status == "ADJUST":
                current_bbox = new_bbox
                iteration += 1
                
            elif status == "MAX_ITERATIONS":
                # Accept current state after max iterations
                final_field, final_widget = None, None
                if field_type in ["text", "date", "email", "phone", "address"]:
                    final_field, final_widget = create_text_field(pdf, page, field_name, current_bbox, page_height)
                elif field_type == "checkbox":
                    final_field, final_widget = create_checkbox_field(pdf, page, field_name, current_bbox, page_height)
                elif field_type == "signature":
                    final_field, final_widget = create_signature_field(pdf, page, field_name, current_bbox, page_height)
                
                pdf.Root.AcroForm.Fields.append(final_field)
                page.Annots.append(final_widget)
                approved_fields.append(field_name)
                break
                
            else:
                rejected_fields.append(field_name)
                break
    
    # Save final PDF
    pdf.save(output_pdf)
    pdf.close()
    
    # Update log
    system.token_log["approved_fields"] = approved_fields
    system.token_log["rejected_fields"] = rejected_fields
    system.token_log["final_field_count"] = len(approved_fields)
    
    # Save log
    system.save_log("visual_field_precision_log.json")
    
    print("\n" + "=" * 60)
    print(f"✓ PDF created: {output_pdf}")
    print(f"✓ Approved fields: {len(approved_fields)}")
    print(f"✗ Rejected fields: {len(rejected_fields)}")
    print("=" * 60)

if __name__ == "__main__":
    create_fillable_petition_pdf(
        "jud-tc-Petition-to-Deem-Satisfied.pdf",
        "jud-tc-Petition-to-Deem-Satisfied_FILLABLE.pdf"
    )
