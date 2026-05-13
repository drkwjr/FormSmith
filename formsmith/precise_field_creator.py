#!/usr/bin/env python3
"""
Precise Field Creator - Uses intelligent analysis to create perfectly positioned fields
"""

import pikepdf
from pikepdf import Array, Dictionary, Name, String, Stream
import json

def create_precise_fillable_pdf(input_pdf, output_pdf):
    """Create fillable PDF using intelligent field positioning"""
    
    # Load the intelligent analysis
    with open('intelligent_field_analysis.json', 'r') as f:
        analysis = json.load(f)
    
    pdf = pikepdf.open(input_pdf)
    page = pdf.pages[0]
    mediabox = page.MediaBox
    page_height = float(mediabox[3])
    
    # Define precise field positions based on actual form layout
    # These are manually refined based on the visual inspection
    precise_fields = [
        # Court information (top section)
        ('docket_number', 100, 65, 100, 15, 'text'),
        ('trial_court_department', 280, 65, 95, 15, 'text'),
        ('trial_court_division', 480, 65, 110, 15, 'text'),
        
        # Party names
        ('plaintiff_name', 22, 108, 280, 15, 'text'),
        ('defendant_name', 310, 108, 280, 15, 'text'),
        
        # Request section - payment date
        ('payment_date', 227, 198, 150, 15, 'text'),
        
        # Petitioner information - Name, Signature, Date row
        ('petitioner_name', 80, 393, 180, 15, 'text'),
        ('petitioner_signature', 335, 393, 150, 15, 'signature'),
        ('petitioner_signature_date', 520, 393, 70, 15, 'text'),
        
        # Petitioner address
        ('petitioner_address', 95, 417, 490, 15, 'text'),
        
        # Petitioner phone and email row
        ('petitioner_phone', 175, 442, 110, 15, 'text'),
        ('petitioner_email', 370, 442, 215, 15, 'text'),
        
        # Attorney information (optional section)
        ('attorney_name', 130, 551, 180, 15, 'text'),
        ('attorney_signature', 380, 551, 120, 15, 'signature'),
        ('attorney_signature_date', 530, 551, 60, 15, 'text'),
        ('attorney_email', 140, 577, 220, 15, 'text'),
        ('attorney_bbo_number', 510, 577, 80, 15, 'text'),
        
        # Certificate of Service - "I gave notice to" checkboxes
        ('service_to_plaintiff', 240, 653, 15, 15, 'checkbox'),
        ('service_to_plaintiff_attorney', 360, 653, 15, 15, 'checkbox'),
        
        # Service date
        ('service_date', 222, 671, 120, 15, 'text'),
        
        # Service method checkboxes (vertical stack)
        ('service_method_mail', 44, 688, 15, 15, 'checkbox'),
        ('service_method_in_person', 44, 704, 15, 15, 'checkbox'),
        ('service_method_email', 44, 720, 15, 15, 'checkbox'),
        
        # Service addresses
        ('service_mail_address', 320, 688, 265, 15, 'text'),
        ('service_email_address', 220, 720, 365, 15, 'text'),
    ]
    
    # Create AcroForm with proper settings
    if not hasattr(pdf.Root, 'AcroForm'):
        pdf.Root.AcroForm = Dictionary({
            '/Fields': Array([]),
            '/NeedAppearances': True,
            '/DR': Dictionary({
                '/Font': Dictionary({
                    '/Helv': Dictionary({
                        '/Type': Name('/Font'),
                        '/Subtype': Name('/Type1'),
                        '/BaseFont': Name('/Helvetica')
                    })
                })
            }),
            '/DA': String('/Helv 0 Tf 0 g'),
        })
    
    fields_array = pdf.Root.AcroForm.Fields
    
    # Initialize page annotations
    if not hasattr(page, 'Annots'):
        page.Annots = Array([])
    
    # Add each field with precise positioning
    for field_name, x, y_from_top, width, height, field_type in precise_fields:
        # Convert y coordinate (PDF uses bottom-left origin)
        y_pdf = page_height - y_from_top - height
        
        # Create field based on type
        if field_type == 'text':
            field = create_precise_text_field(pdf, page, field_name, x, y_pdf, width, height)
        elif field_type == 'signature':
            field = create_precise_signature_field(pdf, page, field_name, x, y_pdf, width, height)
        elif field_type == 'checkbox':
            field = create_precise_checkbox_field(pdf, page, field_name, x, y_pdf, width, height)
        
        fields_array.append(field)
    
    # Save the fillable PDF
    pdf.save(output_pdf)
    pdf.close()
    
    print(f"✅ Created precise fillable PDF: {output_pdf}")
    print(f"   Total fields added: {len(precise_fields)}")
    
    return output_pdf

def create_precise_text_field(pdf, page, name, x, y, width, height):
    """Create a precisely positioned text field"""
    
    # Create the field object
    field = Dictionary({
        '/FT': Name('/Tx'),
        '/T': String(name),
        '/V': String(''),
        '/DA': String('/Helv 10 Tf 0 g'),
        '/Ff': 0,
        '/DV': String(''),
    })
    field_obj = pdf.make_indirect(field)
    
    # Create widget annotation with precise positioning
    widget = Dictionary({
        '/Type': Name('/Annot'),
        '/Subtype': Name('/Widget'),
        '/Rect': Array([x, y, x + width, y + height]),
        '/Parent': field_obj,
        '/P': page.obj,
        '/F': 4,  # Print flag
        '/BS': Dictionary({
            '/W': 1,
            '/S': Name('/S')  # Solid border
        }),
        '/MK': Dictionary({
            '/BC': Array([0, 0, 0]),  # Border color (black)
            '/BG': Array([1, 1, 1])   # Background (white)
        })
    })
    widget_obj = pdf.make_indirect(widget)
    
    # Link field to widget
    field['/Kids'] = Array([widget_obj])
    
    # Add widget to page annotations
    page.Annots.append(widget_obj)
    
    # Create appearance stream
    appearance = create_text_appearance(pdf, width, height, '')
    widget['/AP'] = Dictionary({
        '/N': appearance
    })
    
    return field_obj

def create_precise_signature_field(pdf, page, name, x, y, width, height):
    """Create a precisely positioned signature field"""
    return create_precise_text_field(pdf, page, name, x, y, width, height)

def create_precise_checkbox_field(pdf, page, name, x, y, width, height):
    """Create a precisely positioned checkbox field"""
    
    # Create the field object
    field = Dictionary({
        '/FT': Name('/Btn'),
        '/T': String(name),
        '/V': Name('/Off'),
        '/DV': Name('/Off'),
        '/Ff': 0,  # Checkbox (not radio, not pushbutton)
    })
    field_obj = pdf.make_indirect(field)
    
    # Create widget annotation with precise positioning
    widget = Dictionary({
        '/Type': Name('/Annot'),
        '/Subtype': Name('/Widget'),
        '/Rect': Array([x, y, x + width, y + height]),
        '/Parent': field_obj,
        '/P': page.obj,
        '/F': 4,  # Print flag
        '/BS': Dictionary({
            '/W': 1,
            '/S': Name('/S')
        }),
        '/MK': Dictionary({
            '/BC': Array([0, 0, 0]),
            '/BG': Array([1, 1, 1]),
            '/CA': String('4')  # Checkmark character
        })
    })
    widget_obj = pdf.make_indirect(widget)
    
    # Link field to widget
    field['/Kids'] = Array([widget_obj])
    
    # Add widget to page annotations
    page.Annots.append(widget_obj)
    
    # Create appearance streams for checked/unchecked states
    ap_off = create_checkbox_appearance(pdf, width, height, False)
    ap_yes = create_checkbox_appearance(pdf, width, height, True)
    
    widget['/AP'] = Dictionary({
        '/N': Dictionary({
            '/Off': ap_off,
            '/Yes': ap_yes
        })
    })
    widget['/AS'] = Name('/Off')  # Appearance state
    
    return field_obj

def create_text_appearance(pdf, width, height, text=''):
    """Create appearance stream for text field"""
    
    commands = f"""q
1 1 1 rg
0 0 {width} {height} re
f
0 0 0 RG
0.5 w
0 0 {width} {height} re
S
BT
/Helv 10 Tf
0 0 0 rg
2 {height/2 - 3} Td
({text}) Tj
ET
Q""".encode()
    
    stream = pdf.make_stream(commands)
    stream['/Type'] = Name('/XObject')
    stream['/Subtype'] = Name('/Form')
    stream['/BBox'] = Array([0, 0, width, height])
    stream['/Resources'] = Dictionary({
        '/Font': Dictionary({
            '/Helv': Dictionary({
                '/Type': Name('/Font'),
                '/Subtype': Name('/Type1'),
                '/BaseFont': Name('/Helvetica')
            })
        })
    })
    
    return pdf.make_indirect(stream)

def create_checkbox_appearance(pdf, width, height, checked=False):
    """Create appearance stream for checkbox"""
    
    if checked:
        # Draw checkmark
        commands = f"""q
1 1 1 rg
0 0 {width} {height} re
f
0 0 0 RG
1 w
0 0 {width} {height} re
S
0 0 0 rg
2 w
{width*0.2} {height*0.5} m
{width*0.4} {height*0.2} l
{width*0.8} {height*0.8} l
S
Q""".encode()
    else:
        # Just draw border
        commands = f"""q
1 1 1 rg
0 0 {width} {height} re
f
0 0 0 RG
1 w
0 0 {width} {height} re
S
Q""".encode()
    
    stream = pdf.make_stream(commands)
    stream['/Type'] = Name('/XObject')
    stream['/Subtype'] = Name('/Form')
    stream['/BBox'] = Array([0, 0, width, height])
    
    return pdf.make_indirect(stream)

if __name__ == "__main__":
    input_pdf = "jud-tc-Petition-to-Deem-Satisfied.pdf"
    output_pdf = "jud-tc-Petition-to-Deem-Satisfied_PRECISE.pdf"
    
    print("Creating PRECISE fillable PDF...")
    print("=" * 70)
    
    create_precise_fillable_pdf(input_pdf, output_pdf)
    
    print("\n" + "=" * 70)
    print("✅ PRECISE fillable PDF created!")
    print(f"   Input:  {input_pdf}")
    print(f"   Output: {output_pdf}")
    print("\nTest this version for accuracy!")
