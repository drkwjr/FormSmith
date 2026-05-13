#!/usr/bin/env python3
"""
Test if the PDF is actually fillable by filling it with data
"""

import pikepdf
from pikepdf import String

def fill_and_test():
    """Fill the PDF and see if it works"""
    
    pdf = pikepdf.open('jud-tc-Petition-to-Deem-Satisfied_FILLABLE.pdf')
    
    # Test data
    test_data = {
        'docket_number': 'TEST-2024-001',
        'trial_court_department': 'Housing Court',
        'trial_court_division': 'Eastern Division',
        'plaintiff_name': 'Jane Landlord',
        'defendant_name': 'John Tenant',
        'payment_date': '12/15/2024',
        'petitioner_name': 'John Tenant',
        'petitioner_signature_date': '12/20/2024',
        'petitioner_address': '123 Main St, Boston, MA 02101',
        'petitioner_phone': '617-555-1234',
        'petitioner_email': 'john@example.com',
    }
    
    print("Attempting to fill fields...")
    print("=" * 60)
    
    # Try to fill fields
    filled_count = 0
    for field in pdf.Root.AcroForm.Fields:
        field_name = str(field.T)
        if field_name in test_data:
            try:
                # Set value on FIELD (not widget)
                field.V = String(test_data[field_name])
                print(f"✅ Filled: {field_name} = {test_data[field_name]}")
                filled_count += 1
            except Exception as e:
                print(f"❌ Failed: {field_name} - {e}")
    
    print(f"\nFilled {filled_count} fields")
    
    # Save filled PDF
    output = 'FILLED_TEST.pdf'
    pdf.save(output)
    pdf.close()
    
    print(f"\n✅ Saved filled PDF to: {output}")
    print("\nNow verify:")
    print("-" * 60)
    
    # Re-open and verify
    verify_pdf = pikepdf.open(output)
    for field in verify_pdf.Root.AcroForm.Fields:
        field_name = str(field.T)
        if field_name in test_data:
            if '/V' in field:
                actual = str(field.V)
                expected = test_data[field_name]
                match = "✅" if actual == expected else "❌"
                print(f"  {match} {field_name}: '{actual}'")
    
    verify_pdf.close()
    
    print("\n" + "=" * 60)
    print("Open FILLED_TEST.pdf in Chrome to see if fields show with values!")

if __name__ == "__main__":
    fill_and_test()

