#!/usr/bin/env python3
"""
Test the multi-agent field detection system
"""

import os
from .create_fillable_pdf import create_fillable_petition_pdf

def main():
    input_pdf = "jud-tc-Petition-to-Deem-Satisfied.pdf"
    output_pdf = "jud-tc-Petition-to-Deem-Satisfied_MULTI_AGENT_TEST.pdf"
    
    if not os.path.exists(input_pdf):
        print(f"❌ Input PDF not found: {input_pdf}")
        return
    
    print("="*80)
    print("MULTI-AGENT FIELD DETECTION SYSTEM TEST")
    print("="*80)
    print()
    
    try:
        result = create_fillable_petition_pdf(input_pdf, output_pdf)
        
        print()
        print("="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"✅ Output PDF: {result}")
        print(f"📝 Check logs: field_detection_log.json, pdf_creation_log.json")
        
    except Exception as e:
        print()
        print("="*80)
        print("TEST FAILED")
        print("="*80)
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

