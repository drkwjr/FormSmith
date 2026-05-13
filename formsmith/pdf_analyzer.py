#!/usr/bin/env python3
"""
PDF Field Analyzer
Extracts and analyzes form fields from PDF files
"""

import pikepdf
import json
from collections import defaultdict

def analyze_pdf(pdf_path):
    """Extract comprehensive field information from PDF"""
    
    results = {
        "pdf_name": pdf_path,
        "fields": [],
        "field_count": 0,
        "field_types": defaultdict(int),
        "field_names": [],
        "pages_with_fields": set(),
        "analysis": {}
    }
    
    try:
        pdf = pikepdf.open(pdf_path)
        
        # Check if PDF has form fields
        if not (hasattr(pdf.Root, 'AcroForm') and hasattr(pdf.Root.AcroForm, 'Fields')):
            results["error"] = "PDF has no form fields (AcroForm)"
            return results
        
        fields = pdf.Root.AcroForm.Fields
        results["field_count"] = len(fields)
        
        # Extract field information
        for i, field in enumerate(fields):
            field_info = extract_field_info(field, i)
            if field_info:
                results["fields"].append(field_info)
                results["field_types"][field_info["type"]] += 1
                results["field_names"].append(field_info["name"])
                if field_info.get("page"):
                    results["pages_with_fields"].add(field_info["page"])
        
        # Analysis
        results["pages_with_fields"] = sorted(list(results["pages_with_fields"]))
        results["field_types"] = dict(results["field_types"])
        
        # Analyze field naming patterns
        results["analysis"] = analyze_naming_patterns(results["field_names"])
        
        pdf.close()
        
    except Exception as e:
        results["error"] = str(e)
    
    return results

def extract_field_info(field, index):
    """Extract information about a single field"""
    try:
        info = {
            "index": index,
            "name": str(field.T) if hasattr(field, 'T') else f"field_{index}",
            "type": get_field_type(field),
            "required": False,
            "read_only": False,
            "page": None,
            "rect": None,
            "default_value": None,
            "max_length": None,
            "options": None
        }
        
        # Field type details
        if hasattr(field, 'FT'):
            info["field_type_code"] = str(field.FT)
        
        # Required?
        if hasattr(field, 'Ff'):
            flags = int(field.Ff)
            info["required"] = bool(flags & 2)  # Bit 1 = required
            info["read_only"] = bool(flags & 1)  # Bit 0 = read-only
        
        # Default value
        if hasattr(field, 'V'):
            info["default_value"] = str(field.V)
        
        # Max length (for text fields)
        if hasattr(field, 'MaxLen'):
            info["max_length"] = int(field.MaxLen)
        
        # Options (for choice fields)
        if hasattr(field, 'Opt'):
            info["options"] = [str(opt) for opt in field.Opt]
        
        # Position (rect)
        if hasattr(field, 'Rect'):
            info["rect"] = [float(x) for x in field.Rect]
        
        # Page number (if available through annotations)
        if hasattr(field, 'P'):
            # Try to get page number
            try:
                page_obj = field.P
                # This is tricky - would need to iterate through pages
                info["page"] = "unknown"
            except:
                pass
        
        return info
        
    except Exception as e:
        return {
            "index": index,
            "name": f"field_{index}",
            "type": "unknown",
            "error": str(e)
        }

def get_field_type(field):
    """Determine field type"""
    if not hasattr(field, 'FT'):
        return "unknown"
    
    field_type = str(field.FT)
    
    type_map = {
        '/Tx': 'text',
        '/Btn': 'button_or_checkbox',
        '/Ch': 'choice',
        '/Sig': 'signature'
    }
    
    base_type = type_map.get(field_type, 'unknown')
    
    # Refine button type
    if base_type == 'button_or_checkbox':
        if hasattr(field, 'Ff'):
            flags = int(field.Ff)
            if flags & 32768:  # Bit 15 = radio button
                return 'radio'
            elif flags & 65536:  # Bit 16 = pushbutton
                return 'button'
            else:
                return 'checkbox'
        return 'checkbox'  # Default
    
    # Refine choice type
    if base_type == 'choice':
        if hasattr(field, 'Ff'):
            flags = int(field.Ff)
            if flags & 131072:  # Bit 17 = combo (dropdown)
                return 'dropdown'
            else:
                return 'listbox'
        return 'dropdown'  # Default
    
    return base_type

def analyze_naming_patterns(field_names):
    """Analyze field naming patterns"""
    analysis = {
        "total_fields": len(field_names),
        "generic_names": 0,
        "descriptive_names": 0,
        "naming_style": None,
        "common_prefixes": defaultdict(int),
        "contains_underscore": 0,
        "all_caps": 0,
        "sample_names": field_names[:10] if len(field_names) > 10 else field_names
    }
    
    # Analyze each name
    generic_patterns = ['text', 'check', 'field', 'box', 'button', 'radio']
    
    for name in field_names:
        name_lower = name.lower()
        
        # Generic vs descriptive
        is_generic = any(pattern in name_lower for pattern in generic_patterns)
        if is_generic and any(char.isdigit() for char in name):
            analysis["generic_names"] += 1
        else:
            analysis["descriptive_names"] += 1
        
        # Underscore usage
        if '_' in name:
            analysis["contains_underscore"] += 1
        
        # All caps
        if name.isupper():
            analysis["all_caps"] += 1
        
        # Common prefixes
        parts = name.split('_')
        if len(parts) > 1:
            analysis["common_prefixes"][parts[0]] += 1
    
    # Determine naming style
    if analysis["generic_names"] > analysis["descriptive_names"]:
        analysis["naming_style"] = "generic (needs mapping)"
    else:
        analysis["naming_style"] = "descriptive (may be usable)"
    
    # Convert defaultdict to dict
    analysis["common_prefixes"] = dict(sorted(
        analysis["common_prefixes"].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10])  # Top 10 prefixes
    
    return analysis

if __name__ == "__main__":
    import sys
    
    pdf_path = "jud-tc-Petition-to-Deem-Satisfied.pdf"
    
    print(f"Analyzing PDF: {pdf_path}")
    print("=" * 60)
    
    results = analyze_pdf(pdf_path)
    
    if "error" in results:
        print(f"ERROR: {results['error']}")
        sys.exit(1)
    
    # Print summary
    print(f"\n📄 PDF: {results['pdf_name']}")
    print(f"📊 Total Fields: {results['field_count']}")
    print(f"📄 Pages with Fields: {results['pages_with_fields']}")
    
    print(f"\n🔤 Field Types:")
    for field_type, count in sorted(results['field_types'].items()):
        print(f"  - {field_type}: {count}")
    
    print(f"\n🏷️  Naming Analysis:")
    analysis = results['analysis']
    print(f"  - Style: {analysis['naming_style']}")
    print(f"  - Generic names: {analysis['generic_names']}")
    print(f"  - Descriptive names: {analysis['descriptive_names']}")
    print(f"  - With underscores: {analysis['contains_underscore']}")
    
    if analysis['common_prefixes']:
        print(f"\n  Common prefixes:")
        for prefix, count in list(analysis['common_prefixes'].items())[:5]:
            print(f"    - '{prefix}': {count} fields")
    
    print(f"\n📝 Sample Field Names:")
    for name in analysis['sample_names'][:15]:
        print(f"  - {name}")
    
    # Save full results
    output_file = "pdf_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Full analysis saved to: {output_file}")
    print(f"\nTotal fields extracted: {len(results['fields'])}")

