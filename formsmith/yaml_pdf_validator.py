"""
YAML-PDF Validator

Validates that YAML interview variables and PDF form fields are properly synchronized.
"""

import fitz  # PyMuPDF
from typing import Dict, List, Any
from pathlib import Path
import logging

from .yaml_field_extractor import YAMLFieldExtractor

logger = logging.getLogger(__name__)


class YAMLPDFValidator:
    """Validate that YAML and PDF are in sync."""
    
    def __init__(self):
        """Initialize validator."""
        self.yaml_extractor = YAMLFieldExtractor()
    
    def validate(
        self,
        yaml_path: str,
        pdf_path: str
    ) -> Dict[str, Any]:
        """
        Validate YAML and PDF synchronization.
        
        Args:
            yaml_path: Path to YAML interview
            pdf_path: Path to PDF form
        
        Returns:
            {
                "is_valid": True/False,
                "yaml_field_count": 25,
                "pdf_field_count": 25,
                "mapped_count": 23,
                "unmapped_yaml": [...],  # YAML vars without PDF fields
                "unmapped_pdf": [...],   # PDF fields without YAML vars
                "type_mismatches": [...], # Type incompatibilities
                "confidence_score": 0.92,
                "recommendations": [...]
            }
        """
        
        # Extract YAML fields
        yaml_fields = self.yaml_extractor.extract_fields(yaml_path)
        yaml_var_names = {f['variable_name'] for f in yaml_fields}
        
        # Extract PDF fields
        pdf_fields = self._extract_pdf_fields(pdf_path)
        pdf_field_names = {f['name'] for f in pdf_fields}
        
        # Find matches
        mapped = yaml_var_names & pdf_field_names
        unmapped_yaml = list(yaml_var_names - pdf_field_names)
        unmapped_pdf = list(pdf_field_names - yaml_var_names)
        
        # Check type mismatches
        type_mismatches = self._check_type_mismatches(
            yaml_fields,
            pdf_fields,
            mapped
        )
        
        # Calculate confidence
        if len(yaml_var_names) > 0:
            confidence = len(mapped) / len(yaml_var_names)
        else:
            confidence = 0.0
        
        # Determine if valid
        is_valid = (
            len(unmapped_yaml) == 0 and
            len(type_mismatches) == 0 and
            confidence >= 0.90
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            unmapped_yaml,
            unmapped_pdf,
            type_mismatches
        )
        
        result = {
            "is_valid": is_valid,
            "yaml_field_count": len(yaml_fields),
            "pdf_field_count": len(pdf_fields),
            "mapped_count": len(mapped),
            "unmapped_yaml": unmapped_yaml,
            "unmapped_pdf": unmapped_pdf,
            "type_mismatches": type_mismatches,
            "confidence_score": confidence,
            "recommendations": recommendations
        }
        
        return result
    
    def _extract_pdf_fields(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract field definitions from PDF."""
        fields = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc):
                for widget in page.widgets():
                    if widget.field_type in [fitz.PDF_WIDGET_TYPE_TEXT,
                                            fitz.PDF_WIDGET_TYPE_CHECKBOX,
                                            fitz.PDF_WIDGET_TYPE_SIGNATURE]:
                        
                        field_type = {
                            fitz.PDF_WIDGET_TYPE_TEXT: "text",
                            fitz.PDF_WIDGET_TYPE_CHECKBOX: "checkbox",
                            fitz.PDF_WIDGET_TYPE_SIGNATURE: "signature"
                        }.get(widget.field_type, "text")
                        
                        fields.append({
                            "name": widget.field_name,
                            "type": field_type,
                            "page": page_num,
                            "bbox": list(widget.rect)
                        })
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Failed to extract PDF fields from {pdf_path}: {e}")
        
        return fields
    
    def _check_type_mismatches(
        self,
        yaml_fields: List[Dict],
        pdf_fields: List[Dict],
        mapped_names: set
    ) -> List[Dict[str, Any]]:
        """Check for type incompatibilities between matched fields."""
        mismatches = []
        
        # Create lookup dicts
        yaml_lookup = {f['variable_name']: f for f in yaml_fields}
        pdf_lookup = {f['name']: f for f in pdf_fields}
        
        for name in mapped_names:
            yaml_field = yaml_lookup.get(name)
            pdf_field = pdf_lookup.get(name)
            
            if not yaml_field or not pdf_field:
                continue
            
            yaml_type = yaml_field.get('datatype', 'text')
            pdf_type = pdf_field.get('type', 'text')
            
            # Check compatibility
            compatible = self._are_types_compatible(yaml_type, pdf_type)
            
            if not compatible:
                mismatches.append({
                    "field_name": name,
                    "yaml_type": yaml_type,
                    "pdf_type": pdf_type,
                    "severity": "error"
                })
        
        return mismatches
    
    def _are_types_compatible(self, yaml_type: str, pdf_type: str) -> bool:
        """Check if YAML and PDF types are compatible."""
        
        compatibility_map = {
            "text": ["text", "signature"],
            "yesno": ["checkbox"],
            "signature": ["text", "signature"],
            "date": ["text"],
            "area": ["text"],
            "email": ["text"],
            "number": ["text"]
        }
        
        compatible_pdf_types = compatibility_map.get(yaml_type, ["text"])
        return pdf_type in compatible_pdf_types
    
    def _generate_recommendations(
        self,
        unmapped_yaml: List[str],
        unmapped_pdf: List[str],
        type_mismatches: List[Dict]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if unmapped_yaml:
            recommendations.append(
                f"⚠️  {len(unmapped_yaml)} YAML variables have no matching PDF fields. "
                f"Add these fields to the PDF or remove from YAML: {', '.join(unmapped_yaml[:5])}"
                + ("..." if len(unmapped_yaml) > 5 else "")
            )
        
        if unmapped_pdf:
            recommendations.append(
                f"ℹ️  {len(unmapped_pdf)} PDF fields have no matching YAML variables. "
                f"Consider adding to YAML or renaming PDF fields: {', '.join(unmapped_pdf[:5])}"
                + ("..." if len(unmapped_pdf) > 5 else "")
            )
        
        if type_mismatches:
            recommendations.append(
                f"🚫 {len(type_mismatches)} type mismatches detected. "
                f"These will cause form filling failures. Fix field types."
            )
        
        if not recommendations:
            recommendations.append("✓ YAML and PDF are properly synchronized!")
        
        return recommendations


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 3:
        print("Usage: python yaml_pdf_validator.py <yaml_file> <pdf_file>")
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    pdf_file = sys.argv[2]
    
    validator = YAMLPDFValidator()
    result = validator.validate(yaml_file, pdf_file)
    
    print("\n" + "="*60)
    print("YAML ↔ PDF Validation Report")
    print("="*60 + "\n")
    
    print(f"YAML Fields: {result['yaml_field_count']}")
    print(f"PDF Fields: {result['pdf_field_count']}")
    print(f"Mapped: {result['mapped_count']}")
    print(f"Confidence: {result['confidence_score']:.1%}")
    print(f"Valid: {'✓ YES' if result['is_valid'] else '✗ NO'}\n")
    
    print("Recommendations:")
    for rec in result['recommendations']:
        print(f"  {rec}")
    
    print("\n" + "="*60)
    
    # Save detailed report
    with open("validation_report.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("\nDetailed report saved to: validation_report.json")

