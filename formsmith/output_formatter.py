#!/usr/bin/env python3
"""
Field Collection Export

Exports detected fields in various formats for different use cases.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from .schemas import FieldDefinition, FormDefinition


class FieldCollectionExporter:
    """
    Exports field collections in multiple formats.
    """
    
    def export_as_json(self, form_def: FormDefinition, output_path: str):
        """
        Export as standard JSON format (current format, for compatibility).
        
        Args:
            form_def: Form definition to export
            output_path: Path to save JSON file
        """
        output = {
            "pdf_name": form_def.pdf_name,
            "total_fields": len(form_def.fields),
            "fields": [
                {
                    "name": f.pdf_name,
                    "type": f.pdf_type,
                    "page": f.page,
                    "bbox": f.bbox
                }
                for f in form_def.fields
            ]
        }
        
        with open(output_path, 'w') as file:
            json.dump(output, file, indent=2)
        
        print(f"💾 Exported {len(form_def.fields)} fields to: {output_path}")
    
    def export_as_da_ready_json(self, form_def: FormDefinition, output_path: str):
        """
        Export as interview-ready JSON with full metadata.
        
        This format includes:
        - Form metadata
        - Field groups
        - Complete field definitions with DA properties
        - Detection metadata for quality assurance
        
        Args:
            form_def: Form definition to export
            output_path: Path to save JSON file
        """
        # Build field groups
        field_groups = {}
        for field in form_def.fields:
            if field.interview_field_group:
                if field.interview_field_group not in field_groups:
                    field_groups[field.interview_field_group] = []
                field_groups[field.interview_field_group].append(field.interview_variable)
        
        # Calculate statistics
        field_types = {}
        for field in form_def.fields:
            field_types[field.pdf_type] = field_types.get(field.pdf_type, 0) + 1
        
        da_datatypes = {}
        for field in form_def.fields:
            da_datatypes[field.interview_datatype] = da_datatypes.get(field.interview_datatype, 0) + 1
        
        # Build output
        output = {
            "form_metadata": {
                "pdf_name": form_def.pdf_name,
                "form_type": form_def.form_type or "unknown",
                "total_fields": len(form_def.fields),
                "interview_module": form_def.interview_module or "interview.custom",
                "field_types": field_types,
                "da_datatypes": da_datatypes
            },
            "field_groups": field_groups,
            "fields": [field.to_dict() for field in form_def.fields],
            "quality_metrics": {
                "avg_confidence": sum(f.confidence for f in form_def.fields) / len(form_def.fields) if form_def.fields else 0.0,
                "high_confidence": len([f for f in form_def.fields if f.confidence >= 0.85]),
                "review_needed": len([f for f in form_def.fields if f.confidence < 0.75])
            }
        }
        
        with open(output_path, 'w') as file:
            json.dump(output, file, indent=2)
        
        print(f"💾 Exported interview-ready JSON to: {output_path}")
        print(f"   • {len(form_def.fields)} fields")
        print(f"   • {len(field_groups)} field groups")
        print(f"   • Avg confidence: {output['quality_metrics']['avg_confidence']:.2f}")
    
    def export_for_manual_review(self, form_def: FormDefinition, output_path: str):
        """
        Export fields needing manual review (low confidence).
        
        Args:
            form_def: Form definition to export
            output_path: Path to save review file
        """
        low_confidence_fields = form_def.get_low_confidence_fields(threshold=0.75)
        
        review_data = {
            "pdf_name": form_def.pdf_name,
            "review_date": None,  # To be filled during review
            "total_fields": len(form_def.fields),
            "fields_needing_review": len(low_confidence_fields),
            "fields": [
                {
                    "field_index": i,
                    "pdf_name": f.pdf_name,
                    "interview_variable": f.interview_variable,
                    "interview_label": f.interview_label,
                    "type": f.pdf_type,
                    "bbox": f.bbox,
                    "page": f.page,
                    "confidence": f.confidence,
                    "detection_method": f.detection_method,
                    "source_label": f.source_label,
                    "review_status": "pending",
                    "reviewer_notes": "",
                    "corrections": {}
                }
                for i, f in enumerate(low_confidence_fields)
            ]
        }
        
        with open(output_path, 'w') as file:
            json.dump(review_data, file, indent=2)
        
        if low_confidence_fields:
            print(f"⚠️  {len(low_confidence_fields)} fields need review")
            print(f"   Review file: {output_path}")
        else:
            print(f"✅ All fields have high confidence (≥0.75)")
    
    def export_field_mapping_csv(self, form_def: FormDefinition, output_path: str):
        """
        Export field mapping as CSV for easy viewing/editing.
        
        Args:
            form_def: Form definition to export
            output_path: Path to save CSV file
        """
        import csv
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = [
                'pdf_name', 'interview_variable', 'interview_label', 'interview_datatype',
                'type', 'page', 'bbox', 'confidence', 'required', 'group'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for field in form_def.fields:
                writer.writerow({
                    'pdf_name': field.pdf_name,
                    'interview_variable': field.interview_variable,
                    'interview_label': field.interview_label,
                    'interview_datatype': field.interview_datatype,
                    'type': field.pdf_type,
                    'page': field.page,
                    'bbox': str(field.bbox),
                    'confidence': f"{field.confidence:.2f}",
                    'required': field.required,
                    'group': field.interview_field_group or ''
                })
        
        print(f"💾 Exported field mapping CSV to: {output_path}")


def export_all_formats(form_def: FormDefinition, output_dir: str, base_name: str):
    """
    Export field collection in all available formats.
    
    Args:
        form_def: Form definition to export
        output_dir: Output directory
        base_name: Base filename (without extension)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exporter = FieldCollectionExporter()
    
    print(f"\n📤 Exporting field collection...")
    
    # Standard JSON
    exporter.export_as_json(
        form_def,
        str(output_dir / f"{base_name}_fields.json")
    )
    
    # interview-ready JSON
    exporter.export_as_da_ready_json(
        form_def,
        str(output_dir / f"{base_name}_da_ready.json")
    )
    
    # Review file (if needed)
    exporter.export_for_manual_review(
        form_def,
        str(output_dir / f"{base_name}_review.json")
    )
    
    # CSV mapping
    exporter.export_field_mapping_csv(
        form_def,
        str(output_dir / f"{base_name}_mapping.csv")
    )
    
    print(f"\n✅ All formats exported to: {output_dir}")

