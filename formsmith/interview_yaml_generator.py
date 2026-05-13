#!/usr/bin/env python3
"""
interview YAML Interview Generator

Generates complete interview YAML interviews from detected field definitions.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

try:
    from .schemas import FieldDefinition, FormDefinition
except ImportError:
    from schemas import FieldDefinition, FormDefinition


class InterviewYAMLGenerator:
    """
    Generates interview YAML interviews from field definitions.
    """
    
    def __init__(self):
        self.generated_sections = []
    
    def generate_interview(self, 
                          form_def: FormDefinition,
                          template_pdf_path: str,
                          interview_title: Optional[str] = None) -> str:
        """
        Generate complete interview YAML.
        
        Args:
            form_def: Form definition with detected fields
            template_pdf_path: Path to PDF template file
            interview_title: Optional custom title
            
        Returns:
            Complete YAML interview as string
        """
        self.generated_sections = []
        
        # 1. Metadata block
        self.generated_sections.append(
            self._generate_metadata(form_def, interview_title)
        )
        
        # 2. Objects block (if needed)
        objects_block = self._generate_objects(form_def.fields)
        if objects_block:
            self.generated_sections.append(objects_block)
        
        # 3. Question blocks (grouped by field_group)
        self.generated_sections.extend(
            self._generate_questions(form_def.fields)
        )
        
        # 4. Attachment block
        self.generated_sections.append(
            self._generate_attachment(template_pdf_path, form_def)
        )
        
        # Join with YAML document separator
        return "\n---\n".join(self.generated_sections)
    
    def _generate_metadata(self, form_def: FormDefinition, title: Optional[str]) -> str:
        """Generate metadata block."""
        title = title or form_def.form_type or f"Form {form_def.pdf_name}"
        
        metadata = f"""metadata:
  title: |
    {title}
  short title: |
    {title}
  authors:
    - name: Auto-generated
      organization: PDF Field Detection System
  revision_date: {datetime.now().strftime('%Y-%m-%d')}"""
        
        return metadata
    
    def _generate_objects(self, fields: List[FieldDefinition]) -> Optional[str]:
        """
        Generate objects block if needed (e.g., ALIndividual for parties).
        """
        # Check if we have party fields (petitioner, defendant, etc.)
        has_petitioner = any('petitioner' in f.interview_variable for f in fields)
        has_defendant = any('defendant' in f.interview_variable for f in fields)
        
        if not (has_petitioner or has_defendant):
            return None
        
        objects = ["objects:"]
        
        if has_petitioner:
            objects.append("  - petitioner: ALIndividual")
        
        if has_defendant:
            objects.append("  - defendant: ALIndividual")
        
        return "\n".join(objects)
    
    def _generate_questions(self, fields: List[FieldDefinition]) -> List[str]:
        """
        Generate question blocks grouped by field_group.
        
        Returns list of question block strings.
        """
        # Group fields by field_group
        grouped_fields = defaultdict(list)
        ungrouped_fields = []
        
        for field in fields:
            if field.interview_field_group:
                grouped_fields[field.interview_field_group].append(field)
            else:
                ungrouped_fields.append(field)
        
        question_blocks = []
        
        # Generate grouped questions
        for group_name, group_fields in sorted(grouped_fields.items()):
            question_block = self._generate_grouped_question(group_name, group_fields)
            question_blocks.append(question_block)
        
        # Generate individual questions for ungrouped fields
        for field in ungrouped_fields:
            question_block = self._generate_individual_question(field)
            question_blocks.append(question_block)
        
        return question_blocks
    
    def _generate_grouped_question(self, group_name: str, fields: List[FieldDefinition]) -> str:
        """Generate a grouped question block."""
        # Create human-readable group title
        group_title = group_name.replace('_', ' ').title()
        
        question_lines = [
            "question: |",
            f"  {group_title}",
            "fields:"
        ]
        
        for field in fields:
            # Field label and variable
            question_lines.append(f"  - {field.interview_label}: {field.interview_variable}")
            
            # Datatype
            if field.interview_datatype != "text":
                question_lines.append(f"    datatype: {field.interview_datatype}")
            
            # Required
            question_lines.append(f"    required: {'True' if field.required else 'False'}")
            
            # Validation rules
            if field.validation_rules:
                for rule_key, rule_value in field.validation_rules.items():
                    if rule_key == "max_length":
                        question_lines.append(f"    maxlength: {rule_value}")
                    elif rule_key == "pattern":
                        question_lines.append(f"    validation_regex: {rule_value}")
                    elif rule_key == "min":
                        question_lines.append(f"    min: {rule_value}")
                    elif rule_key == "max":
                        question_lines.append(f"    max: {rule_value}")
        
        return "\n".join(question_lines)
    
    def _generate_individual_question(self, field: FieldDefinition) -> str:
        """Generate an individual question block."""
        question_lines = [
            "question: |",
            f"  {field.interview_label}",
            "fields:"
        ]
        
        # Field variable
        question_lines.append(f"  - {field.interview_label}: {field.interview_variable}")
        
        # Datatype
        if field.interview_datatype != "text":
            question_lines.append(f"    datatype: {field.interview_datatype}")
        
        # Required
        question_lines.append(f"    required: {'True' if field.required else 'False'}")
        
        # Validation rules
        if field.validation_rules:
            for rule_key, rule_value in field.validation_rules.items():
                if rule_key == "max_length":
                    question_lines.append(f"    maxlength: {rule_value}")
                elif rule_key == "pattern":
                    question_lines.append(f"    validation_regex: {rule_value}")
        
        return "\n".join(question_lines)
    
    def _generate_attachment(self, template_pdf: str, form_def: FormDefinition) -> str:
        """Generate attachment block with field mapping."""
        # Get PDF filename
        pdf_filename = Path(template_pdf).name
        
        attachment_lines = [
            "attachment:",
            f"  name: {form_def.form_type or form_def.pdf_name}",
            f"  filename: {Path(pdf_filename).stem}",
            f"  pdf template file: {pdf_filename}",
            "  fields:"
        ]
        
        # Add field mappings
        for field in form_def.fields:
            # PDF field name -> interview variable
            attachment_lines.append(f'    - "{field.pdf_name}": ${{ {field.interview_variable} }}')
        
        return "\n".join(attachment_lines)
    
    def save(self, yaml_content: str, output_path: str):
        """Save generated YAML to file."""
        with open(output_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"💾 Generated interview: {output_path}")


def generate_yaml_from_json(fields_json_path: str, 
                            template_pdf_path: str,
                            output_yaml_path: str,
                            title: Optional[str] = None):
    """
    Convenience function to generate YAML from field JSON file.
    
    Args:
        fields_json_path: Path to field definitions JSON
        template_pdf_path: Path to PDF template
        output_yaml_path: Path to save generated YAML
        title: Optional interview title
    """
    # Load form definition
    form_def = FormDefinition.load(fields_json_path)
    
    # Generate YAML
    generator = InterviewYAMLGenerator()
    yaml_content = generator.generate_interview(form_def, template_pdf_path, title)
    
    # Save
    generator.save(yaml_content, output_yaml_path)
    
    return yaml_content


def main():
    """Main entry point."""
    if len(sys.argv) < 4:
        print("Usage: python -m formsmith.interview_yaml_generator <fields_json> <template_pdf> <output_yaml> [title]")
        print("\nExample:")
        print("  python -m formsmith.interview_yaml_generator fields.json template.pdf interview.yml \"My Interview\"")
        sys.exit(1)
    
    fields_json = sys.argv[1]
    template_pdf = sys.argv[2]
    output_yaml = sys.argv[3]
    title = sys.argv[4] if len(sys.argv) > 4 else None
    
    # Generate YAML
    yaml_content = generate_yaml_from_json(fields_json, template_pdf, output_yaml, title)
    
    print(f"\n✅ YAML interview generated successfully!")
    print(f"   Lines: {len(yaml_content.splitlines())}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

