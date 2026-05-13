"""
YAML Field Extractor

Extracts field definitions from interview YAML interview files.
"""

import yaml
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class YAMLFieldExtractor:
    """Extract field definitions from interview YAML interview."""
    
    def __init__(self):
        """Initialize the extractor."""
        self.fields = []
        self.current_file = None
    
    def extract_fields(self, yaml_path: str) -> List[Dict[str, Any]]:
        """
        Extract all field definitions from a interview YAML file.
        
        Args:
            yaml_path: Path to YAML interview file
        
        Returns:
            List of field definitions:
            [
                {
                    "variable_name": "petitioner_name",
                    "datatype": "text",
                    "label": "Petitioner's Name",
                    "required": False,
                    "source_block": "fields",
                    "line_number": 45,
                    "choices": None,
                    "default": None
                },
                ...
            ]
        """
        self.current_file = yaml_path
        self.fields = []
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                content = f.read()
                docs = list(yaml.safe_load_all(content))
            
            # Extract from each YAML document
            for doc_idx, doc in enumerate(docs):
                if not doc:
                    continue
                
                # Extract from fields blocks
                if 'fields' in doc:
                    self._extract_from_fields_block(doc['fields'], doc_idx)
                
                # Extract from question blocks
                if 'question' in doc or 'subquestion' in doc:
                    self._extract_from_question_block(doc, doc_idx)
                
                # Extract from code blocks (simple assignments)
                if 'code' in doc:
                    self._extract_from_code_block(doc['code'], doc_idx)
            
            logger.info(f"Extracted {len(self.fields)} fields from {yaml_path}")
            return self.fields
        
        except Exception as e:
            logger.error(f"Failed to extract fields from {yaml_path}: {e}")
            return []
    
    def _extract_from_fields_block(self, fields_block: Any, doc_idx: int):
        """Extract fields from a 'fields:' block."""
        if not isinstance(fields_block, list):
            return
        
        for item_idx, item in enumerate(fields_block):
            if not isinstance(item, dict):
                continue
            
            # Simple format: "Label": variable_name
            # Complex format: {"label": "...", "field": "...", "datatype": "..."}
            
            for key, value in item.items():
                field_def = {
                    "variable_name": None,
                    "datatype": "text",
                    "label": None,
                    "required": False,
                    "source_block": "fields",
                    "line_number": None,  # Would need line-by-line parsing
                    "choices": None,
                    "default": None
                }
                
                if isinstance(value, str):
                    # Simple format: "Petitioner's Name": petitioner_name
                    field_def["label"] = key
                    field_def["variable_name"] = value
                
                elif isinstance(value, dict):
                    # Complex format with multiple attributes
                    field_def["label"] = key
                    field_def["variable_name"] = value.get("field", key.lower().replace(" ", "_"))
                    field_def["datatype"] = value.get("datatype", "text")
                    field_def["required"] = value.get("required", False)
                    field_def["choices"] = value.get("choices")
                    field_def["default"] = value.get("default")
                
                # Handle special keys like "note", "html"
                if key in ["note", "html", "css", "script"]:
                    continue
                
                if field_def["variable_name"]:
                    self.fields.append(field_def)
    
    def _extract_from_question_block(self, block: Dict, doc_idx: int):
        """Extract fields from a question block with fields."""
        if 'fields' not in block:
            return
        
        # Same as fields block
        self._extract_from_fields_block(block['fields'], doc_idx)
    
    def _extract_from_code_block(self, code: str, doc_idx: int):
        """Extract simple variable assignments from code blocks."""
        # Look for patterns like: variable_name = "value"
        # This is limited - won't catch complex logic
        
        if not isinstance(code, str):
            return
        
        # Pattern: variable = ...
        assignment_pattern = r'^\s*([a-z_][a-z0-9_]*)\s*='
        
        for line in code.split('\n'):
            match = re.match(assignment_pattern, line.strip())
            if match:
                var_name = match.group(1)
                
                # Skip private variables
                if var_name.startswith('_'):
                    continue
                
                field_def = {
                    "variable_name": var_name,
                    "datatype": "text",
                    "label": self._var_name_to_label(var_name),
                    "required": False,
                    "source_block": "code",
                    "line_number": None,
                    "choices": None,
                    "default": None
                }
                
                self.fields.append(field_def)
    
    def _var_name_to_label(self, var_name: str) -> str:
        """Convert variable name to human-readable label."""
        # petitioner_name → Petitioner Name
        return var_name.replace('_', ' ').title()
    
    def get_field_by_name(self, variable_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific field by variable name."""
        for field in self.fields:
            if field.get("variable_name") == variable_name:
                return field
        return None
    
    def get_fields_by_datatype(self, datatype: str) -> List[Dict[str, Any]]:
        """Get all fields of a specific datatype."""
        return [f for f in self.fields if f.get("datatype") == datatype]


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python yaml_field_extractor.py <yaml_file>")
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    extractor = YAMLFieldExtractor()
    fields = extractor.extract_fields(yaml_file)
    
    print(f"\n✓ Extracted {len(fields)} fields from {yaml_file}\n")
    
    for field in fields[:10]:  # Show first 10
        print(f"  - {field['variable_name']} ({field['datatype']})")
        print(f"    Label: {field['label']}")
        print(f"    Source: {field['source_block']}")
        print()
    
    if len(fields) > 10:
        print(f"  ... and {len(fields) - 10} more fields")

