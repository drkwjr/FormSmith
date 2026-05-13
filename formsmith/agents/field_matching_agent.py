"""
Field Matching Agent

LLM-powered agent that matches YAML interview variables to PDF form fields.
Uses GPT-4 Vision to understand label positioning and field relationships.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
from .base_agent import VisionAgent

logger = logging.getLogger(__name__)


class FieldMapping(BaseModel):
    """Structured output for field mapping."""
    yaml_variable: str = Field(description="Variable name from YAML interview")
    pdf_field: Optional[str] = Field(
        None,
        description="PDF field name that matches, or null if no match found"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in this mapping (0.0-1.0)"
    )
    reasoning: str = Field(description="Why this mapping was chosen")
    action: str = Field(
        description="Action to take: rename_pdf_field | add_yaml_field | flag_mismatch | flag_missing | accept"
    )
    new_pdf_name: Optional[str] = Field(
        None,
        description="New name for PDF field (for rename_pdf_field action)"
    )
    type_mismatch: bool = Field(
        False,
        description="True if YAML and PDF field types are incompatible"
    )


class FieldMatchingAgent(VisionAgent):
    """
    Matches YAML interview variables to PDF form fields.
    
    Uses:
    - Label text similarity
    - Spatial reasoning (page layout)
    - Field type compatibility
    - Semantic understanding
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-2024-08-06"):
        super().__init__(
            api_key=api_key,
            model=model,
            agent_name="field_matching"
        )
    
    def get_system_prompt(self) -> str:
        return """You are a field mapping specialist for interview YAML interviews and PDF forms.

Your job: Match each YAML variable to the correct PDF form field.

You will see:
- YAML interview field definitions (variable names, labels, datatypes)
- PDF form field definitions (field names, types, positions, bounding boxes)
- PDF page image showing the actual form

Matching Criteria (in order of priority):

1. **Label Text Similarity**: YAML label should match text near PDF field
   - "Petitioner's Name" (YAML) ↔ "Petitioner's Name:" (PDF label near field)
   
2. **Field Type Compatibility**:
   - text (YAML) ↔ text (PDF) ✓
   - yesno (YAML) ↔ checkbox (PDF) ✓
   - signature (YAML) ↔ text or signature (PDF) ✓
   - date (YAML) ↔ text or date (PDF) ✓
   - text (YAML) ↔ checkbox (PDF) ✗ TYPE MISMATCH
   
3. **Spatial Reasoning**: Field position on page
   - Fields in same section likely related
   - Top-to-bottom, left-to-right reading order
   
4. **Semantic Understanding**:
   - `date_of_birth` likely near "Date of Birth" or "DOB" label
   - `petitioner_name` likely near "Petitioner" section
   - `signature` likely at bottom of form

Actions to propose:

**rename_pdf_field**: PDF field should be renamed to match YAML variable
- Use when: Label matches, types compatible, names different
- Provide: new_pdf_name = YAML variable name

**add_yaml_field**: PDF has field but YAML doesn't (suggest adding to YAML)
- Use when: PDF field has no YAML match
- Note: This requires manual YAML editing

**flag_mismatch**: Type incompatibility (checkbox vs text, etc.)
- Use when: Label matches but types are incompatible
- Set: type_mismatch = true

**flag_missing**: YAML variable has no matching PDF field
- Use when: YAML defines field but PDF doesn't have it
- Set: pdf_field = null

**accept**: Names already match perfectly
- Use when: YAML variable == PDF field name

Guidelines:
- Be conservative with confidence scores
- confidence ≥ 0.90: Very clear match (label + type + position all align)
- confidence 0.70-0.89: Good match (label + type align, position reasonable)
- confidence 0.50-0.69: Uncertain (partial label match or ambiguous position)
- confidence < 0.50: Weak match (guessing)

- Always prefer exact label matches over semantic similarity
- Check field types carefully - type mismatches cause form filling failures
- Consider spatial context (fields in same section, similar y-coordinates)

Output format:
{
  "yaml_variable": "petitioner_name",
  "pdf_field": "field_0",
  "confidence": 0.95,
  "reasoning": "YAML label 'Petitioner's Name' matches text near PDF field_0 at (100, 200)",
  "action": "rename_pdf_field",
  "new_pdf_name": "petitioner_name",
  "type_mismatch": false
}

EXAMPLES (5 varied scenarios):

Example 1 - Clear match, needs rename:
Input:
  YAML: {"variable_name": "petitioner_name", "label": "Petitioner's Name", "datatype": "text"}
  PDF: {"field_name": "field_0", "type": "text", "bbox": [100, 200, 250, 215], "nearby_label": "Petitioner's Name:"}
Output: {
  "yaml_variable": "petitioner_name",
  "pdf_field": "field_0",
  "confidence": 0.95,
  "reasoning": "YAML label 'Petitioner's Name' exactly matches PDF label near field_0",
  "action": "rename_pdf_field",
  "new_pdf_name": "petitioner_name",
  "type_mismatch": false
}

Example 2 - Checkbox type match:
Input:
  YAML: {"variable_name": "has_children", "label": "Has children?", "datatype": "yesno"}
  PDF: {"field_name": "checkbox_3", "type": "checkbox", "bbox": [100, 300, 112, 312], "nearby_label": "Children?"}
Output: {
  "yaml_variable": "has_children",
  "pdf_field": "checkbox_3",
  "confidence": 0.92,
  "reasoning": "YAML yesno field matches PDF checkbox near 'Children?' label",
  "action": "rename_pdf_field",
  "new_pdf_name": "has_children",
  "type_mismatch": false
}

Example 3 - Type mismatch:
Input:
  YAML: {"variable_name": "agree_to_terms", "label": "I agree", "datatype": "yesno"}
  PDF: {"field_name": "field_15", "type": "text", "bbox": [100, 400, 250, 415], "nearby_label": "I agree"}
Output: {
  "yaml_variable": "agree_to_terms",
  "pdf_field": "field_15",
  "confidence": 0.85,
  "reasoning": "Label matches but type mismatch: YAML expects checkbox (yesno) but PDF has text field",
  "action": "flag_mismatch",
  "new_pdf_name": null,
  "type_mismatch": true
}

Example 4 - Missing PDF field:
Input:
  YAML: {"variable_name": "middle_name", "label": "Middle Name", "datatype": "text"}
  PDF: No matching field found
Output: {
  "yaml_variable": "middle_name",
  "pdf_field": null,
  "confidence": 0.0,
  "reasoning": "No PDF field found near 'Middle Name' label or semantically similar",
  "action": "flag_missing",
  "new_pdf_name": null,
  "type_mismatch": false
}

Example 5 - Already matched:
Input:
  YAML: {"variable_name": "petitioner_signature", "label": "Signature", "datatype": "signature"}
  PDF: {"field_name": "petitioner_signature", "type": "signature", "bbox": [100, 500, 350, 525]}
Output: {
  "yaml_variable": "petitioner_signature",
  "pdf_field": "petitioner_signature",
  "confidence": 1.0,
  "reasoning": "Names already match perfectly, types compatible",
  "action": "accept",
  "new_pdf_name": null,
  "type_mismatch": false
}

These examples show the format and level of reasoning expected."""
    
    def match_fields(
        self,
        yaml_fields: List[Dict],
        pdf_fields: List[Dict],
        pdf_image_path: str
    ) -> List[Dict]:
        """
        Match YAML variables to PDF fields.
        
        Args:
            yaml_fields: List of YAML field definitions
            pdf_fields: List of PDF field definitions
            pdf_image_path: Path to PDF page image
        
        Returns:
            List of mapping proposals
        """
        mappings = []
        
        # Process in batches to avoid overwhelming the LLM
        batch_size = 10
        
        for i in range(0, len(yaml_fields), batch_size):
            batch = yaml_fields[i:i+batch_size]
            
            # Prepare batch prompt
            user_prompt = self._create_batch_prompt(batch, pdf_fields)
            
            # Call LLM with vision
            result = self.analyze_image(
                image_path=pdf_image_path,
                prompt=user_prompt,
                response_format="json_object"  # We'll parse multiple FieldMapping objects
            )
            
            if result.get('success') and result.get('analysis'):
                # Parse batch results
                try:
                    batch_mappings = result['analysis'].get('mappings', [])
                    
                    # Validate each mapping
                    for mapping_data in batch_mappings:
                        try:
                            mapping = FieldMapping(**mapping_data)
                            mappings.append(mapping.model_dump())
                        except Exception as e:
                            logger.warning(f"Invalid mapping in batch: {e}")
                            continue
                
                except Exception as e:
                    logger.error(f"Failed to parse batch mappings: {e}")
            
            logger.info(f"Processed {len(batch)} YAML fields, {len(mappings)} mappings so far")
        
        return mappings
    
    def _create_batch_prompt(
        self,
        yaml_fields: List[Dict],
        pdf_fields: List[Dict]
    ) -> str:
        """Create prompt for a batch of YAML fields."""
        
        yaml_summary = "\n".join([
            f"  - {f['variable_name']}: \"{f.get('label', '')}\" ({f.get('datatype', 'text')})"
            for f in yaml_fields
        ])
        
        pdf_summary = "\n".join([
            f"  - {f.get('pdf_name', f.get('name', 'unknown'))}: "
            f"{f.get('pdf_type', f.get('type', 'text'))} at "
            f"[{f.get('bbox', [0,0,0,0])[0]:.0f}, {f.get('bbox', [0,0,0,0])[1]:.0f}]"
            for f in pdf_fields[:30]  # Limit to avoid prompt overflow
        ])
        
        prompt = f"""Match these YAML variables to PDF fields visible in the image.

YAML Fields to match:
{yaml_summary}

Available PDF Fields:
{pdf_summary}

For each YAML field, propose a mapping. Return JSON:
{{
  "mappings": [
    {{
      "yaml_variable": "...",
      "pdf_field": "..." or null,
      "confidence": 0.0-1.0,
      "reasoning": "...",
      "action": "rename_pdf_field|add_yaml_field|flag_mismatch|flag_missing|accept",
      "new_pdf_name": "..." or null,
      "type_mismatch": true/false
    }},
    ...
  ]
}}

Look at the PDF image to see label positions and field locations.
Match based on label similarity, field types, and spatial relationships."""
        
        return prompt


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        exit(1)
    
    agent = FieldMatchingAgent(api_key=api_key)
    
    # Test with sample data
    yaml_fields = [
        {
            "variable_name": "petitioner_name",
            "label": "Petitioner's Name",
            "datatype": "text"
        }
    ]
    
    pdf_fields = [
        {
            "pdf_name": "field_0",
            "pdf_type": "text",
            "bbox": [100, 200, 250, 215]
        }
    ]
    
    print("FieldMatchingAgent initialized successfully")
    print(f"Ready to match {len(yaml_fields)} YAML fields to {len(pdf_fields)} PDF fields")

