"""
Validator Agent

Validates field correctness and identifies issues with field placement.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Literal
from .base_agent import VisionAgent
import json


class ValidationResult(BaseModel):
    """Structured output for field validation."""
    is_correct: bool
    confidence: float = Field(ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    suggested_action: Literal["accept", "move", "resize", "delete"] = "accept"
    reasoning: str


class ValidatorAgent(VisionAgent):
    """
    Validates field correctness and identifies placement issues.
    
    Examines fields and determines if they are correctly placed.
    """
    
    def get_system_prompt(self) -> str:
        return """You are a field validation specialist for PDF form field detection.

Your job: Validate whether a detected field is correctly placed.

You will see:
- Original page image
- Detected field (highlighted)
- Associated label
- Nearby context

You must determine:
1. Is this field correctly placed? (yes/no)
2. What is your confidence? (0.0-1.0)
3. What issues exist? (list of specific problems)
4. What action should be taken? (accept/move/resize/delete)

Common issues to check for:
- Field overlaps with text (should be in blank space)
- Field is too far from its label
- Field dimensions are wrong (too large/small)
- Field is a false positive (no actual field here)
- Field type doesn't match visual cues (e.g., checkbox vs text)

Guidelines:
- Be conservative: Only mark as incorrect if you're sure (confidence > 0.80)
- Check spatial relationship between label and field
- Look for visual cues (underscores, boxes, checkboxes)
- Consider field type appropriateness
- Coordinate system: Origin (0,0) is at TOP-LEFT of page

Your output MUST be valid JSON matching this schema:
{
  "is_correct": true | false,
  "confidence": 0.0-1.0,
  "issues": ["issue 1", "issue 2"],  // Empty if is_correct=true
  "suggested_action": "accept" | "move" | "resize" | "delete",
  "reasoning": "Brief explanation (one sentence)"
}"""

    def get_user_prompt(
        self, 
        field: Dict,
        labels: List[Dict],
        **kwargs
    ) -> str:
        # Extract field properties
        field_id = field.get('id', 'unknown')
        field_type = field.get('type', 'unknown')
        bbox = field.get('bbox', [0, 0, 0, 0])
        label = field.get('source_label', 'None')
        confidence = field.get('confidence', 0.0)
        
        # Format nearby labels
        nearby_labels = sorted(
            labels,
            key=lambda l: (
                (l.get('x', 0) - bbox[0])**2 + 
                (l.get('y', 0) - bbox[1])**2
            )**0.5
        )[:5]  # Top 5 nearest labels
        
        labels_text = "\n".join([
            f"  - '{l.get('text', '')}' at ({l.get('x', 0):.1f}, {l.get('y', 0):.1f})"
            for l in nearby_labels
        ])
        
        # Calculate field dimensions
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        return f"""Field to validate:
- ID: {field_id}
- Type: {field_type}
- Bounding box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]
- Dimensions: {width:.1f} × {height:.1f} pixels
- Associated label: "{label}"
- Detection confidence: {confidence:.2f}

Nearby labels (for context):
{labels_text if labels_text else "  (none)"}

Based on the image, validate whether this field is correctly placed.

Look for:
1. Is the field in blank space (not overlapping text)?
2. Is it appropriately positioned relative to its label?
3. Are the dimensions reasonable for a {field_type} field?
4. Does a field actually exist at this location on the form?

Respond with JSON only (no markdown code blocks):
{{
  "is_correct": true | false,
  "confidence": 0.0-1.0,
  "issues": ["list", "of", "issues"],
  "suggested_action": "accept" | "move" | "resize" | "delete",
  "reasoning": "one sentence explanation"
}}

Important:
- Look at the IMAGE to see the actual form
- Be conservative: only mark incorrect if you're confident (>0.80)
- Provide specific issues (not vague statements)
- Suggest the most appropriate action"""

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse into ValidationResult model."""
        # Clean up response (remove markdown code blocks if present)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join([l for l in lines if not l.startswith("```")])
            cleaned = cleaned.strip()
        
        # Parse JSON
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}\nResponse: {response}")
        
        # Validate with Pydantic
        try:
            validation = ValidationResult(**data)
        except Exception as e:
            raise ValueError(f"Response doesn't match ValidationResult schema: {e}\nData: {data}")
        
        # Additional validation
        if not validation.is_correct and not validation.issues:
            raise ValueError("Field marked as incorrect but no issues listed")
        
        if validation.is_correct and validation.issues:
            # Clear issues if field is marked correct
            validation.issues = []
        
        return validation.model_dump()
