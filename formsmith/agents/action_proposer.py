"""
Action Proposer Agent

Proposes specific actions to fix field placement issues using structured outputs.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any
from .base_agent import VisionAgent
import json


class FieldAdjustment(BaseModel):
    """Structured output for field adjustments."""
    action: Literal["move", "resize", "delete", "accept"]
    field_id: str
    reasoning: str
    
    # For move action
    new_bbox: Optional[list[float]] = None
    
    # For resize action
    new_width: Optional[float] = None
    new_height: Optional[float] = None
    
    # Confidence in this action
    confidence: float = Field(ge=0.0, le=1.0)


class ActionProposerAgent(VisionAgent):
    """
    Proposes specific actions to fix field placement issues.
    
    Uses structured outputs (Pydantic) to return actionable commands.
    """
    
    def get_system_prompt(self) -> str:
        return """You are a field adjustment specialist for PDF form field detection.

Your job: Propose specific actions to fix incorrectly placed fields.

You will see:
- Original page image
- Detected field (highlighted in red)
- Nearby labels and context

You must propose ONE action:
1. MOVE: Relocate field to correct position (provide exact bbox [x0, y0, x1, y1])
2. RESIZE: Adjust field dimensions (provide new width/height)
3. DELETE: Remove false positive (no bbox needed)
4. ACCEPT: Field is correct as-is (no changes needed)

Guidelines:
- Be precise: Provide exact pixel coordinates for moves/resizes
- Be confident: Only propose high-confidence actions (>0.85)
- Be conservative: If unsure, suggest ACCEPT and flag for review
- Coordinate system: Origin (0,0) is at TOP-LEFT of page
- Bounding box format: [x0, y0, x1, y1] where x0 < x1 and y0 < y1

Your output MUST be valid JSON matching this schema:
{
  "action": "move" | "resize" | "delete" | "accept",
  "field_id": "field_123",
  "reasoning": "Brief explanation (one sentence)",
  "new_bbox": [x0, y0, x1, y1],  // Required for "move"
  "new_width": 120.0,            // Optional for "resize"
  "new_height": 15.0,            // Optional for "resize"
  "confidence": 0.0-1.0          // How confident you are
}

Rules:
- Only MOVE if you can see exactly where the field should be
- Only RESIZE if dimensions are clearly wrong (too large/small)
- DELETE if this is obviously a false positive (no actual field here)
- ACCEPT if field looks correct or you're unsure"""

    def get_user_prompt(
        self, 
        field: Dict,
        issue: str,
        context: Dict,
        **kwargs
    ) -> str:
        # Extract key field properties
        field_id = field.get('id', 'unknown')
        field_type = field.get('type', 'unknown')
        bbox = field.get('bbox', [0, 0, 0, 0])
        label = field.get('source_label', 'None')
        confidence = field.get('confidence', 0.0)
        
        # Format context nicely
        nearby_labels = context.get('labels', [])
        validation = context.get('validation', {})
        
        labels_text = "\n".join([
            f"  - '{l.get('text', '')}' at ({l.get('x', 0):.1f}, {l.get('y', 0):.1f})"
            for l in nearby_labels[:5]  # Top 5 nearby labels
        ])
        
        return f"""Current field that needs review:
- ID: {field_id}
- Type: {field_type}
- Current bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]
- Associated label: "{label}"
- Detection confidence: {confidence:.2f}

Issue detected: {issue}

Nearby labels (for context):
{labels_text if labels_text else "  (none)"}

Validation details:
{json.dumps(validation, indent=2)}

Based on the image and this information, propose an action to fix this field.

Respond with JSON only (no markdown code blocks):
{{
  "action": "move" | "resize" | "delete" | "accept",
  "field_id": "{field_id}",
  "reasoning": "one sentence explaining why",
  "new_bbox": [x0, y0, x1, y1],  // if action is "move"
  "new_width": 120.0,  // if action is "resize"
  "new_height": 15.0,  // if action is "resize"
  "confidence": 0.0-1.0
}}

Important:
- Look at the IMAGE to see the actual form
- Find where the blank space actually is (not where the label is)
- Only propose high-confidence adjustments (>0.85)
- If you can't see a clear field location, use "accept" with low confidence"""

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse into FieldAdjustment model."""
        # Clean up response (remove markdown code blocks if present)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            # Remove markdown code blocks
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
            adjustment = FieldAdjustment(**data)
        except Exception as e:
            raise ValueError(f"Response doesn't match FieldAdjustment schema: {e}\nData: {data}")
        
        # Additional validation
        if adjustment.action == "move" and not adjustment.new_bbox:
            raise ValueError("'move' action requires 'new_bbox'")
        
        if adjustment.action == "move" and adjustment.new_bbox:
            # Validate bbox
            bbox = adjustment.new_bbox
            if len(bbox) != 4:
                raise ValueError(f"new_bbox must have 4 values, got {len(bbox)}")
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError(f"Invalid bbox: x1 must be > x0 and y1 must be > y0, got {bbox}")
        
        if adjustment.action == "resize" and not (adjustment.new_width or adjustment.new_height):
            raise ValueError("'resize' action requires 'new_width' and/or 'new_height'")
        
        return adjustment.model_dump()

