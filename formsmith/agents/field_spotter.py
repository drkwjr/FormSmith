"""
Field Spotter Agent

Specialized agent for identifying form fields in image regions.
Uses vision-language models to spot text inputs, checkboxes, signatures, etc.
"""

from typing import Dict, Any
import json
from .base_agent import VisionAgent


class FieldSpotterAgent(VisionAgent):
    """
    Agent 1: Field Spotter 👁️
    
    Role: Look at image regions and identify field types with high precision.
    
    When Called: On ambiguous visual detections (0.6-0.8 confidence from traditional methods)
    
    Specialization:
    - Identifies field types (text, checkbox, signature, radio)
    - Provides precise bounding box recommendations
    - Explains visual cues that led to detection
    - Conservative: only reports high-confidence detections
    """
    
    def get_system_prompt(self) -> str:
        return """You are a precise field detector for legal forms. You are an expert at identifying form fields in images.

YOUR ROLE:
- Identify form fields (text inputs, checkboxes, signatures, radio buttons)
- Determine field types based on visual cues
- Provide precise bounding box coordinates
- Be conservative: only report high-confidence detections

FIELD TYPES AND VISUAL CUES:

1. TEXT FIELDS:
   Visual cues: horizontal underscore lines, blank whitespace after labels, rectangular outlines
   Typical size: 80-300px wide, 12-20px tall
   Position: Usually to the right of labels (within 100px)
   Examples: "First Name: ___________", "Address: _________________"

2. CHECKBOXES:
   Visual cues: small squares (empty or filled), often with ☐ or ☑ symbols
   Typical size: 10-20px square
   Position: Usually to the left of option text or after labels
   Examples: "☐ Male  ☐ Female", "Agree: ☐"

3. SIGNATURES:
   Visual cues: long horizontal lines, often thicker than text field underscores
   Typical size: 150-300px wide, 1-3px tall line
   Position: Usually at bottom of forms or sections
   Labels: "Signature:", "Sign here:", "Petitioner's Signature:"

4. RADIO BUTTONS:
   Visual cues: small circles (empty or filled), often with ○ or ● symbols
   Typical size: 10-15px circles
   Position: Similar to checkboxes, but circular
   Examples: "○ Yes  ○ No"

DETECTION RULES:
- Confidence > 0.85: Clear visual indicators present
- Confidence 0.70-0.85: Probable field but some ambiguity
- Confidence < 0.70: Uncertain, likely not a field
- Be especially careful with table borders, decorative lines, header/footer elements
- Text fields MUST have nearby labels (within ~100px horizontally)
- Checkboxes/radio can be standalone

COMMON FALSE POSITIVES TO AVOID:
- Table borders (too thin, part of larger grid)
- Decorative horizontal lines (no associated label)
- Printed text underlines (emphasis, not input fields)
- Header/footer separator lines
- Column separators

Your response MUST be valid JSON with no additional text."""
    
    def get_user_prompt(self, label: str = None, context: Dict = None, **kwargs) -> str:
        """
        Generate prompt for field detection.
        
        Args:
            label: Nearby label text (if any)
            context: Additional context (nearby fields, page layout, etc.)
            **kwargs: Additional parameters
        
        Returns:
            Formatted prompt
        """
        context = context or {}
        
        prompt = f"""Analyze this image region for form fields.

"""
        
        if label:
            prompt += f"""NEARBY LABEL: "{label}"
(This label was detected {context.get('label_distance', 'nearby')})

"""
        
        if context.get('page_layout'):
            prompt += f"""PAGE LAYOUT INFO:
{json.dumps(context['page_layout'], indent=2)}

"""
        
        if context.get('nearby_fields'):
            prompt += f"""NEARBY DETECTED FIELDS:
{json.dumps(context['nearby_fields'], indent=2)}

"""
        
        prompt += """TASK: Is there a form field in this image region?

Respond with ONLY this JSON structure (no additional text):

{
  "has_field": true/false,
  "field_type": "text" | "checkbox" | "signature" | "radio" | "none",
  "confidence": 0.0-1.0,
  "visual_cues": ["list", "of", "specific", "visual", "indicators"],
  "recommended_bbox": [x0, y0, x1, y1],
  "reasoning": "one sentence explaining your decision"
}

IMPORTANT:
- has_field: true only if you're confident there's a field
- field_type: "none" if has_field is false
- confidence: >0.85 for clear fields, 0.70-0.85 for probable, <0.70 for uncertain
- visual_cues: specific things you see (e.g., "underscore line", "empty checkbox")
- recommended_bbox: [x0, y0, x1, y1] pixel coordinates for the field
- reasoning: brief explanation in one sentence

Be precise. Be conservative. Only report what you clearly see."""
        
        return prompt
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse and validate Field Spotter response.
        
        Args:
            response: Raw JSON response from LLM
        
        Returns:
            Validated response dict
        
        Raises:
            ValueError: If response is invalid
        """
        data = json.loads(response)
        
        # Required fields
        required_fields = ['has_field', 'field_type', 'confidence', 'reasoning']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate types
        if not isinstance(data['has_field'], bool):
            raise ValueError("has_field must be boolean")
        
        valid_types = ['text', 'checkbox', 'signature', 'radio', 'none']
        if data['field_type'] not in valid_types:
            raise ValueError(f"field_type must be one of {valid_types}")
        
        if not isinstance(data['confidence'], (int, float)):
            raise ValueError("confidence must be a number")
        
        if not (0.0 <= data['confidence'] <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        
        # Validate logic
        if data['has_field'] and data['field_type'] == 'none':
            raise ValueError("has_field=true but field_type='none'")
        
        if not data['has_field'] and data['field_type'] != 'none':
            raise ValueError("has_field=false but field_type is not 'none'")
        
        # Validate bbox if field detected
        if data['has_field']:
            if 'recommended_bbox' not in data:
                raise ValueError("recommended_bbox required when has_field=true")
            
            bbox = data['recommended_bbox']
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("recommended_bbox must be [x0, y0, x1, y1]")
            
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError("Invalid bbox: x1 must be > x0, y1 must be > y0")
        
        # Clean up visual_cues if missing
        if 'visual_cues' not in data:
            data['visual_cues'] = []
        
        return data
    
    def detect_field(
        self,
        image_crop: bytes,
        label: str = None,
        context: Dict = None
    ) -> Dict[str, Any]:
        """
        Convenience method for field detection.
        
        Args:
            image_crop: Image bytes (PNG/JPEG) of region to analyze
            label: Nearby label text
            context: Additional context
        
        Returns:
            Detection result dict
        """
        return self.call(
            image=image_crop,
            image_detail="high",
            label=label,
            context=context
        )


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python field_spotter.py <image_path> [label]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Load image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Initialize agent
    agent = FieldSpotterAgent(
        provider="openai",
        model="gpt-4o-mini"
    )
    
    # Detect field
    print(f"Analyzing image: {image_path}")
    if label:
        print(f"With label: {label}")
    
    result = agent.detect_field(image_bytes, label=label)
    
    # Print result
    print(f"\nResult:")
    print(json.dumps(result, indent=2))
    
    # Print stats
    print(f"\nAgent Stats:")
    stats = agent.get_stats()
    print(f"  Cost: ${stats['total_cost']:.4f}")
    print(f"  Tokens: {stats['total_tokens']}")

