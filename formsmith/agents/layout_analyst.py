"""
Layout Analyst Agent - Analyzes document structure and spatial relationships.

This agent understands document layout patterns and helps determine where fields
should be placed based on visual structure, alignment, and spacing.
"""

from typing import Dict, List, Any, Optional
import base64
from pathlib import Path
import logging

from .base_agent import VisionAgent

logger = logging.getLogger(__name__)


class LayoutAnalystAgent(VisionAgent):
    """
    Analyzes document layout to understand structure and spatial relationships.
    
    This agent is specialized in:
    - Understanding document structure (sections, columns, tables)
    - Identifying alignment patterns
    - Detecting field groupings
    - Recognizing spacing patterns
    - Suggesting field placement based on layout conventions
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-2024-08-06"):
        super().__init__(api_key=api_key, model=model, agent_name="layout_analyst")
    
    def analyze_layout(
        self,
        image_path: Path,
        existing_fields: Optional[List[Dict]] = None,
        focus_region: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze document layout and spatial structure.
        
        Args:
            image_path: Path to PDF page image
            existing_fields: List of already detected fields
            focus_region: Optional region to focus on (bbox dict)
        
        Returns:
            Analysis result with structure, patterns, and suggestions
        """
        # Build context
        context_parts = []
        
        if existing_fields:
            context_parts.append(f"There are {len(existing_fields)} fields already detected.")
            field_types = {}
            for f in existing_fields:
                ft = f.get('type', 'unknown')
                field_types[ft] = field_types.get(ft, 0) + 1
            context_parts.append(
                f"Field types: {', '.join(f'{k}: {v}' for k, v in field_types.items())}"
            )
        
        if focus_region:
            context_parts.append(
                f"Focus on region: x={focus_region['x']}, y={focus_region['y']}, "
                f"width={focus_region['width']}, height={focus_region['height']}"
            )
        
        context = " ".join(context_parts) if context_parts else "Analyze the entire page."
        
        # Craft expert prompt
        prompt = f"""You are a document layout analysis expert specializing in legal forms.

Your task: Analyze the document structure and spatial relationships to understand field placement patterns.

{context}

Analyze:
1. **Document Structure**:
   - How is the document organized? (sections, columns, tables, etc.)
   - Are there clear visual hierarchies or groupings?

2. **Alignment Patterns**:
   - Are fields aligned vertically or horizontally?
   - What are the common alignment axes (x or y coordinates)?
   - Is there a grid pattern?

3. **Spacing Patterns**:
   - What's the typical spacing between fields?
   - Are there consistent margins or padding values?
   - How far are fields from their labels?

4. **Field Groupings**:
   - Which fields appear to be related (same section, same purpose)?
   - Are there repeating patterns (e.g., address blocks, date fields)?

5. **Layout Conventions**:
   - Does this follow standard form conventions?
   - Are there any unusual or non-standard layout elements?

Respond with a structured JSON object:
{{
    "structure": {{
        "type": "single_column | multi_column | table | mixed",
        "sections": ["section1", "section2"],
        "has_grid": true/false
    }},
    "alignment": {{
        "vertical_axes": [x1, x2, x3],
        "horizontal_axes": [y1, y2, y3],
        "dominant_direction": "vertical | horizontal"
    }},
    "spacing": {{
        "field_spacing_avg": number_px,
        "label_to_field_distance": number_px,
        "margin_left": number_px,
        "margin_right": number_px
    }},
    "groupings": [
        {{
            "group_name": "plaintiff_info",
            "fields": ["field1", "field2"],
            "bbox": [x0, y0, x1, y1]
        }}
    ],
    "suggestions": [
        {{
            "type": "alignment",
            "message": "Fields should align at x=100px",
            "confidence": 0.0-1.0
        }}
    ],
    "confidence": 0.0-1.0,
    "reasoning": "Explanation of analysis"
}}"""
        
        result = self.analyze_image(
            image_path=image_path,
            prompt=prompt,
            response_format="json_object"
        )
        
        # Validate structure
        if result.get('success') and 'analysis' in result:
            analysis = result['analysis']
            
            # Ensure required fields
            if 'structure' not in analysis:
                analysis['structure'] = {'type': 'unknown'}
            if 'alignment' not in analysis:
                analysis['alignment'] = {'vertical_axes': [], 'horizontal_axes': []}
            if 'spacing' not in analysis:
                analysis['spacing'] = {}
            if 'groupings' not in analysis:
                analysis['groupings'] = []
            if 'suggestions' not in analysis:
                analysis['suggestions'] = []
            if 'confidence' not in analysis:
                analysis['confidence'] = 0.5
            if 'reasoning' not in analysis:
                analysis['reasoning'] = 'No reasoning provided'
        
        return result
    
    def suggest_field_placement(
        self,
        image_path: Path,
        label_bbox: Dict,
        label_text: str,
        field_type: str,
        layout_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Suggest optimal field placement based on label and layout context.
        
        Args:
            image_path: Path to PDF page image
            label_bbox: Bounding box of the field label
            label_text: Text of the label
            field_type: Expected field type ('text', 'checkbox', etc.)
            layout_context: Optional layout analysis from analyze_layout()
        
        Returns:
            Suggestion with bbox and confidence
        """
        # Build context
        context = f"Label: '{label_text}' at {label_bbox}"
        if layout_context:
            spacing = layout_context.get('spacing', {})
            if 'label_to_field_distance' in spacing:
                context += f"\nTypical label-to-field distance: {spacing['label_to_field_distance']}px"
        
        prompt = f"""You are a form layout expert. Suggest the optimal placement for a field.

{context}
Field type: {field_type}

Based on the label position and document layout, where should the input field be placed?

Consider:
1. Typical distance from label to field for this form
2. Alignment with other fields
3. Standard conventions for {field_type} fields
4. Available blank space

Respond with JSON:
{{
    "suggested_bbox": [x0, y0, x1, y1],
    "confidence": 0.0-1.0,
    "reasoning": "Why this placement is optimal",
    "alternatives": [
        {{"bbox": [x0, y0, x1, y1], "reason": "Alternative placement"}}
    ]
}}"""
        
        result = self.analyze_image(
            image_path=image_path,
            prompt=prompt,
            response_format="json_object"
        )
        
        return result
    
    def validate_field_alignment(
        self,
        image_path: Path,
        fields: List[Dict],
        layout_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Validate that fields are properly aligned with the document structure.
        
        Args:
            image_path: Path to PDF page image
            fields: List of field dictionaries with bbox
            layout_context: Optional layout analysis
        
        Returns:
            Validation result with misalignments identified
        """
        # Prepare field summary
        field_summary = []
        for i, f in enumerate(fields):
            bbox = f.get('bbox', f.get('bounding_box', []))
            field_summary.append(f"Field {i}: {f.get('name', 'unnamed')} at {bbox}")
        
        field_list = "\n".join(field_summary[:20])  # Limit to first 20
        if len(fields) > 20:
            field_list += f"\n... and {len(fields) - 20} more fields"
        
        prompt = f"""You are a document quality control expert. Validate field alignment.

Fields to validate:
{field_list}

Check for:
1. **Alignment Issues**: Are fields aligned to a consistent grid or axes?
2. **Spacing Issues**: Is spacing between fields consistent?
3. **Overlap**: Do any fields overlap inappropriately?
4. **Margin Violations**: Are fields too close to page edges?
5. **Visual Balance**: Is the layout visually balanced?

Respond with JSON:
{{
    "is_well_aligned": true/false,
    "issues": [
        {{
            "type": "misalignment | spacing | overlap | margin | other",
            "severity": "low | medium | high",
            "affected_fields": ["field1", "field2"],
            "description": "What's wrong",
            "suggested_fix": "How to fix it"
        }}
    ],
    "confidence": 0.0-1.0,
    "overall_quality": "excellent | good | fair | poor"
}}"""
        
        result = self.analyze_image(
            image_path=image_path,
            prompt=prompt,
            response_format="json_object"
        )
        
        return result


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    logging.basicConfig(level=logging.INFO)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        exit(1)
    
    agent = LayoutAnalystAgent(api_key=api_key)
    
    # Example: Analyze layout
    test_image = Path("test_form_page.png")
    if test_image.exists():
        result = agent.analyze_layout(test_image)
        if result.get('success'):
            print("Layout Analysis:")
            print(f"  Structure: {result['analysis'].get('structure')}")
            print(f"  Confidence: {result['analysis'].get('confidence')}")
            print(f"  Cost: ${result['cost']:.4f}")
        else:
            print(f"Error: {result.get('error')}")
    else:
        print(f"Test image not found: {test_image}")
