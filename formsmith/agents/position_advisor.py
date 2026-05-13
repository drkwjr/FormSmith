"""
Position Advisor Agent - Provides precise guidance on field positioning.

This agent specializes in determining exact pixel-perfect coordinates for fields,
providing detailed spatial guidance to move fields to optimal positions.
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

from .base_agent import VisionAgent

logger = logging.getLogger(__name__)


class PositionAdvisorAgent(VisionAgent):
    """
    Provides precise positioning guidance for fields.
    
    This agent is specialized in:
    - Calculating pixel-perfect field coordinates
    - Providing directional guidance (move up/down/left/right)
    - Determining optimal field dimensions
    - Ensuring proper spacing and alignment
    - Fine-tuning position adjustments
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-2024-08-06"):
        super().__init__(api_key=api_key, model=model, agent_name="position_advisor")
    
    def get_optimal_position(
        self,
        image_path: Path,
        current_bbox: Tuple[float, float, float, float],
        field_type: str,
        label_text: Optional[str] = None,
        nearby_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Determine the optimal position for a field.
        
        Args:
            image_path: Path to PDF page image
            current_bbox: Current bounding box [x0, y0, x1, y1]
            field_type: Type of field ('text', 'checkbox', etc.)
            label_text: Optional label text for context
            nearby_context: Optional dict with nearby fields/labels
        
        Returns:
            Positioning guidance with optimal bbox and adjustment instructions
        """
        x0, y0, x1, y1 = current_bbox
        width = x1 - x0
        height = y1 - y0
        
        # Build context
        context_parts = []
        if label_text:
            context_parts.append(f"Associated label: '{label_text}'")
        
        if nearby_context:
            if 'nearby_fields' in nearby_context:
                nf = nearby_context['nearby_fields']
                context_parts.append(f"{len(nf)} nearby fields for alignment reference")
            if 'alignment_axis' in nearby_context:
                axis = nearby_context['alignment_axis']
                context_parts.append(f"Alignment axis: x={axis.get('x')}, y={axis.get('y')}")
        
        context_str = "\n".join(context_parts) if context_parts else "No additional context."
        
        prompt = f"""You are a position optimization expert. Provide precise positioning guidance.

Current Field:
- Type: {field_type}
- Current Position: x={x0:.1f}, y={y0:.1f}, width={width:.1f}, height={height:.1f}
- Bounding Box: [{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}]

{context_str}

Your task: Analyze the image and determine the OPTIMAL position for this field.

Provide:
1. **Optimal Bounding Box**: Precise coordinates where the field should be
2. **Adjustment Vector**: How to move from current to optimal position
3. **Dimensional Guidance**: Should width/height be adjusted?
4. **Alignment Correction**: Should this align with other elements?
5. **Confidence**: How certain are you about this position?

Be VERY PRECISE - your guidance will directly move the field.

Respond with JSON:
{{
    "optimal_bbox": [x0, y0, x1, y1],
    "adjustment": {{
        "move_x": number_px,  // positive = right, negative = left
        "move_y": number_px,  // positive = down, negative = up
        "resize_width": number_px,  // positive = wider, negative = narrower
        "resize_height": number_px  // positive = taller, negative = shorter
    }},
    "directional_guidance": "move 5px right, 10px up, expand width by 20px",
    "reasoning": "Why this position is optimal",
    "alignment_note": "Should align with field X at y=200",
    "confidence": 0.0-1.0,
    "precision_level": "pixel_perfect | close | approximate"
}}"""
        
        result = self.analyze_image(
            image_path=image_path,
            prompt=prompt,
            response_format="json_object"
        )
        
        if result.get('success'):
            result['original_bbox'] = current_bbox
        
        return result
    
    def compare_positions(
        self,
        image_path: Path,
        field_name: str,
        position_a: Tuple[float, float, float, float],
        position_b: Tuple[float, float, float, float],
        field_type: str
    ) -> Dict[str, Any]:
        """
        Compare two possible positions and recommend the better one.
        
        Args:
            image_path: Path to PDF page image
            field_name: Name of the field
            position_a: First position option [x0, y0, x1, y1]
            position_b: Second position option [x0, y0, x1, y1]
            field_type: Type of field
        
        Returns:
            Recommendation for which position is better
        """
        prompt = f"""You are a positioning expert. Compare two field placement options.

Field: {field_name} (type: {field_type})

Option A: {position_a}
Option B: {position_b}

Your task: Determine which position is MORE ACCURATE for this field.

Examine the image carefully and compare both positions against the actual form.

Respond with JSON:
{{
    "better_position": "A | B | neither",
    "confidence": 0.0-1.0,
    "reasoning": "Why this position is better",
    "position_a_score": 0.0-1.0,
    "position_b_score": 0.0-1.0,
    "suggested_improvement": "If neither is perfect, what would be better?"
}}"""
        
        result = self.analyze_image(
            image_path=image_path,
            prompt=prompt,
            response_format="json_object"
        )
        
        return result
    
    def refine_field_dimensions(
        self,
        image_path: Path,
        field: Dict
    ) -> Dict[str, Any]:
        """
        Refine field dimensions to perfectly match the visual element.
        
        Args:
            image_path: Path to PDF page image
            field: Field dictionary with current bbox
        
        Returns:
            Refined dimensions
        """
        bbox = field.get('bbox', field.get('bounding_box', []))
        field_type = field.get('type', 'unknown')
        field_name = field.get('name', 'unnamed')
        
        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0
        
        prompt = f"""You are a dimension optimization expert. Refine field dimensions for perfect fit.

Field: {field_name} (type: {field_type})
Current Dimensions: {width:.1f}px × {height:.1f}px
Current Bbox: [{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}]

Your task: Examine the actual visual element and provide REFINED dimensions.

The bounding box should:
1. Perfectly encompass the entire field (no clipping)
2. Not include extra whitespace
3. Match the visual element boundaries exactly
4. Be appropriate for the field type

Respond with JSON:
{{
    "refined_bbox": [x0, y0, x1, y1],
    "refined_dimensions": {{"width": px, "height": px}},
    "adjustment_needed": {{
        "left_edge": number_px,  // positive = move out, negative = move in
        "right_edge": number_px,
        "top_edge": number_px,
        "bottom_edge": number_px
    }},
    "confidence": 0.0-1.0,
    "reasoning": "Why these dimensions are optimal"
}}"""
        
        result = self.analyze_image(
            image_path=image_path,
            prompt=prompt,
            response_format="json_object"
        )
        
        if result.get('success'):
            result['field'] = field
        
        return result
    
    def batch_align_fields(
        self,
        image_path: Path,
        fields: List[Dict],
        alignment_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Provide alignment guidance for multiple fields at once.
        
        Args:
            image_path: Path to PDF page image
            fields: List of fields to align
            alignment_type: 'horizontal' | 'vertical' | 'grid' | 'auto'
        
        Returns:
            Alignment adjustments for all fields
        """
        fields_summary = []
        for i, f in enumerate(fields):
            bbox = f.get('bbox', f.get('bounding_box', []))
            fields_summary.append(
                f"{i+1}. {f.get('name', 'unnamed')}: {bbox}"
            )
        
        fields_text = "\n".join(fields_summary)
        
        prompt = f"""You are an alignment expert. Align {len(fields)} fields for professional appearance.

Alignment Type: {alignment_type}

Fields to align:
{fields_text}

Your task: Provide precise adjustment instructions to align these fields.

Consider:
1. Visual alignment axes
2. Consistent spacing
3. Grid patterns
4. Professional appearance

Respond with JSON:
{{
    "alignment_axis": {{"x": px, "y": px}},
    "adjustments": [
        {{
            "field_name": "name",
            "move_to": {{"x": px, "y": px}},
            "adjustment": {{"dx": px, "dy": px}}
        }}
    ],
    "confidence": 0.0-1.0,
    "reasoning": "Why this alignment is optimal"
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
    
    agent = PositionAdvisorAgent(api_key=api_key)
    
    # Example: Get optimal position
    test_image = Path("test_form_page.png")
    if test_image.exists():
        current_bbox = (100, 200, 300, 230)
        
        result = agent.get_optimal_position(
            test_image,
            current_bbox,
            field_type='text',
            label_text='Plaintiff Name'
        )
        
        if result.get('success'):
            print("Position Guidance:")
            print(f"  Optimal Bbox: {result['analysis'].get('optimal_bbox')}")
            print(f"  Adjustment: {result['analysis'].get('directional_guidance')}")
            print(f"  Confidence: {result['analysis'].get('confidence')}")
            print(f"  Cost: ${result['cost']:.4f}")
        else:
            print(f"Error: {result.get('error')}")
    else:
        print(f"Test image not found: {test_image}")
