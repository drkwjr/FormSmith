"""
Visual Field Detection and Precision Placement System
Uses gpt-4o-mini with vision capabilities for field-by-field precision
"""

import fitz  # PyMuPDF
from openai import OpenAI
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import base64
from io import BytesIO
from PIL import Image

class VisualFieldSystem:
    """Orchestrates visual field detection and precision placement"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-5-mini"
        
        # Tracking
        self.token_log = {
            "timestamp": datetime.now().isoformat(),
            "pdf_name": os.path.basename(pdf_path),
            "agents": {
                "detector": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0},
                "precision_placer": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0, "iterations_per_field": {}}
            },
            "total_cost_usd": 0,
            "fields_processed": 0,
            "fields_approved": 0
        }
        
    def _pdf_page_to_image(self, page_num: int = 0, dpi: int = 150) -> str:
        """Convert PDF page to base64 image for vision API"""
        doc = fitz.open(self.pdf_path)
        page = doc[page_num]
        
        # Render page to image
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        doc.close()
        return f"data:image/png;base64,{img_str}"
        
    def _extract_pdf_structure(self) -> Dict:
        """Extract text positions and visual elements for context"""
        doc = fitz.open(self.pdf_path)
        page = doc[0]
        page_rect = page.rect
        
        # Extract text with positions
        text_blocks = []
        for block in page.get_text("dict")["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text_blocks.append({
                            "text": span["text"],
                            "bbox": span["bbox"],  # (x0, y0, x1, y1)
                            "size": span["size"]
                        })
        
        # Extract drawings (lines, rectangles)
        drawings = []
        for drawing in page.get_drawings():
            drawings.append({
                "type": drawing["type"],
                "rect": drawing.get("rect"),
                "items": drawing.get("items", [])
            })
        
        page_size = (page_rect.width, page_rect.height)
        doc.close()
        
        return {
            "text_blocks": text_blocks,
            "drawings": drawings,
            "page_size": page_size
        }
    
    def _call_vision_api(self, agent_name: str, prompt: str, image_base64: str, response_format: str = "json_object") -> Dict:
        """Call GPT-4o-mini with vision"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_base64, "detail": "high"}}
                ]
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": response_format}
        )
        
        # Track tokens
        usage = response.usage
        self.token_log["agents"][agent_name]["calls"] += 1
        self.token_log["agents"][agent_name]["input_tokens"] += usage.prompt_tokens
        self.token_log["agents"][agent_name]["output_tokens"] += usage.completion_tokens
        
        # Calculate cost (gpt-5-mini pricing - using gpt-4o-mini as estimate: $0.15/1M input, $0.60/1M output)
        cost = (usage.prompt_tokens * 0.15 / 1_000_000) + (usage.completion_tokens * 0.60 / 1_000_000)
        self.token_log["agents"][agent_name]["cost_usd"] += cost
        self.token_log["total_cost_usd"] += cost
        
        return json.loads(response.choices[0].message.content)
    
    def detect_fields(self) -> List[Dict]:
        """Agent 1: Detect all fields that visually exist on the form"""
        print("\n=== AGENT 1: FIELD DETECTOR ===")
        
        # Get visual and structural data
        image = self._pdf_page_to_image()
        structure = self._extract_pdf_structure()
        
        prompt = f"""You are a precise form field detector. Analyze this legal court form and identify ALL form fields that need to be filled out.

For each field you detect, provide:
1. field_name: A descriptive variable name (lowercase_with_underscores)
2. field_type: One of [text, date, checkbox, signature, address, email, phone]
3. label: The text label near the field
4. bbox: Estimated bounding box [x0, y0, x1, y1] in PDF coordinates
5. visual_cue: What indicates this is a field (e.g., "underline", "checkbox symbol", "signature line")

IMPORTANT RULES:
- Only detect fields that have CLEAR visual indicators (underlines, checkboxes, signature lines, etc.)
- Do NOT hallucinate fields that don't visually exist
- Be conservative - if you're not sure a field exists, don't include it
- Pay attention to field labels and their proximity to visual cues

PDF Structure Context:
- Page size: {structure['page_size']}
- Number of text blocks: {len(structure['text_blocks'])}
- Number of visual elements: {len(structure['drawings'])}

Return JSON format:
{{
    "fields": [
        {{"field_name": "...", "field_type": "...", "label": "...", "bbox": [...], "visual_cue": "..."}}
    ],
    "confidence": "high|medium|low",
    "notes": "Any observations about the form structure"
}}
"""
        
        result = self._call_vision_api("detector", prompt, image)
        
        print(f"✓ Detected {len(result['fields'])} fields")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Tokens used: {self.token_log['agents']['detector']['input_tokens']} in, {self.token_log['agents']['detector']['output_tokens']} out")
        
        return result['fields']
    
    def refine_field_placement(self, field: Dict, current_bbox: Tuple[float, float, float, float], iteration: int = 1, max_iterations: int = 10) -> Tuple[Optional[Tuple], str, Dict]:
        """Agent 2: Iteratively refine field placement until perfect
        
        Returns: (new_bbox or None, status, quality_score)
        """
        field_name = field['field_name']
        
        if iteration == 1:
            print(f"\n--- Refining: {field_name} ({field['field_type']}) ---")
        
        # Get current state image
        image = self._pdf_page_to_image()
        
        prompt = f"""You are a STRICT quality assurance validator for legal form fields. Your user will be "hella disappointed" if fields are misaligned, incomplete, or look unprofessional.

Field Information:
- Name: {field_name}
- Type: {field['field_type']}
- Label: {field['label']}
- Visual Cue: {field['visual_cue']}
- Current Position: x={current_bbox[0]:.1f}, y={current_bbox[1]:.1f}, width={current_bbox[2]-current_bbox[0]:.1f}, height={current_bbox[3]-current_bbox[1]:.1f}
- Iteration: {iteration}/{max_iterations}

QUALITY RUBRIC (Score 0-100 for each):
1. **Completeness** (0-100): Is this field actually needed? Is it detecting a real field on the form?
   - 100: Field clearly corresponds to a fillable area on the form
   - 0: Field doesn't belong here, no visual indicator exists

2. **Location Accuracy** (0-100): Is the field EXACTLY where it should be?
   - 100: Perfectly aligned with underline/box/visual cue (within 1-2 pixels)
   - 80-99: Very close but minor misalignment visible
   - 60-79: Noticeable misalignment but in right general area
   - 0-59: Significantly off, wrong position

3. **Size Appropriateness** (0-100): Is the field the right dimensions?
   - 100: Perfect width and height for the visual indicator
   - 80-99: Slightly too wide/narrow/tall/short
   - 0-79: Significantly wrong size

4. **Visual Integration** (0-100): Does it look professional and natural?
   - 100: Looks like it belongs, doesn't disrupt form aesthetics
   - 80-99: Minor visual awkwardness
   - 0-79: Looks out of place, intrusive, or amateur

5. **Overall Quality** (0-100): Would the user be disappointed?
   - 95-100: Production-ready, user will be thrilled
   - 85-94: Good but could be better
   - 70-84: Acceptable but user might notice issues
   - 0-69: User will be disappointed, needs work

RESPONSE FORMAT:
{{
    "status": "APPROVED" or "ADJUST" or "REJECT",
    "quality_scores": {{
        "completeness": <0-100>,
        "location_accuracy": <0-100>,
        "size_appropriateness": <0-100>,
        "visual_integration": <0-100>,
        "overall": <0-100>
    }},
    "reasoning": "Detailed explanation of scores",
    "adjustments": {{
        "x_offset": <pixels, negative for left>,
        "y_offset": <pixels, negative for up>,
        "width_adjust": <pixels, negative to reduce>,
        "height_adjust": <pixels, negative to reduce>
    }} // Only if status is ADJUST
}}

DECISION RULES:
- status = "APPROVED" if overall >= 95 AND all other scores >= 90
- status = "REJECT" if completeness < 70 (field shouldn't exist)
- status = "ADJUST" otherwise

Be RUTHLESSLY HONEST. User wants pixel-perfect accuracy.
"""
        
        result = self._call_vision_api("precision_placer", prompt, image)
        
        # Track iterations
        if field_name not in self.token_log["agents"]["precision_placer"]["iterations_per_field"]:
            self.token_log["agents"]["precision_placer"]["iterations_per_field"][field_name] = {
                "iterations": 0,
                "quality_scores": []
            }
        
        self.token_log["agents"]["precision_placer"]["iterations_per_field"][field_name]["iterations"] += 1
        self.token_log["agents"]["precision_placer"]["iterations_per_field"][field_name]["quality_scores"].append(result["quality_scores"])
        
        status = result["status"]
        scores = result["quality_scores"]
        
        # Print quality scores
        print(f"  Quality Scores (Iteration {iteration}):")
        print(f"    Completeness: {scores['completeness']}/100")
        print(f"    Location Accuracy: {scores['location_accuracy']}/100")
        print(f"    Size: {scores['size_appropriateness']}/100")
        print(f"    Visual Integration: {scores['visual_integration']}/100")
        print(f"    OVERALL: {scores['overall']}/100")
        print(f"  Reasoning: {result['reasoning']}")
        
        if status == "APPROVED":
            print(f"  ✓ APPROVED after {iteration} iteration(s) - Production ready!")
            self.token_log["fields_approved"] += 1
            return None, "APPROVED", scores
        
        elif status == "REJECT":
            print(f"  ✗ REJECTED - Field should not exist (completeness: {scores['completeness']}/100)")
            return None, "REJECT", scores
        
        elif status == "ADJUST":
            if iteration >= max_iterations:
                print(f"  ⚠ Max iterations reached ({max_iterations}), accepting current state")
                return None, "MAX_ITERATIONS", scores
            
            adj = result["adjustments"]
            print(f"  → Adjusting: x{adj['x_offset']:+.1f} y{adj['y_offset']:+.1f} w{adj['width_adjust']:+.1f} h{adj['height_adjust']:+.1f}")
            
            # Calculate new bbox
            new_bbox = (
                current_bbox[0] + adj['x_offset'],
                current_bbox[1] + adj['y_offset'],
                current_bbox[2] + adj['x_offset'] + adj['width_adjust'],
                current_bbox[3] + adj['y_offset'] + adj['height_adjust']
            )
            
            return new_bbox, "ADJUST", scores
        
        return None, "UNKNOWN", scores
    
    def save_log(self, output_path: str = "visual_field_log.json"):
        """Save comprehensive token usage log"""
        with open(output_path, 'w') as f:
            json.dump(self.token_log, f, indent=2)
        
        print(f"\n=== TOKEN USAGE SUMMARY ===")
        print(f"Total Cost: ${self.token_log['total_cost_usd']:.4f}")
        print(f"Fields Processed: {self.token_log['fields_processed']}")
        print(f"Fields Approved: {self.token_log['fields_approved']}")
        print(f"\nDetector: {self.token_log['agents']['detector']['calls']} calls, ${self.token_log['agents']['detector']['cost_usd']:.4f}")
        print(f"Precision Placer: {self.token_log['agents']['precision_placer']['calls']} calls, ${self.token_log['agents']['precision_placer']['cost_usd']:.4f}")
        print(f"\nLog saved to: {output_path}")

