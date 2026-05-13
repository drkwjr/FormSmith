"""
Multi-Agent Field Detector

Orchestrates all detection witnesses (traditional + LLM agents) for robust field detection.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import fitz  # PyMuPDF
from PIL import Image
import io
import os

# Import agents
from .agents.field_spotter import FieldSpotterAgent
from .agents.layout_analyst import LayoutAnalystAgent
from .agents.validator import ValidatorAgent
from .agents.referee import RefereeAgent
from .agents.position_advisor import PositionAdvisorAgent
from .agents.learning_agent import LearningAgent

# Import traditional detection methods
from .learned_field_detector import LearnedFieldDetector
from .pattern_learner import PatternLearner


class MultiAgentDetector:
    """
    Multi-Agent Field Detection System
    
    Combines traditional detection methods (OpenCV, Text, Pattern) with
    specialized LLM agents for robust, accurate field detection.
    
    Architecture:
    1. Traditional Detection (fast, free)
    2. LLM Enhancement (targeted, cost-optimized)
    3. Referee Decision (consensus-based)
    4. Learning Analysis (continuous improvement)
    """
    
    def __init__(
        self,
        patterns_path: str = None,
        use_llm_agents: bool = True,
        llm_mode: str = "balanced",  # "conservative", "balanced", "aggressive"
        provider: str = "openai",
        api_key: Optional[str] = None,
        ground_truth_path: Optional[str] = None,
        learning_mode: bool = False
    ):
        """
        Initialize multi-agent detector.
        
        Args:
            patterns_path: Path to learned patterns JSON
            use_llm_agents: Whether to use LLM agents
            llm_mode: How aggressively to use LLM agents
                - "conservative": Only on low confidence (<0.70)
                - "balanced": On medium confidence (0.60-0.85)
                - "aggressive": On all detections
            provider: LLM provider ("openai" or "anthropic")
            api_key: API key for LLM provider
            ground_truth_path: Path to ground truth for learning
            learning_mode: Whether to enable learning agent
        """
        # Traditional detector
        self.traditional_detector = LearnedFieldDetector(patterns_path)
        
        # LLM settings
        self.use_llm_agents = use_llm_agents
        self.llm_mode = llm_mode
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Ground truth and learning
        self.ground_truth_path = ground_truth_path
        self.learning_mode = learning_mode
        
        # Initialize agents (lazy loading)
        self._field_spotter = None
        self._layout_analyst = None
        self._validator = None
        self._referee = None
        self._position_advisor = None
        self._learning_agent = None
        
        # Statistics
        self.stats = {
            "total_calls": 0,
            "traditional_only": 0,
            "llm_enhanced": 0,
            "referee_decisions": 0,
            "total_cost": 0.0,
            "agent_calls": {}
        }
    
    # Lazy agent initialization
    @property
    def field_spotter(self) -> FieldSpotterAgent:
        if self._field_spotter is None:
            self._field_spotter = FieldSpotterAgent(
                provider=self.provider,
                model="gpt-4o-mini",
                api_key=self.api_key
            )
        return self._field_spotter
    
    @property
    def layout_analyst(self) -> LayoutAnalystAgent:
        if self._layout_analyst is None:
            self._layout_analyst = LayoutAnalystAgent(
                provider=self.provider,
                model="gpt-4o-mini",
                api_key=self.api_key
            )
        return self._layout_analyst
    
    @property
    def validator(self) -> ValidatorAgent:
        if self._validator is None:
            self._validator = ValidatorAgent(
                provider=self.provider,
                model="gpt-4o-mini",
                api_key=self.api_key
            )
        return self._validator
    
    @property
    def referee(self) -> RefereeAgent:
        if self._referee is None:
            self._referee = RefereeAgent(
                provider=self.provider,
                model="gpt-4o-mini",
                api_key=self.api_key
            )
        return self._referee
    
    @property
    def position_advisor(self) -> PositionAdvisorAgent:
        if self._position_advisor is None:
            self._position_advisor = PositionAdvisorAgent(
                provider=self.provider,
                model="gpt-4o-mini",
                api_key=self.api_key
            )
        return self._position_advisor
    
    @property
    def learning_agent(self) -> LearningAgent:
        if self._learning_agent is None:
            self._learning_agent = LearningAgent(
                provider=self.provider,
                model="gpt-4o-mini",
                api_key=self.api_key
            )
        return self._learning_agent
    
    def detect(
        self,
        pdf_path: str,
        page_num: int = 0,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect fields using multi-agent system.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number to process
            output_path: Optional path to save results
        
        Returns:
            Detection results with fields and metadata
        """
        self.stats["total_calls"] += 1
        
        print(f"\n{'='*60}")
        print(f"Multi-Agent Field Detection")
        print(f"{'='*60}")
        print(f"PDF: {Path(pdf_path).name}")
        print(f"Page: {page_num}")
        print(f"LLM Mode: {'Enabled (' + self.llm_mode + ')' if self.use_llm_agents else 'Disabled'}")
        print(f"{'='*60}\n")
        
        # PHASE 1: Traditional Detection
        print("🔍 PHASE 1: Traditional Detection (OpenCV + Text + Patterns)")
        traditional_result = self.traditional_detector.detect(pdf_path)
        traditional_fields = traditional_result.fields
        
        print(f"   ✓ Detected {len(traditional_fields)} fields using traditional methods")
        
        # If LLM agents disabled, return traditional results
        if not self.use_llm_agents:
            self.stats["traditional_only"] += 1
            result = {
                "fields": traditional_fields,
                "method": "traditional_only",
                "stats": self.stats.copy()
            }
            
            if output_path:
                self._save_results(result, output_path)
            
            return result
        
        # PHASE 2: LLM Enhancement
        print(f"\n🤖 PHASE 2: LLM Enhancement ({self.llm_mode} mode)")
        
        # Render page to image
        page_image = self._render_page_to_image(pdf_path, page_num)
        
        # Get layout analysis (once per page)
        print("   📐 Layout Analyst: analyzing form structure...")
        labels = [f.interview_label for f in traditional_fields if hasattr(f, 'interview_label') and f.interview_label]
        layout_info = self.layout_analyst.analyze_layout(page_image, labels)
        print(f"   ✓ Layout type: {layout_info.get('layout_type', 'unknown')}")
        print(f"     Columns: {len(layout_info.get('columns', []))}")
        print(f"     Sections: {len(layout_info.get('sections', []))}")
        
        self._track_agent_call('layout_analyst', layout_info.get('_metadata', {}))
        
        # Process each field
        enhanced_fields = []
        needs_referee = []
        
        for field in traditional_fields:
            confidence = getattr(field, 'confidence', 0.5)
            
            # Determine if we should call LLM agents based on mode
            should_enhance = self._should_enhance(confidence)
            
            if not should_enhance:
                # High confidence - trust traditional methods
                enhanced_fields.append(field)
                continue
            
            # Medium/Low confidence - get LLM opinions
            print(f"\n   👁️  Enhancing: {getattr(field, 'interview_label', 'unlabeled')} (conf: {confidence:.2f})")
            
            # Extract region around field
            crop = self._extract_region(pdf_path, page_num, field.bbox, padding=50)
            
            # Field Spotter analysis
            spotter_result = self.field_spotter.detect_field(
                crop,
                label=getattr(field, 'interview_label', None),
                context={'page_layout': layout_info}
            )
            
            print(f"      Field Spotter: {spotter_result.get('has_field', False)} (conf: {spotter_result.get('confidence', 0):.2f})")
            self._track_agent_call('field_spotter', spotter_result.get('_metadata', {}))
            
            # Validator analysis (for medium confidence)
            if 0.6 <= confidence <= 0.85:
                # Create overlay image for validator
                overlay_image = self._create_overlay_image(pdf_path, page_num, [field])
                
                validator_result = self.validator.validate_field(
                    overlay_image,
                    field_label=getattr(field, 'interview_label', ''),
                    field_bbox=field.bbox,
                    field_type=getattr(field, 'interview_type', 'text'),
                    confidence=confidence,
                    context={'layout_info': layout_info}
                )
                
                print(f"      Validator: {validator_result.get('is_correct', False)} (conf: {validator_result.get('confidence', 0):.2f})")
                self._track_agent_call('validator', validator_result.get('_metadata', {}))
            else:
                validator_result = None
            
            # Decide based on LLM feedback
            if spotter_result.get('has_field') and (validator_result is None or validator_result.get('is_correct')):
                # LLM agents agree - boost confidence
                setattr(field, 'confidence', max(confidence, 0.85))
                enhanced_fields.append(field)
                print(f"      ✓ Accepted (boosted confidence)")
            else:
                # Disagreement - needs referee
                needs_referee.append({
                    'field': field,
                    'spotter': spotter_result,
                    'validator': validator_result
                })
                print(f"      ⚖️  Needs referee decision")
        
        # PHASE 3: Referee Decisions
        if needs_referee:
            print(f"\n⚖️  PHASE 3: Referee ({len(needs_referee)} disputed cases)")
            
            for case in needs_referee:
                field = case['field']
                
                # Prepare witness reports
                witness_reports = [
                    {
                        'witness': 'Traditional Detection',
                        'decision': 'accept',
                        'confidence': getattr(field, 'confidence', 0.5),
                        'details': {'method': 'opencv+text+pattern'}
                    },
                    {
                        'witness': 'Field Spotter Agent',
                        'decision': 'accept' if case['spotter'].get('has_field') else 'reject',
                        'confidence': case['spotter'].get('confidence', 0),
                        'details': case['spotter']
                    }
                ]
                
                if case['validator']:
                    witness_reports.append({
                        'witness': 'Validator Agent',
                        'decision': 'accept' if case['validator'].get('is_correct') else 'reject',
                        'confidence': case['validator'].get('confidence', 0),
                        'details': case['validator']
                    })
                
                # Get referee decision
                field_info = {
                    'label': getattr(field, 'interview_label', ''),
                    'bbox': field.bbox,
                    'type': getattr(field, 'interview_type', 'text')
                }
                
                decision = self.referee.make_decision(field_info, witness_reports)
                
                print(f"   {getattr(field, 'interview_label', 'unlabeled')}: {decision['decision']} (conf: {decision['final_confidence']:.2f})")
                self._track_agent_call('referee', decision.get('_metadata', {}))
                self.stats["referee_decisions"] += 1
                
                if decision['decision'] == 'accept':
                    setattr(field, 'confidence', decision['final_confidence'])
                    enhanced_fields.append(field)
                elif decision['decision'] == 'needs_manual_review':
                    setattr(field, 'needs_review', True)
                    setattr(field, 'confidence', decision['final_confidence'])
                    enhanced_fields.append(field)
                # else: reject (don't add to enhanced_fields)
        
        print(f"\n✓ Enhanced detection complete: {len(enhanced_fields)} fields")
        self.stats["llm_enhanced"] += 1
        
        # PHASE 4: Learning (if enabled and ground truth available)
        learning_insights = None
        if self.learning_mode and self.ground_truth_path:
            print(f"\n🧠 PHASE 4: Learning Analysis")
            learning_insights = self._run_learning_analysis(
                enhanced_fields,
                traditional_fields
            )
            print(f"   Grade: {learning_insights.get('overall_assessment', {}).get('detection_quality', 'N/A')}")
        
        # Prepare result
        result = {
            "fields": enhanced_fields,
            "method": "multi_agent_enhanced",
            "layout_info": layout_info,
            "learning_insights": learning_insights,
            "stats": self.stats.copy()
        }
        
        if output_path:
            self._save_results(result, output_path)
        
        # Print cost summary
        print(f"\n💰 Cost Summary:")
        for agent_name, calls in self.stats["agent_calls"].items():
            print(f"   {agent_name}: {calls['count']} calls, ${calls['cost']:.4f}")
        print(f"   TOTAL: ${self.stats['total_cost']:.4f}")
        
        return result
    
    def _should_enhance(self, confidence: float) -> bool:
        """Determine if field should be enhanced with LLM agents."""
        if self.llm_mode == "conservative":
            return confidence < 0.70
        elif self.llm_mode == "balanced":
            return confidence < 0.85
        elif self.llm_mode == "aggressive":
            return True
        return False
    
    def _render_page_to_image(self, pdf_path: str, page_num: int) -> bytes:
        """Render PDF page to image bytes."""
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Render at 150 DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
        
        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
        
        doc.close()
        return img_bytes
    
    def _extract_region(
        self,
        pdf_path: str,
        page_num: int,
        bbox: List[int],
        padding: int = 50
    ) -> bytes:
        """Extract image region around bbox."""
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Expand bbox with padding
        x0, y0, x1, y1 = bbox
        x0 = max(0, x0 - padding)
        y0 = max(0, y0 - padding)
        x1 = min(page.rect.width, x1 + padding)
        y1 = min(page.rect.height, y1 + padding)
        
        # Render full page
        pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        
        # Crop to region (adjust coordinates for 150 DPI)
        scale = 150 / 72
        crop_bbox = (
            int(x0 * scale),
            int(y0 * scale),
            int(x1 * scale),
            int(y1 * scale)
        )
        cropped = img.crop(crop_bbox)
        
        # Convert back to bytes
        output = io.BytesIO()
        cropped.save(output, format='PNG')
        
        doc.close()
        return output.getvalue()
    
    def _create_overlay_image(
        self,
        pdf_path: str,
        page_num: int,
        fields: List[Any]
    ) -> bytes:
        """Create image with fields overlaid in green."""
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Render page
        pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        
        # Draw fields
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img, 'RGBA')
        
        scale = 150 / 72
        for field in fields:
            x0, y0, x1, y1 = field.bbox
            scaled_bbox = (
                int(x0 * scale),
                int(y0 * scale),
                int(x1 * scale),
                int(y1 * scale)
            )
            # Green box with transparency
            draw.rectangle(scaled_bbox, outline=(0, 255, 0, 255), width=2)
            draw.rectangle(scaled_bbox, fill=(0, 255, 0, 50))
        
        # Convert to bytes
        output = io.BytesIO()
        img.save(output, format='PNG')
        
        doc.close()
        return output.getvalue()
    
    def _track_agent_call(self, agent_name: str, metadata: Dict):
        """Track agent call statistics."""
        if agent_name not in self.stats["agent_calls"]:
            self.stats["agent_calls"][agent_name] = {"count": 0, "cost": 0.0}
        
        self.stats["agent_calls"][agent_name]["count"] += 1
        cost = metadata.get('cost', 0.0)
        self.stats["agent_calls"][agent_name]["cost"] += cost
        self.stats["total_cost"] += cost
    
    def _run_learning_analysis(
        self,
        enhanced_fields: List,
        traditional_fields: List
    ) -> Dict[str, Any]:
        """Run learning agent analysis."""
        # Load ground truth
        with open(self.ground_truth_path, 'r') as f:
            ground_truth = json.load(f)
        
        # Prepare detection results
        detection_results = {
            'fields': [
                {
                    'label': getattr(f, 'interview_label', ''),
                    'bbox': f.bbox,
                    'type': getattr(f, 'interview_type', 'text'),
                    'confidence': getattr(f, 'confidence', 0.5)
                }
                for f in enhanced_fields
            ],
            'true_positives': 0,  # TODO: Calculate
            'false_positives': 0,  # TODO: Calculate
            'false_negatives': 0   # TODO: Calculate
        }
        
        # Get current thresholds
        current_thresholds = {
            'confidence_threshold': 0.80,
            'min_width': 25,
            'spatial_tolerance': 1.5
        }
        
        # Run learning analysis
        insights = self.learning_agent.analyze_and_learn(
            detection_results,
            ground_truth,
            current_thresholds=current_thresholds
        )
        
        self._track_agent_call('learning_agent', insights.get('_metadata', {}))
        
        return insights
    
    def _save_results(self, result: Dict, output_path: str):
        """Save results to JSON file."""
        # Convert fields to dicts for JSON serialization
        if 'fields' in result:
            result['fields'] = [
                {
                    'label': getattr(f, 'interview_label', ''),
                    'bbox': f.bbox,
                    'type': getattr(f, 'interview_type', 'text'),
                    'confidence': getattr(f, 'confidence', 0.5)
                }
                for f in result['fields']
            ]
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_calls": 0,
            "traditional_only": 0,
            "llm_enhanced": 0,
            "referee_decisions": 0,
            "total_cost": 0.0,
            "agent_calls": {}
        }


# CLI interface
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent Field Detector")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--patterns", help="Path to learned patterns JSON")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM agents")
    parser.add_argument("--mode", choices=["conservative", "balanced", "aggressive"],
                       default="balanced", help="LLM usage mode")
    parser.add_argument("--provider", choices=["openai", "anthropic"],
                       default="openai", help="LLM provider")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--ground-truth", help="Ground truth JSON for learning")
    parser.add_argument("--learning", action="store_true", help="Enable learning mode")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = MultiAgentDetector(
        patterns_path=args.patterns,
        use_llm_agents=not args.no_llm,
        llm_mode=args.mode,
        provider=args.provider,
        ground_truth_path=args.ground_truth,
        learning_mode=args.learning
    )
    
    # Run detection
    result = detector.detect(
        args.pdf_path,
        output_path=args.output
    )
    
    print(f"\n{'='*60}")
    print(f"Detection Complete")
    print(f"{'='*60}")
    print(f"Fields detected: {len(result['fields'])}")
    print(f"Method: {result['method']}")
    print(f"Total cost: ${result['stats']['total_cost']:.4f}")
