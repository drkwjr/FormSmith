"""
Multi-Agent Field Detector

Orchestrates all LLM agents to enhance traditional field detection.
Provides intelligent fallbacks, consensus-based decisions, and learning capabilities.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from io import BytesIO

from PIL import Image, ImageDraw
import fitz  # PyMuPDF

from .field_spotter import FieldSpotterAgent
from .layout_analyst import LayoutAnalystAgent
from .validator import ValidatorAgent
from .referee import RefereeAgent
from .position_advisor import PositionAdvisorAgent
from .learning_agent import LearningAgent


class MultiAgentDetector:
    """
    Multi-Agent Field Detector
    
    Orchestrates traditional detection methods + LLM agents for intelligent field detection.
    
    Architecture:
    1. Traditional detection (OpenCV, Text, Pattern) runs first (free, fast)
    2. Layout Analyst provides structural context (once per page)
    3. Field Spotter validates ambiguous detections (targeted)
    4. Validator checks medium-confidence placements (targeted)
    5. Referee resolves disagreements (when needed)
    6. Position Advisor finds missed fields (if ground truth available)
    7. Learning Agent analyzes results and suggests improvements (learning mode)
    """
    
    def __init__(
        self,
        use_llm_agents: bool = True,
        llm_mode: str = "balanced",  # "conservative", "balanced", "aggressive"
        provider: str = "openai",
        api_key: Optional[str] = None,
        max_llm_calls: Optional[int] = None,
        learning_mode: bool = False,
        ground_truth_path: Optional[str] = None
    ):
        """
        Initialize multi-agent detector.
        
        Args:
            use_llm_agents: Whether to use LLM agents (vs traditional only)
            llm_mode: How aggressively to use LLMs
                - "conservative": Only on low confidence (<0.75)
                - "balanced": On medium confidence (0.70-0.85)
                - "aggressive": On all detections for validation
            provider: LLM provider ("openai" or "anthropic")
            api_key: API key for LLM provider
            max_llm_calls: Budget limit for LLM calls per form
            learning_mode: Enable learning agent analysis
            ground_truth_path: Path to ground truth JSON (for learning/validation)
        """
        self.use_llm_agents = use_llm_agents
        self.llm_mode = llm_mode
        self.learning_mode = learning_mode
        self.ground_truth_path = ground_truth_path
        self.max_llm_calls = max_llm_calls
        
        # Initialize agents (only if LLM enabled)
        self.agents = {}
        if use_llm_agents:
            self.agents['field_spotter'] = FieldSpotterAgent(provider=provider, api_key=api_key)
            self.agents['layout_analyst'] = LayoutAnalystAgent(provider=provider, api_key=api_key)
            self.agents['validator'] = ValidatorAgent(provider=provider, api_key=api_key)
            self.agents['referee'] = RefereeAgent(provider=provider, api_key=api_key)
            self.agents['position_advisor'] = PositionAdvisorAgent(provider=provider, api_key=api_key)
            
            if learning_mode:
                self.agents['learning_agent'] = LearningAgent(provider=provider, api_key=api_key)
        
        # Tracking
        self.llm_call_count = 0
        self.detection_log = []
    
    def enhance_detection(
        self,
        traditional_fields: List[Dict],
        page_image_path: str,
        labels: List[Dict] = None
    ) -> List[Dict]:
        """
        Enhance traditional detection with LLM agents.
        
        Args:
            traditional_fields: Fields detected by traditional methods
            page_image_path: Path to page image
            labels: Detected labels with positions
        
        Returns:
            Enhanced field list
        """
        if not self.use_llm_agents:
            return traditional_fields
        
        # Load page image
        with open(page_image_path, 'rb') as f:
            page_image_bytes = f.read()
        
        # Step 1: Get layout analysis (once per page)
        layout = None
        if self._should_call_agent('layout_analyst'):
            try:
                layout = self.agents['layout_analyst'].analyze_layout(
                    page_image_bytes,
                    labels=labels
                )
                self.llm_call_count += 1
                self.detection_log.append({
                    "agent": "layout_analyst",
                    "action": "analyze_layout",
                    "result": layout
                })
            except Exception as e:
                print(f"Warning: Layout analysis failed: {e}")
        
        # Step 2: Process each field based on confidence
        enhanced_fields = []
        needs_referee = []
        
        for field in traditional_fields:
            confidence = field.get('confidence', 0.5)
            
            # High confidence - accept as-is
            if confidence >= 0.85:
                enhanced_fields.append(field)
                continue
            
            # Low confidence - likely false positive, but check with LLM
            if confidence < 0.70:
                if self._should_call_agent('field_spotter'):
                    spotter_result = self._check_with_spotter(
                        page_image_bytes,
                        field,
                        layout
                    )
                    
                    if spotter_result and spotter_result.get('has_field'):
                        # LLM sees something traditional methods missed
                        if spotter_result['confidence'] > 0.85:
                            field['confidence'] = spotter_result['confidence']
                            field['llm_validated'] = True
                            enhanced_fields.append(field)
                        else:
                            # Send to referee
                            needs_referee.append({
                                'field': field,
                                'traditional_confidence': confidence,
                                'llm_report': spotter_result
                            })
                    # else: LLM agrees it's not a field, discard
                continue
            
            # Medium confidence (0.70-0.85) - validate with both agents
            if self._should_call_agent('field_spotter') or self._should_call_agent('validator'):
                spotter_result = self._check_with_spotter(
                    page_image_bytes,
                    field,
                    layout
                ) if self._should_call_agent('field_spotter') else None
                
                validator_result = self._check_with_validator(
                    page_image_bytes,
                    field,
                    enhanced_fields,
                    layout
                ) if self._should_call_agent('validator') else None
                
                # Both LLMs agree it's good
                if (spotter_result and spotter_result.get('has_field') and
                    validator_result and validator_result.get('is_correct')):
                    field['confidence'] = max(confidence, 0.85)
                    field['llm_validated'] = True
                    enhanced_fields.append(field)
                
                # Disagreement - send to referee
                elif (spotter_result or validator_result):
                    needs_referee.append({
                        'field': field,
                        'traditional_confidence': confidence,
                        'spotter_report': spotter_result,
                        'validator_report': validator_result
                    })
                else:
                    # No LLM validation available, use traditional confidence
                    enhanced_fields.append(field)
        
        # Step 3: Referee resolves disputes
        if needs_referee and self._should_call_agent('referee'):
            for case in needs_referee:
                traditional_reports = [
                    {"witness": "traditional", "confidence": case['traditional_confidence']}
                ]
                llm_reports = []
                
                if case.get('spotter_report'):
                    llm_reports.append(case['spotter_report'])
                if case.get('validator_report'):
                    llm_reports.append(case['validator_report'])
                
                try:
                    decision = self.agents['referee'].make_decision(
                        traditional_reports=traditional_reports,
                        llm_reports=llm_reports,
                        field_info=case['field']
                    )
                    self.llm_call_count += 1
                    
                    if decision['decision'] == 'accept':
                        case['field']['confidence'] = decision['final_confidence']
                        case['field']['referee_decision'] = decision
                        enhanced_fields.append(case['field'])
                    elif decision['decision'] == 'needs_manual_review':
                        case['field']['needs_human_review'] = True
                        enhanced_fields.append(case['field'])
                    # else: reject, don't add to enhanced_fields
                    
                except Exception as e:
                    print(f"Warning: Referee decision failed: {e}")
                    # Fallback: use traditional confidence
                    enhanced_fields.append(case['field'])
        
        return enhanced_fields
    
    def find_missed_fields(
        self,
        page_image_path: str,
        detected_fields: List[Dict],
        ground_truth: List[Dict]
    ) -> List[Dict]:
        """
        Use Position Advisor to find missed fields.
        
        Args:
            page_image_path: Path to page image
            detected_fields: Fields that were detected
            ground_truth: Ground truth fields
        
        Returns:
            Additional fields found
        """
        if not self._should_call_agent('position_advisor'):
            return []
        
        # Identify missed fields
        detected_labels = {f.get('label', '') for f in detected_fields}
        gt_labels = {f.get('label', '') for f in ground_truth}
        missed_labels = gt_labels - detected_labels
        
        if not missed_labels:
            return []  # Nothing missed!
        
        missed_fields = [f for f in ground_truth if f.get('label') in missed_labels]
        
        # Get advice from Position Advisor
        with open(page_image_path, 'rb') as f:
            page_image_bytes = f.read()
        
        try:
            advice = self.agents['position_advisor'].get_advice(
                page_image_bytes,
                detected_fields=detected_fields,
                missed_fields=missed_fields,
                ground_truth=ground_truth
            )
            self.llm_call_count += 1
            
            # TODO: Re-run detection on suggested regions
            # For now, just log the advice
            self.detection_log.append({
                "agent": "position_advisor",
                "action": "find_missed",
                "result": advice
            })
            
            return []  # Would return re-detected fields
        
        except Exception as e:
            print(f"Warning: Position advisor failed: {e}")
            return []
    
    def learn_from_results(
        self,
        detection_results: Dict,
        ground_truth: List[Dict]
    ) -> Dict[str, Any]:
        """
        Use Learning Agent to analyze results and suggest improvements.
        
        Args:
            detection_results: Full detection results
            ground_truth: Ground truth fields
        
        Returns:
            Learning insights dict
        """
        if not self.learning_mode or not self._should_call_agent('learning_agent'):
            return {}
        
        try:
            insights = self.agents['learning_agent'].analyze_and_learn(
                detection_results=detection_results,
                ground_truth=ground_truth,
                all_witness_reports=self.detection_log
            )
            self.llm_call_count += 1
            
            return insights
        
        except Exception as e:
            print(f"Warning: Learning analysis failed: {e}")
            return {}
    
    def _should_call_agent(self, agent_name: str) -> bool:
        """Check if we should call this agent based on budget and mode."""
        if not self.use_llm_agents:
            return False
        
        if agent_name not in self.agents:
            return False
        
        # Check budget
        if self.max_llm_calls and self.llm_call_count >= self.max_llm_calls:
            return False
        
        # Mode-based filtering
        if self.llm_mode == "conservative":
            # Only essential agents
            return agent_name in ['field_spotter', 'referee']
        elif self.llm_mode == "balanced":
            # Most agents except learning (unless explicitly enabled)
            if agent_name == 'learning_agent':
                return self.learning_mode
            return True
        else:  # aggressive
            return True
    
    def _check_with_spotter(
        self,
        page_image_bytes: bytes,
        field: Dict,
        layout: Dict = None
    ) -> Optional[Dict]:
        """Call Field Spotter agent on a field."""
        try:
            # Extract region around field
            crop_bytes = self._extract_field_region(page_image_bytes, field['bbox'])
            
            result = self.agents['field_spotter'].detect_field(
                crop_bytes,
                label=field.get('label'),
                context={"layout": layout} if layout else None
            )
            self.llm_call_count += 1
            
            self.detection_log.append({
                "agent": "field_spotter",
                "field": field,
                "result": result
            })
            
            return result
        
        except Exception as e:
            print(f"Warning: Field Spotter failed on {field.get('label', 'unknown')}: {e}")
            return None
    
    def _check_with_validator(
        self,
        page_image_bytes: bytes,
        field: Dict,
        nearby_fields: List[Dict],
        layout: Dict = None
    ) -> Optional[Dict]:
        """Call Validator agent on a field."""
        try:
            # Create image with field highlighted
            overlay_bytes = self._create_field_overlay(page_image_bytes, field['bbox'])
            
            result = self.agents['validator'].validate_field(
                overlay_bytes,
                field=field,
                label=field.get('label'),
                nearby_fields=nearby_fields[-5:] if nearby_fields else [],  # Last 5
                layout=layout
            )
            self.llm_call_count += 1
            
            self.detection_log.append({
                "agent": "validator",
                "field": field,
                "result": result
            })
            
            return result
        
        except Exception as e:
            print(f"Warning: Validator failed on {field.get('label', 'unknown')}: {e}")
            return None
    
    def _extract_field_region(
        self,
        page_image_bytes: bytes,
        bbox: List[int],
        padding: int = 50
    ) -> bytes:
        """Extract region around field with padding."""
        img = Image.open(BytesIO(page_image_bytes))
        
        x0, y0, x1, y1 = bbox
        crop_box = (
            max(0, x0 - padding),
            max(0, y0 - padding),
            min(img.width, x1 + padding),
            min(img.height, y1 + padding)
        )
        
        cropped = img.crop(crop_box)
        
        # Convert to bytes
        output = BytesIO()
        cropped.save(output, format='PNG')
        return output.getvalue()
    
    def _create_field_overlay(
        self,
        page_image_bytes: bytes,
        bbox: List[int]
    ) -> bytes:
        """Create image with field highlighted in green."""
        img = Image.open(BytesIO(page_image_bytes))
        draw = ImageDraw.Draw(img, 'RGBA')
        
        x0, y0, x1, y1 = bbox
        
        # Draw green semi-transparent rectangle
        draw.rectangle(
            [x0, y0, x1, y1],
            outline=(0, 255, 0, 255),
            width=3
        )
        draw.rectangle(
            [x0, y0, x1, y1],
            fill=(0, 255, 0, 50)
        )
        
        # Convert to bytes
        output = BytesIO()
        img.save(output, format='PNG')
        return output.getvalue()
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics for all agents."""
        stats = {
            "total_llm_calls": self.llm_call_count,
            "total_cost": 0.0,
            "agents": {}
        }
        
        for name, agent in self.agents.items():
            agent_stats = agent.get_stats()
            stats["agents"][name] = agent_stats
            stats["total_cost"] += agent_stats["total_cost"]
        
        stats["total_cost"] = round(stats["total_cost"], 4)
        
        return stats
    
    def get_detection_log(self) -> List[Dict]:
        """Get full detection log."""
        return self.detection_log.copy()


# Example usage
if __name__ == "__main__":
    import sys
    
    # Example: Initialize detector with LLM agents
    detector = MultiAgentDetector(
        use_llm_agents=True,
        llm_mode="balanced",
        provider="openai",
        learning_mode=True
    )
    
    # Example: Traditional fields from detection
    traditional_fields = [
        {"label": "First Name:", "bbox": [210, 150, 410, 165], "confidence": 0.72},
        {"label": "Last Name:", "bbox": [210, 200, 410, 215], "confidence": 0.88},
    ]
    
    # Enhance with LLM agents
    if len(sys.argv) >= 2:
        page_image_path = sys.argv[1]
        
        print("Enhancing detection with multi-agent system...")
        enhanced_fields = detector.enhance_detection(
            traditional_fields,
            page_image_path=page_image_path
        )
        
        print(f"\nResults:")
        print(f"  Input fields: {len(traditional_fields)}")
        print(f"  Enhanced fields: {len(enhanced_fields)}")
        print(f"\nAgent Stats:")
        print(json.dumps(detector.get_agent_stats(), indent=2))
    else:
        print("Usage: python multi_agent_detector.py <page_image_path>")
        print("\nThis will enhance detection using the multi-agent system.")

