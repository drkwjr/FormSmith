"""
Learning Agent - Analyzes performance and suggests system improvements.

This agent learns from successes and failures to continuously improve the
detection system over time.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import logging

from .base_agent import VisionAgent

logger = logging.getLogger(__name__)


class LearningAgent(VisionAgent):
    """
    Learns from detection results to improve the system.
    
    This agent is specialized in:
    - Analyzing detection successes and failures
    - Identifying systematic errors or patterns
    - Suggesting parameter tuning
    - Recommending algorithm improvements
    - Building institutional knowledge
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-2024-08-06"):
        super().__init__(api_key=api_key, model=model, agent_name="learning_agent")
        self.learning_history: List[Dict] = []
    
    def analyze_detection_session(
        self,
        session_data: Dict,
        ground_truth: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a complete detection session to extract learnings.
        
        Args:
            session_data: Dict containing:
                - detected_fields: List of fields detected
                - agent_decisions: List of agent decisions made
                - execution_report: Report from action executor
                - metadata: Form type, difficulty, etc.
            ground_truth: Optional ground truth for accuracy measurement
        
        Returns:
            Learning insights and improvement suggestions
        """
        detected_count = len(session_data.get('detected_fields', []))
        agent_decisions = session_data.get('agent_decisions', [])
        execution_report = session_data.get('execution_report', {})
        
        gt_count = len(ground_truth) if ground_truth else "unknown"
        
        # Prepare summary
        summary = f"""Detection Session Analysis:
- Fields Detected: {detected_count}
- Ground Truth Count: {gt_count}
- Agent Decisions Made: {len(agent_decisions)}
- Actions Executed: {execution_report.get('actions_executed', 0)}
- Fields Created: {execution_report.get('fields_created', 0)}
- Fields Modified: {execution_report.get('fields_modified', 0)}
- Fields Deleted: {execution_report.get('fields_deleted', 0)}
- Errors: {len(execution_report.get('errors', []))}
"""
        
        # Accuracy metrics if ground truth available
        accuracy_str = ""
        if ground_truth:
            accuracy_str = f"\nAccuracy Metrics Available: {len(ground_truth)} ground truth fields"
        
        prompt = f"""You are a machine learning expert analyzing detection system performance.

{summary}{accuracy_str}

Your task: Extract ACTIONABLE INSIGHTS to improve the detection system.

Analyze:
1. **What Went Well**: Patterns of successful detection
2. **What Went Wrong**: Systematic errors or failures
3. **Agent Performance**: Which agents were most/least effective?
4. **Parameter Tuning**: Should confidence thresholds be adjusted?
5. **Algorithm Improvements**: Suggestions for better detection logic
6. **Pattern Recognition**: Recurring issues that need addressing

Provide CONCRETE, IMPLEMENTABLE recommendations.

Respond with JSON:
{{
    "session_quality": "excellent | good | fair | poor",
    "key_successes": [
        {{
            "pattern": "description of what worked",
            "frequency": number,
            "why_it_worked": "explanation"
        }}
    ],
    "key_failures": [
        {{
            "pattern": "description of what failed",
            "frequency": number,
            "why_it_failed": "explanation",
            "suggested_fix": "how to prevent this"
        }}
    ],
    "parameter_recommendations": [
        {{
            "parameter": "confidence_threshold | spatial_tolerance | etc",
            "current_value": number,
            "recommended_value": number,
            "reasoning": "why this change would help"
        }}
    ],
    "algorithm_improvements": [
        {{
            "component": "text_extraction | visual_detection | spatial_matching | etc",
            "issue": "what's not working well",
            "suggestion": "how to improve it",
            "priority": "high | medium | low"
        }}
    ],
    "agent_performance": {{
        "field_spotter": {{"effectiveness": 0.0-1.0, "notes": "..."}},
        "layout_analyst": {{"effectiveness": 0.0-1.0, "notes": "..."}},
        "validator": {{"effectiveness": 0.0-1.0, "notes": "..."}}
    }},
    "overall_recommendations": [
        "High-level recommendations for system improvement"
    ]
}}"""
        
        # This is a text-only analysis, not image-based
        result = self._call_llm(prompt, response_format="json_object")
        
        # Store learning
        self.learning_history.append({
            'session_data': session_data,
            'insights': result.get('analysis'),
            'timestamp': Path.cwd().name  # Placeholder
        })
        
        return result
    
    def compare_forms(
        self,
        form_results: List[Dict]
    ) -> Dict[str, Any]:
        """
        Compare detection results across multiple forms to identify patterns.
        
        Args:
            form_results: List of detection results from different forms
        
        Returns:
            Cross-form insights and patterns
        """
        # Prepare summary
        summary_parts = []
        for i, result in enumerate(form_results[:5]):  # Limit to 5 forms
            summary_parts.append(
                f"Form {i+1}: {result.get('detected_count', 0)} fields, "
                f"quality={result.get('quality_grade', 'unknown')}"
            )
        
        summary = "\n".join(summary_parts)
        if len(form_results) > 5:
            summary += f"\n... and {len(form_results) - 5} more forms"
        
        prompt = f"""You are analyzing detection performance across {len(form_results)} different forms.

Form Results Summary:
{summary}

Your task: Identify PATTERNS across multiple forms.

Analyze:
1. **Consistent Strengths**: What detection features work well across all forms?
2. **Consistent Weaknesses**: What fails repeatedly?
3. **Form-Specific Issues**: Do certain form types have unique challenges?
4. **Generalization**: Is the system generalizing well to new forms?
5. **Improvement Trajectory**: Is performance improving over time?

Respond with JSON:
{{
    "cross_form_patterns": {{
        "strengths": ["pattern1", "pattern2"],
        "weaknesses": ["pattern1", "pattern2"]
    }},
    "form_type_insights": [
        {{
            "form_type": "simple | complex | table-heavy | etc",
            "performance": "good | fair | poor",
            "specific_challenges": ["challenge1", "challenge2"]
        }}
    ],
    "generalization_assessment": {{
        "is_generalizing_well": true/false,
        "evidence": "explanation",
        "recommendation": "how to improve generalization"
    }},
    "priority_improvements": [
        {{
            "improvement": "description",
            "impact": "high | medium | low",
            "effort": "high | medium | low",
            "rationale": "why this is important"
        }}
    ]
}}"""
        
        result = self._call_llm(prompt, response_format="json_object")
        
        return result
    
    def suggest_training_data(
        self,
        current_performance: Dict,
        weak_areas: List[str]
    ) -> Dict[str, Any]:
        """
        Suggest what kind of training data would most improve the system.
        
        Args:
            current_performance: Current system performance metrics
            weak_areas: List of identified weak areas
        
        Returns:
            Recommendations for training data collection
        """
        perf_summary = json.dumps(current_performance, indent=2)
        weak_summary = "\n".join(f"- {area}" for area in weak_areas)
        
        prompt = f"""You are a machine learning expert recommending training data collection.

Current Performance:
{perf_summary}

Identified Weak Areas:
{weak_summary}

Your task: Recommend SPECIFIC training data that would most improve the system.

Consider:
1. What types of examples are missing?
2. What edge cases need more coverage?
3. What field types are under-represented?
4. What form layouts are challenging?

Respond with JSON:
{{
    "priority_data_needs": [
        {{
            "data_type": "description of needed data",
            "quantity_needed": number,
            "why_needed": "explanation",
            "expected_impact": "high | medium | low"
        }}
    ],
    "collection_strategy": [
        "Step-by-step guide for collecting this data"
    ],
    "labeling_guidance": [
        "How to label/annotate the training data"
    ]
}}"""
        
        result = self._call_llm(prompt, response_format="json_object")
        
        return result
    
    def save_learnings(self, output_path: Path):
        """Save accumulated learnings to disk."""
        with open(output_path, 'w') as f:
            json.dump({
                'learning_sessions': len(self.learning_history),
                'history': self.learning_history
            }, f, indent=2)
        
        logger.info(f"Saved {len(self.learning_history)} learning sessions to {output_path}")
    
    def _call_llm(self, prompt: str, response_format: str = "json_object") -> Dict[str, Any]:
        """Call LLM for text-only analysis (no image)."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": response_format},
                temperature=0.2
            )
            
            # Track cost
            cost = self._calculate_cost(response.usage)
            self.total_cost += cost
            
            # Parse response
            content = response.choices[0].message.content
            analysis = json.loads(content) if response_format == "json_object" else content
            
            return {
                'success': True,
                'analysis': analysis,
                'cost': cost,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'cost': 0
            }


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    logging.basicConfig(level=logging.INFO)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        exit(1)
    
    agent = LearningAgent(api_key=api_key)
    
    # Example: Analyze a detection session
    session_data = {
        'detected_fields': [{'name': f'field_{i}'} for i in range(45)],
        'agent_decisions': [{'agent': 'field_spotter', 'decision': 'approve'}] * 10,
        'execution_report': {
            'actions_executed': 10,
            'fields_created': 5,
            'fields_modified': 3,
            'fields_deleted': 2,
            'errors': []
        },
        'metadata': {
            'form_type': 'divorce_petition',
            'difficulty': 'medium'
        }
    }
    
    ground_truth = [{'name': f'gt_field_{i}'} for i in range(47)]
    
    result = agent.analyze_detection_session(session_data, ground_truth)
    if result.get('success'):
        print("Learning Insights:")
        print(f"  Session Quality: {result['analysis'].get('session_quality')}")
        print(f"  Key Successes: {len(result['analysis'].get('key_successes', []))}")
        print(f"  Key Failures: {len(result['analysis'].get('key_failures', []))}")
        print(f"  Recommendations: {len(result['analysis'].get('overall_recommendations', []))}")
        print(f"  Cost: ${result['cost']:.4f}")
    else:
        print(f"Error: {result.get('error')}")
