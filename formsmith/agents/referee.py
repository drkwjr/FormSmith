"""
Referee Agent - Makes final decisions when other agents disagree.

This agent acts as the tiebreaker and final authority when multiple agents
provide conflicting recommendations.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from .base_agent import VisionAgent

logger = logging.getLogger(__name__)


class RefereeAgent(VisionAgent):
    """
    Makes final decisions when agents disagree.
    
    This agent is specialized in:
    - Resolving conflicts between agent recommendations
    - Making tie-breaking decisions
    - Providing authoritative final judgment
    - Synthesizing multiple perspectives
    - High-stakes decision making
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-2024-08-06"):
        super().__init__(api_key=api_key, model=model, agent_name="referee")
    
    def resolve_conflict(
        self,
        image_path: Path,
        field_candidate: Dict,
        agent_opinions: List[Dict]
    ) -> Dict[str, Any]:
        """
        Resolve conflicting opinions about a field.
        
        Args:
            image_path: Path to PDF page image
            field_candidate: The field in question
            agent_opinions: List of opinions from different agents
        
        Returns:
            Final decision with reasoning
        """
        bbox = field_candidate.get('bbox', field_candidate.get('bounding_box', []))
        field_type = field_candidate.get('type', 'unknown')
        field_name = field_candidate.get('name', 'unnamed')
        
        # Summarize opinions
        opinions_summary = []
        for i, opinion in enumerate(agent_opinions):
            agent_name = opinion.get('agent', f'Agent {i+1}')
            verdict = opinion.get('verdict', 'unknown')
            confidence = opinion.get('confidence', 0)
            reasoning = opinion.get('reasoning', 'No reasoning')
            
            opinions_summary.append(
                f"**{agent_name}**: {verdict} (confidence: {confidence:.2f})\n"
                f"  Reasoning: {reasoning}"
            )
        
        opinions_text = "\n\n".join(opinions_summary)
        
        prompt = f"""You are the Referee - the final authority on field detection decisions.

Field Under Review:
- Name: {field_name}
- Type: {field_type}
- Position: {bbox}

Agent Opinions ({len(agent_opinions)} agents weighed in):
{opinions_text}

Your task: Make the FINAL, AUTHORITATIVE decision on this field.

Consider:
1. Weight each agent's opinion by their confidence and reasoning quality
2. Examine the image yourself to form an independent judgment
3. Identify which agents have the strongest evidence
4. Make a clear, decisive ruling

This is high-stakes: Your decision is FINAL and will be implemented.

Respond with JSON:
{{
    "final_decision": "approve | modify | reject",
    "confidence": 0.0-1.0,
    "modified_bbox": [x0, y0, x1, y1],  // only if decision is "modify"
    "modified_type": "type",  // only if decision is "modify"
    "reasoning": "Clear explanation of why you made this decision",
    "agent_consensus": {{
        "agreed_with": ["agent names that you sided with"],
        "disagreed_with": ["agent names you overruled"],
        "synthesis": "How you synthesized multiple viewpoints"
    }},
    "certainty": "absolute | high | medium | low"
}}"""
        
        result = self.analyze_image(
            image_path=image_path,
            prompt=prompt,
            response_format="json_object"
        )
        
        if result.get('success'):
            result['field_candidate'] = field_candidate
            result['agent_opinions'] = agent_opinions
        
        return result
    
    def review_action_plan(
        self,
        image_path: Path,
        proposed_actions: List[Dict],
        agent_votes: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Review and approve/reject an entire action plan.
        
        Args:
            image_path: Path to PDF page image
            proposed_actions: List of proposed field actions
            agent_votes: Dict of agent_name -> vote
        
        Returns:
            Final approval decision
        """
        # Summarize actions
        actions_summary = []
        for i, action in enumerate(proposed_actions[:10]):
            actions_summary.append(
                f"{i+1}. {action.get('action_type', 'unknown').upper()}: "
                f"{action.get('field_name', 'unnamed')} "
                f"(confidence: {action.get('confidence', 0):.2f})"
            )
        if len(proposed_actions) > 10:
            actions_summary.append(f"... and {len(proposed_actions) - 10} more actions")
        
        actions_text = "\n".join(actions_summary)
        
        # Summarize votes
        votes_summary = []
        for agent, vote in agent_votes.items():
            votes_summary.append(f"- {agent}: {vote}")
        votes_text = "\n".join(votes_summary)
        
        prompt = f"""You are the Referee - final authority on approving action plans.

Proposed Action Plan ({len(proposed_actions)} actions):
{actions_text}

Agent Votes:
{votes_text}

Your task: Review the ENTIRE plan and make a final approval decision.

Consider:
1. Overall quality and consistency of the plan
2. Risk of implementing these changes
3. Consensus level among agents
4. Potential for introducing errors

Make a decisive ruling on whether this plan should be:
- **APPROVED**: Implement all actions as proposed
- **APPROVED_WITH_MODIFICATIONS**: Approve but filter some actions
- **REJECTED**: Do not implement this plan

Respond with JSON:
{{
    "decision": "approved | approved_with_modifications | rejected",
    "confidence": 0.0-1.0,
    "actions_to_remove": ["field_id1", "field_id2"],  // if approved_with_modifications
    "reasoning": "Clear explanation of your decision",
    "risk_assessment": "low | medium | high",
    "recommendation": "Any additional guidance for implementation"
}}"""
        
        result = self.analyze_image(
            image_path=image_path,
            prompt=prompt,
            response_format="json_object"
        )
        
        return result
    
    def final_quality_check(
        self,
        image_path: Path,
        final_fields: List[Dict],
        original_detection_count: int
    ) -> Dict[str, Any]:
        """
        Perform final quality check before saving output.
        
        Args:
            image_path: Path to PDF page image
            final_fields: Final field list after all processing
            original_detection_count: How many fields were originally detected
        
        Returns:
            Final quality assessment
        """
        fields_summary = []
        for i, f in enumerate(final_fields[:15]):
            bbox = f.get('bbox', f.get('bounding_box', []))
            fields_summary.append(
                f"{i+1}. {f.get('name', 'unnamed')} ({f.get('type', 'unknown')}) at {bbox}"
            )
        if len(final_fields) > 15:
            fields_summary.append(f"... and {len(final_fields) - 15} more")
        
        fields_text = "\n".join(fields_summary)
        
        prompt = f"""You are the Referee performing the FINAL quality check before output.

Final Field Set ({len(final_fields)} fields, originally {original_detection_count}):
{fields_text}

Your task: This is the LAST checkpoint before these fields are saved to PDF.

Perform a comprehensive final review:
1. **Completeness**: Are all expected fields present?
2. **Accuracy**: Do fields appear correctly positioned?
3. **No False Positives**: Are there any suspicious fields?
4. **Quality**: Is this production-ready?
5. **Confidence**: Can we trust this output?

This is your chance to STOP a bad output from being saved.

Respond with JSON:
{{
    "final_approval": true/false,
    "quality_grade": "A | B | C | D | F",
    "confidence": 0.0-1.0,
    "critical_issues": [
        {{
            "severity": "critical | major | minor",
            "issue": "description",
            "affected_fields": ["field names"]
        }}
    ],
    "save_recommendation": "save | save_with_warning | do_not_save",
    "reasoning": "Final assessment and reasoning"
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
    
    agent = RefereeAgent(api_key=api_key)
    
    # Example: Resolve conflict
    test_image = Path("test_form_page.png")
    if test_image.exists():
        field = {
            'name': 'plaintiff_name',
            'type': 'text',
            'bbox': [100, 200, 300, 230]
        }
        
        opinions = [
            {
                'agent': 'field_spotter',
                'verdict': 'approve',
                'confidence': 0.85,
                'reasoning': 'Clear text field detected'
            },
            {
                'agent': 'validator',
                'verdict': 'reject',
                'confidence': 0.70,
                'reasoning': 'Position seems too low'
            }
        ]
        
        result = agent.resolve_conflict(test_image, field, opinions)
        if result.get('success'):
            print("Referee Decision:")
            print(f"  Decision: {result['analysis'].get('final_decision')}")
            print(f"  Confidence: {result['analysis'].get('confidence')}")
            print(f"  Certainty: {result['analysis'].get('certainty')}")
            print(f"  Cost: ${result['cost']:.4f}")
        else:
            print(f"Error: {result.get('error')}")
    else:
        print(f"Test image not found: {test_image}")
