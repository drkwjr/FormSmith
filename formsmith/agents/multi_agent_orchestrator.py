"""
Multi-Agent Orchestrator - Coordinates all agents in the vision system.

This is the central controller that manages agent interactions, consensus building,
and decision execution. It implements the chain of responsibility pattern and
ensures cost-effective agent usage.
"""

import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

# Import all agents
from .field_spotter import FieldSpotterAgent
from .layout_analyst import LayoutAnalystAgent
from .validator import ValidatorAgent
from .referee import RefereeAgent
from .position_advisor import PositionAdvisorAgent
from .learning_agent import LearningAgent
from .action_executor import ActionExecutor, FieldAction, ActionPlan

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the multi-agent orchestrator."""
    api_key: str
    model: str = "gpt-4o-2024-08-06"
    confidence_threshold: float = 0.75
    consensus_required: float = 0.66
    max_cost_per_form: float = 1.00
    enable_learning: bool = True
    save_decisions: bool = True
    
    # Agent enablement
    use_field_spotter: bool = True
    use_layout_analyst: bool = True
    use_validator: bool = True
    use_referee: bool = True
    use_position_advisor: bool = True
    use_learning_agent: bool = True
    
    # Cost optimization
    only_call_llm_on_ambiguous: bool = True
    ambiguity_threshold: float = 0.70


class MultiAgentOrchestrator:
    """
    Orchestrates multiple AI agents for intelligent field detection.
    
    This class implements:
    1. Chain of Responsibility: Agents are called in order based on need
    2. Multi-Agent Consensus: Multiple agents vote on decisions
    3. Cost Optimization: LLMs only called when traditional methods are uncertain
    4. Action Execution: Converts agent recommendations into concrete changes
    5. Learning Loop: Accumulates knowledge to improve over time
    """
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        
        # Initialize agents (lazy initialization for cost control)
        self.field_spotter = None
        self.layout_analyst = None
        self.validator = None
        self.referee = None
        self.position_advisor = None
        self.learning_agent = None
        
        # Initialize action executor
        self.action_executor = ActionExecutor(
            confidence_threshold=config.confidence_threshold
        )
        
        # Session tracking
        self.total_cost = 0.0
        self.decisions_made: List[Dict] = []
        self.session_start = datetime.now()
        
        logger.info("Multi-Agent Orchestrator initialized")
    
    def _get_field_spotter(self) -> FieldSpotterAgent:
        """Lazy load field spotter agent."""
        if self.field_spotter is None:
            self.field_spotter = FieldSpotterAgent(
                api_key=self.config.api_key,
                model=self.config.model
            )
        return self.field_spotter
    
    def _get_layout_analyst(self) -> LayoutAnalystAgent:
        """Lazy load layout analyst agent."""
        if self.layout_analyst is None:
            self.layout_analyst = LayoutAnalystAgent(
                api_key=self.config.api_key,
                model=self.config.model
            )
        return self.layout_analyst
    
    def _get_validator(self) -> ValidatorAgent:
        """Lazy load validator agent."""
        if self.validator is None:
            self.validator = ValidatorAgent(
                api_key=self.config.api_key,
                model=self.config.model
            )
        return self.validator
    
    def _get_referee(self) -> RefereeAgent:
        """Lazy load referee agent."""
        if self.referee is None:
            self.referee = RefereeAgent(
                api_key=self.config.api_key,
                model=self.config.model
            )
        return self.referee
    
    def _get_position_advisor(self) -> PositionAdvisorAgent:
        """Lazy load position advisor agent."""
        if self.position_advisor is None:
            self.position_advisor = PositionAdvisorAgent(
                api_key=self.config.api_key,
                model=self.config.model
            )
        return self.position_advisor
    
    def _get_learning_agent(self) -> LearningAgent:
        """Lazy load learning agent."""
        if self.learning_agent is None:
            self.learning_agent = LearningAgent(
                api_key=self.config.api_key,
                model=self.config.model
            )
        return self.learning_agent
    
    def process_candidate_field(
        self,
        image_path: Path,
        candidate: Dict,
        context: Optional[Dict] = None
    ) -> Tuple[str, Dict]:
        """
        Process a single candidate field through the multi-agent system.
        
        Args:
            image_path: Path to PDF page image
            candidate: Candidate field dictionary
            context: Optional context (layout, nearby fields, etc.)
        
        Returns:
            (decision, reasoning_dict)
            decision: 'approve', 'modify', 'reject'
            reasoning_dict: Detailed reasoning from agents
        """
        logger.info(f"Processing candidate: {candidate.get('name', 'unnamed')}")
        
        # Check if we're within budget
        if self.total_cost >= self.config.max_cost_per_form:
            logger.warning(f"Cost limit reached (${self.total_cost:.4f}), skipping LLM agents")
            return 'approve', {'reason': 'cost_limit_reached', 'confidence': 0.5}
        
        # Step 1: Check if candidate is ambiguous (only call LLMs if needed)
        candidate_confidence = candidate.get('confidence', 0)
        
        if self.config.only_call_llm_on_ambiguous:
            if candidate_confidence >= self.config.ambiguity_threshold:
                logger.info(f"High confidence ({candidate_confidence:.2f}), skipping LLM agents")
                return 'approve', {
                    'reason': 'high_confidence_traditional',
                    'confidence': candidate_confidence
                }
        
        # Step 2: Field Spotter (verify field exists)
        agent_opinions = []
        
        if self.config.use_field_spotter:
            logger.info("Calling Field Spotter agent...")
            spotter_result = self._get_field_spotter().spot_field(
                image_path=image_path,
                region_bbox=candidate.get('bbox')
            )
            
            if spotter_result.get('success'):
                analysis = spotter_result['analysis']
                agent_opinions.append({
                    'agent': 'field_spotter',
                    'verdict': 'approve' if analysis.get('field_exists') else 'reject',
                    'confidence': analysis.get('confidence', 0),
                    'reasoning': analysis.get('reasoning', '')
                })
                self.total_cost += spotter_result.get('cost', 0)
                
                # Early rejection if field doesn't exist
                if not analysis.get('field_exists'):
                    logger.info("Field Spotter rejected - field doesn't exist")
                    return 'reject', {
                        'reason': 'field_spotter_rejected',
                        'agent_opinions': agent_opinions
                    }
        
        # Step 3: Validator (check quality)
        if self.config.use_validator:
            logger.info("Calling Validator agent...")
            validator_result = self._get_validator().validate_field(
                image_path=image_path,
                field=candidate,
                context=context
            )
            
            if validator_result.get('success'):
                analysis = validator_result['analysis']
                agent_opinions.append({
                    'agent': 'validator',
                    'verdict': 'approve' if analysis.get('is_valid') else 'reject',
                    'confidence': analysis.get('confidence', 0),
                    'reasoning': analysis.get('reasoning', '')
                })
                self.total_cost += validator_result.get('cost', 0)
        
        # Step 4: Position Advisor (suggest optimal position if needed)
        suggested_position = None
        if self.config.use_position_advisor and len(agent_opinions) > 0:
            # Only if previous agents had concerns
            concerns = [op for op in agent_opinions if op['verdict'] != 'approve']
            if len(concerns) > 0:
                logger.info("Calling Position Advisor for position guidance...")
                position_result = self._get_position_advisor().get_optimal_position(
                    image_path=image_path,
                    current_bbox=tuple(candidate.get('bbox', [])),
                    field_type=candidate.get('type', 'text'),
                    label_text=candidate.get('label', None),
                    nearby_context=context
                )
                
                if position_result.get('success'):
                    analysis = position_result['analysis']
                    suggested_position = analysis.get('optimal_bbox')
                    agent_opinions.append({
                        'agent': 'position_advisor',
                        'verdict': 'modify',
                        'confidence': analysis.get('confidence', 0),
                        'reasoning': analysis.get('reasoning', ''),
                        'suggested_bbox': suggested_position
                    })
                    self.total_cost += position_result.get('cost', 0)
        
        # Step 5: Calculate consensus
        if not agent_opinions:
            # No agents called, approve by default
            return 'approve', {
                'reason': 'no_agents_called',
                'confidence': candidate_confidence
            }
        
        # Count votes
        votes = {'approve': 0, 'modify': 0, 'reject': 0}
        for opinion in agent_opinions:
            verdict = opinion.get('verdict', 'approve')
            votes[verdict] = votes.get(verdict, 0) + 1
        
        total_votes = sum(votes.values())
        consensus_level = max(votes.values()) / total_votes if total_votes > 0 else 0
        
        # Determine winning vote
        winning_verdict = max(votes, key=votes.get)
        
        # Step 6: Referee if low consensus
        if consensus_level < self.config.consensus_required and self.config.use_referee:
            logger.info("Low consensus, calling Referee agent...")
            referee_result = self._get_referee().resolve_conflict(
                image_path=image_path,
                field_candidate=candidate,
                agent_opinions=agent_opinions
            )
            
            if referee_result.get('success'):
                analysis = referee_result['analysis']
                final_decision = analysis.get('final_decision', winning_verdict)
                self.total_cost += referee_result.get('cost', 0)
                
                return final_decision, {
                    'reason': 'referee_decision',
                    'final_decision': final_decision,
                    'referee_confidence': analysis.get('confidence', 0),
                    'agent_opinions': agent_opinions,
                    'consensus_level': consensus_level,
                    'referee_reasoning': analysis.get('reasoning', '')
                }
        
        # Return consensus decision
        reasoning = {
            'reason': 'agent_consensus',
            'decision': winning_verdict,
            'consensus_level': consensus_level,
            'votes': votes,
            'agent_opinions': agent_opinions
        }
        
        if suggested_position and winning_verdict == 'modify':
            reasoning['suggested_position'] = suggested_position
        
        return winning_verdict, reasoning
    
    def enhance_detection(
        self,
        image_path: Path,
        initial_fields: List[Dict],
        ground_truth: Optional[List[Dict]] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Enhance a set of initially detected fields using the multi-agent system.
        
        Args:
            image_path: Path to PDF page image
            initial_fields: Fields detected by traditional methods
            ground_truth: Optional ground truth for learning
        
        Returns:
            (enhanced_fields, session_report)
        """
        logger.info(f"Enhancing {len(initial_fields)} initially detected fields")
        
        # Step 1: Get layout analysis (once for entire page)
        layout_context = None
        if self.config.use_layout_analyst:
            logger.info("Analyzing page layout...")
            layout_result = self._get_layout_analyst().analyze_layout(
                image_path=image_path,
                existing_fields=initial_fields
            )
            
            if layout_result.get('success'):
                layout_context = layout_result['analysis']
                self.total_cost += layout_result.get('cost', 0)
        
        # Step 2: Process each field through the agent system
        actions = []
        
        for i, field in enumerate(initial_fields):
            logger.info(f"Processing field {i+1}/{len(initial_fields)}: {field.get('name')}")
            
            # Prepare context
            context = {
                'layout': layout_context,
                'nearby_fields': initial_fields,  # Simplified
                'field_index': i
            }
            
            # Get agent decision
            decision, reasoning = self.process_candidate_field(
                image_path=image_path,
                candidate=field,
                context=context
            )
            
            # Convert decision to action
            if decision == 'reject':
                action = FieldAction(
                    action_type='delete',
                    field_id=field.get('name', f'field_{i}'),
                    confidence=reasoning.get('confidence', 0.5),
                    agent_source='multi_agent_system',
                    reasoning=str(reasoning)
                )
                actions.append(action)
                
            elif decision == 'modify':
                suggested_pos = reasoning.get('suggested_position')
                action = FieldAction(
                    action_type='modify',
                    field_id=field.get('name', f'field_{i}'),
                    confidence=reasoning.get('confidence', 0.8),
                    agent_source='multi_agent_system',
                    bbox=tuple(suggested_pos) if suggested_pos else None,
                    reasoning=str(reasoning)
                )
                actions.append(action)
            
            # Store decision
            self.decisions_made.append({
                'field': field,
                'decision': decision,
                'reasoning': reasoning,
                'cost': self.total_cost
            })
            
            # Check cost limit
            if self.total_cost >= self.config.max_cost_per_form:
                logger.warning(f"Cost limit reached at field {i+1}/{len(initial_fields)}")
                break
        
        # Step 3: Build and validate action plan
        agent_votes = {}
        for decision in self.decisions_made:
            agent_opinions = decision.get('reasoning', {}).get('agent_opinions', [])
            for opinion in agent_opinions:
                agent_name = opinion['agent']
                if agent_name not in agent_votes:
                    agent_votes[agent_name] = []
                agent_votes[agent_name].append(opinion['verdict'])
        
        # Calculate overall confidence
        confidences = []
        for decision in self.decisions_made:
            conf = decision.get('reasoning', {}).get('confidence', 0.5)
            if isinstance(conf, (int, float)):
                confidences.append(conf)
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Calculate consensus
        if len(self.decisions_made) > 0:
            decision_types = [d['decision'] for d in self.decisions_made]
            most_common = max(set(decision_types), key=decision_types.count)
            consensus = decision_types.count(most_common) / len(decision_types)
        else:
            consensus = 1.0
        
        action_plan = ActionPlan(
            actions=actions,
            overall_confidence=overall_confidence,
            consensus_level=consensus,
            agent_votes=agent_votes,
            validation_status='approved' if consensus >= self.config.consensus_required else 'needs_review'
        )
        
        # Step 4: Execute action plan
        enhanced_fields, execution_report = self.action_executor.execute_action_plan(
            plan=action_plan,
            current_fields=initial_fields
        )
        
        # Get active fields (non-deleted)
        final_fields = self.action_executor.get_active_fields(enhanced_fields)
        
        # Step 5: Final quality check with Referee
        final_approval = None
        if self.config.use_referee and len(final_fields) > 0:
            logger.info("Performing final quality check...")
            final_check = self._get_referee().final_quality_check(
                image_path=image_path,
                final_fields=final_fields,
                original_detection_count=len(initial_fields)
            )
            
            if final_check.get('success'):
                final_approval = final_check['analysis']
                self.total_cost += final_check.get('cost', 0)
        
        # Step 6: Learning (if enabled)
        learning_insights = None
        if self.config.enable_learning and self.config.use_learning_agent:
            logger.info("Generating learning insights...")
            session_data = {
                'detected_fields': initial_fields,
                'final_fields': final_fields,
                'agent_decisions': self.decisions_made,
                'execution_report': execution_report,
                'metadata': {
                    'image_path': str(image_path),
                    'initial_count': len(initial_fields),
                    'final_count': len(final_fields)
                }
            }
            
            learning_result = self._get_learning_agent().analyze_detection_session(
                session_data=session_data,
                ground_truth=ground_truth
            )
            
            if learning_result.get('success'):
                learning_insights = learning_result['analysis']
                self.total_cost += learning_result.get('cost', 0)
        
        # Step 7: Build session report
        session_report = {
            'initial_field_count': len(initial_fields),
            'final_field_count': len(final_fields),
            'actions_executed': len(actions),
            'execution_report': execution_report,
            'total_cost': self.total_cost,
            'decisions_made': len(self.decisions_made),
            'final_approval': final_approval,
            'learning_insights': learning_insights,
            'session_duration': (datetime.now() - self.session_start).total_seconds()
        }
        
        logger.info(
            f"Enhancement complete: {len(initial_fields)} → {len(final_fields)} fields, "
            f"cost ${self.total_cost:.4f}"
        )
        
        return final_fields, session_report
    
    def save_session(self, output_dir: Path):
        """Save complete session data to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save decisions
        if self.config.save_decisions:
            decisions_path = output_dir / 'agent_decisions.json'
            with open(decisions_path, 'w') as f:
                json.dump({
                    'total_decisions': len(self.decisions_made),
                    'decisions': self.decisions_made
                }, f, indent=2)
        
        # Save execution log
        exec_log_path = output_dir / 'execution_log.json'
        self.action_executor.save_execution_log(exec_log_path)
        
        # Save learning insights
        if self.config.enable_learning and self.learning_agent is not None:
            learning_path = output_dir / 'learning_insights.json'
            self.learning_agent.save_learnings(learning_path)
        
        # Save summary
        summary_path = output_dir / 'session_summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'total_cost': self.total_cost,
                'decisions_made': len(self.decisions_made),
                'execution_stats': self.action_executor.get_statistics(),
                'session_duration': (datetime.now() - self.session_start).total_seconds(),
                'config': asdict(self.config)
            }, f, indent=2)
        
        logger.info(f"Session data saved to {output_dir}")


def create_orchestrator_from_env() -> MultiAgentOrchestrator:
    """Create orchestrator from environment variables."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    config = OrchestratorConfig(
        api_key=api_key,
        model=os.getenv('OPENAI_MODEL', 'gpt-4o-2024-08-06'),
        confidence_threshold=float(os.getenv('AGENT_CONFIDENCE_THRESHOLD', '0.75')),
        consensus_required=float(os.getenv('MIN_AGENT_CONSENSUS', '0.66')),
        max_cost_per_form=float(os.getenv('AGENT_MAX_COST_PER_FORM', '1.00')),
        enable_learning=os.getenv('ENABLE_LEARNING_AGENT', 'true').lower() == 'true',
        save_decisions=os.getenv('SAVE_AGENT_DECISIONS', 'true').lower() == 'true'
    )
    
    return MultiAgentOrchestrator(config)


if __name__ == '__main__':
    import sys
    from dotenv import load_dotenv
    
    load_dotenv()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python multi_agent_orchestrator.py <image_path>")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Create orchestrator
    orchestrator = create_orchestrator_from_env()
    
    # Example: Process some initial fields
    initial_fields = [
        {
            'name': 'plaintiff_name',
            'type': 'text',
            'bbox': [100, 200, 300, 230],
            'confidence': 0.65
        },
        {
            'name': 'defendant_name',
            'type': 'text',
            'bbox': [100, 250, 300, 280],
            'confidence': 0.92
        }
    ]
    
    # Enhance fields
    enhanced_fields, report = orchestrator.enhance_detection(
        image_path=image_path,
        initial_fields=initial_fields
    )
    
    print(f"\nResults:")
    print(f"  Initial fields: {report['initial_field_count']}")
    print(f"  Final fields: {report['final_field_count']}")
    print(f"  Total cost: ${report['total_cost']:.4f}")
    print(f"  Duration: {report['session_duration']:.2f}s")
    
    if report.get('final_approval'):
        approval = report['final_approval']
        print(f"  Quality Grade: {approval.get('quality_grade')}")
        print(f"  Save Recommendation: {approval.get('save_recommendation')}")
    
    # Save session
    output_dir = Path('agent_sessions') / datetime.now().strftime('%Y%m%d_%H%M%S')
    orchestrator.save_session(output_dir)
    print(f"\nSession saved to: {output_dir}")

