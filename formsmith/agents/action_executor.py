"""
Action Executor - Converts agent recommendations into concrete field modifications.

This module bridges the gap between LLM agent analysis and actual PDF field manipulation.
It takes structured agent decisions and translates them into precise field adjustments.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class FieldAction:
    """Represents a single action to be taken on a field."""
    action_type: str  # 'create', 'modify', 'delete', 'validate'
    field_id: str
    confidence: float
    agent_source: str
    
    # Field properties
    bbox: Optional[Tuple[float, float, float, float]] = None
    field_type: Optional[str] = None
    field_name: Optional[str] = None
    page: Optional[int] = None
    
    # Reasoning
    reasoning: str = ""
    supporting_evidence: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.supporting_evidence is None:
            self.supporting_evidence = {}


@dataclass
class ActionPlan:
    """A complete plan of actions recommended by the multi-agent system."""
    actions: List[FieldAction]
    overall_confidence: float
    consensus_level: float
    agent_votes: Dict[str, Any]
    validation_status: str  # 'approved', 'needs_review', 'rejected'
    
    def to_dict(self) -> Dict:
        return {
            'actions': [asdict(a) for a in self.actions],
            'overall_confidence': self.overall_confidence,
            'consensus_level': self.consensus_level,
            'agent_votes': self.agent_votes,
            'validation_status': self.validation_status
        }


class ActionExecutor:
    """
    Executes field modification actions recommended by agents.
    
    This class is responsible for:
    1. Validating action plans before execution
    2. Applying field modifications to the detection pipeline
    3. Tracking what actions were taken and why
    4. Providing rollback capability
    5. Ensuring changes are persisted correctly
    """
    
    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold
        self.action_history: List[ActionPlan] = []
        self.execution_log: List[Dict] = []
        
    def validate_action_plan(self, plan: ActionPlan) -> Tuple[bool, List[str]]:
        """
        Validate an action plan before execution.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check overall confidence
        if plan.overall_confidence < self.confidence_threshold:
            issues.append(
                f"Overall confidence {plan.overall_confidence:.2f} below threshold "
                f"{self.confidence_threshold:.2f}"
            )
        
        # Check consensus level
        if plan.consensus_level < 0.66:
            issues.append(
                f"Consensus level {plan.consensus_level:.2f} too low (need >= 0.66)"
            )
        
        # Validate individual actions
        for i, action in enumerate(plan.actions):
            if action.action_type not in ['create', 'modify', 'delete', 'validate']:
                issues.append(f"Action {i}: Invalid action type '{action.action_type}'")
            
            if action.action_type in ['create', 'modify']:
                if action.bbox is None:
                    issues.append(f"Action {i}: Missing bbox for {action.action_type}")
                if action.field_type is None:
                    issues.append(f"Action {i}: Missing field_type for {action.action_type}")
                if action.page is None:
                    issues.append(f"Action {i}: Missing page for {action.action_type}")
                    
                # Validate bbox coordinates
                if action.bbox is not None:
                    x0, y0, x1, y1 = action.bbox
                    if x1 <= x0 or y1 <= y0:
                        issues.append(
                            f"Action {i}: Invalid bbox dimensions "
                            f"({x0}, {y0}, {x1}, {y1})"
                        )
                    if x0 < 0 or y0 < 0:
                        issues.append(f"Action {i}: Negative bbox coordinates")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def execute_action_plan(
        self,
        plan: ActionPlan,
        current_fields: List[Dict],
        force: bool = False
    ) -> Tuple[List[Dict], Dict]:
        """
        Execute an action plan to modify the field list.
        
        Args:
            plan: The action plan to execute
            current_fields: Current list of detected fields
            force: If True, execute even if validation fails (use with caution)
        
        Returns:
            (modified_fields, execution_report)
        """
        # Validate first
        is_valid, issues = self.validate_action_plan(plan)
        
        if not is_valid and not force:
            logger.warning(f"Action plan validation failed: {issues}")
            return current_fields, {
                'status': 'rejected',
                'reason': 'validation_failed',
                'issues': issues,
                'actions_executed': 0
            }
        
        # Track execution
        execution_report = {
            'status': 'success',
            'actions_executed': 0,
            'actions_skipped': 0,
            'fields_created': 0,
            'fields_modified': 0,
            'fields_deleted': 0,
            'errors': []
        }
        
        # Make a copy to modify
        modified_fields = [f.copy() for f in current_fields]
        
        # Execute each action
        for action in plan.actions:
            try:
                if action.action_type == 'create':
                    new_field = self._create_field(action)
                    modified_fields.append(new_field)
                    execution_report['fields_created'] += 1
                    
                elif action.action_type == 'modify':
                    success = self._modify_field(modified_fields, action)
                    if success:
                        execution_report['fields_modified'] += 1
                    else:
                        execution_report['actions_skipped'] += 1
                        
                elif action.action_type == 'delete':
                    success = self._delete_field(modified_fields, action)
                    if success:
                        execution_report['fields_deleted'] += 1
                    else:
                        execution_report['actions_skipped'] += 1
                
                execution_report['actions_executed'] += 1
                
            except Exception as e:
                logger.error(f"Error executing action {action.action_type}: {e}")
                execution_report['errors'].append({
                    'action': action.action_type,
                    'field_id': action.field_id,
                    'error': str(e)
                })
        
        # Log execution
        self.execution_log.append({
            'plan': plan.to_dict(),
            'report': execution_report,
            'field_count_before': len(current_fields),
            'field_count_after': len(modified_fields)
        })
        
        # Store in history
        self.action_history.append(plan)
        
        return modified_fields, execution_report
    
    def _create_field(self, action: FieldAction) -> Dict:
        """Create a new field from an action."""
        x0, y0, x1, y1 = action.bbox
        
        field = {
            'name': action.field_name or f"field_{action.field_id}",
            'type': action.field_type,
            'page': action.page,
            'bbox': action.bbox,
            'x': x0,
            'y': y0,
            'width': x1 - x0,
            'height': y1 - y0,
            'confidence': action.confidence,
            'source': action.agent_source,
            'reasoning': action.reasoning,
            'metadata': {
                'agent_created': True,
                'supporting_evidence': action.supporting_evidence
            }
        }
        
        logger.info(
            f"Created field '{field['name']}' (type={action.field_type}, "
            f"confidence={action.confidence:.2f})"
        )
        
        return field
    
    def _modify_field(self, fields: List[Dict], action: FieldAction) -> bool:
        """Modify an existing field."""
        # Find the field to modify
        for field in fields:
            if field.get('name') == action.field_id or \
               field.get('id') == action.field_id:
                
                # Update properties
                if action.bbox is not None:
                    x0, y0, x1, y1 = action.bbox
                    field['bbox'] = action.bbox
                    field['x'] = x0
                    field['y'] = y0
                    field['width'] = x1 - x0
                    field['height'] = y1 - y0
                
                if action.field_type is not None:
                    field['type'] = action.field_type
                
                if action.field_name is not None:
                    field['name'] = action.field_name
                
                # Add metadata
                if 'metadata' not in field:
                    field['metadata'] = {}
                field['metadata']['agent_modified'] = True
                field['metadata']['modification_reason'] = action.reasoning
                field['metadata']['modified_by'] = action.agent_source
                field['confidence'] = action.confidence
                
                logger.info(
                    f"Modified field '{action.field_id}' "
                    f"(confidence={action.confidence:.2f})"
                )
                
                return True
        
        logger.warning(f"Field '{action.field_id}' not found for modification")
        return False
    
    def _delete_field(self, fields: List[Dict], action: FieldAction) -> bool:
        """Delete a field (mark as deleted rather than removing)."""
        for i, field in enumerate(fields):
            if field.get('name') == action.field_id or \
               field.get('id') == action.field_id:
                
                # Mark as deleted rather than removing
                # (allows for audit trail and potential rollback)
                field['deleted'] = True
                field['deletion_reason'] = action.reasoning
                field['deleted_by'] = action.agent_source
                
                logger.info(
                    f"Deleted field '{action.field_id}' "
                    f"(reason: {action.reasoning})"
                )
                
                return True
        
        logger.warning(f"Field '{action.field_id}' not found for deletion")
        return False
    
    def get_active_fields(self, fields: List[Dict]) -> List[Dict]:
        """Filter out deleted fields."""
        return [f for f in fields if not f.get('deleted', False)]
    
    def save_execution_log(self, output_path: Path):
        """Save the execution log to a file for audit trail."""
        with open(output_path, 'w') as f:
            json.dump({
                'total_actions': len(self.action_history),
                'execution_log': self.execution_log
            }, f, indent=2)
        
        logger.info(f"Saved execution log to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about actions executed."""
        if not self.execution_log:
            return {
                'total_actions': 0,
                'fields_created': 0,
                'fields_modified': 0,
                'fields_deleted': 0,
                'errors': 0
            }
        
        stats = {
            'total_actions': sum(e['report']['actions_executed'] for e in self.execution_log),
            'fields_created': sum(e['report']['fields_created'] for e in self.execution_log),
            'fields_modified': sum(e['report']['fields_modified'] for e in self.execution_log),
            'fields_deleted': sum(e['report']['fields_deleted'] for e in self.execution_log),
            'errors': sum(len(e['report']['errors']) for e in self.execution_log)
        }
        
        return stats


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    executor = ActionExecutor(confidence_threshold=0.75)
    
    # Create a sample action plan
    action1 = FieldAction(
        action_type='create',
        field_id='test_field_1',
        confidence=0.92,
        agent_source='field_spotter',
        bbox=(100, 200, 300, 230),
        field_type='text',
        field_name='plaintiff_name',
        page=0,
        reasoning='High confidence text field detected near "Plaintiff:" label'
    )
    
    plan = ActionPlan(
        actions=[action1],
        overall_confidence=0.92,
        consensus_level=1.0,
        agent_votes={'field_spotter': 'approve'},
        validation_status='approved'
    )
    
    # Validate and execute
    is_valid, issues = executor.validate_action_plan(plan)
    print(f"Plan valid: {is_valid}")
    
    if is_valid:
        current_fields = []
        modified_fields, report = executor.execute_action_plan(plan, current_fields)
        print(f"Execution report: {report}")
        print(f"Fields after execution: {len(modified_fields)}")
        print(f"Statistics: {executor.get_statistics()}")

