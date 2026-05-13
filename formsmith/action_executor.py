"""
Action Executor

Applies agent-proposed actions to field definitions.
Handles: applying adjustments, tracking changes, backup/rollback, verification.
"""

from typing import List, Dict, Optional
from pathlib import Path
import json
import shutil
from datetime import datetime
import copy


class ActionExecutor:
    """
    Executes agent-proposed actions on field definitions.
    
    Handles:
    - Applying adjustments
    - Tracking changes
    - Backup/rollback
    - Verification
    """
    
    def __init__(self, config):
        """
        Initialize executor.
        
        Args:
            config: Config instance with settings
        """
        self.config = config
        self.action_log = []
    
    def execute_adjustment(
        self,
        fields: List[Dict],
        adjustment: Dict,
        dry_run: bool = False
    ) -> tuple[List[Dict], Dict]:
        """
        Execute a field adjustment.
        
        Args:
            fields: Current field list
            adjustment: FieldAdjustment from agent
            dry_run: If True, don't actually modify
        
        Returns:
            (modified_fields, execution_result)
        """
        field_id = adjustment['field_id']
        action = adjustment['action']
        
        # Find the field
        field_idx = next((i for i, f in enumerate(fields) if f.get('id') == field_id), None)
        
        if field_idx is None:
            return fields, {
                'success': False,
                'error': f'Field {field_id} not found',
                'field_id': field_id,
                'action': action
            }
        
        # Make a copy for safety
        fields_copy = [copy.deepcopy(f) for f in fields]
        field = fields_copy[field_idx]
        original_field = copy.deepcopy(field)
        
        # Execute action
        if action == 'move':
            field['bbox'] = adjustment['new_bbox']
            field['adjusted'] = True
            field['adjustment_reason'] = adjustment['reasoning']
            field['adjustment_confidence'] = adjustment['confidence']
            
        elif action == 'resize':
            bbox = field['bbox']
            if adjustment.get('new_width'):
                bbox[2] = bbox[0] + adjustment['new_width']
            if adjustment.get('new_height'):
                bbox[3] = bbox[1] + adjustment['new_height']
            field['bbox'] = bbox
            field['adjusted'] = True
            field['adjustment_reason'] = adjustment['reasoning']
            field['adjustment_confidence'] = adjustment['confidence']
            
        elif action == 'delete':
            if not dry_run:
                fields_copy.pop(field_idx)
            return fields_copy, {
                'success': True,
                'action': 'delete',
                'field_id': field_id,
                'reasoning': adjustment['reasoning'],
                'confidence': adjustment['confidence']
            }
            
        elif action == 'accept':
            field['verified'] = True
            field['verification_confidence'] = adjustment['confidence']
            field['verification_reason'] = adjustment['reasoning']
        
        # Log action
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'field_id': field_id,
            'action': action,
            'original': original_field,
            'modified': field,
            'agent_reasoning': adjustment['reasoning'],
            'confidence': adjustment['confidence']
        }
        
        if not dry_run:
            self.action_log.append(log_entry)
        
        return fields_copy, {
            'success': True,
            'action': action,
            'field_id': field_id,
            'changes': log_entry,
            'reasoning': adjustment['reasoning'],
            'confidence': adjustment['confidence']
        }
    
    def execute_batch(
        self,
        fields: List[Dict],
        adjustments: List[Dict],
        verify_after: bool = True
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Execute multiple adjustments in sequence.
        
        Args:
            fields: Current field list
            adjustments: List of FieldAdjustments
            verify_after: Run verification after all adjustments
        
        Returns:
            (final_fields, results)
        """
        results = []
        current_fields = [copy.deepcopy(f) for f in fields]
        
        for adjustment in adjustments:
            current_fields, result = self.execute_adjustment(
                current_fields,
                adjustment
            )
            results.append(result)
            
            # Stop if an adjustment failed
            if not result.get('success', False):
                print(f"⚠️  Adjustment failed: {result.get('error', 'Unknown error')}")
                break
        
        # Verify all fields if requested
        if verify_after:
            verification_results = self.verify_all_fields(current_fields)
            results.append({
                'action': 'verification',
                'results': verification_results
            })
        
        return current_fields, results
    
    def verify_all_fields(self, fields: List[Dict]) -> Dict:
        """
        Verify all fields have valid properties.
        
        Returns verification report.
        """
        issues = []
        
        for field in fields:
            bbox = field.get('bbox', [0, 0, 0, 0])
            field_id = field.get('id', 'unknown')
            
            # Check bbox validity
            if len(bbox) != 4:
                issues.append({
                    'field_id': field_id,
                    'issue': 'invalid_bbox_length',
                    'bbox': bbox
                })
                continue
            
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                issues.append({
                    'field_id': field_id,
                    'issue': 'invalid_bbox',
                    'bbox': bbox,
                    'details': 'x1 must be > x0 and y1 must be > y0'
                })
            
            # Check reasonable dimensions
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            if width < 10 or height < 5:
                issues.append({
                    'field_id': field_id,
                    'issue': 'too_small',
                    'dimensions': f"{width:.1f}×{height:.1f}",
                    'bbox': bbox
                })
            
            if width > 500 or height > 100:
                issues.append({
                    'field_id': field_id,
                    'issue': 'too_large',
                    'dimensions': f"{width:.1f}×{height:.1f}",
                    'bbox': bbox
                })
        
        return {
            'total_fields': len(fields),
            'valid_fields': len(fields) - len(issues),
            'issues': issues,
            'pass': len(issues) == 0
        }
    
    def get_action_log(self) -> List[Dict]:
        """Get complete action log."""
        return copy.deepcopy(self.action_log)
    
    def save_action_log(self, filepath: Path):
        """Save action log to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump({
                'total_actions': len(self.action_log),
                'actions': self.action_log
            }, f, indent=2)
        
        print(f"📝 Action log saved to: {filepath}")
    
    def clear_log(self):
        """Clear action log."""
        self.action_log = []

