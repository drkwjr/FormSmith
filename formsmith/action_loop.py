"""
Action Loop

Orchestrates the observe-propose-execute-verify loop.

Flow:
1. Detect fields (traditional methods)
2. Identify issues (validator agent)
3. Propose fixes (action proposer agent)
4. Execute fixes (action executor)
5. Verify fixes (validator agent again)
6. Loop until convergence or max iterations
"""

from typing import List, Dict, Optional
import asyncio
from pathlib import Path


class ActionLoop:
    """
    Orchestrates the observe-propose-execute-verify loop.
    
    Flow:
    1. Detect fields (traditional methods)
    2. Identify issues (validator agent)
    3. Propose fixes (action proposer agent)
    4. Execute fixes (action executor)
    5. Verify fixes (validator agent again)
    6. Loop until convergence or max iterations
    """
    
    def __init__(
        self,
        action_proposer_agent,
        validator_agent,
        action_executor,
        max_iterations: int = 3,
        min_confidence: float = 0.85
    ):
        """
        Initialize action loop.
        
        Args:
            action_proposer_agent: Agent that proposes field adjustments
            validator_agent: Agent that validates field correctness
            action_executor: Executor that applies adjustments
            max_iterations: Maximum loop iterations
            min_confidence: Minimum confidence for applying adjustments
        """
        self.action_proposer = action_proposer_agent
        self.validator = validator_agent
        self.executor = action_executor
        self.max_iterations = max_iterations
        self.min_confidence = min_confidence
    
    async def run_loop(
        self,
        page_image: bytes,
        initial_fields: List[Dict],
        labels: List[Dict],
        verbose: bool = True
    ) -> tuple[List[Dict], Dict]:
        """
        Run the complete action loop.
        
        Args:
            page_image: PNG/JPEG bytes of page
            initial_fields: Initially detected fields
            labels: Extracted text labels
            verbose: Print progress messages
        
        Returns:
            (final_fields, loop_report)
        """
        fields = [f.copy() for f in initial_fields]
        iteration = 0
        loop_report = {
            'initial_field_count': len(fields),
            'iterations': [],
            'total_adjustments': 0,
            'convergence': False,
            'final_field_count': 0
        }
        
        if verbose:
            print(f"\n🔄 Starting Action Loop")
            print(f"   Initial fields: {len(fields)}")
            print(f"   Max iterations: {self.max_iterations}")
            print(f"   Min confidence: {self.min_confidence}")
        
        while iteration < self.max_iterations:
            iteration += 1
            if verbose:
                print(f"\n🔄 Action Loop Iteration {iteration}/{self.max_iterations}")
            
            # Step 1: Validate all fields
            if verbose:
                print(f"   Validating {len(fields)} fields...")
            
            issues = []
            
            for field in fields:
                try:
                    validation = self.validator.call(
                        image=page_image,
                        field=field,
                        labels=labels
                    )
                    
                    # Check if field has issues
                    is_correct = validation.get('is_correct', True)
                    confidence = validation.get('confidence', 1.0)
                    
                    if not is_correct or confidence < self.min_confidence:
                        issues.append({
                            'field': field,
                            'validation': validation
                        })
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Validation failed for field {field.get('id')}: {e}")
                    # Continue with other fields
                    continue
            
            if verbose:
                print(f"   Found {len(issues)} fields needing adjustment")
            
            # If no issues, we've converged!
            if len(issues) == 0:
                if verbose:
                    print(f"   ✅ Converged! All fields validated.")
                loop_report['convergence'] = True
                break
            
            # Step 2: Propose actions for each issue
            if verbose:
                print(f"   Proposing actions...")
            
            adjustments = []
            
            for issue_data in issues:
                field = issue_data['field']
                validation = issue_data['validation']
                
                try:
                    # Get action proposal
                    adjustment = self.action_proposer.call(
                        image=page_image,
                        field=field,
                        issue="; ".join(validation.get('issues', [])),
                        context={'labels': labels, 'validation': validation}
                    )
                    
                    # Only apply high-confidence adjustments
                    if adjustment['confidence'] >= self.min_confidence:
                        adjustments.append(adjustment)
                    else:
                        if verbose:
                            print(f"   ⏭️  Skipping low-confidence adjustment for {field.get('id')} (confidence: {adjustment['confidence']:.2f})")
                
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Action proposal failed for field {field.get('id')}: {e}")
                    continue
            
            if verbose:
                print(f"   Proposed {len(adjustments)} high-confidence adjustments")
            
            # Step 3: Execute adjustments
            if adjustments:
                fields, results = self.executor.execute_batch(
                    fields,
                    adjustments,
                    verify_after=True
                )
                
                # Count successful adjustments
                successful = sum(1 for r in results if r.get('success', False))
                
                loop_report['iterations'].append({
                    'iteration': iteration,
                    'issues_found': len(issues),
                    'adjustments_proposed': len(adjustments),
                    'adjustments_applied': successful,
                    'results': results
                })
                
                loop_report['total_adjustments'] += successful
                
                if verbose:
                    print(f"   ✅ Applied {successful} adjustments")
            else:
                if verbose:
                    print(f"   No high-confidence adjustments, stopping loop")
                break
        
        # Final verification
        final_verification = self.executor.verify_all_fields(fields)
        loop_report['final_verification'] = final_verification
        loop_report['final_field_count'] = len(fields)
        
        if verbose:
            print(f"\n📊 Action Loop Complete")
            print(f"   Iterations: {iteration}")
            print(f"   Total adjustments: {loop_report['total_adjustments']}")
            print(f"   Final fields: {len(fields)}")
            print(f"   Converged: {loop_report['convergence']}")
            print(f"   Final verification: {'✅ PASS' if final_verification['pass'] else '❌ FAIL'}")
        
        return fields, loop_report
    
    def run_loop_sync(
        self,
        page_image: bytes,
        initial_fields: List[Dict],
        labels: List[Dict],
        verbose: bool = True
    ) -> tuple[List[Dict], Dict]:
        """
        Synchronous wrapper for run_loop.
        
        Use this if you don't want to deal with async/await.
        """
        return asyncio.run(self.run_loop(
            page_image,
            initial_fields,
            labels,
            verbose
        ))

