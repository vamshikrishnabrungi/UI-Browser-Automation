import os
import re
import json
import datetime
from typing import Dict, List, Any, Optional
from state import TestState, StepResult

class ValidationResult:
    """Class to hold validation results for a test step"""
    
    def __init__(self, step: int, is_valid: bool, validation_message: str, suggested_fixes: Optional[List[str]] = None):
        """Initialize the validation result
        
        Args:
            step: Step number
            is_valid: Whether the step result is valid
            validation_message: Validation message
            suggested_fixes: Suggested fixes (optional)
        """
        self.step = step
        self.is_valid = is_valid
        self.validation_message = validation_message
        self.suggested_fixes = suggested_fixes or []

class TestValidationAgent:
    """Agent that validates test results"""
    
    def __init__(self):
        """Initialize the validation agent"""
        pass
    
    def validate_results(self, state: TestState) -> TestState:
        """Validate test results and update the state
        
        Args:
            state: TestState with execution results
            
        Returns:
            Updated TestState with validation results
        """
        print(f"Validating {len(state.results) if state.results else 0} test results...")
        
        # Create validation results
        validation_results = []
        
        # Process each step result
        for result in state.results:
            # Default to valid for passed steps
            is_valid = result.status == "passed"
            validation_message = "Step executed successfully" if is_valid else "Step execution failed"
            suggested_fixes = []
            
            if not is_valid:
                # For failed steps, provide some validation feedback
                if result.error:
                    validation_message = f"Execution error: {result.error}"
                    
                    # Suggest fixes based on error patterns
                    if "element not found" in result.error.lower():
                        suggested_fixes.append("Verify element selector or element presence on page")
                    elif "timeout" in result.error.lower():
                        suggested_fixes.append("Increase wait time before action")
                    elif "navigation" in result.error.lower():
                        suggested_fixes.append("Check URL and network connectivity")
                
                # If the step was skipped, suggest appropriate feedback
                if result.status == "skipped":
                    validation_message = "Step was skipped during execution"
                    suggested_fixes.append("Check previous step dependencies")
            
            # Create validation result
            validation_result = ValidationResult(
                step=result.step,
                is_valid=is_valid,
                validation_message=validation_message,
                suggested_fixes=suggested_fixes
            )
            validation_results.append(validation_result)
        
        # Update state with validation results
        state.validation_results = validation_results
        
        # Calculate overall validation score
        valid_count = sum(1 for vr in validation_results if vr.is_valid)
        total_count = len(validation_results)
        state.validator_confidence = valid_count / total_count if total_count > 0 else 0.0
        
        print(f"Validation complete: {valid_count}/{total_count} steps valid, confidence: {state.validator_confidence:.2f}")
        
        return state