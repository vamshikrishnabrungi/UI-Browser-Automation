from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class TestStep(BaseModel):
    """Model for a single test step"""
    step_number: Union[int, str]  # Allow both int and string step IDs like TC_001_S1
    action: str
    expected_outcome: Optional[str] = None
    action_type: str
    dependencies: Dict[str, Any] = Field(default_factory=dict)
    # Add fields for feasibility tracking
    feasible: bool = True
    feasibility_reason: str = ""
    feasibility_confidence: float = 0.8


class TestPlan(BaseModel):
    """Model for the overall test plan"""
    title: str = ""
    description: str = ""
    prerequisites: List[str] = Field(default_factory=list)
    test_data: List[str] = Field(default_factory=list)
    full_text: str = ""
    critical_paths: List[List[Union[int, str]]] = Field(default_factory=list)
    verification_points: List[Union[int, str]] = Field(default_factory=list)


class StepResult(BaseModel):
    """Model for the result of a test step execution"""
    step: Union[int, str]  # Match the step_number type in TestStep
    action: str
    status: str  # "passed" or "failed"
    error: Optional[str] = None
    url: Optional[str] = None
    expected_outcome: Optional[str] = None
    actual_outcome: Optional[str] = None
    screenshot: Optional[str] = None
    confidence: float = 1.0  # Confidence score for this result


class ValidationResult(BaseModel):
    """Model for validation results"""
    step: Union[int, str]  # Match the step_number type in TestStep
    is_valid: bool
    validation_message: str
    confidence: float = 1.0
    suggested_fixes: Optional[List[str]] = None


class TestState(BaseModel):
    """Shared state for the LangGraph pipeline"""
    # Test definition
    test_steps: List[TestStep] = Field(default_factory=list)
    test_plan: Optional[TestPlan] = None
    
    # Execution tracking
    current_step: int = 0
    max_steps: int = 0
    is_complete: bool = False
    
    # Results
    results: List[StepResult] = Field(default_factory=list)
    validation_results: List[ValidationResult] = Field(default_factory=list)
    
    # Environment
    browser: str = "chrome"
    current_url: str = ""
    error: Optional[str] = None
    timestamp: str = ""
    
    # Pipeline control
    parsing_complete: bool = False
    execution_complete: bool = False
    validation_complete: bool = False
    
    # Confidence scores
    parser_confidence: float = 0.0
    executor_confidence: float = 0.0
    validator_confidence: float = 0.0
    overall_feasibility: float = 0.0  # Added for feasibility tracking
    
    # Raw input and output
    raw_test_input: str = ""
    raw_execution_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Paths to saved files
    parsed_json_path: str = ""