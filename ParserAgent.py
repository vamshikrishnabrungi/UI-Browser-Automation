import os
import json
import re
import datetime
import pandas as pd
import argparse
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from state import TestState, TestStep, TestPlan

# Load environment variables
load_dotenv()

# Check if state module functionality is available for test execution
STATE_AVAILABLE = True

class TestParserAgent:
    """Agent that parses test steps from Excel into structured JSON using LLM or rule-based methods"""
    
    def __init__(self, output_dir: str = "parsed_tests", use_llm: bool = True):
        """Initialize the parser agent with optional Azure OpenAI API"""
        self.output_dir = output_dir
        self.use_llm = use_llm
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.use_llm:
            try:
                # Configure Azure OpenAI
                self.llm = AzureChatOpenAI(
                    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    model_name="gpt-4",
                    temperature=0.0
                )
                
                # System prompt for the parser with enhanced capabilities
                self.system_prompt = """
                You are an expert test automation engineer specializing in converting manual test steps into automated test scripts.
                Your task is to analyze test steps written in natural language and extract structured information for test automation.
                
                For each test step, extract the following components:
                1. Action: The core action to perform (e.g., navigate, click, enter, verify)
                2. Target: The UI element or object being acted upon
                3. Input: Any data being entered or used (if applicable)
                4. Expected Outcome: The expected result of the action
                5. Dependencies: Any steps this step depends on
                
                IMPORTANT: If a single test step contains multiple distinct actions or multiple expected outcomes to verify,
                break it down into separate smaller steps, each with a single action or verification point.
                For example, "Enter username and password then click Login" should be split into:
                1. Enter username
                2. Enter password  
                3. Click Login
                
                Similarly, if a step has multiple verification points like "Verify user is logged in and dashboard is displayed",
                break it into:
                1. Verify user is logged in
                2. Verify dashboard is displayed
                
                You'll perform the following NLP tasks:
                - Intent Classification: Determine the primary action (navigate, click, input, verify, etc.)
                - Named Entity Recognition: Identify UI elements, input values, and conditions
                - Dependency Parsing: Detect temporal or conditional relationships between steps
                - Step Decomposition: Break down complex steps with multiple actions/verifications
                
                Return your analysis as a structured JSON object.
                """
                print("LLM initialized successfully for enhanced parsing")
            except Exception as e:
                print(f"Warning: Could not initialize LLM: {e}")
                print("Falling back to rule-based parsing")
                self.use_llm = False
        
    def parse_test_steps(self, test_content: str) -> TestState:
        """Parse test steps from text or direct Excel file path"""
        try:
            # Check if test_content is a path to an Excel file
            if test_content.endswith('.xlsx') and os.path.exists(test_content):
                print(f"Parsing Excel file: {test_content}")
                return self._parse_excel_file(test_content)
            else:
                # It's raw text, extract as before
                print(f"Parsing text content ({len(test_content)} chars)")
                return self._parse_text_content(test_content)
                
        except Exception as e:
            print(f"Error parsing test steps: {e}")
            import traceback
            traceback.print_exc()
            # Return empty state with error
            return TestState(
                test_steps=[],
                test_plan=TestPlan(),
                raw_test_input=test_content,
                parser_confidence=0.0,
                error=str(e)
            )
            
    def _parse_text_content(self, test_content: str) -> TestState:
        """Parse test steps from text content into structured format"""
        # Extract actions and expected outcomes from the text
        actions = []
        expected_outcomes = []
        for line in test_content.split('\n'):
            if not line.strip():
                continue
            
            # Split by pipe to separate action and expected outcome
            if '|' in line:
                parts = line.split('|')
                action = parts[0].strip()
                expected = parts[1].strip() if len(parts) > 1 else None
                
                if expected and expected.lower().startswith('expected:'):
                    expected = expected[9:].strip()  # Remove "Expected: " prefix
            else:
                action = line.strip()
                expected = None
            
            # Extract step number from action
            match = re.match(r'^(\d+\.|\w+\.)\s*(.+)$', action)
            if match:
                action = match.group(2).strip()
            
            actions.append(action)
            expected_outcomes.append(expected)
        
        # Process the extracted actions
        test_steps = []
        for i, (action, expected) in enumerate(zip(actions, expected_outcomes)):
            step_num = str(i + 1)
            
            # Determine action type
            action_type = self._determine_action_type(action)
            
            # Create dependencies dict
            dependencies = {}
            if i > 0:
                dependencies = {
                    "requires_previous_success": True,
                    "dependent_on_steps": [str(i)],
                    "critical_dependency": False
                }
            
            # Create test step
            test_step = TestStep(
                step_number=step_num,
                action=action,
                action_type=action_type,
                expected_outcome=expected,
                dependencies=dependencies
            )
            test_steps.append(test_step)
        
        # Create test plan
        test_plan = TestPlan(
            title="Test from Text Input",
            description="Test steps parsed from text input",
            prerequisites=[],
            test_data=[],
            full_text=test_content,
            critical_paths=[],
            verification_points=[]
        )
        
        # Create test state
        state = TestState(
            test_steps=test_steps,
            test_plan=test_plan,
            raw_test_input=test_content,
            parser_confidence=0.9,
            parsing_complete=True,
            max_steps=len(test_steps)
        )
        
        # Save parsed steps for reference
        self._save_parsed_steps(state)
        
        return state
    
    def _parse_excel_file(self, excel_file: str) -> TestState:
        """Parse an Excel file directly into TestState"""
        try:
            # Get sheet names
            xl = pd.ExcelFile(excel_file)
            sheet_names = xl.sheet_names
            
            if not sheet_names:
                raise ValueError(f"No sheets found in {excel_file}")
                
            # Use first sheet that's not "Summary"
            sheet_name = sheet_names[0]
            for name in sheet_names:
                if name.lower() != "summary":
                    sheet_name = name
                    break
                    
            print(f"Using sheet: {sheet_name}")
            
            # Parse the sheet
            structured_steps = self._parse_sheet(excel_file, sheet_name)
            
            if not structured_steps:
                raise ValueError(f"No test steps found in sheet '{sheet_name}'")
                
            # Convert structured steps to TestStep objects
            test_steps = []
            step_id_map = {}  # Map to track original step IDs to new sequential IDs
            
            # First pass: Create TestStep objects and build ID mapping
            for i, step_data in enumerate(structured_steps):
                # Skip test case marker steps
                if step_data.get("Action") == "TestCase":
                    continue
                    
                step_id = step_data["Step ID"]
                new_step_id = str(i + 1)
                step_id_map[step_id] = new_step_id
                
                action = f"{step_data['Action']} {step_data['Target'] or ''}".strip()
                
                # Add input value to action if present
                if step_data["Input"]:
                    if isinstance(step_data["Input"], dict):
                        # Handle structured input (e.g., username/password)
                        input_str = ", ".join([f"{k}: {v}" for k, v in step_data["Input"].items()])
                        action += f" with inputs {input_str}"
                    else:
                        action += f" with input '{step_data['Input']}'"
                    
                # Determine action type
                action_type = self._determine_action_type(action)
                
                # Create empty dependencies dictionary (will be updated in second pass)
                dependencies = {}
                
                # Create TestStep object
                test_step = TestStep(
                    step_number=new_step_id,
                    action=action,
                    expected_outcome=step_data["Expected Outcome"],
                    action_type=action_type,
                    dependencies=dependencies
                )
                test_steps.append(test_step)
            
            # Second pass: Update dependencies with correct step IDs
            for i, step_data in enumerate(structured_steps):
                # Skip test case marker steps
                if step_data.get("Action") == "TestCase":
                    continue
                    
                if step_data["Depends On"]:
                    # Get the correct step number from our mapping
                    original_dependency = step_data["Depends On"]
                    if original_dependency in step_id_map:
                        mapped_dependency = step_id_map[original_dependency]
                        if i < len(test_steps):  # Ensure index is valid
                            test_steps[i].dependencies = {
                                "requires_previous_success": True,
                                "dependent_on_steps": [mapped_dependency],
                                "critical_dependency": False
                            }
                    else:
                        # If dependency refers to a step ID that doesn't exist, default to previous step
                        if i > 0 and i < len(test_steps):  # Ensure index is valid
                            test_steps[i].dependencies = {
                                "requires_previous_success": True,
                                "dependent_on_steps": [str(i)],
                                "critical_dependency": False
                            }
            
            # Create test plan
            test_plan = TestPlan(
                title=f"Test from {excel_file}",
                description=f"Test steps imported from {excel_file}, sheet: {sheet_name}",
                prerequisites=[],
                test_data=[],
                full_text=f"Excel file: {excel_file}, sheet: {sheet_name}",
                critical_paths=[],
                verification_points=[]
            )
            
            # Create raw text representation for debugging
            raw_text = "\n".join([
                f"{step.step_number}. {step.action} | Expected: {step.expected_outcome or ''}"
                for step in test_steps
            ])
            
            # Create TestState
            state = TestState(
                test_steps=test_steps,
                test_plan=test_plan,
                raw_test_input=raw_text,
                parser_confidence=0.95,
                parsing_complete=True,
                max_steps=len(test_steps)
            )
            
            # Save parsed steps for reference
            self._save_parsed_steps(state)
            
            return state
            
        except Exception as e:
            print(f"Error parsing Excel file: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty state with error
            return TestState(
                test_steps=[],
                test_plan=TestPlan(title=f"Error parsing {excel_file}"),
                raw_test_input=f"Excel file: {excel_file}",
                parser_confidence=0.0,
                error=str(e)
            )

    def _determine_action_type(self, action_text: str) -> str:
        """Determine the type of action from the action text"""
        action_text = str(action_text).lower()
        
        if "navigate" in action_text or "go to" in action_text or "open" in action_text:
            return "navigation"
        elif "click" in action_text or "press" in action_text or "select" in action_text or "tap" in action_text:
            return "click"
        elif "enter" in action_text or "input" in action_text or "type" in action_text or "fill" in action_text:
            return "input"
        elif "wait" in action_text or "pause" in action_text:
            return "wait"
        elif "verify" in action_text or "check" in action_text or "validate" in action_text or "confirm" in action_text:
            return "verification"
        elif "search" in action_text or "find" in action_text:
            return "search"
        elif "launch" in action_text or "start" in action_text:
            return "launch"
        elif "switch" in action_text or "change" in action_text:
            return "navigation"
        else:
            return "other"
            
    def _save_parsed_steps(self, state: TestState) -> None:
        """Save parsed test steps to JSON file with timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"excel_parsed_{timestamp}.json")
        
        # Convert state to dictionary
        state_dict = {
            "test_steps": [
                {
                    "step_number": step.step_number,
                    "action": step.action,
                    "action_type": step.action_type,
                    "expected_outcome": step.expected_outcome,
                    "dependencies": step.dependencies
                }
                for step in state.test_steps
            ],
            "test_plan": {
                "title": state.test_plan.title,
                "description": state.test_plan.description,
                "prerequisites": state.test_plan.prerequisites,
                "test_data": state.test_plan.test_data,
                "critical_paths": state.test_plan.critical_paths,
                "verification_points": state.test_plan.verification_points
            },
            "parser_confidence": state.parser_confidence
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)
            
        print(f"✅ Parsed test steps saved to: {filepath}")
        
        # Store the path in the state for reference
        state.parsed_json_path = filepath

    def _parse_sheet(self, excel_file: str, sheet_name: str) -> List[Dict[str, Any]]:
        """Parse a single sheet to extract structured test steps
        
        Args:
            excel_file: Path to the Excel file
            sheet_name: Name of the sheet to parse
            
        Returns:
            List of structured test steps
        """
        # Read the Excel sheet
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Identify test cases (rows with a Sr. No./Test case No.)
        test_case_rows = []
        for idx, row in df.iterrows():
            if 'Sr. No./Test case No.' in df.columns and pd.notna(row['Sr. No./Test case No.']):
                test_case_rows.append(idx)
            elif 'Test Case ID' in df.columns and pd.notna(row['Test Case ID']):
                test_case_rows.append(idx)
        
        if not test_case_rows:
            print(f"No test cases found in sheet '{sheet_name}'")
            return []
        
        print(f"Found {len(test_case_rows)} test cases in sheet '{sheet_name}'")
        
        # Extract and process test steps using NLP techniques
        all_structured_steps = []
        step_counter = 0  # Counter to maintain continuous step numbering across test cases
        
        for i, test_case_idx in enumerate(test_case_rows):
            # Determine where this test case ends
            next_test_case_idx = test_case_rows[i+1] if i+1 < len(test_case_rows) else len(df)
            
            # Get test case details
            row = df.iloc[test_case_idx]
            test_case_id = row.get('Sr. No./Test case No.', '') if 'Sr. No./Test case No.' in df.columns else row.get('Test Case ID', '')
            test_case_title = row.get('Test cases', '') if 'Test cases' in df.columns else row.get('Test Case', '')
            
            if pd.isna(test_case_id):
                test_case_id = f"TC_{i+1}"
            if pd.isna(test_case_title):
                test_case_title = f"Test Case {i+1}"
                
            print(f"Processing Test Case {test_case_id}: {str(test_case_title)[:50]}...")
            
            # Save test case information in a comment step
            test_case_comment = {
                "Step ID": f"TC{i+1}",
                "Action": "TestCase",
                "Target": f"{test_case_id}: {test_case_title}",
                "Input": None,
                "Expected Outcome": None,
                "Depends On": None,
                "is_test_case_marker": True  # Flag to identify this as a test case marker
            }
            all_structured_steps.append(test_case_comment)
            
            # Extract steps for this test case
            test_steps = []
            
            # Collect the raw test steps
            for idx in range(test_case_idx, next_test_case_idx):
                row = df.iloc[idx]
                
                # Get step data from appropriate columns
                action_col = 'Test steps' if 'Test steps' in df.columns else 'Action'
                expected_col = 'Expected Result ' if 'Expected Result ' in df.columns else 'Expected Outcome'
                
                action = row.get(action_col, '')
                expected = row.get(expected_col, '')
                
                # Only include rows with actual content
                if pd.notna(action) or pd.notna(expected):
                    step = {
                        "raw_action": str(action) if pd.notna(action) else "",
                        "raw_expected": str(expected) if pd.notna(expected) else ""
                    }
                    test_steps.append(step)
            
            # Process this test case's steps
            if self.use_llm:
                # Use LLM for advanced processing with step breakdown
                structured_steps = self._apply_nlp_processing(test_steps, test_case_id, step_counter)
            else:
                # Use rule-based processing with a step offset
                structured_steps = self._rule_based_processing(test_steps, step_counter)
                
            # Update the step counter for the next test case
            step_counter += len(test_steps)
            
            # Add the processed steps to the overall list
            for step in structured_steps:
                if not step.get("is_test_case_marker", False):  # Skip test case markers when counting
                    all_structured_steps.append(step)
        
        # Remove the test case marker flag from all steps
        for step in all_structured_steps:
            if "is_test_case_marker" in step:
                del step["is_test_case_marker"]
                
        return all_structured_steps
        
    def _create_test_state_from_steps(self, structured_steps: List[Dict[str, Any]], excel_file: str, sheet_name: str) -> TestState:
        """Create a TestState object from structured steps
        
        Args:
            structured_steps: List of structured test steps from parsing
            excel_file: Path to the Excel file
            sheet_name: Name of the sheet the steps came from
            
        Returns:
            TestState object with test steps, test plan, etc.
        """
        # Convert structured steps to TestStep objects
        test_steps = []
        step_id_map = {}  # Map to track original step IDs to new sequential IDs
        
        # First pass: Create TestStep objects and build ID mapping
        for i, step_data in enumerate(structured_steps):
            # Skip test case marker steps
            if step_data.get("Action") == "TestCase":
                continue
            
            # Get step ID, ensuring it exists
            step_id = step_data.get("Step ID", str(i + 1))
            new_step_id = str(i + 1)
            step_id_map[step_id] = new_step_id
            
            # Create the action text from Action and Target
            target = step_data.get("Target", "")
            action = f"{step_data.get('Action', '')} {target if target else ''}".strip()
            
            # Add input value to action if present
            input_val = step_data.get("Input")
            if input_val:
                if isinstance(input_val, dict):
                    # Handle structured input (e.g., username/password)
                    input_str = ", ".join([f"{k}: {v}" for k, v in input_val.items()])
                    action += f" with inputs {input_str}"
                else:
                    action += f" with input '{input_val}'"
                
            # Determine action type from the action or use the one provided
            action_type = self._determine_action_type(action) if "action_type" not in step_data else step_data.get("action_type")
            
            # Create empty dependencies dictionary (will be updated in second pass)
            dependencies = {}
            
            # Create TestStep object
            test_step = TestStep(
                step_number=new_step_id,
                action=action,
                expected_outcome=step_data.get("Expected Outcome"),
                action_type=action_type,
                dependencies=dependencies
            )
            test_steps.append(test_step)
        
        # Second pass: Update dependencies with correct step IDs
        for i, step_data in enumerate(structured_steps):
            # Skip test case marker steps
            if step_data.get("Action") == "TestCase" or i >= len(test_steps):
                continue
                
            depends_on = step_data.get("Depends On")
            if depends_on:
                # Get the correct step number from our mapping
                if depends_on in step_id_map:
                    mapped_dependency = step_id_map[depends_on]
                    test_steps[i].dependencies = {
                        "requires_previous_success": True,
                        "dependent_on_steps": [mapped_dependency],
                        "critical_dependency": False
                    }
                else:
                    # If dependency refers to a step ID that doesn't exist, default to previous step
                    if i > 0:
                        test_steps[i].dependencies = {
                            "requires_previous_success": True,
                            "dependent_on_steps": [str(i)],
                            "critical_dependency": False
                        }
        
        # Create test plan
        test_plan = TestPlan(
            title=f"Test from {excel_file}",
            description=f"Test steps imported from {excel_file}, sheet: {sheet_name}",
            prerequisites=[],
            test_data=[],
            full_text=f"Excel file: {excel_file}, sheet: {sheet_name}",
            critical_paths=[],
            verification_points=[]
        )
        
        # Create raw text representation for debugging
        raw_text = "\n".join([
            f"{step.step_number}. {step.action} | Expected: {step.expected_outcome or ''}"
            for step in test_steps
        ])
        
        # Create TestState
        state = TestState(
            test_steps=test_steps,
            test_plan=test_plan,
            raw_test_input=raw_text,
            parser_confidence=0.95,
            parsing_complete=True,
            max_steps=len(test_steps)
        )
        
        return state
        
    def parse_excel_file(self, excel_file: str, prefix: str = "parsed") -> Dict[str, List[Dict[str, Any]]]:
        """Parse all sheets in an Excel file
        
        Args:
            excel_file: Path to the Excel file
            prefix: Prefix for output JSON files
            
        Returns:
            Dictionary mapping sheet names to parsed steps
        """
        # Get sheet names
        xl = pd.ExcelFile(excel_file)
        sheet_names = xl.sheet_names
        
        if not sheet_names:
            print(f"No sheets found in {excel_file}")
            return {}
            
        results = {}
        
        # Process each sheet
        for sheet_name in sheet_names:
            print(f"\nParsing sheet: {sheet_name}")
            parsed_steps = self._parse_sheet(excel_file, sheet_name)
            
            if parsed_steps:
                results[sheet_name] = parsed_steps
                # Remove the JSON file save operation from here to avoid multiple files
                print(f"  ✅ Parsed sheet '{sheet_name}'")
        
        return results
        
    def _apply_nlp_processing(self, raw_steps: List[Dict[str, str]], test_case_id: str, step_counter: int) -> List[Dict[str, Any]]:
        """Apply NLP techniques to process raw test steps into structured format
        
        Args:
            raw_steps: List of raw test steps with action and expected outcome
            test_case_id: ID of the test case
            step_counter: Counter to maintain continuous step numbering
            
        Returns:
            List of structured test steps with action type, target, input, etc.
        """
        if not raw_steps:
            return []
            
        # Prepare the input for LLM processing
        step_descriptions = []
        for i, step in enumerate(raw_steps):
            step_num = i + 1 + step_counter
            action = step["raw_action"].strip()
            expected = step["raw_expected"].strip()
            step_descriptions.append(f"Step {step_num}:\nAction: {action}\nExpected: {expected}")
        
        # Create the full prompt for LLM
        steps_text = "\n\n".join(step_descriptions)
        prompt = f"""
        I need to convert the following test steps into structured test automation steps.
        Test Case ID: {test_case_id}
        
        {steps_text}
        
        For each step, extract:
        1. Action (navigate, click, enter, select, verify, etc.)
        2. Target (what UI element to act on)
        3. Input value (if applicable)
        4. Expected outcome
        5. Dependencies on previous steps
        
        IMPORTANT: If any step contains multiple actions (e.g., "enter username and password") or multiple verification points, 
        break it down into smaller steps with a single action or verification each. Make sure to establish proper dependencies
        between these sub-steps.
        
        Return JSON in this exact format for all steps:
        [
          {{
            "Step ID": "1",
            "Action": "Navigate",
            "Target": "https://example.com",
            "Input": null,
            "Expected Outcome": "Login page appears",
            "Depends On": null
          }},
          {{
            "Step ID": "2",
            "Action": "Enter",
            "Target": "Username field",
            "Input": "testuser",
            "Expected Outcome": null,
            "Depends On": "1"
          }}
        ]
        """
        
        # Process with LLM for NLP-enhanced parsing
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Extract JSON result from LLM response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = response_text[json_start:json_end]
                structured_steps = json.loads(json_content)
                
                # Validate and clean up the structured steps
                for step in structured_steps:
                    # Ensure all required fields are present
                    for field in ["Step ID", "Action", "Target", "Input", "Expected Outcome", "Depends On"]:
                        if field not in step:
                            step[field] = None
                            
                    # Convert empty strings to null
                    for key, value in step.items():
                        if value == "":
                            step[key] = None
                    
                    # Fix Step IDs to ensure they're in the expected format
                    if "Step ID" in step and step["Step ID"]:
                        # Check if Step ID has "Step " prefix and remove it if present
                        if isinstance(step["Step ID"], str) and step["Step ID"].startswith("Step "):
                            step["Step ID"] = step["Step ID"][5:]
                    
                    # Fix Depends On to ensure it's in the expected format
                    if "Depends On" in step and step["Depends On"]:
                        # Check if Depends On has "Step " prefix and remove it if present
                        if isinstance(step["Depends On"], str) and step["Depends On"].startswith("Step "):
                            step["Depends On"] = step["Depends On"][5:]
                
                return structured_steps
            else:
                print(f"Error: Could not extract valid JSON from LLM response")
                print(f"Response: {response_text[:100]}...")
                return self._rule_based_processing(raw_steps, step_counter)
                
        except Exception as e:
            print(f"Error during LLM processing: {e}")
            # Fall back to rule-based processing if LLM fails
            return self._rule_based_processing(raw_steps, step_counter)

    def _rule_based_processing(self, raw_steps: List[Dict[str, str]], step_counter: int = 0) -> List[Dict[str, Any]]:
        """Rule-based processing for when LLM is not available
        
        Args:
            raw_steps: List of raw test steps
            step_counter: Counter to maintain continuous step numbering
            
        Returns:
            List of structured test steps
        """
        print("Using rule-based processing")
        structured_steps = []
        
        # Define patterns for different action types
        action_patterns = {
            "Navigate": ["navigate", "go to", "open", "launch", "browse"],
            "Click": ["click", "press", "select", "tap", "choose"],
            "Enter": ["enter", "input", "type", "fill", "provide"],
            "Verify": ["verify", "check", "validate", "confirm", "ensure"],
            "Wait": ["wait", "pause"],
            "Switch": ["switch", "change"]
        }
        
        # Process each step
        for i, step in enumerate(raw_steps):
            step_id = str(i + 1 + step_counter)
            raw_action = step["raw_action"].lower()
            raw_expected = step["raw_expected"]
            
            # Try to break down complex steps
            substeps = self._break_down_step(step_id, raw_action, raw_expected, action_patterns)
            
            if substeps:
                # We successfully broke down the step
                structured_steps.extend(substeps)
            else:
                # Process as a single step
                # Determine action type using regex patterns
                action_type = "Other"
                for action, patterns in action_patterns.items():
                    if any(re.search(rf'\b{pattern}\b', raw_action) for pattern in patterns):
                        action_type = action
                        break
                
                # Extract target (simplified approach)
                target = None
                if action_type == "Navigate" and "http" in raw_action:
                    # Extract URL
                    url_match = re.search(r'https?://\S+', raw_action)
                    if url_match:
                        target = url_match.group(0)
                elif action_type in ["Click", "Enter"]:
                    # Try to extract UI element after the action word
                    for pattern in action_patterns[action_type]:
                        match = re.search(rf'\b{pattern}\b\s+(?:on|in|the)?\s*(.+?)(?:\s+and|\s*$)', raw_action)
                        if match:
                            target = match.group(1).strip()
                            break
                
                # Extract input value (for Enter action)
                input_value = None
                if action_type == "Enter":
                    # Look for quotes that might contain input value
                    quotes_match = re.search(r'[\'"]([^\'"]+)[\'"]', raw_action)
                    if quotes_match:
                        input_value = quotes_match.group(1)
                
                # Determine dependencies
                depends_on = None
                if i > 0:  # Most steps depend on the previous step
                    depends_on = str(i + step_counter)
                    
                # Create structured step
                structured_step = {
                    "Step ID": step_id,
                    "Action": action_type,
                    "Target": target,
                    "Input": input_value,
                    "Expected Outcome": raw_expected if raw_expected else None,
                    "Depends On": depends_on
                }
                
                structured_steps.append(structured_step)
        
        return structured_steps
    
    def _break_down_step(self, step_id: str, raw_action: str, raw_expected: str, 
                         action_patterns: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Break down a complex step with multiple actions into smaller steps
        
        Args:
            step_id: Original step ID
            raw_action: Raw action text
            raw_expected: Raw expected outcome text
            action_patterns: Patterns for different action types
            
        Returns:
            List of broken down steps or empty list if step cannot be broken down
        """
        substeps = []
        
        # Check for "and" or multiple action verbs in the step
        actions_found = []
        for action_type, patterns in action_patterns.items():
            for pattern in patterns:
                matches = re.finditer(rf'\b{pattern}\b', raw_action.lower())
                for match in matches:
                    actions_found.append((match.start(), action_type, pattern))
        
        # Sort actions by position in text
        actions_found.sort()
        
        if len(actions_found) <= 1:
            # Not enough actions to break down
            return []
            
        # Break down the step by actions found
        for i, (pos, action_type, pattern) in enumerate(actions_found):
            # Get the text after this action verb until the next action or end of string
            if i < len(actions_found) - 1:
                next_pos = actions_found[i+1][0]
                action_text = raw_action[pos:next_pos]
            else:
                action_text = raw_action[pos:]
                
            # Extract target for this action
            target_match = re.search(rf'\b{pattern}\b\s+(?:on|in|the)?\s*(.+?)(?:\s+and|\s*$)', action_text)
            target = target_match.group(1).strip() if target_match else None
            
            # Extract input value for Enter actions
            input_value = None
            if action_type == "Enter":
                quotes_match = re.search(r'[\'"]([^\'"]+)[\'"]', action_text)
                if quotes_match:
                    input_value = quotes_match.group(1)
            
            # Only add expected outcome to the last substep
            expected = None
            if i == len(actions_found) - 1:
                expected = raw_expected
                
            # Create sub-step ID
            substep_id = f"{step_id}.{i+1}"
            
            # Determine dependencies
            depends_on = None
            if i > 0:
                # This sub-step depends on the previous sub-step
                depends_on = f"{step_id}.{i}"
            elif step_id != "1" and not step_id.startswith("TC"):
                # First sub-step depends on the previous main step, using numeric comparison
                # to ensure we don't create invalid dependencies
                try:
                    step_num = int(float(step_id))
                    depends_on = str(step_num-1)
                except ValueError:
                    # If step_id isn't a number (e.g., "TC1"), don't set dependency
                    pass
                
            # Create structured sub-step
            substep = {
                "Step ID": substep_id,
                "Action": action_type,
                "Target": target,
                "Input": input_value,
                "Expected Outcome": expected,
                "Depends On": depends_on
            }
            
            substeps.append(substep)
            
        return substeps

    def convert_to_test_state(self, structured_steps: List[Dict[str, Any]], sheet_name: str) -> TestState:
        """Convert structured steps to a TestState object for execution
        
        Args:
            structured_steps: List of structured test steps
            sheet_name: Name of the sheet the steps came from
            
        Returns:
            TestState object ready for execution
        """
        # This is a simplified wrapper around _create_test_state_from_steps
        return self._create_test_state_from_steps(structured_steps, f"Execution of {sheet_name}", sheet_name)

def main():
    """
    Run the Excel parser to extract and break down test steps from Excel files
    """
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Parse Excel test cases with automatic step breakdown')
    parser.add_argument('--file', type=str, default='Project Onboarding UI Test Cases.xlsx',
                       help='Path to the Excel file containing test cases')
    parser.add_argument('--sheet', type=str, 
                       help='Specific sheet to parse (optional, will process all sheets if not specified)')
    parser.add_argument('--output', type=str, default='parsed_tests',
                       help='Directory to save parsed results (default: parsed_tests)')
    parser.add_argument('--prefix', type=str, default='parsed',
                       help='Prefix for output JSON files (default: parsed)')
    parser.add_argument('--rule-based', action='store_true',
                       help='Use rule-based parsing instead of LLM')
    args = parser.parse_args()
    
    # Print banner
    print("=" * 80)
    print("Excel Test Parser")
    print("=" * 80)
    
    # Check if the Excel file exists
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found")
        return
    
    # Create parser and parse Excel file
    parser = TestParserAgent(output_dir=args.output, use_llm=not args.rule_based)
    
    if args.sheet:
        print(f"Parsing sheet '{args.sheet}' from {args.file}")
        # Parse specific sheet
        results = {}
        parsed_steps = parser._parse_sheet(args.file, args.sheet)
        if parsed_steps:
            results[args.sheet] = parsed_steps
            # Convert to TestState to save in standard format
            state = parser._create_test_state_from_steps(parsed_steps, args.file, args.sheet)
            # Save parsed steps using the standard method to ensure consistent file naming
            parser._save_parsed_steps(state)
            print(f"Successfully parsed sheet '{args.sheet}'")
    else:
        # Parse all sheets
        print(f"Parsing all sheets from {args.file}")
        results = parser.parse_excel_file(args.file, args.prefix)
    
    # Print summary
    if results:
        total_sheets = len(results)
        total_steps = sum(len(steps) for steps in results.values())
        print("\n" + "=" * 50)
        print(f"Parsing Summary:")
        print(f"Processed {total_sheets} sheets with a total of {total_steps} test steps")
        
        for sheet, steps in results.items():
            print(f"- {sheet}: {len(steps)} steps")
            
            # Count how many steps have dependencies
            steps_with_deps = sum(1 for step in steps if step.get("Depends On"))
            print(f"  - {steps_with_deps} steps have dependencies")
            
            # Analyze step breakdown
            decimal_steps = sum(1 for step in steps if '.' in step.get("Step ID", ""))
            if decimal_steps > 0:
                print(f"  - {decimal_steps} steps are broken down substeps")
    else:
        print("No results were parsed.")
    
    print("\nParsing complete!")

if __name__ == "__main__":
    main() 