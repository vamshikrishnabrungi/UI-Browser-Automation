import os
import json
import asyncio
import datetime
import traceback
import re
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from browser_use import Agent, Browser, BrowserConfig

# Import our custom agents and state from local files
from state import TestState, TestStep, StepResult
from ParserAgent import TestParserAgent

# Load environment variables
load_dotenv()

class TestExecutor:
    """Class to execute tests and track results"""
    
    def __init__(self, screenshots_dir=None, results_dir=None):
        """Initialize the test executor with directories for screenshots and results"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configure directories
        self.screenshots_dir = screenshots_dir or 'screenshots'
        self.results_dir = results_dir or 'execution_results'
        
        # Create a ParserAgent instance to use its methods
        self.parser_agent = TestParserAgent(use_llm=False)
        
        # Ensure directories use absolute paths
        if not os.path.isabs(self.screenshots_dir):
            self.screenshots_dir = os.path.abspath(self.screenshots_dir)
        if not os.path.isabs(self.results_dir):
            self.results_dir = os.path.abspath(self.results_dir)
        
        print(f"TestExecutor initialized with screenshots_dir={self.screenshots_dir}, results_dir={self.results_dir}")
        
        # Create run-specific directories
        self.run_id = f"run_{timestamp}"
        self.run_screenshots_dir = os.path.join(self.screenshots_dir, self.run_id)
        
        # Create directories with explicit permissions
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.run_screenshots_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize test results
        self.results = {
            "test_steps": [],
            "results": [],
            "is_complete": False,
            "current_step": 0,
            "current_url": "",
            "error": None,
            "browser": "chrome",
            "timestamp": timestamp,
            "test_plan": {},
            "auth_state": {
                "auth_completed": False,
                "auth_method": "",
                "login_attempts": 0,
                "auth_step": 0,
                "credentials_used": {},
                "current_auth_flow": ""
            }
        }
        
        # Screenshot counter for unique filenames
        self.screenshot_counter = 0
        
        # Track login state to prevent loops
        self.authentication_completed = False
        self.login_attempts = 0
        self.max_login_attempts = 3
    
    def setup_from_parsed_steps(self, parsed_state):
        """Set up the test executor from parsed test steps"""
        # Store test metadata
        self.results["test_plan"]["full_text"] = parsed_state.raw_test_input
        if parsed_state.test_plan and parsed_state.test_plan.title:
            self.results["test_plan"]["title"] = parsed_state.test_plan.title
            
        # Convert TestStep objects to dictionaries for the test executor
        self.results["test_steps"] = [
            {
                "step_number": step.step_number,
                "action": step.action,
                "expected_outcome": step.expected_outcome,
                "action_type": step.action_type or self.parser_agent._determine_action_type(step.action),
                "dependencies": step.dependencies or self._identify_dependencies(step.action, i+1)
            }
            for i, step in enumerate(parsed_state.test_steps)
        ]
        
        # Update max steps
        self.results["max_steps"] = len(self.results["test_steps"])
        
        # Analyze test steps for planning
        self._analyze_test_flow()
        self._preprocess_auth_flows()
    
    def _identify_dependencies(self, action_text, step_num):
        """Identify step dependencies for better error recovery"""
        dependencies = {
            "requires_previous_success": True,
            "dependent_on_steps": [],
            "critical_dependency": False
        }
        
        # Most steps depend on previous step
        if step_num > 1:
            dependencies["dependent_on_steps"] = [str(step_num - 1)]
        
        # Handle authentication steps with special logic
        if any(term in action_text.lower() for term in ["authentication", "password", "login", "verify"]):
            dependencies["critical_dependency"] = True
            
        return dependencies
    
    def _analyze_test_flow(self):
        """Analyze the test flow to identify critical paths and verification points"""
        steps = self.results["test_steps"]
        
        # Find verification points - steps that validate expected outcomes
        verification_points = [
            step["step_number"] for step in steps 
            if step.get("action_type") == "verification" or "should" in step.get("expected_outcome", "").lower()
        ]
        self.results["test_plan"]["verification_points"] = verification_points
        
        # Identify authentication steps
        auth_steps = []
        for step in steps:
            action_text = step.get("action", "").lower()
            if any(term in action_text for term in ["authentication", "password", "login", "verify", "push notification"]):
                auth_steps.append(step["step_number"])
        
        if auth_steps:
            self.results["auth_state"]["auth_steps"] = auth_steps
            self.results["test_plan"]["auth_steps"] = auth_steps

    def _preprocess_auth_flows(self):
        """Pre-process test steps to identify and annotate authentication flows"""
        auth_steps = self.results.get("auth_state", {}).get("auth_steps", [])
        if not auth_steps:
            return
        
        # Mark authentication-related steps
        for step in self.results["test_steps"]:
            step_num = step["step_number"]
            
            # Convert to consistent types for comparison
            try:
                step_num_int = int(step_num) if isinstance(step_num, str) else step_num
                auth_steps_int = [int(s) if isinstance(s, str) else s for s in auth_steps]
                
                if step_num_int in auth_steps_int:
                    step["is_auth_step"] = True
                    
                    # Set context based on step content
                    action = step.get("action", "").lower()
                    if "gpid" in action or "username" in action:
                        step["auth_context"] = "username_input"
                    elif "password" in action:
                        step["auth_context"] = "password_input"
                    elif "verify" in action and "button" in action:
                        step["auth_context"] = "verification_action"
                    elif "push" in action and "notification" in action:
                        step["auth_context"] = "push_notification"
            except ValueError:
                pass

    async def take_screenshot(self, agent, step_number, action_name="unknown"):
        """Take a screenshot using the agent's browser"""
        try:
            self.screenshot_counter += 1
            filename = f"step_{step_number:02d}_{self.screenshot_counter:03d}_{action_name}.png"
            screenshot_path = os.path.join(self.run_screenshots_dir, filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
            
            # Take screenshot if browser is available
            if agent and hasattr(agent, 'browser') and hasattr(agent.browser, 'page'):
                await agent.browser.page.screenshot(path=screenshot_path, full_page=True)
                print(f"Screenshot saved to {screenshot_path}")
                return screenshot_path
            else:
                print(f"No page available for screenshot")
                return None
        except Exception as e:
            print(f"Failed to take screenshot: {e}")
            return None
    
    def record_step_result(self, step_number, action, status, error=None, url=None, expected_outcome=None, actual_outcome=None, screenshot=None):
        """Record the result of a test step"""
        result = {
            "step": step_number,
            "action": action,
            "status": status,  # "passed" or "failed"
            "error": error,
            "url": url,
            "expected_outcome": expected_outcome,
            "actual_outcome": actual_outcome,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "screenshot": screenshot
        }
        self.results["results"].append(result)
        self.results["current_step"] = step_number
        
        if url:
            self.results["current_url"] = url
        
        if error:
            self.results["error"] = error
            
        # Print result for easier debugging
        outcome_status = "✅ PASSED" if status == "passed" else "❌ FAILED"
        print(f"Step {step_number} - {outcome_status}: {action}")
        if error:
            print(f"  Error: {error}")
    
    def save_results(self):
        """Save the test results to a JSON file"""
        results_file = os.path.join(self.results_dir, f"execution_results_{self.run_id}.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Test results saved to: {results_file}")
        return results_file
        
    def get_step_details(self, step_number):
        """Get the details for a specific test step"""
        if 1 <= step_number <= len(self.results["test_steps"]):
            return self.results["test_steps"][step_number - 1]
        return None

class BrowserExecutorAgent:
    """Agent that manages browser test execution within the LangGraph pipeline"""
    
    def __init__(self):
        """Initialize the browser executor agent"""
        self.test_executor = None
        self.browser = None
        self.agent = None
        
    async def setup_executor(self, state: TestState) -> None:
        """Set up the browser executor with the test steps from the state"""
        # Create test executor with explicit screenshot directory
        screenshots_dir = os.path.join(os.getcwd(), 'screenshots')
        results_dir = os.path.join(os.getcwd(), 'execution_results')
        
        self.test_executor = TestExecutor(screenshots_dir=screenshots_dir, results_dir=results_dir)
        
        # Configure browser
        browser_config = BrowserConfig(
            headless=False,
            browser=state.browser,
            viewport="1920,1080",
            slow_mo=100
        )
        
        # Configure LLM for the agent
        llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        
        # Set up test executor from parsed state
        self.test_executor.setup_from_parsed_steps(state)
        
        # Create agent with task
        self.agent = Agent(
            task="\n".join([
                f"{step.step_number}. {step.action}\n   Expected: {step.expected_outcome or 'N/A'}"
                for step in state.test_steps
            ]),
            llm=llm
        )
        
        # Configure browser after agent creation
        self.agent.browser = Browser(browser_config)
        self.browser = self.agent.browser
    
    async def execute_tests(self, state: TestState) -> TestState:
        """Execute the tests and update the state with results"""
        if not self.test_executor or not self.agent:
            await self.setup_executor(state)
        
        print("Starting test execution with browser...")
        
        # Ensure screenshot directory exists
        os.makedirs(self.test_executor.run_screenshots_dir, exist_ok=True)
        
        try:
            # Take initial screenshot
            await self.test_executor.take_screenshot(self.agent, 0, "initial_state")
            
            # Create sanitized execution instructions to avoid content filter issues
            safe_instructions = """
            IMPORTANT INSTRUCTIONS FOR TEST EXECUTION:
            
            1. SEQUENTIAL EXECUTION: Execute each test step in sequence
            2. ELEMENT RECOGNITION: Scan the page for elements thoroughly
            3. LOGIN PROCESS: Complete login steps in sequence
            4. HANDLING DELAYS: Wait for page loads before proceeding
            5. ERROR HANDLING: Try alternative approaches if a step fails
            
            Each test step is critical and must be attempted in sequence.
            """
            
            # Update agent's task with safe instructions
            if hasattr(self.agent, 'task'):
                modified_task = f"{safe_instructions}\n\n" + self.agent.task
                if hasattr(self.agent, '_task'):
                    self.agent._task = modified_task
            
            # Configure timeouts for better element detection
            if hasattr(self.agent.browser, 'page') and self.agent.browser.page:
                await self.agent.browser.page.set_default_timeout(10000)  # 10 seconds
                await self.agent.browser.page.set_default_navigation_timeout(60000)  # 60 seconds
            
            # Execute test steps
            try:
                # Allow multiple steps for retries and error handling
                await self.agent.run(max_steps=len(state.test_steps) * 3)
                
                # Check if any steps were skipped
                if hasattr(self.agent, 'history') and self.agent.history:
                    executed_steps = []
                    for entry in self.agent.history:
                        if isinstance(entry, dict) and 'output' in entry:
                            output = entry['output']
                            step_match = re.search(r'Step\s+(\d+)', output) if isinstance(output, str) else None
                            if step_match:
                                executed_steps.append(int(step_match.group(1)))
                    
                    # Record skipped steps
                    all_steps = set(range(1, len(state.test_steps) + 1))
                    executed_steps_set = set(executed_steps)
                    skipped_steps = all_steps - executed_steps_set
                    
                    if skipped_steps:
                        print(f"Warning: The following steps were skipped: {sorted(skipped_steps)}")
                        
                        # Record placeholder results for skipped steps
                        for step_num in skipped_steps:
                            step = next((s for s in state.test_steps if 
                                       (isinstance(s.step_number, int) and s.step_number == step_num) or
                                       (isinstance(s.step_number, str) and int(s.step_number) == step_num)), None)
                            if step:
                                self.test_executor.record_step_result(
                                    step_number=step_num,
                                    action=step.action,
                                    status="skipped",
                                    error="Step was skipped during test execution",
                                    expected_outcome=step.expected_outcome,
                                    actual_outcome="Step was not executed by the agent"
                                )
            except Exception as e:
                print(f"Error during test execution: {e}")
                traceback.print_exc()
            
            # Take final screenshot
            await self.test_executor.take_screenshot(self.agent, 999, "final_state")
            
            # Mark test as complete
            self.test_executor.results["is_complete"] = True
            
            # Copy results from TestExecutor to TestState
            state.is_complete = True
            state.execution_complete = True
            state.raw_execution_results = self.test_executor.results
            
            # Convert results to StepResult objects
            results = []
            
            if self.test_executor.results.get("results", []):
                for r in self.test_executor.results.get("results", []):
                    result = StepResult(
                        step=r.get("step", 0),
                        action=r.get("action", "Unknown action"),
                        status=r.get("status", "failed"),
                        error=r.get("error"),
                        url=r.get("url"),
                        expected_outcome=r.get("expected_outcome"),
                        actual_outcome=r.get("actual_outcome"),
                        screenshot=r.get("screenshot"),
                        confidence=0.9 if r.get("status") == "passed" else 0.7
                    )
                    results.append(result)
            
            # If no results, generate basic ones from agent history
            if not results and hasattr(self.agent, 'history') and self.agent.history:
                print("Generating results from agent history...")
                
                # Extract steps and their status from history
                step_results = {}
                
                for entry in self.agent.history:
                    if isinstance(entry, dict) and 'output' in entry:
                        output = entry['output']
                        
                        if isinstance(output, str) and 'Step' in output and ':' in output:
                            parts = output.split(':')
                            if len(parts) >= 2:
                                step_info = parts[0].strip()
                                step_result = parts[1].strip()
                                
                                # Try to extract step number
                                step_match = re.search(r'Step\s+(\d+)', step_info)
                                if step_match:
                                    step_num = int(step_match.group(1))
                                    
                                    # Determine if step passed or failed
                                    status = "passed"
                                    if "fail" in step_result.lower() or "error" in step_result.lower():
                                        status = "failed"
                                        
                                    # Store step result
                                    step_results[step_num] = {
                                        "status": status,
                                        "message": step_result
                                    }
                
                # Create results from history data
                for step in state.test_steps:
                    step_num = step.step_number
                    if isinstance(step_num, str):
                        try:
                            step_num = int(step_num)
                        except:
                            step_num = 0
                            
                    if step_num in step_results:
                        result = StepResult(
                            step=step_num,
                            action=step.action,
                            status=step_results[step_num]["status"],
                            error=None if step_results[step_num]["status"] == "passed" else step_results[step_num]["message"],
                            expected_outcome=step.expected_outcome,
                            actual_outcome=step_results[step_num]["message"],
                            confidence=0.9 if step_results[step_num]["status"] == "passed" else 0.7
                        )
                    else:
                        # Step wasn't executed
                        result = StepResult(
                            step=step_num,
                            action=step.action,
                            status="skipped",
                            error="No explicit result found in agent history",
                            expected_outcome=step.expected_outcome,
                            actual_outcome="Step may have been skipped or not properly recorded",
                            confidence=0.5
                        )
                        
                    results.append(result)
            
            state.results = results
            
            # Set overall executor confidence based on pass/fail ratio
            passed = sum(1 for r in state.results if r.status == "passed")
            total = len(state.results) if state.results else 1
            state.executor_confidence = passed / total if total > 0 else 0.0
            
            # Save results to a file
            results_file = self.test_executor.save_results()
            
            # Ensure parser confidence is a float
            if hasattr(state, 'parser_confidence') and not isinstance(state.parser_confidence, float):
                state.parser_confidence = float(state.parser_confidence) if state.parser_confidence else 0.8
            
            return state
            
        finally:
            # Clean up browser
            if self.agent and self.agent.browser:
                await self.agent.browser.close()

# Create a simple function to run a standalone execution
async def run_standalone_execution(test_steps_file: str, browser: str = "chrome") -> TestState:
    """Run a standalone browser execution without the full pipeline"""
    
    # Read the test steps file
    with open(test_steps_file, 'r') as f:
        test_content = f.read()
        
    # Create parser to parse the steps
    parser = TestParserAgent()
    state = parser.parse_test_steps(test_content)
    
    # Set browser
    state.browser = browser
    
    # Execute tests
    executor = BrowserExecutorAgent()
    final_state = await executor.execute_tests(state)
    
    return final_state

# Standalone execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run browser test execution')
    parser.add_argument('--file', type=str, required=True, help='Path to test steps file')
    parser.add_argument('--browser', type=str, default='chrome', choices=['chrome', 'edge'], 
                      help='Browser to use for execution')
    args = parser.parse_args()
    
    print(f"Running standalone browser execution on {args.file} using {args.browser}")
    final_state = asyncio.run(run_standalone_execution(args.file, args.browser))
    
    # Print results
    print("\n=== TEST EXECUTION SUMMARY ===")
    print(f"Total steps: {len(final_state.test_steps)}")
    print(f"Executed steps: {len(final_state.results)}")
    print(f"Passed steps: {sum(1 for r in final_state.results if r.status == 'passed')}")
    print(f"Failed steps: {sum(1 for r in final_state.results if r.status == 'failed')}")