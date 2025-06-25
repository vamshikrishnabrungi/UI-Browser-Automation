# Browser Test Automation Framework

An advanced AI-powered browser test automation framework for executing UI test cases defined in Excel spreadsheets or text format. This framework leverages natural language processing (NLP) to interpret test steps and employs LLM-based agents to execute and validate tests in a browser environment.

## Overview

This framework automates UI testing by:
1. Parsing test cases from Excel files or text descriptions
2. Executing test steps in Chrome or Edge browsers
3. Validating test results and generating comprehensive reports
4. Handling authentication and complex UI interactions
5. Providing detailed screenshots and execution logs

## Architecture

The framework follows a modular architecture with three main agents:

1. **TestParserAgent**: Parses test cases from Excel or text input into structured test steps
2. **BrowserExecutorAgent**: Executes test steps in a browser environment
3. **TestValidationAgent**: Validates test results and generates reports

The flow is orchestrated using a LangGraph pipeline:

```
Parser → Executor → Validator
```

## Key Features

- **Natural Language Processing**: Understand test steps written in plain English
- **Multi-browser Support**: Works with Chrome and Edge
- **Authentication Handling**: Sophisticated authentication flow detection and handling
- **Screenshot Capture**: Automatic screenshots at each step
- **Dependency Analysis**: Smart detection of dependencies between test steps
- **Error Recovery**: Continues execution after failures with graceful recovery
- **Detailed Reporting**: Comprehensive execution reports and summaries

## Requirements

```
browser-use
python-dotenv
openai
pytest
pytest-html
pytest-xdist
langgraph
langchain
langchain-openai
azure-identity
azure-core
pydantic
pandas
openpyxl
playwright
```

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables in a `.env` file:
   ```
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_API_VERSION=your_api_version
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
   ```

## Usage

Update the details in the Manual test Case sheet in the Excel sheet

use the below command to run the pipeline

 python run_pipeline.py --test-file "Project Onboarding UI Test Cases.xlsx" --sheet "Manual Test Case" --filter-resilient

## Project Structure

- `BrowserExecutorAgent.py`: Main agent for executing tests in a browser
- `ParserAgent.py`: Agent for parsing test steps from various formats
- `ValidationAgent.py`: Agent for validating test results
- `state.py`: Definition of the test state data structures
- `run_pipeline.py`: Main orchestrator for running the test pipeline
- `requirements.txt`: Required dependencies
- `execution_results/`: Directory containing execution results and reports
- `parsed_tests/`: Directory containing parsed test step files
- `screenshots/`: Directory containing test execution screenshots

## Results and Reporting

The framework generates detailed execution results in the `execution_results/` directory:
- Execution results for each test run
- Summary reports with statistics and metrics
- Pass/fail information with detailed error messages
- Validation feedback and suggested fixes

Screenshot evidence is stored in the `screenshots/` directory organized by run timestamp.

## Advanced Features

### Authentication Flow Detection

The framework can intelligently detect and handle multi-step authentication processes, including:
- Username/password authentication
- Push notifications
- Security verification steps

### Dependency Analysis

The system automatically analyzes dependencies between test steps to:
- Identify critical paths in the test
- Determine which steps depend on previous steps
- Handle failures gracefully by understanding the impact

### Error Recovery

When errors occur, the framework implements recovery strategies:
- Batched execution of remaining steps
- Alternative approaches for failed steps
- Comprehensive error documentation

## License

This project is proprietary software.

## Contact

For support or questions, please contact the development team.