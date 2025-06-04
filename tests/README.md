# Tests for LLM OCR Package

This directory contains comprehensive tests for the LLM OCR package.

## Structure

```
tests/
├── conftest.py                    # Shared fixtures and pytest configuration
├── run_tests.py                   # Test runner script (in parent directory)
├── unit/                          # Unit tests
├── integration/                   # Integration tests
├── performance/                   # Performance tests
├── mocks/                         # Mock utilities and providers
└── test_fixtures/                 # Test data and sample files
    ├── sample_data/               # Sample ALTO XML, images, ground truth
    └── mock_responses/            # Mock API responses
```

## Running Tests

### Basic Test Run (No API Keys Required)
```bash
# Run all tests except API-dependent ones
pytest tests/

# Run specific test categories
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
```

### Running Tests with API Integration (Requires API Keys)
```bash
# Skip API-dependent tests (default behavior)
pytest tests/ -m "not requires_api"

# Run only API-dependent tests (requires environment variables)
pytest tests/ -m "requires_api"
```

### Test Coverage
```bash
pytest tests/ --cov=llm_ocr --cov-report=term-missing
```

### Environment Setup for API Tests
To run API-dependent tests, copy the `.env.template` file and fill in your API key values:

```bash
cp .env.template .env
# Edit .env and add your API key values
```

## Test Configuration

### Quick Start

```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --type unit

# Run with coverage
python run_tests.py --coverage

# Run specific markers
python run_tests.py --markers "not slow"
```

### Using pytest directly

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests only  
pytest tests/integration/

# With coverage
pytest --cov=llm_ocr --cov-report=term-missing

# Parallel execution
pytest -n 4

# Specific test file
pytest tests/unit/test_models.py

# Specific test function
pytest tests/unit/test_models.py::TestOCRMetrics::test_default_initialization
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual components in isolation:

- **Models** (`test_models.py`) - Data model validation
- **Evaluators** (`evaluators/`) - Metrics calculation and evaluation logic
- **LLM Providers** (`llm/`) - Individual LLM provider implementations
- **Pipelines** (`pipelines/`) - Processing pipeline components
- **Processors** (`processors/`) - Input format processors (ALTO XML)
- **Utils** (`utils/`) - Utility functions

### Integration Tests (`tests/integration/`)

Test component interactions:

- **Workflow Integration** - End-to-end workflow execution
- **Cross-Component** - Multiple components working together
- **Data Flow** - Data passing between components

### Performance Tests (`tests/performance/`)

Test performance characteristics:

- **Memory Usage** - Memory consumption under load
- **Processing Speed** - Timing benchmarks
- **Scalability** - Performance with large datasets

## Test Markers

Use pytest markers to categorize and select tests:

```python
@pytest.mark.unit          # Unit tests
@pytest.mark.integration   # Integration tests  
@pytest.mark.performance   # Performance tests
@pytest.mark.slow          # Slow-running tests
@pytest.mark.requires_api  # Tests requiring API keys
```

Examples:
```bash
# Run only fast tests
pytest -m "not slow"

# Run only unit tests
pytest -m unit

# Skip API-dependent tests
pytest -m "not requires_api"
```

## Fixtures

### Shared Fixtures (`conftest.py`)

- **sample_text_pairs** - Ground truth/extracted text pairs
- **sample_old_russian_text** - Old Russian text with historical characters
- **sample_line_objects** - Line objects for testing
- **mock_llm_model** - Mock LLM model
- **temp_dir** - Temporary directory for test files
- **ocr_evaluator** - Configured OCR evaluator

### Configuration Fixtures

- **default_model_config** - Standard model configuration
- **default_evaluation_config** - Standard evaluation configuration

### Mock Fixtures

- **mock_api_responses** - Mock responses for different providers
- **mock_env_vars** - Mock environment variables for API keys

## Mocking Strategy

### API Calls
All LLM provider API calls are mocked to avoid:
- Network dependencies
- API rate limits
- API costs
- Flaky tests

### File I/O
Test files are created in temporary directories and cleaned up automatically.

### External Dependencies
External services and network calls are mocked or stubbed.

## Test Data

### Sample Data (`test_fixtures/sample_data/`)

- **test_document.xml** - Sample ALTO XML with text coordinates
- **test_document.txt** - Corresponding ground truth text
- Sample images (created programmatically in tests)

### Mock Responses (`test_fixtures/mock_responses/`)

- **claude_responses.json** - Mock Claude API responses
- **openai_responses.json** - Mock OpenAI API responses
- Various error conditions and edge cases

## Writing New Tests

### Test Structure

```python
import pytest
from llm_ocr.your_module import YourClass

class TestYourClass:
    """Test YourClass functionality."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        instance = YourClass()
        
        # Act
        result = instance.method()
        
        # Assert
        assert result == expected_value
    
    @pytest.mark.parametrize("input,expected", [
        ("input1", "output1"),
        ("input2", "output2"),
    ])
    def test_multiple_inputs(self, input, expected):
        """Test with multiple input values."""
        instance = YourClass()
        assert instance.method(input) == expected
```

### Using Fixtures

```python
def test_with_fixtures(self, ocr_evaluator, sample_text_pairs):
    """Test using shared fixtures."""
    for pair in sample_text_pairs:
        result = ocr_evaluator.evaluate_line(
            pair["ground_truth"], 
            pair["extracted"]
        )
        assert result.char_accuracy >= 0.0
```

### Mocking External Dependencies

```python
from unittest.mock import patch, MagicMock

@patch('llm_ocr.llm.claude.anthropic.Anthropic')
def test_with_mock_api(self, mock_anthropic):
    """Test with mocked API."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="test response")]
    mock_anthropic.return_value.messages.create.return_value = mock_response
    
    # Your test code here
```

## Coverage Goals

- **Unit Tests**: >90% line coverage
- **Integration Tests**: Critical workflows covered
- **Performance Tests**: Key performance metrics tracked

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH includes package root
2. **Fixture Not Found**: Check fixture is defined in conftest.py or imported
3. **Mock Not Working**: Verify mock patch target is correct
4. **Slow Tests**: Use `@pytest.mark.slow` and run with `-m "not slow"`

### Running Specific Tests

```bash
# Run tests matching pattern
pytest -k "test_models"

# Run tests in specific file
pytest tests/unit/test_models.py

# Run with verbose output
pytest -v

# Stop on first failure
pytest -x

# Debug mode
pytest --pdb
```