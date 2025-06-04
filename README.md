# LLM OCR Package

A LLM-powered OCR evaluation and correction package that supports multiple language models for OCR processing and text correction tasks.

## Features

- **Multi-Provider LLM Support**: Claude, GPT-4, Gemini, and Together AI
- **Multiple Processing Modes**: Single-line, sliding window, and full-page OCR
- **Evaluation**: Character accuracy, word accuracy, case preservation, and error analysis
- **OCR Correction**: LLM-based text correction with configurable output formats
- **ALTO XML Support**: Process ALTO XML files with corresponding images
- **Detailed Metrics**: Extensive evaluation metrics with error pattern analysis
- **Workflow Management**: Complete pipeline orchestration with result tracking

## Installation

### Using pip (recommended)

```bash
pip install llm-ocr
```

### From source

```bash
git clone https://github.com/mary-lev/llm-ocr.git
cd llm-ocr
pip install -e .
```

### Development installation

```bash
git clone https://github.com/mary-lev/llm-ocr.git
cd llm-ocr
pip install -e ".[dev]"
```

## Quick Start

### 1. Set up API keys

Copy the `.env.template` file and fill in your API key values:

```bash
cp .env.template .env
# Edit .env and add your API key values
```

### 2. Basic usage

```python
from llm_ocr.workflow import OCRPipelineWorkflow
from llm_ocr.models import ProcessingMode
from llm_ocr.prompts.prompt import PromptVersion

# Initialize workflow
workflow = OCRPipelineWorkflow(
    id="document_001",
    folder="ground_truth",  # Contains .xml, .jpeg files
    model_name="claude-3-7-sonnet-20250219",
    modes=[ProcessingMode.FULL_PAGE],
    prompt_version=PromptVersion.V3
)

# Run complete pipeline
results = workflow.run_pipeline()

# Or run individual steps
workflow.run_ocr()
workflow.evaluate_ocr()
workflow.run_correction()
workflow.evaluate_correction()
```

## Architecture

### Core Components

- **Pipelines** (`llm_ocr/pipelines/`): OCR and correction processing workflows
- **LLM Models** (`llm_ocr/llm/`): Multi-provider LLM support with unified interface
- **Evaluators** (`llm_ocr/evaluators/`): Comprehensive metrics and evaluation framework
- **Processors** (`llm_ocr/processors/`): Input format handling (ALTO XML)
- **Workflow** (`llm_ocr/workflow.py`): Main orchestration and result management

### Processing Modes

- **SINGLE_LINE**: Process each text line individually
- **SLIDING_WINDOW**: Process lines with context window
- **FULL_PAGE**: Process entire page at once

### Supported Models

- **Claude**: Anthropic's Claude models (3.5 Sonnet, etc.)
- **GPT-4**: OpenAI's GPT-4 models
- **Gemini**: Google's Gemini models
- **Together AI**: Various open-source models via Together

## Configuration

### Model Configuration

```python
from llm_ocr.config import ModelConfig

config = ModelConfig(
    max_tokens=2048,
    temperature=0.0,
    sliding_window_size=3,
    batch_size=10
)
```

### Evaluation Configuration

```python
from llm_ocr.config import EvaluationConfig

eval_config = EvaluationConfig(
    use_char_accuracy=True,
    use_word_accuracy=True,
    use_old_char_preservation=True,
    include_detailed_analysis=True
)
```

## Data Format

### Input Requirements

Your data folder should contain:
- `{id}.xml`: ALTO XML file with text coordinates
- `{id}.jpeg`: Corresponding image file
- `{id}.txt`: Ground truth text (optional, for evaluation)

### Output Format

Results are saved as JSON files with complete metrics and analysis:

```json
{
  "document_info": {
    "document_name": "document_001",
    "timestamp": "20250124_143022"
  },
  "models": {
    "claude-3-7-sonnet-20250219": {
      "ocr_results": {
        "fullpage": {
          "lines": [...],
          "metrics": {...}
        }
      },
      "correction_results": {
        "original_ocr_text": "...",
        "corrected_text": "...",
        "metrics": {...}
      }
    }
  }
}
```

## Advanced Usage

### Custom Model Integration

```python
from llm_ocr.llm.base import BaseOCRModel

class CustomOCRModel(BaseOCRModel):
    def process_single_line(self, image_base64: str):
        # Implement single line processing
        pass
    
    def process_full_page(self, page_image_base64: str, id: str):
        # Implement full page processing  
        pass
    
    def correct_text(self, text: str, image_base64: str):
        # Implement text correction
        pass
```

### Batch Processing

```python
from llm_ocr.workflow import run_multi_model_workflow

results = run_multi_model_workflow(
    xml_path="data/document.xml",
    image_path="data/document.jpeg", 
    ground_truth_path="data/document.txt",
    model_names=["claude-3-7-sonnet-20250219", "gpt-4o-2024-08-06"],
    output_dir="results"
)
```

## Development

### Running Tests

```bash
# Run all tests
python -m unittest discover llm_ocr/

# Run specific test files
python -m unittest llm_ocr.test_metrics
python -m unittest llm_ocr.evaluators.test_evaluator
```

### Code Quality

```bash
# Format code
black llm_ocr/
isort llm_ocr/

# Lint code  
flake8 llm_ocr/
mypy llm_ocr/
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `python -m unittest discover`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{llm_ocr_package,
  title = {LLM OCR: Multi-Provider OCR Evaluation and Correction},
  author = {Maria Levchenko},
  year = {2025},
  url = {https://github.com/mary-lev/llm-ocr}
}
```