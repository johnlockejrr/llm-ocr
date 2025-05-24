# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM-powered OCR evaluation and correction package that supports multiple language models (Claude, GPT-4, Gemini, Together AI) for OCR processing and text correction tasks. The package processes ALTO XML files with corresponding images to perform OCR and evaluate results against ground truth data.

## Architecture

### Core Components

- **Pipelines** (`llm_ocr/pipelines/`): OCR and correction processing workflows
  - `ocr.py`: Main OCR pipeline with single-line, sliding window, and full-page processing modes
  - `correction.py`: LLM-based text correction pipeline
  - `base.py`: Abstract base class for all pipelines

- **LLM Models** (`llm_ocr/llm/`): Multi-provider LLM support
  - `base.py`: Abstract base class defining the model interface
  - `claude.py`, `openai.py`, `gemini.py`, `together.py`: Provider-specific implementations
  - `model_factory.py`: Factory for creating model instances

- **Evaluators** (`llm_ocr/evaluators/`): Comprehensive metrics and evaluation
  - `evaluator.py`: Main evaluation engine with multiple metric types
  - `metrics/`: Character accuracy, word accuracy, case preservation, error analysis
  - `evaluation.py`: High-level evaluation service

- **Processors** (`llm_ocr/processors/`): Input format handling
  - `alto.py`: ALTO XML format processor for extracting text and line coordinates

- **Workflow** (`llm_ocr/workflow.py`): Main orchestration class that manages the complete OCR pipeline

### Data Models

- **ProcessingMode**: Enum defining processing strategies (SINGLE_LINE, SLIDING_WINDOW, FULL_PAGE)
- **Line**: Text line with optional image data and coordinates
- **OCRResult**: Complete OCR result with metrics and analysis
- **OCRMetrics**: Standardized evaluation metrics

## Development Commands

### Testing

```bash
python -m unittest discover llm_ocr/
```

Individual test files can be run with:
```bash
python -m unittest llm_ocr.test_metrics
python -m unittest llm_ocr.evaluators.test_evaluator
python -m unittest llm_ocr.pipelines.test_correction
```

### Running OCR Workflows

The main entry point is the `OCRPipelineWorkflow` class in `workflow.py`. Example usage:

```python
from llm_ocr.workflow import OCRPipelineWorkflow
from llm_ocr.models import ProcessingMode

workflow = OCRPipelineWorkflow(
    id="document_id",
    folder="ground_truth",  # Contains .xml, .jpeg files
    model_name="claude-3-7-sonnet-20250219",
    modes=[ProcessingMode.FULL_PAGE],
    prompt_version=PromptVersion.V3
)

results = workflow.run_pipeline()
```

## Configuration

- **Model Configuration**: `config.py` contains model parameters, evaluation settings, and defaults
- **Prompt Versions**: Different prompt versions are defined in `prompts/prompt.py`
- **Default Models**: Claude Sonnet, GPT-4o, with support for custom model parameters

## Key Patterns

- All LLM models implement the `BaseOCRModel` interface with standardized methods
- Pipelines use the evaluator pattern for consistent metrics calculation
- Results are stored in consolidated JSON files per document
- Error handling with graceful fallbacks and detailed logging
- Multi-model support with unified result structures

## Ground Truth Data

The `ground_truth/` directory contains test data with ALTO XML files, corresponding JPEG images, and expected text outputs for evaluation.