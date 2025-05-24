"""
Configuration classes for OCR language models and evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type, Union

from llm_ocr.prompts.prompt import ModelType, PromptVersion

# Default configuration parameters
DEFAULT_MODEL_PARAMS = {
    "max_tokens": 2048,
    "temperature": 0.0,
    "sliding_window_size": 3,
    "batch_size": 10,
    "retry_attempts": 3,
    "retry_delay": 1.0,
}

# Default evaluation parameters
DEFAULT_EVAL_PARAMS = {
    "old_russian_chars": "ѣѲѳѵѢѴъьїi",
    "include_detailed_analysis": True,
    "match_threshold": 0.5,
}

# Default model names
DEFAULT_CLAUDE_MODEL = "claude-3-7-sonnet-20250219"
DEFAULT_GPT_MODEL = "gpt-4o-2024-08-06"
WINDOW_SIZE = 3


@dataclass
class ModelConfig:
    """Base configuration for OCR language models."""

    # Common parameters
    max_tokens: int = 2048
    temperature: float = 0.0
    sliding_window_size: int = 3
    batch_size: int = 10
    retry_attempts: int = 3
    retry_delay: float = 1.0

    # Model-specific parameters
    model_name: str = "default-model"
    model_type: ModelType = field(default=ModelType.GEMINI)  # Will be set by subclasses
    prompt_version: PromptVersion = PromptVersion.V1

    # Optional advanced parameters that can be overridden
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """Configuration for OCR evaluation."""

    # General evaluation settings
    old_russian_chars: str = "ѣѲѳѵѢѴъьїi"
    include_detailed_analysis: bool = True
    match_threshold: float = 0.5

    # Metrics configuration
    use_char_accuracy: bool = True
    use_word_accuracy: bool = True
    use_old_char_preservation: bool = True
    use_case_accuracy: bool = True

    # Similarity calculation weights
    char_similarity_weight: float = 0.7
    word_similarity_weight: float = 0.3

    # Advanced metrics can be toggled
    use_levenshtein_distance: bool = True
    use_jaro_winkler: bool = False

    # Export settings
    export_csv: bool = True
    export_json: bool = True

    # Visualization settings
    create_diff_visualizations: bool = False
