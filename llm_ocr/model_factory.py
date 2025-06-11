# in llm_ocr/factory.py
import logging
from typing import Optional

from llm_ocr.config import settings
from llm_ocr.llm.base import BaseOCRModel
from llm_ocr.llm.claude import ClaudeOCRModel
from llm_ocr.llm.gemini import GeminiOCRModel
from llm_ocr.llm.openai import OpenAIOCRModel
from llm_ocr.llm.together import TogetherOCRModel
from llm_ocr.prompts.prompt import ModelType, PromptVersion

logger = logging.getLogger(__name__)

api_keys = {"anthropic": settings.ANTHROPIC_API_KEY, "openai": settings.OPENAI_API_KEY}

# Simple model registry mapping model names to their types
MODEL_REGISTRY = {
    # Claude models
    "claude-3-haiku-20240307": ModelType.CLAUDE,
    "claude-3-7-sonnet-20250219": ModelType.CLAUDE,
    "claude-3-opus-20240229": ModelType.CLAUDE,
    "claude-3-sonnet-20240229": ModelType.CLAUDE,
    # OpenAI models
    "gpt-4o-2024-08-06": ModelType.GPT,
    "gpt-4.1-2025-04-14": ModelType.GPT,
    "gpt-4-turbo": ModelType.GPT,
    # Gemini models
    "gemini-1.5-pro": ModelType.GEMINI,
    "gemini-1.5-flash": ModelType.GEMINI,
    "gemini-2.0-flash": ModelType.GEMINI,
    "Qwen/Qwen2.5-VL-72B-Instruct": ModelType.TOGETHER,
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": ModelType.TOGETHER,
}


def get_model_type(model_name: str) -> ModelType:
    """
    Get model type from the model name.

    Args:
        model_name: Name of the model

    Returns:
        ModelType enum value
    """
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]

    # Try to infer from name prefix if not in registry
    if model_name.startswith("claude"):
        return ModelType.CLAUDE
    elif model_name.startswith(("gpt", "o", "text-davinci")):
        return ModelType.GPT
    elif model_name.startswith("gemini"):
        return ModelType.GEMINI
    else:
        return ModelType.TOGETHER


def create_model(model_name: str, prompt_version: Optional[PromptVersion] = None) -> BaseOCRModel:
    """
    Create a model instance based on model name.

    Args:
        model_name: Name of the model (e.g., "claude-3-7-sonnet-20250219")
        prompt_version: Version of the prompt to use (deprecated, handled by pipeline)

    Returns:
        Model instance
    """
    # Get model type from name
    model_type = get_model_type(model_name)
    logger.debug("Model type for '%s': %s", model_name, model_type)

    # Create appropriate config (prompt_version is handled by the pipeline, not individual models)
    if model_type == ModelType.CLAUDE:
        logger.debug("Creating Claude model")
        return ClaudeOCRModel(model_name=model_name)

    elif model_type == ModelType.GPT:
        logger.debug("Creating OpenAI model")
        return OpenAIOCRModel(model_name=model_name)

    elif model_type == ModelType.GEMINI:
        logger.debug("Creating Gemini model")
        return GeminiOCRModel(model_name=model_name)

    elif model_type == ModelType.TOGETHER:
        logger.debug("Creating Together model")
        return TogetherOCRModel(model_name=model_name)

    else:
        raise ValueError(f"Unsupported model type for '{model_name}'")
