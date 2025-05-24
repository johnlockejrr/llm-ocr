# in llm_ocr/factory.py
from llm_ocr.llm.base import BaseOCRModel
from llm_ocr.llm.claude import ClaudeOCRModel
from llm_ocr.llm.gemini import GeminiOCRModel
from llm_ocr.llm.openai import OpenAIOCRModel
from llm_ocr.llm.together import TogetherOCRModel
from llm_ocr.prompts.prompt import ModelType, PromptVersion
from llm_ocr.settings import ANTHROPIC_API_KEY, OPENAI_API_KEY

api_keys = {"anthropic": ANTHROPIC_API_KEY, "openai": OPENAI_API_KEY}

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


def create_model(model_name: str, prompt_version: PromptVersion) -> BaseOCRModel:
    """
    Create a model instance based on model name.

    Args:
        model_name: Name of the model (e.g., "claude-3-7-sonnet-20250219")
        prompt_version: Version of the prompt to use (optional)

    Returns:
        Model instance
    """
    # Get model type from name
    model_type = get_model_type(model_name)
    print(f"Model type for '{model_name}': {model_type}")

    # Create appropriate config
    if model_type == ModelType.CLAUDE:
        print("Creating Claude model...")
        return ClaudeOCRModel(model_name=model_name, prompt_version=prompt_version)

    elif model_type == ModelType.GPT:
        print("Creating OpenAI model...")
        return OpenAIOCRModel(model_name=model_name, prompt_version=prompt_version)

    elif model_type == ModelType.GEMINI:
        print("Creating Gemini model...")
        return GeminiOCRModel(model_name=model_name, prompt_version=prompt_version)

    elif model_type == ModelType.TOGETHER:
        print("Creating Together model...")
        return TogetherOCRModel(model_name=model_name, prompt_version=prompt_version)

    else:
        raise ValueError(f"Unsupported model type for '{model_name}'")
