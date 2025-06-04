"""
Settings module for LLM OCR package.
Loads configuration from environment variables.
"""
from dotenv import load_dotenv
load_dotenv()

import os
from typing import Optional


def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with optional default."""
    return os.getenv(name, default)

# API Keys for LLM providers
OPENAI_API_KEY = get_env_var("OPENAI_API_KEY")
ANTHROPIC_API_KEY = get_env_var("ANTHROPIC_API_KEY") 
GEMINI_API_KEY = get_env_var("GEMINI_API_KEY")
TOGETHER_API_KEY = get_env_var("TOGETHER_API_KEY")

# Optional providers
DEEP_SEEK_API_KEY = get_env_var("DEEP_SEEK_API_KEY")
DEEPINFRA_API_KEY = get_env_var("DEEPINFRA_API_KEY")

# Validation function for required API keys
def validate_api_key(key_name: str, key_value: Optional[str]) -> None:
    """Validate that required API key is set."""
    if not key_value:
        raise ValueError(
            f"{key_name} is not set. Please set it as an environment variable "
            f"or create a .env file based on .env.template"
        )