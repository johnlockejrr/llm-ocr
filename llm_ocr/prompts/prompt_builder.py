"""Simple, independent prompt builder with JSON configuration and metadata integration."""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Type of prompt output format."""

    STRUCTURED = "structured"  # JSON output required
    SIMPLE = "simple"  # Plain text output


class PromptVersion(Enum):
    """Prompt versions for context enrichment experiments."""

    V1 = "v1"  # Basic prompts
    V2 = "v2"  # With book metadata
    V3 = "v3"  # Enhanced context
    V4 = "v4"  # Russian language


class PromptBuilder:
    """Build prompts from JSON configuration with automatic metadata enrichment."""

    def __init__(self, config_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """Initialize prompt builder.

        Args:
            config_path: Path to prompts_config.json
            metadata_path: Path to dataset JSON file with book metadata
        """
        self.config_path = Path(config_path or self._default_config_path())
        self.metadata_path = Path(metadata_path or self._default_metadata_path())
        self._config = None
        self._metadata = None

    def _default_config_path(self) -> str:
        """Get default config path relative to this file."""
        return str(Path(__file__).parent / "prompts_config.json")

    def _default_metadata_path(self) -> str:
        """Get default metadata path - assumes it's in project root."""
        return str(Path(__file__).parent.parent.parent / "filtered_100_dataset.json")

    @property
    def config(self) -> Dict[str, Any]:
        """Lazy load prompt configuration."""
        if self._config is None:
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
                logger.debug(f"Loaded prompt config from {self.config_path}")
            except FileNotFoundError:
                logger.error(f"Prompt config not found at {self.config_path}")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in prompt config: {e}")
                raise
        return self._config

    def get_metadata(self, document_id: str) -> Dict[str, Any]:
        """Get document metadata from JSON file.

        Args:
            document_id: Document ID to look up

        Returns:
            Dictionary with book metadata or empty dict if not found
        """
        if self._metadata is None:
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
                logger.debug(f"Loaded metadata from {self.metadata_path}")
            except FileNotFoundError:
                logger.warning(f"Metadata file not found at {self.metadata_path}")
                return {}
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in metadata file: {e}")
                return {}

        # Find book containing this document_id
        for book in self._metadata.get("books", []):
            if document_id in book.get("image_ids", []):
                metadata = {
                    "book_title": book.get("title", ""),
                    "book_year": book.get("year", ""),
                    "publication_info": book.get("publication_info", ""),
                }
                logger.debug(f"Found metadata for {document_id}: {metadata}")
                return metadata

        logger.warning(f"No metadata found for document_id: {document_id}")
        return {}

    def build_prompt(
        self, mode: str, prompt_type: PromptType, version: PromptVersion, **kwargs
    ) -> str:
        """Build prompt from JSON components.

        Args:
            mode: Processing mode (single_line, sliding_window, full_page, correction)
            prompt_type: Output format type (structured or simple)
            version: Prompt version for context enrichment
            **kwargs: Template variables, including optional document_id for auto-enrichment

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If mode/type/version combination not found in config
        """
        # Auto-enrich with metadata if document_id provided
        if "document_id" in kwargs:
            metadata = self.get_metadata(kwargs["document_id"])
            kwargs.update(metadata)
            del kwargs["document_id"]  # Remove from template vars

        try:
            # Get template components
            config = self.config
            # Handle both enum and string versions
            version_str = version.value if hasattr(version, 'value') else str(version)
            prompt_type_str = prompt_type.value if hasattr(prompt_type, 'value') else str(prompt_type)
            
            context = config["context_enrichment"][version_str]
            mode_instruction = config["mode_instructions"][mode.lower()]
            output_format = config["output_formats"][prompt_type_str][mode.lower()]

            # Build template
            template = (
                f"{config['components']['base_ocr']}{context}. "
                f"{mode_instruction}. "
                f"{config['components']['orthography']}. "
                f"{output_format}"
            )

            # First replace component references like {json_format}
            for key, value in config["components"].items():
                template = template.replace(f"{{{key}}}", value)

            # Then format with user kwargs - use string safe formatting to avoid issues with JSON
            if kwargs:
                try:
                    # Build replacement dict to avoid issues with JSON braces
                    for key, value in kwargs.items():
                        template = template.replace(f"{{{key}}}", str(value))
                except Exception as e:
                    logger.error(f"Error during template variable replacement: {e}")
                    # Continue with unformatted template rather than failing

            result = template.strip()
            logger.debug(
                f"Built prompt for {mode}/{prompt_type_str}/{version_str}: {len(result)} chars"
            )
            return result

        except KeyError as e:
            raise ValueError(
                f"Invalid configuration: {e}. "
                f"Mode: {mode}, Type: {prompt_type_str}, Version: {version_str}"
            )


# Global convenience instance
_default_builder = None


def get_default_builder() -> PromptBuilder:
    """Get default shared builder instance."""
    global _default_builder
    if _default_builder is None:
        _default_builder = PromptBuilder()
    return _default_builder


def get_prompt(mode: str, prompt_type: PromptType, version: PromptVersion, **kwargs) -> str:
    """Convenience function using default builder.

    Args:
        mode: Processing mode
        prompt_type: Output format type
        version: Prompt version
        **kwargs: Template variables

    Returns:
        Formatted prompt string
    """
    return get_default_builder().build_prompt(mode, prompt_type, version, **kwargs)
