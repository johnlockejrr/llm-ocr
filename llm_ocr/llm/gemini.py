import base64
import logging
import time
from typing import Any, Dict, List, Optional, Union

import google.generativeai as genai

from llm_ocr.llm.base import BaseOCRModel
from llm_ocr.prompts.prompt import ModelType, PromptVersion, get_prompt
from llm_ocr.settings import GEMINI_API_KEY

# If you need more specific types from the library for hinting:
# from google.generativeai.types import PartDict # Example


# Define a more specific type for the parts we're constructing, if helpful for clarity
# A Part can be a string (for text) or a dictionary (for inline data like images)
# This is a simplified representation; the actual PartDict can be more complex.
ContentPart = Union[str, Dict[str, Any]]


class GeminiOCRModel(BaseOCRModel):
    """Gemini implementation of OCR language model."""

    def __init__(self, model_name: str, prompt_version: PromptVersion):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name  # Keep if needed for logging or other purposes
        self.model_type = ModelType.GEMINI
        self.prompt_version = prompt_version
        self.logger = logging.getLogger(__name__)

    def _construct_image_part(self, image_base64: str) -> Dict[str, Any]:
        """Helper to decode base64 image and structure it as an image part."""
        image_bytes = base64.b64decode(image_base64)
        # Correct structure for an inline image part
        return {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}}

    def _get_response_text(self, response: genai.types.GenerateContentResponse) -> Optional[str]:
        """Safely extracts text from the response."""
        try:
            if response and response.candidates:
                # Concatenate text from all parts in the first candidate's content
                text_parts = []
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
                if text_parts:
                    return "".join(text_parts)
            # Fallback or if no text parts specifically (e.g. if response.text is preferred and reliable)
            # elif response and hasattr(response, 'text') and response.text:
            #     return response.text
        except Exception as e:
            self.logger.error(f"Error extracting text from response: {e}")

        if response and response.prompt_feedback:
            self.logger.warning(f"Request was blocked or had issues: {response.prompt_feedback}")
        return None

    def process_single_line(self, image_base64: str, id: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        # Assuming prompt_version is always set via __init__
        prompt_str = get_prompt("SINGLE_LINE", self.model_type, self.prompt_version)

        try:
            image_part = self._construct_image_part(image_base64)

            # The 'contents' argument takes an iterable of parts.
            # A string is a valid part, and our image_part dict is also a valid part.
            contents_payload: List[ContentPart] = [prompt_str, image_part]

            response = self.model.generate_content(
                contents_payload
            )  # Removed model=self.model_name

            response_text = self._get_response_text(response)

            if response_text is not None:  # Check for None explicitly
                self.logger.info(f"Response text received: {response_text[:200]}...")  # Log snippet
                result = self._extract_json_from_response(response_text)
                self.logger.info(f"Result: {result}")

                if isinstance(result, dict):
                    result["processing_time"] = time.time() - start_time
                    return result
                else:
                    return {
                        "line": str(result) if result is not None else "",
                        "processing_time": time.time() - start_time,
                        "error": "Unexpected response format or empty result",
                    }
            else:
                self.logger.warning(
                    "No text content in the response or response was empty/blocked."
                )
                error_msg = "No text content in response"
                if response and response.prompt_feedback:
                    error_msg = f"Request issue: {response.prompt_feedback.block_reason}"
                return {
                    "line": "",
                    "processing_time": time.time() - start_time,
                    "error": error_msg,
                }

        except Exception as e:
            self.logger.error(f"Error processing single line: {str(e)}", exc_info=True)
            return {"line": "", "processing_time": time.time() - start_time, "error": str(e)}

    def process_sliding_window(
        self, images_base64: List[str], id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        start_time = time.time()
        prompt_str = get_prompt("SLIDING_WINDOW", self.model_type, self.prompt_version)

        try:
            # Initialize content_parts with the prompt string first
            # The type hint helps MyPy understand the list can contain mixed valid part types.
            content_parts: List[ContentPart] = [prompt_str]

            for img_b64 in images_base64:
                image_part = self._construct_image_part(img_b64)
                content_parts.append(image_part)

            response = self.model.generate_content(content_parts)  # Removed model= argument

            response_text = self._get_response_text(response)

            if response_text is not None:
                self.logger.info(f"Sliding window response: {response_text[:200]}...")
                result = self._extract_json_from_response(response_text)

                if isinstance(result, dict):
                    result["processing_time"] = time.time() - start_time
                    return result
                else:  # Handle case where result is not a dict (e.g. list or string)
                    return {
                        "lines": (
                            result
                            if isinstance(result, list)
                            else ([str(result)] if result is not None else [])
                        ),
                        "processing_time": time.time() - start_time,
                        "error": "Unexpected response format from JSON extraction",
                    }
            else:
                self.logger.warning(
                    "No text content in the response for sliding window or response was empty/blocked."
                )
                # Optionally, return a structured error
                error_msg = "No text content in response for sliding window"
                if response and response.prompt_feedback:
                    error_msg = f"Request issue: {response.prompt_feedback.block_reason}"
                return {
                    "lines": [],
                    "processing_time": time.time() - start_time,
                    "error": error_msg,
                }

        except Exception as e:
            self.logger.error(f"Error processing sliding window: {str(e)}", exc_info=True)
            return None  # Or a dict with error info

    def process_full_page(self, page_image_base64: str, document_id: str) -> str:
        start_time = time.time()
        prompt_str = get_prompt(
            "FULL_PAGE", self.model_type, self.prompt_version, document_id=document_id
        )

        try:
            image_part = self._construct_image_part(page_image_base64)
            content_parts: List[ContentPart] = [prompt_str, image_part]
            response = self.model.generate_content(content_parts)

            response_text = self._get_response_text(response)

            if response_text is None:
                self.logger.warning(
                    "No text content in the response for full page or response was empty/blocked."
                )
                return ""

            self.logger.info(f"Full page response text received: {response_text[:200]}...")
            result = self._extract_json_from_response(response_text)

            if isinstance(result, list):
                processed_lines = []
                for item in result:
                    if isinstance(item, dict) and "line" in item:
                        processed_lines.append(item["line"])
                    elif isinstance(item, str):  # If model returns list of strings
                        processed_lines.append(item)
                    else:
                        self.logger.warning(f"Unexpected item format in full page results: {item}")

                if processed_lines:
                    combined_text = "\n".join(processed_lines)
                    self.logger.info(f"Combined text: {combined_text[:200]}...")
                    return combined_text
                else:
                    self.logger.warning("No valid line items found in results for full page.")
                    return ""
            elif isinstance(result, str):  # If the JSON extraction returns a single string
                self.logger.info(f"Full page result was a single string: {result[:200]}...")
                return result
            else:
                self.logger.error(
                    f"Expected list or string from JSON extraction for full page, but got {type(result)}"
                )
                return ""

        except Exception as e:
            self.logger.error(f"Error processing full page: {str(e)}", exc_info=True)
            return ""

    def correct_text(self, text_to_correct: str, image_base64: str, mode: str = "single") -> str:
        prompt_str = get_prompt(
            "TEXT_CORRECTION", self.model_type, self.prompt_version, text=text_to_correct
        )

        try:
            image_part = self._construct_image_part(image_base64)
            content_parts: List[ContentPart] = [prompt_str, image_part]
            response = self.model.generate_content(content_parts)

            corrected_text = self._get_response_text(response)

            if corrected_text is None:
                self.logger.warning(
                    "No text content in the response for text correction. Returning original."
                )
                return text_to_correct

            return corrected_text.strip()

        except Exception as e:
            self.logger.error(f"Error correcting text: {str(e)}", exc_info=True)
            return text_to_correct

    def correct_text_with_paragraphs(
        self, text_to_correct: str, image_base64: str
    ) -> Union[str, List[str]]:
        prompt_str = get_prompt(
            "TEXT_CORRECTION_WITH_PARAGRAPHS",
            self.model_type,
            self.prompt_version,
            text=text_to_correct,
        )

        # Define fallback based on original text structure
        if "\n\n" in text_to_correct:
            fallback_result = text_to_correct.split("\n\n")
        elif "\n" in text_to_correct:
            fallback_result = text_to_correct.split("\n")
        else:
            fallback_result = text_to_correct

        try:
            image_part = self._construct_image_part(image_base64)
            content_parts: List[ContentPart] = [prompt_str, image_part]
            response = self.model.generate_content(content_parts)

            corrected_text_full = self._get_response_text(response)

            if corrected_text_full is None:
                self.logger.warning(
                    "No text content in the response for paragraph correction. Returning original structure."
                )
                return fallback_result

            corrected_text_stripped = corrected_text_full.strip()

            if "\n\n" in corrected_text_stripped:
                return corrected_text_stripped.split("\n\n")
            elif "\n" in corrected_text_stripped:
                return corrected_text_stripped.split("\n")
            else:
                return corrected_text_stripped

        except Exception as e:
            self.logger.error(f"Error correcting text with paragraphs: {str(e)}", exc_info=True)
            # Fallback to splitting the simply corrected text, or the original if that fails
            try:
                simple_corrected = self.correct_text(text_to_correct, image_base64)
                if "\n\n" in simple_corrected:
                    return simple_corrected.split("\n\n")
                elif "\n" in simple_corrected:
                    return simple_corrected.split("\n")
                else:
                    return simple_corrected
            except Exception:  # If simple correction also fails
                return fallback_result
