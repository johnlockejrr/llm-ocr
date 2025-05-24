import logging
import time
from typing import Any, Dict, List, Optional, Union
from google.genai import Client
import google.generativeai as genai

from llm_ocr.llm.base import BaseOCRModel
from llm_ocr.prompts.prompt import ModelType, PromptVersion, get_prompt
from llm_ocr.settings import GEMINI_API_KEY


class GeminiOCRModel(BaseOCRModel):
    """Gemini implementation of OCR language model."""

    def __init__(self, model_name: str, prompt_version: Optional[PromptVersion] = None):
        self.client = Client(api_key=GEMINI_API_KEY)
        self.model_name = model_name
        self.model_type = ModelType.GEMINI
        self.prompt_version = prompt_version
        self.logger = logging.getLogger(__name__)

    def process_single_line(self, image_base64: str, id: Optional[str] = None) -> Dict[str, Any]:
        """Process a single line image."""
        start_time = time.time()
        prompt = get_prompt("SINGLE_LINE", self.model_type, self.prompt_version or PromptVersion.V1)

        try:
            # Prepare the image for Gemini
            # image_data = base64.b64decode(image_base64)
            image_part = genai.Part.from_bytes(data=image_base64, mime_type="image/jpeg")

            # Send the request
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"parts": [{"text": prompt}, image_part]}],
            )

            # Process the response
            if response:
                self.logger.info(f"Response received: {response}")
                response_text: str = response.candidates[0].content.parts[0].text
                print(f"Response text: {response_text}")
                result = self._extract_json_from_response(response_text)
                print(f"Result: {result}")

                # Ensure result is a dictionary before adding processing_time
                if isinstance(result, dict):
                    result["processing_time"] = time.time() - start_time
                    return result
                else:
                    # Handle case where result is not a dict
                    return {
                        "line": str(result) if result else "",
                        "processing_time": time.time() - start_time,
                        "error": "Unexpected response format",
                    }
            else:
                self.logger.warning("No text content in the response.")
                return {
                    "line": "",
                    "processing_time": time.time() - start_time,
                    "error": "No text content in response",
                }

        except Exception as e:
            self.logger.error(f"Error processing single line: {str(e)}")
            return {"line": "", "processing_time": time.time() - start_time, "error": str(e)}

    def process_sliding_window(
        self, images_base64: List[str], id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Process window of lines."""
        start_time = time.time()
        prompt = get_prompt(
            "SLIDING_WINDOW", self.model_type, self.prompt_version or PromptVersion.V1
        )

        try:
            # Prepare all parts (prompt + images)
            parts = [{"text": prompt}]

            # Add image parts
            for img_base64 in images_base64:
                image_part = genai.Part.from_bytes(data=img_base64, mime_type="image/jpeg")
                parts.append(image_part)

            # Send the request
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"parts": parts}],
            )

            # Process the response
            if response:
                response_text: str = response.text
                result = self._extract_json_from_response(response_text)
                print(f"Sliding window response: {response_text}")

                # Ensure result is a dictionary before adding processing_time
                if isinstance(result, dict):
                    result["processing_time"] = time.time() - start_time
                    return result
                else:
                    # Handle case where result is not a dict
                    return {
                        "lines": result if isinstance(result, list) else [str(result)],
                        "processing_time": time.time() - start_time,
                        "error": "Unexpected response format",
                    }
            else:
                self.logger.warning("No text content in the response for sliding window.")
                return None

        except Exception as e:
            self.logger.error(f"Error processing sliding window: {str(e)}")
            return None

    def process_full_page(self, page_image_base64: str, document_id: str) -> str:
        """Process full page."""
        prompt = get_prompt(
            "FULL_PAGE",
            self.model_type,
            self.prompt_version or PromptVersion.V1,
            document_id=document_id,
        )

        try:
            # Prepare the image part
            image_part = genai.Part.from_bytes(data=page_image_base64, mime_type="image/jpeg")

            # Send the request
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"parts": [{"text": prompt}, image_part]}],
            )

            if not response:
                self.logger.warning("No text content in the response for full page.")
                return ""

            self.logger.info(f"Response received: {response}")

            # Process the response
            response_text: str = response.text
            result = self._extract_json_from_response(response_text)

            # Check if result is a list (array of lines)
            if isinstance(result, list):
                # Process each line in the results
                processed_results = []
                for item in result:
                    if isinstance(item, dict) and "line" in item:
                        processed_results.append(item)
                    else:
                        self.logger.warning(f"Unexpected item format in full page results: {item}")

                self.logger.info(f"Processed results: {processed_results}")

                if processed_results:
                    combined_text = "\n".join([item["line"] for item in processed_results])
                    self.logger.info(f"Combined text: {combined_text}")
                    return combined_text
                else:
                    self.logger.warning("No valid line items found in results")
                    return ""
            else:
                self.logger.error(f"Expected list but got {type(result)}: {result}")
                return ""

        except Exception as e:
            self.logger.error(f"Error processing full page: {str(e)}")
            return ""

    def correct_text(self, text: str, image_base64: str, mode: str = "single") -> str:
        """Correct OCR text and format as a single paragraph."""
        prompt = get_prompt(
            "TEXT_CORRECTION", self.model_type, self.prompt_version or PromptVersion.V1, text=text
        )

        try:
            # Prepare the image part
            image_part = genai.Part.from_bytes(data=image_base64, mime_type="image/jpeg")

            # Send the request
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"parts": [{"text": prompt}, image_part]}],
            )

            if not response:
                self.logger.warning("No text content in the response for text correction.")
                return text

            # For text correction, we want the raw response text
            response_text: str = response.text
            return response_text.strip()

        except Exception as e:
            self.logger.error(f"Error correcting text: {str(e)}")
            return text  # Return original on error

    def correct_text_with_paragraphs(self, text: str, image_base64: str) -> Union[str, List[str]]:
        """Correct OCR text preserving paragraph structure."""
        prompt = get_prompt(
            "TEXT_CORRECTION_WITH_PARAGRAPHS",
            self.model_type,
            self.prompt_version or PromptVersion.V1,
            text=text,
        )

        try:
            # Prepare the image part
            image_part = genai.Part.from_bytes(data=image_base64, mime_type="image/jpeg")

            # Send the request
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"parts": [{"text": prompt}, image_part]}],
            )

            if not response:
                self.logger.warning("No text content in the response for paragraph correction.")
                # Fall back to original text
                if "\n\n" in text:
                    return text.split("\n\n")
                elif "\n" in text:
                    return text.split("\n")
                else:
                    return text

            response_text: str = response.text
            corrected_text = response_text.strip()

            # Split into paragraphs
            if "\n\n" in corrected_text:
                return corrected_text.split("\n\n")
            elif "\n" in corrected_text:
                return corrected_text.split("\n")
            else:
                return corrected_text

        except Exception as e:
            self.logger.error(f"Error correcting text with paragraphs: {str(e)}")
            # Fallback to simple paragraph splitting
            corrected = self.correct_text(text, image_base64)
            if "\n\n" in corrected:
                return corrected.split("\n\n")
            elif "\n" in corrected:
                return corrected.split("\n")
            else:
                return corrected
