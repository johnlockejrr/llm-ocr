"""Together AI OCR Model Implementation - Simplified without prompt logic."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

from together import Together

from llm_ocr.config import settings
from llm_ocr.llm.base import BaseOCRModel


class TogetherOCRModel(BaseOCRModel):
    """Together AI implementation of OCR language model for open source models."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = Together(api_key=settings.TOGETHER_API_KEY)
        self.logger = logging.getLogger(__name__)
        self.max_width = 800
        self.max_height = 1000
        self.image_quality = 70

    def _make_api_request(self, prompt: str, image_data: Union[str, List[str]]) -> Dict[str, Any]:
        """Make request to Together AI API using the official library."""
        # Combine prompt with image(s)
        self.logger.debug("Image data: %s, %d", type(image_data), len(image_data))

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                            },
                        ],
                    }
                ],
                temperature=0.2,
                top_p=0.9,
                max_tokens=1024,
            )

            # Extract response content with proper typing
            response_content: str = response.choices[0].message.content or ""

            return {"choices": [{"text": response_content}]}
        except Exception as e:
            self.logger.error(f"API request failed: {str(e)}")
            raise

    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from the model response."""
        # Look for JSON blocks between ```json and ```
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)

        if json_match:
            json_str = json_match.group(1)
            try:
                parsed_json = json.loads(json_str)
                # Ensure we return a Dict[str, Any]
                if isinstance(parsed_json, dict):
                    return parsed_json
                else:
                    return {"data": parsed_json}
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse JSON: {json_str}")
                return {"error": "Invalid JSON format"}

        # If no JSON block, try to find any JSON-like content
        try:
            # Try to find JSON-like content with brackets anywhere in text
            potential_json = re.search(r"(\{.*\})", response_text, re.DOTALL)
            if potential_json:
                parsed_json = json.loads(potential_json.group(1))
                # Ensure we return a Dict[str, Any]
                if isinstance(parsed_json, dict):
                    return parsed_json
                else:
                    return {"data": parsed_json}
        except json.JSONDecodeError:
            pass

        # Last resort: Return the raw text
        self.logger.warning("Could not extract JSON from response")
        return {"line": response_text.strip(), "confidence": 0.0}

    def process_single_line(self, prompt: str, image_base64: str) -> Dict[str, Any]:
        """Process a single line image with pre-built prompt."""

        try:

            # Send the request
            response = self._make_api_request(prompt, image_base64)
            self.logger.info(f"Response received: {response}")

            # Process the response
            if response and "choices" in response and response["choices"]:
                response_text: str = response["choices"][0]["text"]
                result = self._extract_json_from_response(response_text)
                return result
            else:
                self.logger.warning("No text content in the response.")
                return {
                    "line": "",
                    "error": "No text content in response",
                }

        except Exception as e:
            self.logger.error(f"Error processing single line: {str(e)}")
            return {"line": "", "error": str(e)}

    def process_sliding_window(
        self, prompt: str, images_base64: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Process window of lines with pre-built prompt."""
        content = []

        for img_base64 in images_base64:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            )
        content.append({"type": "text", "text": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                temperature=0.2,
                top_p=0.9,
                max_tokens=1024,
            )
            self.logger.info(f"Response received: {response}")

            # Process the response
            if response and response.choices:
                response_text: str = response.choices[0].message.content or ""
                result = self._extract_json_from_response(response_text)
                return result
            else:
                self.logger.warning("No text content in the response for sliding window.")
                return None

        except Exception as e:
            self.logger.error(f"Error processing sliding window: {str(e)}")
            return None

    def process_full_page(self, prompt: str, page_image_base64: str) -> str:
        """Process full page with pre-built prompt."""
        self.logger.debug("Prompt: %s", prompt)

        try:
            response = self._make_api_request(prompt, page_image_base64)

            if not response or "choices" not in response or not response["choices"]:
                self.logger.warning("No text content in the response for full page.")
                return ""

            response_text: str = response["choices"][0]["text"]
            self.logger.info(f"Response received: {response_text}")

            # Check for JSON code blocks
            json_code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
            if json_code_block:
                try:
                    code_content = json_code_block.group(1).strip()
                    if code_content.startswith("["):
                        # Parse the JSON array inside the code block
                        json_array = json.loads(code_content)
                        if isinstance(json_array, list):
                            lines = []
                            for item in json_array:
                                if isinstance(item, dict) and "line" in item:
                                    lines.append(item["line"])
                            if lines:
                                return "\n".join(lines)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON in code block: {e}")

            # Direct array parsing
            if response_text.strip().startswith("["):
                try:
                    direct_json = json.loads(response_text)
                    if isinstance(direct_json, list):
                        lines = []
                        for item in direct_json:
                            if isinstance(item, dict) and "line" in item:
                                lines.append(item["line"])
                        if lines:
                            return "\n".join(lines)
                except json.JSONDecodeError:
                    pass

            # Fall back to regular extraction
            result = self._extract_json_from_response(response_text)

            # Check if result["line"] contains a JSON array string
            if (
                isinstance(result, dict)
                and "line" in result
                and result["line"].strip().startswith("[")
            ):
                try:
                    line_array = json.loads(result["line"])
                    if isinstance(line_array, list):
                        lines = []
                        for item in line_array:
                            if isinstance(item, dict) and "line" in item:
                                lines.append(item["line"])
                        if lines:
                            return "\n".join(lines)
                except json.JSONDecodeError:
                    pass

            # Just return result["line"] if nothing else worked
            if isinstance(result, dict) and "line" in result and isinstance(result["line"], str):
                return result["line"]
            else:
                self.logger.error("Could not extract text from response")
                return ""

        except Exception as e:
            self.logger.error(f"Error processing full page: {str(e)}")
            return ""

    def correct_text(self, prompt: str, text: str, image_base64: str) -> str:
        """Correct OCR text with pre-built prompt."""

        try:
            # Send the request
            response = self._make_api_request(prompt, image_base64)

            if not response or "choices" not in response or not response["choices"]:
                self.logger.warning("No text content in the response for text correction.")
                return text

            # For text correction, we want the raw response text
            response_text: str = response["choices"][0]["text"]
            return response_text.strip()

        except Exception as e:
            self.logger.error(f"Error correcting text: {str(e)}")
            return text  # Return original on error
