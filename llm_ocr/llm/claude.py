import json
import time
import logging
from typing import List, Dict, Any, Optional, Union
import base64
import anthropic

from llm_ocr.settings import ANTHROPIC_API_KEY
from llm_ocr.models import ProcessingMode
from llm_ocr.llm.base import BaseOCRModel
from llm_ocr.prompts.prompt import PromptVersion, ModelType, get_prompt



class ClaudeOCRModel(BaseOCRModel):
    """Claude implementation of OCR language model."""
    
    def __init__(self, model_name: str, prompt_version: Optional[PromptVersion] = None):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model_name = model_name
        print(f"Model name: {model_name}")
        self.model_type = ModelType.CLAUDE
        print(f"Model type: {self.model_type}")
        self.prompt_version = prompt_version
        print(f"Prompt version: {self.prompt_version}")
        self.logger = logging.getLogger(__name__)
    
    def process_single_line(self, image_base64: str) -> Dict[str, Any]:
        """Process a single line image."""
        start_time = time.time()
 
        prompt = get_prompt("SINGLE_LINE", self.model_type, self.prompt_version)
                
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=2000,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        
        result = self._extract_json_from_response(response.content[0].text)
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
                
        return result
    
    def process_sliding_window(self, images_base64: List[str]) -> Optional[Dict[str, Any]]:
        """Process window of lines."""
        content = []
        for img_base64 in images_base64:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_base64
                }
            })
                
        prompt = get_prompt("SLIDING_WINDOW", self.model_type, self.prompt_version)
        content.append({"type": "text", "text": prompt})
        
        try:
            start_time = time.time()
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=2000,
                temperature=0.0,
                messages=[{"role": "user", "content": content}]
            )
            
            result = self._extract_json_from_response(response.content[0].text)
            result["processing_time"] = time.time() - start_time
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing sliding window: {str(e)}")
            return None

    def process_full_page(self, page_image_base64: str, document_id: str) -> str:
        """Process full page."""
        prompt = get_prompt("FULL_PAGE", self.model_type, self.prompt_version, document_id=document_id)

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=2000,
                temperature=0.0,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": page_image_base64
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            self.logger.info(f"Response received: {response}")
        except Exception as e:
            self.logger.error(f"Error processing full page: {str(e)}")
            return []
            
        try:
            # Use your existing method to extract JSON
            result = self._extract_json_from_response(response.content[0].text)
            
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
                    combined_text = "\n".join([item['line'] for item in processed_results])
                    self.logger.info(f"Combined text: {combined_text}")
                    return combined_text
                else:
                    self.logger.warning("No valid line items found in results")
                    return ""
            else:
                self.logger.error(f"Expected list but got {type(result)}: {result}")
                return ""
                
        except (ValueError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to parse JSON from response: {e}")
            self.logger.debug(f"Response text: {response.content[0].text}")
    
    def correct_text(self, text: str, image_base64: str, mode: str = "line") -> str:
        """Correct OCR text and format as a single line."""
        if mode == "line":
            prompt = get_prompt("TEXT_CORRECTION", self.model_type, self.prompt_version, text=text)
        else:
            prompt = get_prompt("TEXT_CORRECTION_WITH_PARAGRAPHS", self.model_type, self.prompt_version, text=text)
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=2000,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        
        return response.content[0].text.strip()
    
    def correct_text_with_paragraphs(self, text: str, image_base64: str) -> Union[str, List[str]]:
        """Correct OCR text preserving paragraph structure."""

        prompt = get_prompt(
            "TEXT_CORRECTION_WITH_PARAGRAPHS",
            self.model_type,
            self.prompt_version,
            text=text
        )
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=2000,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        
        corrected_text = response.content[0].text.strip()
        
        # Split into paragraphs
        if "\n\n" in corrected_text:
            return corrected_text.split("\n\n")
        elif "\n" in corrected_text:
            return corrected_text.split("\n")
        else:
            return corrected_text