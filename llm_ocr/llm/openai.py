import json
import time
import logging
from typing import List, Dict, Any, Optional, Union
import base64
from openai import OpenAI
from pydantic import BaseModel

from llm_ocr.settings import OPENAI_API_KEY
from llm_ocr.llm.base import BaseOCRModel
from llm_ocr.prompts.prompt import PromptVersion, ModelType, get_prompt


# Pydantic models for response parsing
class Line(BaseModel):
    line: str

class LineGroup(BaseModel):
    lines: List[Line]

class Page(BaseModel):
    lines: List[Line]

class Page_with_Paragraphs(BaseModel):
    paragraphs: List[str]


class OpenAIOCRModel(BaseOCRModel):
    """OpenAI implementation of OCR language model."""
    
    def __init__(self, model_name: str, prompt_version: Optional[PromptVersion] = None):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name
        self.model_type = ModelType.GPT
        self.prompt_version = prompt_version
        self.logger = logging.getLogger(__name__)
            
    def process_single_line(self, image_base64: str) -> Dict[str, Any]:
        """Process a single line image."""
        start_time = time.time()   
        prompt = get_prompt("SINGLE_LINE", self.model_type, self.prompt_version)
                
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]},
                ],
                response_format=Line,
            )
            
            result = completion.choices[0].message.parsed.model_dump()
            result["processing_time"] = time.time() - start_time
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing single line: {str(e)}")
            return {"line": "", "processing_time": time.time() - start_time, "error": str(e)}
    
    def process_sliding_window(self, images_base64: List[str]) -> Union[List[Dict[str, Any]], Dict[str, Any], None]:
        """Process window of lines."""
        start_time = time.time()
        content = []
        
        for img_base64 in images_base64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
                
        prompt = get_prompt("SLIDING_WINDOW", self.model_type, self.prompt_version)
        content.append({"type": "text", "text": prompt})
        
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": content},
                ],
                response_format=LineGroup,
            )
            
            result = completion.choices[0].message.parsed
            
            if len(result.lines) == 1:
                # Return single line result
                return result.lines[0].model_dump()
            else:
                # Return middle line for sliding window
                middle_idx = len(images_base64) // 2
                if middle_idx < len(result.lines):
                    line_result = result.lines[middle_idx].model_dump()
                    line_result["processing_time"] = time.time() - start_time
                    return line_result
                else:
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error processing sliding window: {str(e)}")
            return None

    def process_full_page(self, page_image_base64: str, document_id: str) -> List[Dict[str, Any]]:
        """Process full page."""
        prompt = get_prompt("FULL_PAGE", self.model_type, self.prompt_version, document_id=document_id)

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{page_image_base64}"
                            }
                        }
                    ]},
                ],
                response_format=LineGroup,
            )
            
            result = completion.choices[0].message.parsed
            self.logger.info(f"Full page: {result}")
            extracted_lines = "\n".join([line.line for line in result.lines])
            self.logger.info(f"Extracted lines: {extracted_lines}")
            
            return extracted_lines
                    
        except Exception as e:
            self.logger.error(f"Error processing full page: {str(e)}")
            return ""
    
    def correct_text(self, text: str, image_base64: str) -> str:
        """Correct OCR text and format as a single paragraph."""
        prompt = get_prompt(
            "TEXT_CORRECTION",
            self.model_type,
            self.prompt_version,
            text=text
        )
        
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]},
                ],
                response_format=Line,
            )
            
            return completion.choices[0].message.parsed.line
            
        except Exception as e:
            self.logger.error(f"Error correcting text: {str(e)}")
            return text  # Return original on error
    
    def correct_text_with_paragraphs(self, text: str, image_base64: str) -> Union[str, List[str]]:
        """Correct OCR text preserving paragraph structure."""
        prompt = get_prompt("TEXT_CORRECTION_WITH_PARAGRAPHS", self.model_type, self.prompt_version, text=text)
        
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]},
                ],
                response_format=Page_with_Paragraphs,
            )
            
            result = completion.choices[0].message.parsed
            return result.paragraphs
            
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