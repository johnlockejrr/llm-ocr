"""Module containing all prompts used in the OCR system with model-specific versioning."""

import json
import logging
import os
from enum import Enum, auto
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enum representing different model types."""

    CLAUDE = auto()
    GPT = auto()
    GEMINI = auto()
    TOGETHER = auto()
    # Add more model types as needed


class PromptVersion(Enum):
    """Enum representing prompt versions."""

    V1 = auto()
    V2 = auto()
    V3 = auto()  # context enhanced
    V4 = auto()  # Russian language specific


class PromptTemplates:
    """Container for prompt templates with model-specific versioning support."""

    # Dictionary structure: {model_type: {version: prompt_text}}

    SINGLE_LINE = {
        ModelType.CLAUDE: {
            PromptVersion.V1: """
            Extract the OCR text from this 18th century Russian book line.
            Preserve the original Old Russian orthography.
            Respond with ONLY a JSON object containing the extracted text in the 'line' field.
            """,
            PromptVersion.V2: """
            Extract the OCR text from this 18th century Russian book line.
            Preserve all original Old Russian orthography including ѣ, Ѳ, ѳ, ѵ, and ъ characters.
            Respond with ONLY a JSON object containing the extracted text in the 'line' field.
            """,
        },
        ModelType.GPT: {
            PromptVersion.V1: """
            You are an expert in Old Russian text recognition.
            Extract the OCR text from this 18th century Russian book line.
            Preserve the original Old Russian orthography.
            Respond with ONLY a JSON object containing the extracted text in the 'line' field.
            """,
            PromptVersion.V2: """
            You are an expert in Old Russian text recognition and paleography.
            Extract the OCR text from this 18th century Russian book line with perfect accuracy.
            Preserve all original Old Russian orthography including ѣ, Ѳ, ѳ, ѵ, and ъ characters.
            Pay special attention to superscript letters, titlos, and abbreviated words.
            Respond with ONLY a JSON object containing the extracted text in the 'line' field.
            Do not include any explanations or notes.
            """,
        },
        ModelType.GEMINI: {
            PromptVersion.V1: """
            Task: Extract the OCR text from this 18th century Russian book line.
            Important: Preserve the original Old Russian orthography.
            Format: Respond with ONLY a JSON object containing the extracted text in the 'line' field.
            """,
            PromptVersion.V2: """
            Task: Extract the OCR text from this 18th century Russian book line.
            Important: 
            - Preserve all original Old Russian orthography including ѣ, Ѳ, ѳ, ѵ, and ъ characters
            Format: Return ONLY a JSON object with {"line": "your extracted text"}
            No explanations needed.
            """,
        },
        ModelType.TOGETHER: {
            PromptVersion.V1: """
            You are an expert in Old Russian text recognition.
            Extract the OCR text from this 18th century Russian book line.
            Preserve the original Old Russian orthography.
            Respond with ONLY a JSON object containing the extracted text in the 'line' field.
            """,
            PromptVersion.V2: """
            You are an expert in Old Russian text recognition and paleography.
            Extract the OCR text from this 18th century Russian book line with perfect accuracy.
            Preserve all original Old Russian orthography including ѣ, Ѳ, ѳ, ѵ, and ъ characters.
            Respond with ONLY a JSON object containing the extracted text in the 'line' field.
            Do not include any explanations or notes.
            """,
        },
    }

    SLIDING_WINDOW = {
        ModelType.CLAUDE: {
            PromptVersion.V1: """
            Extract the text from these consecutive lines of an 18th century Russian book.
            Focus on the middle line while using surrounding lines as context.
            Preserve the original Old Russian orthography.
            Respond with ONLY a JSON object containing the extracted text of the middle line in the 'line' field.
            Do not include any additional explanations.
            """,
            PromptVersion.V2: """
            Extract the text from these consecutive lines of an 18th century Russian book.
            Focus specifically on extracting the middle line while using surrounding lines for context.
            Preserve all original Old Russian orthography including ѣ, Ѳ, ѳ, ѵ, and ъ characters.
            Respond with ONLY a JSON object containing the extracted text of the middle line in the 'line' field.
            Do not include any explanations.
            """,
        },
        ModelType.GPT: {
            PromptVersion.V1: """
            You are an expert in Old Russian text recognition.
            Extract the text from these consecutive lines of an 18th century Russian book.
            Focus on the middle line while using surrounding lines as context.
            Preserve the original Old Russian orthography.
            Respond with ONLY a JSON object containing the extracted text of the middle line in the 'line' field.
            Do not include any additional explanations.
            """,
            PromptVersion.V2: """
            You are an expert in Old Russian text recognition and paleography.
            Task: Extract the text from these consecutive lines of an 18th century Russian book.
            Instruction: Focus specifically on the middle line, using surrounding lines only for context.
            Requirements:
            - Preserve all original Old Russian orthography including ѣ, Ѳ, ѳ, ѵ, and ъ characters
            - Be extremely precise with punctuation
            Format: Respond with ONLY a JSON object containing the extracted text in the format {"line": "your extracted text"}
            Do not provide any explanations, comments, or additional information.
            """,
        },
        ModelType.GEMINI: {
            PromptVersion.V1: """
            Task: Extract the text from these consecutive lines of an 18th century Russian book.
            Focus: The middle line (use surrounding lines as context only).
            Important: Preserve the original Old Russian orthography.
            Format: Respond with ONLY a JSON object {"line": "extracted text"}
            No explanations.
            """,
            PromptVersion.V2: """
            Task: Extract text from consecutive lines in this 18th century Russian book.
            Focus: ONLY extract the middle line (use surrounding lines as context).
            Requirements:
            - Preserve all Old Russian characters (ѣ, Ѳ, ѳ, ѵ, ъ, etc.)
            - Maintain all punctuation
            Format: Return only {"line": "extracted text"}
            """,
        },
        ModelType.TOGETHER: {
            PromptVersion.V1: """
            You are an expert in Old Russian text recognition.
            Extract the text from these consecutive lines of an 18th century Russian book.
            Focus on the middle line while using surrounding lines as context.
            Preserve the original Old Russian orthography.
            Respond with ONLY a JSON object containing the extracted text of the middle line in the 'line' field.
            """,
            PromptVersion.V2: """
            You are an expert in Old Russian text recognition and paleography.
            Task: Extract the text from these consecutive lines of an 18th century Russian book.
            Instruction: Focus specifically on the middle line, using surrounding lines only for context.
            Requirements:
            - Preserve all original Old Russian orthography including ѣ, Ѳ, ѳ, ѵ, and ъ characters
            - Be extremely precise with punctuation
            Format: Respond with ONLY a JSON object containing the extracted text in the format {"line": "your extracted text"}
            Do not provide any explanations, comments, or additional information.
            """,
        },
    }

    FULL_PAGE = {
        ModelType.CLAUDE: {
            PromptVersion.V1: """
            Extract the OCR text from this full page of an 18th century Russian book.
            Preserve the original Old Russian orthography.
            Process each line independently.
            
            Respond with ONLY a JSON array where each object has a 'line' field containing the corrected text.
            Do not include any additional explanations.
            """,
            PromptVersion.V2: """
            Extract the OCR text from this full page of an 18th century Russian book.
            Preserve all original Old Russian orthography including ѣ, Ѳ, ѳ, ѵ, and ъ characters.
            Process each line independently, maintaining the exact layout of the page.
            
            Respond with ONLY a JSON array where each object has a 'line' field containing the extracted text.
            Do not include any explanations or formatting.
            """,
            PromptVersion.V3: """
            You are an expert OCR system specialized in processing 18th century Russian texts. 
            Your task is to accurately transcribe text from an image of a page from a {book_year} Russian book titled "{book_title}" published in {publication_info}.
            Instructions:
            1. Analyze the entire image thoroughly before beginning transcription
            2. Process the text line by line, maintaining the exact layout of the original page
            3. Preserve all original Old Russian orthography, including: 
                - special characters: ѣ, Ѳ, ѳ, ѵ, і, ї and ъ,
                - Original punctuation,
                - Capitalization as it appears in the original text.

            Respond with ONLY a JSON array where each object has a 'line' field containing the extracted text.
            Do not include any explanations or additional formatting in your response.
            """,
            PromptVersion.V4: """
            Вы являетесь экспертной OCR-системой, специализирующейся на обработке русских текстов XVIII века, напечатанных гражданским шрифтом после реформы Петра I (1708-1710 гг.), но до реформы орфографии 1918 года.
            Ваша задача - точно транскрибировать текст с изображения страницы из русской книги {book_year} года под названием "{book_title}", опубликованной в {publication_info}.
            Особенности орфографии этого периода включают:
            - Наличие специфических букв: ѣ (ять), і (и десятеричное) или ї, ѳ (фита), ѵ (ижица), ъ (твёрдый знак на конце слов)
            - Отсутствие букв церковнославянского алфавита (ѡ, ѧ, ѫ, ѯ, ѱ, etc.)
            - Использование гражданского шрифта вместо устава или полуустава

            Инструкции:
            1. Тщательно проанализируйте всё изображение перед началом транскрипции
            2. Обрабатывайте текст построчно, сохраняя точное расположение оригинальной страницы
            3. Сохраняйте всю оригинальную старорусскую орфографию, включая:
                - Специальные символы: ѣ, Ѳ, ѳ, ѵ, і, ї и ъ,
                - Оригинальную пунктуацию,
                - Заглавные буквы так, как они представлены в оригинальном тексте.

            Отвечайте ТОЛЬКО JSON-массивом, где каждый объект имеет поле 'line', содержащее каждую извлеченную строку текста.
            Не включайте никаких пояснений или дополнительного форматирования в ваш ответ.
            """,
        },
        ModelType.GPT: {
            PromptVersion.V1: """
            You are an expert in Old Russian text recognition.
            Extract the OCR text from this full page of an 18th century Russian book.
            Preserve the original Old Russian orthography.
            Process each line independently.
            
            Respond with ONLY a JSON array where each object has a 'line' field containing the corrected text.
            Do not include any additional explanations.
            """,
            PromptVersion.V2: """
            You are an expert in Old Russian text recognition, paleography, and manuscript analysis.
            Task: Extract all text from this full page of an 18th century Russian book with perfect accuracy.
            Requirements:
            - Preserve all original Old Russian orthography (ѣ, Ѳ, ѳ, ѵ, ъ, etc.)
            - Process each line independently, maintaining the exact layout
            - Preserve original punctuation exactly
            
            Format: Respond with ONLY a JSON array where each line is represented as {"line": "extracted text"}           
            No explanations or additional text allowed in your response.
            """,
            PromptVersion.V3: """
            You are an expert OCR system specialized in processing 18th century Russian texts. 
            Your task is to accurately transcribe text from an image of a page from a {book_year} Russian book titled "{book_title}" published in {publication_info}.
            Instructions:
            1. Analyze the entire image thoroughly before beginning transcription
            2. Process the text line by line, maintaining the exact layout of the original page
            3. Preserve all original Old Russian orthography, including: 
                - Special characters: ѣ, Ѳ, ѳ, ѵ, і, ї and ъ,
                - Original punctuation,
                - Capitalization as it appears in the original text.

            Respond with ONLY a JSON array where each object has a 'line' field containing the corrected text.
            Do not include any explanations or additional formatting in your response.
            """,
            PromptVersion.V4: """
            Вы являетесь экспертной OCR-системой, специализирующейся на обработке русских текстов XVIII века, напечатанных гражданским шрифтом после реформы Петра I (1708-1710 гг.), но до реформы орфографии 1918 года.
            Ваша задача - точно транскрибировать текст с изображения страницы из русской книги {book_year} года под названием "{book_title}", опубликованной в {publication_info}.
            Особенности орфографии этого периода включают:
            - Наличие специфических букв: ѣ (ять), і (и десятеричное) или ї, ѳ (фита), ѵ (ижица), ъ (твёрдый знак на конце слов)
            - Отсутствие букв церковнославянского алфавита (ѡ, ѧ, ѫ, ѯ, ѱ, etc.)
            - Использование гражданского шрифта вместо устава или полуустава

            Инструкции:
            1. Тщательно проанализируйте всё изображение перед началом транскрипции
            2. Обрабатывайте текст построчно, сохраняя точное расположение оригинальной страницы
            3. Сохраняйте всю оригинальную старорусскую орфографию, включая:
                - Специальные символы: ѣ, Ѳ, ѳ, ѵ, і, ї и ъ,
                - Оригинальную пунктуацию,
                - Заглавные буквы так, как они представлены в оригинальном тексте.

            Отвечайте ТОЛЬКО JSON-массивом, где каждый объект имеет поле 'line', содержащее каждую извлеченную строку текста.
            Не включайте никаких пояснений или дополнительного форматирования в ваш ответ.
            """,
        },
        ModelType.GEMINI: {
            PromptVersion.V1: """
            Task: Extract OCR text from this full page of an 18th century Russian book.
            Requirements:
            - Preserve original Old Russian orthography
            - Process each line independently
            Respond with ONLY a JSON array where each object has a 'line' field containing the corrected text.
            No explanations needed.
            """,
            PromptVersion.V2: """
            Task: Extract all text from this 18th century Russian book page.
            Requirements:
            - Preserve all Old Russian characters (ѣ, Ѳ, ѳ, ѵ, ъ, etc.)
            - Maintain exact line-by-line structure
            - Keep original punctuation
            Format: Return only a JSON array: [{"line": "text1"}, {"line": "text2"}]
            """,
            PromptVersion.V3: """
            You are an expert OCR system specialized in processing 18th century Russian texts. 
            Your task is to accurately transcribe text from an image of a page from a {book_year} Russian book titled "{book_title}" published in {publication_info}.
            Instructions:
            1. Analyze the entire image thoroughly before beginning transcription
            2. Process the text line by line, maintaining the exact layout of the original page
            3. Preserve all original Old Russian orthography, including: 
                - Special characters: ѣ, Ѳ, ѳ, ѵ, і, ї and ъ,
                - Original punctuation,
                - Capitalization as it appears in the original text.

            Respond ONLY with a JSON array where each object has a 'line' field containing each extracted line of text. 
            Do not include any explanations or additional formatting in your response.
            """,
            PromptVersion.V4: """
            Вы являетесь экспертной OCR-системой, специализирующейся на обработке русских текстов XVIII века, напечатанных гражданским шрифтом после реформы Петра I (1708-1710 гг.), но до реформы орфографии 1918 года.
            Ваша задача - точно транскрибировать текст с изображения страницы из русской книги {book_year} года под названием "{book_title}", опубликованной в {publication_info}.
            Особенности орфографии этого периода включают:
            - Наличие специфических букв: ѣ (ять), і (и десятеричное) или ї, ѳ (фита), ѵ (ижица), ъ (твёрдый знак на конце слов)
            - Отсутствие букв церковнославянского алфавита (ѡ, ѧ, ѫ, ѯ, ѱ, etc.)
            - Использование гражданского шрифта вместо устава или полуустава

            Инструкции:
            1. Тщательно проанализируйте всё изображение перед началом транскрипции
            2. Обрабатывайте текст построчно, сохраняя точное расположение оригинальной страницы
            3. Сохраняйте всю оригинальную старорусскую орфографию, включая:
                - Специальные символы: ѣ, Ѳ, ѳ, ѵ, і, ї и ъ,
                - Оригинальную пунктуацию,
                - Заглавные буквы так, как они представлены в оригинальном тексте.

            Отвечайте ТОЛЬКО JSON-массивом, где каждый объект имеет поле 'line', содержащее каждую извлеченную строку текста.
            Не включайте никаких пояснений или дополнительного форматирования в ваш ответ.
            """,
        },
        ModelType.TOGETHER: {
            PromptVersion.V1: """
            Task: Extract OCR text from this full page of an 18th century Russian book.
            Requirements:
            - Preserve original Old Russian orthography
            - Process each line independently
            Respond ONLY with a JSON array where each object has a 'line' field containing each extracted line of text. 
            No explanations needed.
            """,
            PromptVersion.V2: """
            Task: Extract all text from this 18th century Russian book page.
            Requirements:
            - Preserve all Old Russian characters (ѣ, Ѳ, ѳ, ѵ, ъ, etc.)
            - Maintain exact line-by-line structure 
            - Keep original punctuation
            Format: Return only a JSON array: [{"line": "text1"}, {"line": "text2"}]
            """,
            PromptVersion.V3: """
            You are an expert OCR system specialized in processing 18th century Russian texts. 
            Your task is to accurately transcribe text from an image of a page from a {book_year} Russian book titled "{book_title}" published in {publication_info}.
            Instructions:
            1. Analyze the entire image thoroughly before beginning transcription
            2. Process the text line by line, maintaining the exact layout of the original page
            3. Preserve all original Old Russian orthography, including special characters: ѣ, Ѳ, ѳ, ѵ, і, ї and ъ.
            4. Maintain original punctuation exactly as it appears.
            5. Reproduce capitalization precisely as it appears in the original text.

            Respond ONLY with a JSON array where each object has a 'line' field containing each extracted line of text. 
            Do not include any explanations or additional formatting in your response.
            """,
            PromptVersion.V4: """
            Вы являетесь экспертной OCR-системой, специализирующейся на обработке русских текстов XVIII века, напечатанных гражданским шрифтом после реформы Петра I (1708-1710 гг.), но до реформы орфографии 1918 года.
            Ваша задача - точно транскрибировать текст с изображения страницы из русской книги {book_year} года под названием "{book_title}", опубликованной в {publication_info}.
            Особенности орфографии этого периода включают:
            - Наличие специфических букв: ѣ (ять), і (и десятеричное) или ї, ѳ (фита), ѵ (ижица), ъ (твёрдый знак на конце слов)
            - Отсутствие букв церковнославянского алфавита (ѡ, ѧ, ѫ, ѯ, ѱ, etc.)
            - Использование гражданского шрифта вместо устава или полуустава

            Инструкции:
            1. Тщательно проанализируйте всё изображение перед началом транскрипции
            2. Обрабатывайте текст построчно, сохраняя точное расположение оригинальной страницы
            3. Сохраняйте всю оригинальную старорусскую орфографию, включая:
                - Специальные символы: ѣ, Ѳ, ѳ, ѵ, і, ї и ъ,
                - Оригинальную пунктуацию,
                - Заглавные буквы так, как они представлены в оригинальном тексте.

            Отвечайте ТОЛЬКО JSON-массивом, где каждый объект имеет поле 'line', содержащее каждую извлеченную строку текста.
            Не включайте никаких пояснений или дополнительного форматирования в ваш ответ.
            """,
        },
    }

    BATCH_FULL_PAGE = {
        ModelType.GPT: {
            PromptVersion.V1: """
            You are an expert in Old Russian text recognition.
            Extract the OCR text from this full page of an 18th century Russian book.
            Preserve the original Old Russian orthography.
            Process each line independently.
            Format: Return a JSON array with one object per line: [{\"line\": \"...\"}].\n
            Do not include any additional explanations.
            """,
            PromptVersion.V2: """
            You are an expert in Old Russian text recognition, paleography, and manuscript analysis.
            Task: Extract all text from this full page of an 18th century Russian book with perfect accuracy.
            Requirements:
            - Preserve all original Old Russian orthography (ѣ, Ѳ, ѳ, ѵ, ъ, etc.)
            - Process each line independently, maintaining the exact layout
            - Preserve original punctuation exactly
            Return a JSON array of objects, each with a 'line' field.
            Example: [{"line": "текст строки 1"}, {"line": "текст строки 2"}]
            No explanations or additional text allowed in your response.
            """,
        },
        ModelType.CLAUDE: {
            PromptVersion.V1: """
            You are an expert in Old Russian text recognition.
            Extract the OCR text from this full page of an 18th century Russian book.
            Preserve the original Old Russian orthography.
            Process each line independently.
            Format: Return a JSON array with one object per line: [{\"line\": \"...\"}].\n
            Do not include any additional explanations.
            """,
            PromptVersion.V2: """
            You are an expert in Old Russian text recognition, paleography, and manuscript analysis.
            Task: Extract all text from this full page of an 18th century Russian book with perfect accuracy.
            Requirements:
            - Preserve all original Old Russian orthography (ѣ, Ѳ, ѳ, ѵ, ъ, etc.)
            - Process each line independently, maintaining the exact layout
            - Preserve original punctuation exactly
            Return a JSON array of objects, each with a 'line' field.
            Example: [{"line": "текст строки 1"}, {"line": "текст строки 2"}]
            No explanations or additional text allowed in your response.
            """,
        },
    }

    TEXT_CORRECTION = {
        ModelType.CLAUDE: {
            PromptVersion.V1: """
            Correct this OCR text from an 18th century Russian book and format it as a single continuous paragraph.
            
            Requirements:
            - Preserve original Old Russian orthography
            - Join hyphenated words at line breaks (e.g., "вне-" + "дряться" -> "внедряться")
            - Remove all line breaks
            - Keep original punctuation
            - Maintain a single continuous paragraph
            - Keep the page number at the start if presents and the catchword at the end if presents
            
            Original text:
            {text}
            
            Return only the corrected continuous text without any markup, explanations, or formatting.
            """,
            PromptVersion.V2: """
            Correct this OCR text from an 18th century Russian book and format it as a single continuous paragraph.
            
            Requirements:
            - Preserve all original Old Russian orthography including ѣ, Ѳ, ѳ, ѵ, and ъ characters
            - Join hyphenated words at line breaks (e.g., "вне-" + "дряться" -> "внедряться")
            - Remove all line breaks
            - Keep original punctuation exactly as it appears
            - Maintain a single continuous paragraph
            - Keep page numbers at the start if present
            
            Original text:
            {text}
            
            Return only the corrected continuous text without any markup, explanations, or formatting.
            """,
        },
        ModelType.GPT: {
            PromptVersion.V1: """
            You are an expert in Old Russian text correction.
            Correct this OCR text from an 18th century Russian book and format it as a single continuous paragraph.
            
            Requirements:
            - Preserve original Old Russian orthography
            - Join hyphenated words at line breaks (e.g., "вне-" + "дряться" -> "внедряться")
            - Remove all line breaks
            - Keep original punctuation
            - Maintain a single continuous paragraph
            - Keep the page number at the start if present
            
            Original text:
            {text}
            
            Return only the corrected continuous text without any markup, explanations, or formatting.
            """,
            PromptVersion.V2: """
            You are an expert in Old Russian language, orthography, and historical texts.
            
            Task: Correct and format this OCR-extracted text from an 18th century Russian book.
            
            Requirements:
            - Preserve all original Old Russian orthography (ѣ, Ѳ, ѳ, ѵ, ъ, etc.)
            - Join hyphenated words that break across lines (e.g., "вне-" + "дряться" -> "внедряться")
            - Remove all line breaks to create one continuous paragraph
            - Maintain all original punctuation exactly as it appears in the source
            - Keep any page numbers at the beginning if present
            
            Original text:
            {text}
            
            Respond with ONLY the corrected text as a single paragraph. No explanations or comments.
            """,
        },
        ModelType.GEMINI: {
            PromptVersion.V1: """
            Task: Correct OCR text from an 18th century Russian book.
            Format: Create a single continuous paragraph.
            Requirements:
            - Keep original Old Russian orthography
            - Join hyphenated words at line breaks
            - Remove line breaks
            - Preserve punctuation and page numbers
            
            Original text:
            {text}
            
            Return only the corrected text. No explanations.
            """,
            PromptVersion.V2: """
            Task: Correct this OCR text from an 18th century Russian book.
            
            Requirements:
            - Preserve all Old Russian characters (ѣ, Ѳ, ѳ, ѵ, ъ, etc.)
            - Join split words at line breaks (e.g., "вне-" + "дряться" → "внедряться")
            - Create a single continuous paragraph
            - Keep original punctuation exactly as shown
            - Maintain page numbers if present
            
            Original text:
            {text}
            
            Return only the corrected text without any comments.
            """,
        },
        ModelType.TOGETHER: {
            PromptVersion.V1: """
            Task: Correct OCR text from an 18th century Russian book.
            Format: Create a single continuous paragraph.
            Requirements:
            - Keep original Old Russian orthography
            - Join hyphenated words at line breaks
            - Remove line breaks
            - Preserve punctuation and page numbers
            
            Original text:
            {text}
            
            Return only the corrected text. No explanations.
            """,
            PromptVersion.V2: """
            Task: Correct this OCR text from an 18th century Russian book.
            
            Requirements:
            - Preserve all Old Russian characters (ѣ, Ѳ, ѳ, ѵ, ъ, etc.)
            - Join split words at line breaks (e.g., "вне-" + "дряться" → "внедряться")
            - Create a single continuous paragraph
            - Keep original punctuation exactly as shown
            - Maintain page numbers if present
            
            Original text:
            {text}
            
            Return only the corrected text without any comments.
            """,
        },
    }

    TEXT_CORRECTION_WITH_PARAGRAPHS = {
        ModelType.CLAUDE: {
            PromptVersion.V1: """
            Correct this OCR text from an 18th century Russian book.
            Preserve the original paragraph structure while fixing OCR errors.
            
            Requirements:
            - Preserve original Old Russian orthography
            - Join hyphenated words at line breaks within paragraphs (e.g., "вне-" + "дряться" -> "внедряться")
            - Keep paragraph breaks if they exist in the original page image
            - Keep original punctuation and spacing between paragraphs
            - Keep the page number as a separate line if present
            - Keep the catchword as a separate line if present
            - Each paragraph should be continuous without artificial line breaks
            
            Original text:
            {text}
            
            Return only the corrected text with original paragraph structure, without any markup or explanations.
            Example format:
            44
            First paragraph text as a continuous line...
            Second paragraph text as a continuous line...
            Third paragraph text as a continuous line...
            catchword
            """,
            PromptVersion.V2: """
            Correct this OCR text from an 18th century Russian book.
            Preserve the original paragraph structure while fixing OCR errors.
            
            Requirements:
            - Preserve all original Old Russian orthography including ѣ, Ѳ, ѳ, ѵ, and ъ characters
            - Pay special attention to superscript letters and titlos
            - Join hyphenated words at line breaks within paragraphs (e.g., "вне-" + "дряться" -> "внедряться")
            - Maintain exact paragraph structure as shown in the image
            - Keep original punctuation and spacing between paragraphs exactly as they appear
            - Preserve page numbers if present
            - Each paragraph should be continuous without artificial line breaks
            
            Original text:
            {text}
            
            Return only the corrected text with original paragraph structure, without any markup or explanations.
            Example format:
            44
            First paragraph text as a continuous line...
            Second paragraph text as a continuous line...
            Third paragraph text as a continuous line...
            """,
        },
        ModelType.GPT: {
            PromptVersion.V1: """
            You are an expert in Old Russian text correction.
            Correct this OCR text from an 18th century Russian book while preserving paragraph structure.
            
            Requirements:
            - Preserve original Old Russian orthography
            - Join hyphenated words at line breaks within paragraphs (e.g., "вне-" + "дряться" -> "внедряться")
            - Keep paragraph breaks as in the original
            - Keep original punctuation and spacing between paragraphs
            - Keep the page number if present
            - Each paragraph should be continuous without artificial line breaks
            
            Original text:
            {text}
            
            Return only the corrected text with original paragraph structure, without any markup or explanations.
            Example format:
            44
            First paragraph text...
            Second paragraph text...
            """,
            PromptVersion.V2: """
            You are an expert in Old Russian language, orthography, and historical texts.
            
            Task: Correct this OCR text from an 18th century Russian book while preserving paragraph structure.
            
            Requirements:
            - Preserve all original Old Russian orthography (ѣ, Ѳ, ѳ, ѵ, ъ, etc.)
            - Pay careful attention to superscript letters, titlos, and abbreviated words
            - Join hyphenated words that break across lines within the same paragraph
            - Maintain the exact original paragraph structure
            - Preserve all original punctuation and spacing between paragraphs
            - Keep page numbers when present
            - Make each paragraph a continuous text without artificial line breaks
            
            Original text:
            {text}
            
            Respond with ONLY the corrected text with proper paragraph breaks. No explanations or comments.
            
            Example format:
            44
            First paragraph as a continuous line of text...
            
            Second paragraph as a continuous line of text...
            
            Third paragraph as a continuous line of text...
            """,
        },
        ModelType.GEMINI: {
            PromptVersion.V1: """
            Task: Correct OCR text from an 18th century Russian book.
            Format: Preserve original paragraph structure.
            Requirements:
            - Keep original Old Russian orthography
            - Join hyphenated words within paragraphs
            - Maintain paragraph breaks
            - Preserve punctuation, spacing, and page numbers
            
            Original text:
            {text}
            
            Return only corrected text with paragraphs intact.
            """,
            PromptVersion.V2: """
            Task: Correct this OCR text from an 18th century Russian book.
            
            Requirements:
            - Preserve all Old Russian characters (ѣ, Ѳ, ѳ, ѵ, ъ, etc.)
            - Join split words within paragraphs
            - Maintain exact paragraph structure as in original
            - Keep original punctuation and spacing
            - Preserve page numbers
            - Each paragraph should be continuous
            
            Original text:
            {text}
            
            Return only the corrected text with paragraph structure intact.
            No comments or explanations.
            """,
        },
    }


def get_prompt(
    prompt_type: str, model_type: ModelType, version: PromptVersion, **kwargs: Any
) -> str:
    """
    Get a prompt with the specified model type, version and format it with the given kwargs.

    Args:
        prompt_type: Type of prompt to retrieve (attribute name from PromptTemplates)
        model_type: Type of model the prompt is for
        version: Version of the prompt to use
        **kwargs: Variables to format into the prompt template

    Returns:
        Formatted prompt string
    """
    if not hasattr(PromptTemplates, prompt_type):
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    prompt_dict = getattr(PromptTemplates, prompt_type)

    if model_type not in prompt_dict:
        raise ValueError(f"Model type {model_type} not found for prompt type {prompt_type}")

    model_prompts = prompt_dict[model_type]

    if version not in model_prompts:
        raise ValueError(
            f"Version {version} not found for model type {model_type} and prompt type {prompt_type}"
        )

    prompt_template = model_prompts[version]

    # If document_id is provided and V3 prompt, enrich kwargs with document metadata
    if "document_id" in kwargs and version == PromptVersion.V3 or version == PromptVersion.V4:
        document_metadata = get_document_metadata(kwargs["document_id"])
        # Update kwargs with document metadata
        kwargs.update(document_metadata)
        # Remove document_id from kwargs as it's not needed in the template
        del kwargs["document_id"]

    # Format the prompt with kwargs if provided
    if kwargs:
        logger.debug("Formatting prompt with kwargs: %s", prompt_template.format(**kwargs))
        return str(prompt_template.format(**kwargs))

    return str(prompt_template)


def get_document_metadata(document_id: str) -> Dict[str, Any]:
    """
    Retrieve document metadata by document_id from the dataset.

    Args:
        document_id: The document ID to look up (present in image_ids list)

    Returns:
        Dictionary containing metadata about the document
    """
    # Path to your JSON dataset - adjust this path as needed
    dataset_path = "filtered_100_dataset.json"

    # Check if file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    # Load the dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Iterate through books to find the one containing this image_id
    for book in dataset.get("books", []):
        # Check if the document_id is in the image_ids list
        if document_id in book.get("image_ids", []):
            logger.debug(
                "Found document ID %s in book: %s", document_id, book.get("title", "Unknown Title")
            )
            # Create metadata dictionary with all relevant fields
            metadata = {
                "book_title": book.get("title", ""),
                "book_author": book.get("author", ""),
                "book_year": book.get("year", ""),
                "book_subject": book.get("subject", ""),
                "book_language": book.get("language", ""),
                "publication_info": book.get("publication_info", ""),
            }

            return metadata

    # If no matching document is found
    raise ValueError(f"Document ID {document_id} not found in dataset")
