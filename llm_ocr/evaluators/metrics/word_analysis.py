import difflib
from typing import Any, Dict

from llm_ocr.evaluators.metrics.base import BaseMetric


class WordAnalysisMetric(BaseMetric):
    """Word-level analysis between ground truth and extracted text."""

    @property
    def name(self) -> str:
        return "word_analysis"

    def evaluate(self, ground_truth: str, extracted: str) -> Dict[str, Any]:
        """
        Analyze word-level differences between ground truth and extracted text.

        Args:
            ground_truth: The known correct text
            extracted: The text extracted by OCR

        Returns:
            Dictionary with word-level error analysis
        """
        gt_words = ground_truth.split()
        ext_words = extracted.split()

        analysis: Dict[str, Any] = {
            "word_substitutions": [],
            "word_deletions": [],
            "word_insertions": [],
            "total_word_errors": 0,
        }

        matcher = difflib.SequenceMatcher(None, gt_words, ext_words)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                analysis["word_substitutions"].append(
                    {"ground_truth": gt_words[i1:i2], "extracted": ext_words[j1:j2], "position": i1}
                )
            elif tag == "delete":
                analysis["word_deletions"].append({"words": gt_words[i1:i2], "position": i1})
            elif tag == "insert":
                analysis["word_insertions"].append({"words": ext_words[j1:j2], "position": i1})

        analysis["total_word_errors"] = (
            len(analysis["word_substitutions"])
            + len(analysis["word_deletions"])
            + len(analysis["word_insertions"])
        )

        return analysis
