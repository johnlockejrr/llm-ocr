"""
Example usage of the flexible OCREvaluationService.
"""
import json
from pathlib import Path
import logging

from ..models import Line
from .evaluator import OCREvaluator
from ..config import EvaluationConfig
from .evaluation import OCREvaluationService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('example')

def main():
    # Initialize the evaluation service
    config = EvaluationConfig(
        old_russian_chars='ѣѲѳѵѡѠѢѴѶѷѸѹѺѻѼѽѾѿъь',
        include_detailed_analysis=True
    )
    evaluator = OCREvaluator(config)
    evaluation_service = OCREvaluationService(evaluator, config)
    
    # Example 1: Compare two individual lines
    logger.info("Example 1: Comparing individual lines")
    ground_truth = "Вѣрую въ единаго Бога Отца Вседержителя"
    extracted = "Верую в единаго Бога Отца Вседержителя"
    logger.info(f"Ground Truth: {ground_truth}")
    logger.info(f"Extracted: {extracted}")
    
    result, metrics = evaluation_service.compare_texts(
        ground_truth=ground_truth, 
        extracted_text=extracted,
        model_name="TestModel"
    )
    
    logger.info(f"Character Accuracy: {metrics['char_accuracy']:.2%}")
    logger.info(f"Word Accuracy: {metrics['word_accuracy']:.2%}")
    logger.info(f"Old Character Preservation: {metrics['old_char_preservation']:.2%}")
    
    # Example 2: Compare lists of texts
    logger.info("\nExample 2: Comparing lists of texts")
    ground_truth_texts = [
        "Вѣрую въ единаго Бога Отца Вседержителя",
        "Творца небу и земли, видимымъ же всѣмъ и невидимымъ.",
        "И во единаго Господа Іисуса Христа, Сына Божія"
    ]
    
    extracted_texts = [
        "Верую в единаго Бога Отца Вседержителя",
        "Творца небу и земли, видимым же всем и невидимым.",
        "И во единаго Господа Иисуса Христа, Сына Божия"
    ]
    logger.info(f"Ground Truth Texts: {ground_truth_texts}")
    logger.info(f"Extracted Texts: {extracted_texts}")
    
    report = evaluation_service.compare_text_lists(
        ground_truth_texts=ground_truth_texts,
        extracted_texts=extracted_texts,
        model_name="TestModel"
    )
    
    logger.info(f"Overall Character Accuracy: {report['character_accuracy']:.2%}")
    logger.info(f"Overall Word Accuracy: {report['word_accuracy']:.2%}")
    logger.info(f"Top Error Patterns:")
    for pattern, count in list(report['error_analysis']['common_char_errors'].items())[:3]:
        logger.info(f"  {pattern}: {count} occurrences")
    
    # Example 3: Compare using Line objects
    logger.info("\nExample 3: Using Line objects")
    
    # Create Line objects (normally these would come from your ALTO processor)
    lines = [
        Line(
            text=ground_truth_texts[0], 
            line_id="1",
            base64_image="",  # You can add base64 image data if needed
            position={"x": 0, "y": 0, "width": 100, "height": 20}
        ),
        Line(
            text=ground_truth_texts[1], 
            line_id="2",
            base64_image="",
            position={"x": 0, "y": 30, "width": 100, "height": 20}
        ),
        Line(
            text=ground_truth_texts[2], 
            line_id="3",
            base64_image="",
            position={"x": 0, "y": 60, "width": 100, "height": 20}
        )
    ]
    
    report = evaluation_service.compare_line_objects(
        ground_truth_lines=lines,
        extracted_texts=extracted_texts,
        model_name="TestModel"
    )
    
    logger.info(f"Line Objects - Character Accuracy: {report['character_accuracy']:.2%}")
    
    # Example 4: Compare pairs of texts
    logger.info("\nExample 4: Using text pairs")
    
    text_pairs = list(zip(ground_truth_texts, extracted_texts))
    
    report = evaluation_service.compare_line_pairs(
        line_pairs=text_pairs,
        model_name="TestModel"
    )
    
    logger.info(f"Text Pairs - Character Accuracy: {report['character_accuracy']:.2%}")
    
    # Example 5: Evaluate results with processing times
    logger.info("\nExample 5: With processing times")
    
    processing_times = [0.15, 0.23, 0.17]  # in seconds
    
    report = evaluation_service.compare_text_lists(
        ground_truth_texts=ground_truth_texts,
        extracted_texts=extracted_texts,
        model_name="TestModel",
        processing_times=processing_times
    )
    
    logger.info(f"Average Processing Time: {report['average_processing_time']:.3f} seconds")
    
    # Save report to file
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    evaluation_service.save_evaluation_report(
        {"TestModel": report},
        output_dir,
        prefix="flexible_example"
    )
    
    logger.info(f"Report saved to {output_dir}/flexible_example_report.json")

if __name__ == "__main__":
    main()