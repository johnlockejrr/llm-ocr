import base64
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_ocr.evaluators.evaluation import OCREvaluationService
from llm_ocr.evaluators.evaluator import OCREvaluator
from llm_ocr.model_factory import create_model
from llm_ocr.models import ProcessingMode
from llm_ocr.pipelines.correction import OCRCorrectionPipeline
from llm_ocr.pipelines.ocr import OCRPipeline
from llm_ocr.processors.alto import ALTOProcessor
from llm_ocr.prompts.prompt import PromptVersion
from llm_ocr.utils.image_tool import resize_image_to_dpi

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class OCRPipelineWorkflow:
    """
    A class to manage the full OCR evaluation workflow with clearly defined steps:
    1. Run OCR with different modes and models
    2. Evaluate the OCR results with ground truth
    3. Run correction step with LLM
    4. Evaluate the correction results

    Results are stored in a single consolidated JSON file per document.
    """

    def __init__(
        self,
        id: str,
        folder: str = "evaluate",
        model_name: str = "claude-3-7-sonnet-20250219",
        modes: List[ProcessingMode] = [ProcessingMode.FULL_PAGE],
        output_dir: str = "outputs",
        prompt_version: PromptVersion = PromptVersion.V3,
        evaluation: bool = True,
        rerun: bool = False,
        target_dpi: int = 150,
    ):
        """
        Initialize the OCR pipeline workflow.

        Args:
            id: Document ID
            folder: Folder containing the document files (xml, jpeg, txt)
            model_name: Model name to use
            modes: Processing modes to use
            output_dir: Directory to save outputs
            prompt_version: Version of prompts to use
        """
        self.id = id
        self.xml_path = f"{folder}/{id}.xml"
        self.image_path = f"{folder}/{id}.jpeg"
        self.image_str = self._get_image_str()
        self.ground_truth_path = f"{folder}/{id}.txt"
        self.evaluation = evaluation
        self.output_dir = Path(output_dir)
        self.prompt_version = prompt_version
        self.model_name = model_name
        self.modes = modes
        self.rerun = rerun
        self.target_dpi = target_dpi  # Store target DPI

        # Initialize document info
        # self.document_name = Path(self.xml_path).stem
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create main output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Path to the consolidated results file
        self.results_file_path = self.output_dir / f"{id}_results.json"

        # Load existing results file if it exists, or create a new one
        self.results = self._load_or_init_results()

        # Read ground truth if available
        self.ground_truth = None
        if self.ground_truth_path:
            try:
                self.ground_truth = Path(self.ground_truth_path).read_text(encoding="utf-8")
                self.results["document_info"]["ground_truth"] = self.ground_truth
            except Exception as e:
                logging.warning(f"Could not read ground truth file: {e}")

        # Initialize model
        self.model = self._initialize_model()
        logging.info(f"Model initialized: {self.model.__class__.__name__}")
        if not self.model:
            raise ValueError("No models could be initialized. Check API keys.")

        # Initialize components
        self.alto_processor = ALTOProcessor()
        self.evaluator = OCREvaluator()

        # Save initial document info
        self._save_results()

    def _load_or_init_results(self) -> Dict[str, Any]:
        """Load existing results file or initialize a new one."""
        if self.results_file_path.exists():
            try:
                with open(self.results_file_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                logging.info(f"Loaded existing results file: {self.results_file_path}")

                # Initialize model entry if it doesn't exist
                if "models" not in results:
                    results["models"] = {}

                if self.model_name not in results["models"]:
                    results["models"][self.model_name] = {
                        "ocr_results": {},
                        "correction_results": None,
                    }
                if not isinstance(results, dict):
                    raise ValueError("Loaded results are not a dictionary")
                return results
            except Exception as e:
                logging.warning(f"Error loading results file: {e}. Creating new file.")

        # Initialize new results structure
        return {
            "document_info": {
                "document_name": self.id,
                "xml_path": str(self.xml_path),
                "image_path": str(self.image_path),
                "ground_truth_path": (
                    str(self.ground_truth_path) if self.ground_truth_path else None
                ),
                "timestamp": self.timestamp,
                "prompt_version": self.prompt_version.name,
            },
            "models": {self.model_name: {"ocr_results": {}, "correction_results": None}},
            "processing_history": [],
        }

    def _initialize_model(self) -> Any:
        """Initialize LLM model."""
        return create_model(
            model_name=self.model_name,
            prompt_version=self.prompt_version,
        )

    def _get_image_str(self) -> str:
        """Convert image to base64 string."""
        with open(self.image_path, "rb") as image_file:
            image_str = base64.b64encode(image_file.read()).decode("utf-8")
        return image_str

    def _add_history_entry(self, step: str, mode: Optional[str]) -> None:
        """Add an entry to the processing history."""
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "step": step,
            "model": self.model_name,
        }

        if mode:
            entry["mode"] = mode

        self.results["processing_history"].append(entry)

    def _save_results(self) -> None:
        """Save the consolidated results to the JSON file."""
        # Update timestamp
        self.results["document_info"]["last_updated"] = datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        with open(self.results_file_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        logging.info(f"Saved results to {self.results_file_path}")

    # STEP 1: Run OCR with different modes
    def run_ocr(self) -> None:
        """
        Step 1: Run OCR with different modes, save the results.

        Returns:
            Dictionary with OCR results by mode
        """
        logging.info(f"STEP 1: Processing document: {self.id} with model: {self.model_name}")

        # Process document with all modes
        for mode in self.modes:
            start_time = time.time()
            logging.info(f"Processing with mode: {mode.value}")
            if not self.rerun:
                if self.results["models"][self.model_name]["ocr_results"].get(mode.value):
                    lines_existed = self.results["models"][self.model_name]["ocr_results"][
                        mode.value
                    ].get("lines")
                    if lines_existed:
                        extracted_text = lines_existed[0].get("extracted_text", "")
                        if extracted_text != "":
                            logging.info(f"Mode {mode.value} already processed. Skipping.")
                            continue

            # Initialize pipeline for OCR
            pipeline = OCRPipeline(model=self.model, evaluator=self.evaluator)

            # Process document
            lines = self.alto_processor.process_alto_file(self.xml_path, self.image_path)

            results = pipeline.ocr_document(
                lines=lines,
                image_str=self.image_str,
                mode=mode,
                id=self.id,
            )

            completion_time = time.time() - start_time
            logging.info(
                f"Processed {len(lines)} lines with mode {mode.value} in {completion_time:.2f} seconds"
            )

            # Add to results structure
            self.results["models"][self.model_name]["ocr_results"][mode.value] = {"lines": []}

            # Convert to serializable format and add to results
            for model_name, model_results in results.items():
                serializable_lines = []
                for line in model_results:
                    if hasattr(line, "__dict__"):
                        serializable_lines.append(dict(line))
                    else:
                        serializable_lines.append(line)

                self.results["models"][self.model_name]["ocr_results"][mode.value][
                    "lines"
                ] = serializable_lines

            # Add to history
            self._add_history_entry(step="ocr", mode=mode.value)

            # Save results after each mode to preserve progress
            self._save_results()

        # return self.results["models"][self.model_name]["ocr_results"]

    # STEP 2: Evaluate OCR results with ground truth
    def evaluate_ocr(self) -> None:
        """
        Step 2: Evaluate the OCR results with ground truth and save metrics.

        Returns:
            Dictionary with comparison metrics
        """
        logging.info(f"STEP 2: Evaluating OCR results for model: {self.model_name}")

        if not self.ground_truth:
            logging.warning("Ground truth not provided. Skipping OCR evaluation.")
            return None

        evaluation_service = OCREvaluationService()

        for mode, mode_results in self.results["models"][self.model_name]["ocr_results"].items():
            start_time = time.time()
            logging.info(f"Evaluating mode: {mode}")

            lines = mode_results.get("lines", [])
            if not lines:
                logging.warning(f"No lines found for mode {mode}. Skipping evaluation.")
                continue

            evaluation_data = []
            for line in lines:
                evaluation_data.append(
                    {
                        "ground_truth_text": line.get("ground_truth_text", ""),
                        "extracted_text": line.get("extracted_text", ""),
                        "line_id": line.get("line_id", "unknown"),
                    }
                )

            try:
                report = evaluation_service.evaluate_ocr_results(
                    evaluation_data, include_details=True
                )

                # Add metrics to results
                self.results["models"][self.model_name]["ocr_results"][mode]["metrics"] = report

                completion_time = time.time() - start_time
                logging.info(
                    f"Evaluation completed for mode {mode} in {completion_time:.2f} seconds"
                )

                # Save results after each evaluation
                self._save_results()

            except Exception as e:
                logging.error(f"Error evaluating OCR results for mode {mode}: {e}")

        # return {mode: results.get("metrics") for mode, results in self.results["models"][self.model_name]["ocr_results"].items()}

    # STEP 3: Run correction with LLM
    def run_correction(self, correction_mode: str = "line") -> Optional[Dict[str, Any]]:
        """
        Step 3: Run correction step with LLM and save results.

        Args:
            correction_mode: "line" for single line output, "para" for paragraph detection

        Returns:
            Dictionary with correction results
        """
        logging.info(f"STEP 3: Running OCR correction with model: {self.model_name}")

        # Extract OCR text from fullpage results (required for correction)
        model_results = self.results["models"][self.model_name]["ocr_results"]

        if "fullpage" not in model_results:
            logging.warning("Fullpage OCR results required for correction but not found.")
            return None

        fullpage_results = model_results.get("fullpage", {})
        lines = fullpage_results.get("lines", [])

        if not lines:
            logging.warning("No fullpage OCR results found. Skipping correction.")
            return None

        ocr_text = lines[0].get("extracted_text", "")

        # Check if correction_results already exists
        if "correction_results" in self.results["models"][self.model_name]:
            correction_results = self.results["models"][self.model_name]["correction_results"]
            if correction_results and not self.rerun:
                logging.info("Correction results already exist. Skipping correction.")
                return correction_results

        if not ocr_text:
            logging.warning("No OCR text found in fullpage results. Skipping correction.")
            return None

        start_time = time.time()

        # Resize image for correction step to optimize token usage
        logging.info(f"Optimizing image for correction step (target DPI: {self.target_dpi})")
        optimized_image_str, _ = resize_image_to_dpi(
            self.image_path, self.target_dpi, max_pixels=2000000
        )

        # Create correction pipeline
        correction_pipeline = OCRCorrectionPipeline(model=self.model, evaluator=self.evaluator)

        try:
            # Run correction with specified mode for prompt formatting
            correction_results = correction_pipeline.run_correction(
                image_str=optimized_image_str, ocr_text=ocr_text, mode=correction_mode
            )
        except Exception as e:
            logging.error(f"Correction failed: {e}")
            return None

        completion_time = time.time() - start_time
        logging.info(f"Correction completed in {completion_time:.2f} seconds")

        # Add to results
        self.results["models"][self.model_name]["correction_results"] = {
            "original_ocr_text": ocr_text,
            "corrected_text": correction_results.get("corrected_text", ""),
            "correction_mode": correction_mode,
        }

        # Add to history
        self._add_history_entry(step="correction", mode=correction_mode)

        # Save results
        self._save_results()

        return self.results["models"][self.model_name]["correction_results"]

    # STEP 4: Evaluate correction results
    def evaluate_correction(self) -> None:
        """
        Step 4: Evaluate the correction results.

        Returns:
            Dictionary with correction metrics
        """
        if not self.ground_truth:
            logging.warning("Ground truth not provided. Skipping correction evaluation.")
            return None

        line_filename = self.ground_truth_path.replace(".txt", "_line.txt")
        print(f"line_filename: {line_filename}")
        with open(line_filename, "r", encoding="utf-8") as f:
            ground_truth_line = f.read()
        logging.info(f"Ground truth line: {ground_truth_line}")

        model_results = self.results["models"][self.model_name]
        if not model_results["correction_results"]:
            logging.warning("No correction results available for evaluation.")
            return None

        logging.info(f"STEP 4: Evaluating correction results for model: {self.model_name}")

        start_time = time.time()

        evaluation_service = OCREvaluationService()
        corrected_text = model_results["correction_results"]["corrected_text"]

        if not corrected_text:
            logging.warning("No corrected text available for evaluation.")
            return None
        logging.info(f"Ground truth line: {ground_truth_line}")
        logging.info(f"Corrected text: {corrected_text}")

        evaluation_data = [
            {
                "ground_truth_text": ground_truth_line,
                "extracted_text": corrected_text,
            }
        ]

        try:
            report = evaluation_service.evaluate_ocr_results(evaluation_data, include_details=True)

            # Add metrics to results
            self.results["models"][self.model_name]["correction_results"]["metrics"] = report

            completion_time = time.time() - start_time
            logging.info(f"Correction evaluation completed in {completion_time:.2f} seconds")

            # Save results
            self._save_results()

        except Exception as e:
            logging.error(f"Error evaluating correction results: {e}")

        # return self.results["models"][self.model_name]["correction_results"].get("metrics")

    # Utility method to run the complete pipeline
    def run_pipeline(self) -> None:
        """
        Run the complete OCR pipeline in sequence:
        1. Run OCR
        2. Evaluate OCR
        3. Run Correction (if ground truth available)
        4. Evaluate Correction (if correction performed)

        Returns:
            Dictionary with all results
        """
        # Step 1: Run OCR
        self.run_ocr()

        # Step 2: Evaluate OCR results
        if self.ground_truth and self.evaluation:
            self.evaluate_ocr()

        # Step 3: Run Correction
        self.run_correction()

        # Step 4: Evaluate Correction results
        if self.ground_truth:
            self.evaluate_correction()

        logging.info(f"All evaluation data saved to: {self.results_file_path}")

        # return self.results
