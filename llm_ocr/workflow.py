import base64
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
            evaluation: Whether to evaluate the results
            rerun: Whether to rerun the pipeline even if results exist
            target_dpi: Target DPI for image resizing
        """
        self.id = id
        self.folder = Path(folder)
        self.output_dir = Path(output_dir)
        self.image_str = self._get_image_str()
        self.evaluation = evaluation
        self.prompt_version = prompt_version
        self.model_name = model_name
        self.modes = modes
        self.rerun = rerun
        self.target_dpi = target_dpi  # Store target DPI

        # Setup paths
        self._setup_paths()

        # Validate required files
        self._validate_required_files()

        # Initialize document info
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

    def _setup_paths(self) -> None:
        """Set up all file paths."""
        self.xml_path = self.folder / f"{self.id}.xml"
        self.image_path = self.folder / f"{self.id}.jpeg"
        self.ground_truth_path = self.folder / f"{self.id}.txt"

    def _validate_required_files(self) -> None:
        """Validate that required files exist."""
        required_files = [(self.xml_path, "XML file"), (self.image_path, "Image file")]

        for file_path, description in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"{description} not found: {file_path}")

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
            lines = (
                self.alto_processor.process_alto_file(str(self.xml_path), str(self.image_path))
                if self.xml_path.exists()
                else []
            )
            if not lines:
                logging.warning("No lines found in the document. Skipping OCR.")
                continue

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
            return

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

    # STEP 3: Run correction with LLM
    def run_correction(self, correction_modes: Union[str, List[str]] = "line") -> None:
        """
        Step 3: Run correction step with LLM and save results.

        Args:
            correction_modes: Single mode string or list of modes ["line", "para"]

        Returns:
            Dictionary with correction results for all modes
        """
        # Normalize input to list
        if isinstance(correction_modes, str):
            correction_modes = [correction_modes]

        logging.info(
            f"STEP 3: Running OCR correction with model: {self.model_name}, modes: {correction_modes}"
        )

        # Extract OCR text from fullpage results (required for correction)
        model_results = self.results["models"][self.model_name]["ocr_results"]

        if "fullpage" not in model_results:
            logging.warning("Fullpage OCR results required for correction but not found.")
            return

        fullpage_results = model_results.get("fullpage", {})
        lines = fullpage_results.get("lines", [])

        if not lines:
            logging.warning("No fullpage OCR results found. Skipping correction.")
            return

        ocr_text = lines[0].get("extracted_text", "")
        if not ocr_text:
            logging.warning("No OCR text found in fullpage results. Skipping correction.")
            return

        # Initialize correction_results as dict if not exists
        if "correction_results" not in self.results["models"][self.model_name]:
            self.results["models"][self.model_name]["correction_results"] = {}

        # Resize image once for all modes
        logging.info(f"Optimizing image for correction step (target DPI: {self.target_dpi})")
        optimized_image_str, _ = resize_image_to_dpi(
            str(self.image_path), self.target_dpi, max_pixels=2000000
        )

        # Create correction pipeline once
        correction_pipeline = OCRCorrectionPipeline(model=self.model, evaluator=self.evaluator)

        all_results = {}

        # Process each mode
        for mode in correction_modes:
            # Check if this specific mode already exists
            if (
                mode in self.results["models"][self.model_name]["correction_results"]
                and not self.rerun
            ):
                logging.info(f"Correction mode '{mode}' already exists. Skipping.")
                all_results[mode] = self.results["models"][self.model_name]["correction_results"][
                    mode
                ]
                continue

            logging.info(f"Processing correction mode: {mode}")
            start_time = time.time()

            try:
                # Run correction with specified mode
                correction_results = correction_pipeline.run_correction(
                    image_str=optimized_image_str, ocr_text=ocr_text, mode=mode
                )

                if not correction_results or not isinstance(correction_results, dict):
                    logging.warning("No correction results available for evaluation.")
                    return

                completion_time = time.time() - start_time
                logging.info(f"Mode '{mode}' completed in {completion_time:.2f} seconds")

                # Store results for this specific mode
                mode_results = {
                    "original_ocr_text": ocr_text,
                    "corrected_text": correction_results.get("corrected_text", ""),
                    "correction_mode": mode,
                    "processing_time": completion_time,
                }

                # Add mode-specific fields
                if mode == "para" and "paragraphs" in correction_results:
                    mode_results["paragraphs"] = correction_results["paragraphs"]
                    mode_results["paragraph_boundaries"] = correction_results.get(
                        "paragraph_boundaries", []
                    )

                # Store under the specific mode
                self.results["models"][self.model_name]["correction_results"][mode] = mode_results
                all_results[mode] = mode_results

                # Add to history
                self._add_history_entry(step="correction", mode=mode)

            except Exception as e:
                logging.error(f"Correction failed for mode '{mode}': {e}")
                all_results[mode] = {"error": str(e)}

        # Save results after all modes
        self._save_results()

    # STEP 4: Evaluate correction results
    def evaluate_correction(self, modes: Union[str, List[str], None] = None) -> None:
        """
        Step 4: Evaluate the correction results.

        Args:
            modes: Specific modes to evaluate, or None to evaluate all available modes
        """
        if not self.ground_truth:
            logging.warning("Ground truth not provided. Skipping correction evaluation.")
            return

        model_results = self.results["models"][self.model_name]
        correction_results = model_results.get("correction_results")

        if not correction_results or not isinstance(correction_results, dict):
            logging.warning("No correction results available for evaluation.")
            return

        if modes is None:
            modes = list(correction_results.keys())  # Now safe to use
        elif isinstance(modes, str):
            modes = [modes]

        logging.info(f"STEP 4: Evaluating correction results for modes: {modes}")

        evaluation_service = OCREvaluationService()

        for mode in modes:
            if mode not in correction_results:
                logging.warning(f"No results found for mode '{mode}'. Skipping.")
                continue

            logging.info(f"Evaluating mode: {mode}")

            # Get appropriate ground truth for the mode
            ground_truth_for_mode = self._get_ground_truth_for_mode(mode)
            if not ground_truth_for_mode:
                logging.warning(f"No ground truth available for mode '{mode}'. Skipping.")
                continue

            mode_results = correction_results[mode]
            corrected_text = mode_results.get("corrected_text", "")

            if not corrected_text:
                logging.warning(f"No corrected text available for mode '{mode}'.")
                continue

            start_time = time.time()

            evaluation_data = [
                {
                    "ground_truth_text": ground_truth_for_mode,
                    "extracted_text": corrected_text,
                }
            ]

            try:
                report = evaluation_service.evaluate_ocr_results(
                    evaluation_data, include_details=True
                )

                # Add metrics to mode-specific results
                mode_results["metrics"] = report

                completion_time = time.time() - start_time
                logging.info(
                    f"Evaluation for mode '{mode}' completed in {completion_time:.2f} seconds"
                )

            except Exception as e:
                logging.error(f"Error evaluating correction results for mode '{mode}': {e}")

        # Save results after all evaluations
        self._save_results()

    def _get_ground_truth_for_mode(self, mode: str) -> Optional[str]:
        """Get appropriate ground truth for the correction mode."""
        if mode == "line":
            # Try to load line-specific ground truth
            line_filename = Path(self.ground_truth_path).with_stem(
                Path(self.ground_truth_path).stem + "_line"
            )
            if line_filename.exists():
                return line_filename.read_text(encoding="utf-8")
        elif mode == "para":
            # Try to load paragraph-specific ground truth
            para_filename = Path(self.ground_truth_path).with_stem(
                Path(self.ground_truth_path).stem + "_para"
            )
            if para_filename.exists():
                return para_filename.read_text(encoding="utf-8")

        # Fallback to regular ground truth
        return self.ground_truth

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
