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
    3. Run correction step with LLM (potentially different model)
    4. Evaluate the correction results

    Results are stored in separate JSON files:
    - _ocr_results.json: OCR processing results by model and mode
    - _correction_results.json: Correction results by OCR+correction model combinations
    """

    def __init__(
        self,
        id: str,
        folder: str = "evaluate",
        ocr_model_name: str = "claude-3-7-sonnet-20250219",
        correction_model_name: Optional[str] = None,
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
            ocr_model_name: Model name to use for OCR
            correction_model_name: Model name to use for correction (defaults to ocr_model_name)
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

        self.evaluation = evaluation
        self.prompt_version = prompt_version
        self.ocr_model_name = ocr_model_name
        self.correction_model_name = correction_model_name or ocr_model_name
        self.modes = modes
        self.rerun = rerun
        self.target_dpi = target_dpi  # Store target DPI

        # Setup paths
        self._setup_paths()

        # Validate required files
        self._validate_required_files()
        self.image_str = self._get_image_str()

        # Initialize document info
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create main output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Paths to separate results files
        self.ocr_results_file_path = self.output_dir / f"{id}_ocr_results.json"
        self.correction_results_file_path = self.output_dir / f"{id}_correction_results.json"

        # Load existing results files if they exist, or create new ones
        self.ocr_results = self._load_or_init_ocr_results()
        self.correction_results = self._load_or_init_correction_results()

        # Read ground truth if available
        self.ground_truth = None
        if self.ground_truth_path:
            try:
                self.ground_truth = Path(self.ground_truth_path).read_text(encoding="utf-8")
                self.ocr_results["document_info"]["ground_truth"] = self.ground_truth
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
        self._save_ocr_results()

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

    def _load_or_init_ocr_results(self) -> Dict[str, Any]:
        """Load existing OCR results file or initialize a new one."""
        if self.ocr_results_file_path.exists():
            try:
                with open(self.ocr_results_file_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                logging.info(f"Loaded existing OCR results file: {self.ocr_results_file_path}")

                # Initialize model entry if it doesn't exist
                if "ocr_models" not in results:
                    results["ocr_models"] = {}

                if self.ocr_model_name not in results["ocr_models"]:
                    results["ocr_models"][self.ocr_model_name] = {}

                if not isinstance(results, dict):
                    raise ValueError("Loaded OCR results are not a dictionary")
                return results
            except Exception as e:
                logging.warning(f"Error loading OCR results file: {e}. Creating new file.")

        # Initialize new OCR results structure
        return {
            "document_info": {
                "document_id": self.id,
                "xml_path": str(self.xml_path),
                "image_path": str(self.image_path),
                "ground_truth_path": (
                    str(self.ground_truth_path) if self.ground_truth_path else None
                ),
                "timestamp": self.timestamp,
                "prompt_version": self.prompt_version.name,
            },
            "ocr_models": {self.ocr_model_name: {}},
            "processing_history": [],
        }

    def _load_or_init_correction_results(self) -> Dict[str, Any]:
        """Load existing correction results file or initialize a new one."""
        if self.correction_results_file_path.exists():
            try:
                with open(self.correction_results_file_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                logging.info(
                    f"Loaded existing correction results file: {self.correction_results_file_path}"
                )

                if "correction_combinations" not in results:
                    # Handle legacy format migration
                    if "correction_models" not in results:
                        results["correction_models"] = {}
                else:
                    # Migrate from old format to new format
                    if "correction_models" not in results:
                        results["correction_models"] = {}
                    # Remove old format
                    del results["correction_combinations"]

                if not isinstance(results, dict):
                    raise ValueError("Loaded correction results are not a dictionary")
                return results
            except Exception as e:
                logging.warning(f"Error loading correction results file: {e}. Creating new file.")

        # Initialize new correction results structure
        return {
            "document_info": {
                "document_id": self.id,
                "xml_path": str(self.xml_path),
                "image_path": str(self.image_path),
                "ground_truth_path": (
                    str(self.ground_truth_path) if self.ground_truth_path else None
                ),
                "timestamp": self.timestamp,
                "prompt_version": self.prompt_version.name,
            },
            "correction_models": {},
            "processing_history": [],
        }

    def _initialize_model(self) -> Any:
        """Initialize LLM model for OCR."""
        return create_model(
            model_name=self.ocr_model_name,
            prompt_version=self.prompt_version,
        )

    def _initialize_correction_model(self) -> Any:
        """Initialize LLM model for correction."""
        return create_model(
            model_name=self.correction_model_name,
            prompt_version=self.prompt_version,
        )

    def _get_image_str(self) -> str:
        """Convert image to base64 string."""
        with open(self.image_path, "rb") as image_file:
            image_str = base64.b64encode(image_file.read()).decode("utf-8")
        return image_str

    def _add_ocr_history_entry(self, mode: Optional[str]) -> None:
        """Add an entry to the OCR processing history."""
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "step": "ocr",
            "model": self.ocr_model_name,
        }

        if mode:
            entry["mode"] = mode

        self.ocr_results["processing_history"].append(entry)

    def _add_correction_history_entry(self, ocr_model: str, correction_mode: str) -> None:
        """Add an entry to the correction processing history."""
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "step": "correction",
            "ocr_model": ocr_model,
            "correction_model": self.correction_model_name,
            "correction_mode": correction_mode,
        }

        self.correction_results["processing_history"].append(entry)

    def _save_ocr_results(self) -> None:
        """Save the OCR results to the JSON file."""
        # Update timestamp
        self.ocr_results["document_info"]["last_updated"] = datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        with open(self.ocr_results_file_path, "w", encoding="utf-8") as f:
            json.dump(self.ocr_results, f, ensure_ascii=False, indent=2)

        logging.info(f"Saved OCR results to {self.ocr_results_file_path}")

    def _save_correction_results(self) -> None:
        """Save the correction results to the JSON file."""
        # Update timestamp
        self.correction_results["document_info"]["last_updated"] = datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        with open(self.correction_results_file_path, "w", encoding="utf-8") as f:
            json.dump(self.correction_results, f, ensure_ascii=False, indent=2)

        logging.info(f"Saved correction results to {self.correction_results_file_path}")

    # STEP 1: Run OCR with different modes
    def run_ocr(self) -> None:
        """
        Step 1: Run OCR with different modes, save the results.

        Returns:
            Dictionary with OCR results by mode
        """
        logging.info(
            f"STEP 1: Processing document: {self.id} with OCR model: {self.ocr_model_name}"
        )

        # Process document with all modes
        for mode in self.modes:
            start_time = time.time()
            logging.info(f"Processing with mode: {mode.value}")
            if not self.rerun:
                if self.ocr_results["ocr_models"][self.ocr_model_name].get(mode.value):
                    lines_existed = self.ocr_results["ocr_models"][self.ocr_model_name][
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
            self.ocr_results["ocr_models"][self.ocr_model_name][mode.value] = {
                "lines": [],
                "processing_time": completion_time,
            }

            # Convert to serializable format and add to results
            for _, model_results in results.items():
                serializable_lines = []
                for line in model_results:
                    if hasattr(line, "__dict__"):
                        serializable_lines.append(dict(line))
                    else:
                        serializable_lines.append(line)

                self.ocr_results["ocr_models"][self.ocr_model_name][mode.value][
                    "lines"
                ] = serializable_lines

            # Add to history
            self._add_ocr_history_entry(mode=mode.value)

            # Save results after each mode to preserve progress
            self._save_ocr_results()

    # STEP 2: Evaluate OCR results with ground truth
    def evaluate_ocr(self) -> None:
        """
        Step 2: Evaluate the OCR results with ground truth and save metrics.

        Returns:
            Dictionary with comparison metrics
        """
        logging.info(f"STEP 2: Evaluating OCR results for model: {self.ocr_model_name}")

        if not self.ground_truth:
            logging.warning("Ground truth not provided. Skipping OCR evaluation.")
            return

        evaluation_service = OCREvaluationService()

        for mode, mode_results in self.ocr_results["ocr_models"][self.ocr_model_name].items():
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
                self.ocr_results["ocr_models"][self.ocr_model_name][mode]["metrics"] = report

                completion_time = time.time() - start_time
                logging.info(
                    f"Evaluation completed for mode {mode} in {completion_time:.2f} seconds"
                )

                # Save results after each evaluation
                self._save_ocr_results()

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
            f"STEP 3: Running OCR correction with correction model: {self.correction_model_name}, modes: {correction_modes}"
        )

        # Extract OCR text from fullpage results (required for correction)
        ocr_model_results = self.ocr_results["ocr_models"][self.ocr_model_name]

        if "fullpage" not in ocr_model_results:
            logging.warning("Fullpage OCR results required for correction but not found.")
            return

        fullpage_results = ocr_model_results.get("fullpage", {})
        lines = fullpage_results.get("lines", [])

        if not lines:
            logging.warning("No fullpage OCR results found. Skipping correction.")
            return

        ocr_text = lines[0].get("extracted_text", "")
        if not ocr_text:
            logging.warning("No OCR text found in fullpage results. Skipping correction.")
            return

        # Initialize correction model entry if not exists
        if self.correction_model_name not in self.correction_results["correction_models"]:
            self.correction_results["correction_models"][self.correction_model_name] = {}

        # Resize image once for all modes
        logging.info(f"Optimizing image for correction step (target DPI: {self.target_dpi})")
        optimized_image_str, _ = resize_image_to_dpi(
            str(self.image_path), self.target_dpi, max_pixels=2000000
        )

        # Create correction pipeline with correction model
        correction_model = self._initialize_correction_model()
        correction_pipeline = OCRCorrectionPipeline(
            model=correction_model, evaluator=self.evaluator
        )

        all_results = {}

        # Process each mode
        for mode in correction_modes:
            # Initialize correction mode if not exists
            if mode not in self.correction_results["correction_models"][self.correction_model_name]:
                self.correction_results["correction_models"][self.correction_model_name][mode] = {
                    "ocr_sources": {}
                }

            # Check if this specific OCR source already exists for this mode
            if (
                self.ocr_model_name
                in self.correction_results["correction_models"][self.correction_model_name][mode][
                    "ocr_sources"
                ]
                and not self.rerun
            ):
                logging.info(
                    f"Correction mode '{mode}' with OCR source '{self.ocr_model_name}' already exists. Skipping."
                )
                all_results[mode] = self.correction_results["correction_models"][
                    self.correction_model_name
                ][mode]["ocr_sources"][self.ocr_model_name]
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

                # Store results for this specific OCR source
                source_results = {
                    "original_ocr_text": ocr_text,
                    "corrected_text": correction_results.get("corrected_text", ""),
                    "processing_time": completion_time,
                }

                # Add mode-specific fields
                if mode == "para" and "paragraphs" in correction_results:
                    source_results["paragraphs"] = correction_results["paragraphs"]
                    source_results["paragraph_boundaries"] = correction_results.get(
                        "paragraph_boundaries", []
                    )

                # Store under the specific correction model -> mode -> OCR source
                self.correction_results["correction_models"][self.correction_model_name][mode][
                    "ocr_sources"
                ][self.ocr_model_name] = source_results
                all_results[mode] = source_results

                # Add to history
                self._add_correction_history_entry(
                    ocr_model=self.ocr_model_name, correction_mode=mode
                )

            except Exception as e:
                logging.error(f"Correction failed for mode '{mode}': {e}")
                all_results[mode] = {"error": str(e)}

        # Save results after all modes
        self._save_correction_results()

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

        if self.correction_model_name not in self.correction_results["correction_models"]:
            logging.warning(
                f"No correction results found for model '{self.correction_model_name}'."
            )
            return

        correction_model_results = self.correction_results["correction_models"][
            self.correction_model_name
        ]

        if not correction_model_results:
            logging.warning("No correction results available for evaluation.")
            return

        if modes is None:
            modes = list(correction_model_results.keys())
        elif isinstance(modes, str):
            modes = [modes]

        logging.info(
            f"STEP 4: Evaluating correction results for model '{self.correction_model_name}', modes: {modes}"
        )

        evaluation_service = OCREvaluationService()

        for mode in modes:
            if mode not in correction_model_results:
                logging.warning(f"No results found for mode '{mode}'. Skipping.")
                continue

            mode_data = correction_model_results[mode]
            ocr_sources = mode_data.get("ocr_sources", {})

            if self.ocr_model_name not in ocr_sources:
                logging.warning(
                    f"No OCR source '{self.ocr_model_name}' found for mode '{mode}'. Skipping."
                )
                continue

            logging.info(f"Evaluating mode: {mode} with OCR source: {self.ocr_model_name}")

            # Get appropriate ground truth for the mode
            ground_truth_for_mode = self._get_ground_truth_for_mode(mode)
            if not ground_truth_for_mode:
                logging.warning(f"No ground truth available for mode '{mode}'. Skipping.")
                continue

            source_results = ocr_sources[self.ocr_model_name]
            corrected_text = source_results.get("corrected_text", "")

            if not corrected_text:
                logging.warning(
                    f"No corrected text available for mode '{mode}' and OCR source '{self.ocr_model_name}'."
                )
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

                # Add metrics to source-specific results
                source_results["metrics"] = report

                completion_time = time.time() - start_time
                logging.info(
                    f"Evaluation for mode '{mode}' with OCR source '{self.ocr_model_name}' completed in {completion_time:.2f} seconds"
                )

            except Exception as e:
                logging.error(f"Error evaluating correction results for mode '{mode}': {e}")

        # Save results after all evaluations
        self._save_correction_results()

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

        logging.info(f"OCR results saved to: {self.ocr_results_file_path}")
        logging.info(f"Correction results saved to: {self.correction_results_file_path}")
