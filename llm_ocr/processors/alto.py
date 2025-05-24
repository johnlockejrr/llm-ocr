import base64
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class ALTOLine:
    """Represents a single line from ALTO XML with its text and image data"""

    text: str
    points: np.ndarray
    image: np.ndarray
    label: Optional[str]
    line_id: str
    block_id: str
    page_id: str
    confidence: float = 0.0
    word_count: int = 0
    line_height: int = 0
    line_width: int = 0

    def get_base64_image(self) -> str:
        """Convert the image to base64 string"""
        _, buffer = cv2.imencode(".jpg", self.image)
        return base64.b64encode(buffer).decode("utf-8")

    def get_dimensions(self) -> Tuple[int, int]:
        """Get the dimensions of the line image"""
        return self.image.shape[:2] if self.image is not None else (0, 0)

    def get_word_count(self) -> int:
        """Get the number of words in the line"""
        return len(self.text.split()) if self.text else 0


class ALTOProcessor:
    """Processes ALTO XML files and extracts line information with images"""

    def __init__(self) -> None:
        self.ns = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
        # Register namespaces for XML handling
        ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
        ET.register_namespace("", "http://www.loc.gov/standards/alto/ns-v4#")

        # Setup logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def extract_line_image(
        self, image: np.ndarray, points: np.ndarray, padding: int = 5
    ) -> np.ndarray:
        """
        Extract line image using polygon points with optional padding

        Args:
            image: Source image
            points: Polygon points defining the line
            padding: Optional padding around the line in pixels

        Returns:
            Extracted line image
        """
        try:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(points)

            # Add padding while keeping within image bounds
            y1 = max(0, y - padding)
            y2 = min(image.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(image.shape[1], x + w + padding)

            # Create and apply mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)

            # Extract region
            masked = cv2.bitwise_and(image, image, mask=mask)
            cropped = masked[y1:y2, x1:x2]

            return cropped

        except Exception as e:
            self.logger.error(f"Error extracting line image: {str(e)}")
            return np.array([])

    def process_alto_file(
        self, xml_path: str, image_path: str, skip_labels: Optional[List[str]] = None
    ) -> List[ALTOLine]:
        """
        Process single ALTO XML file and corresponding image

        Args:
            xml_path: Path to ALTO XML file
            image_path: Path to corresponding image file
            skip_labels: Optional list of labels to skip

        Returns:
            List of ALTOLine objects
        """
        try:
            # Load XML and image
            tree = ET.parse(xml_path)
            root = tree.getroot()
            image = cv2.imread(image_path)

            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            skip_labels = skip_labels or ["LT16"]
            lines = []

            # Process each page
            for page in root.findall(".//alto:Page", self.ns):
                page_id = page.get("ID", "")

                # Process each text block
                for text_block in page.findall(".//alto:TextBlock", self.ns):
                    block_id = text_block.get("ID", "")

                    # Process each line in the block
                    for text_line in text_block.findall(".//alto:TextLine", self.ns):
                        line = self._process_text_line(
                            text_line=text_line,
                            image=image,
                            page_id=page_id,
                            block_id=block_id,
                            skip_labels=skip_labels,
                        )
                        if line:
                            lines.append(line)

            return lines

        except Exception as e:
            self.logger.error(f"Error processing {xml_path}: {str(e)}")
            raise

    def _process_text_line(
        self,
        text_line: ET.Element,
        image: np.ndarray,
        page_id: str,
        block_id: str,
        skip_labels: List[str],
    ) -> Optional[ALTOLine]:
        """
        Process single text line from ALTO XML

        Args:
            text_line: XML element containing line information
            image: Source image
            page_id: ID of the parent page
            block_id: ID of the parent text block
            skip_labels: Labels to skip

        Returns:
            ALTOLine object or None if line should be skipped
        """
        try:
            # Get line attributes
            label = text_line.get("TAGREFS")
            if label in skip_labels:
                return None

            line_id = text_line.get("ID", "")

            # Get polygon coordinates
            polygon = text_line.find(".//alto:Polygon", self.ns)
            if polygon is None:
                self.logger.warning(f"No polygon found for line {line_id}")
                return None

            coords = polygon.attrib["POINTS"]
            points = np.array(
                [[int(n) for n in point.split(",")] for point in coords.split()], np.int32
            )
            points = points.reshape((-1, 1, 2))

            # Get text content
            string_elem = text_line.find(".//alto:String", self.ns)
            if string_elem is None:
                self.logger.warning(f"No string element found for line {line_id}")
                return None

            text = string_elem.attrib["CONTENT"]

            # Extract line image
            line_image = self.extract_line_image(image, points)
            if line_image.size == 0:
                self.logger.warning(f"Failed to extract image for line {line_id}")
                return None

            # Create ALTOLine object
            line = ALTOLine(
                text=text,
                points=points,
                image=line_image,
                label=label,
                line_id=line_id,
                block_id=block_id,
                page_id=page_id,
                confidence=float(string_elem.get("WC", 0.0)),
                word_count=len(text.split()),
                line_height=line_image.shape[0],
                line_width=line_image.shape[1],
            )

            return line

        except Exception as e:
            self.logger.error(f"Error processing text line: {str(e)}")
            return None

    def process_directory(
        self, directory: Union[str, Path], pattern: str = "*.xml"
    ) -> Generator[List[ALTOLine], None, None]:
        """
        Process all ALTO XML files in a directory

        Args:
            directory: Directory containing ALTO XML files
            pattern: Glob pattern for XML files

        Yields:
            List of ALTOLine objects for each file
        """
        directory = Path(directory)
        for xml_file in directory.glob(pattern):
            image_file = xml_file.with_suffix(".jpeg")
            if not image_file.exists():
                image_file = xml_file.with_suffix(".jpg")

            if image_file.exists():
                try:
                    yield self.process_alto_file(str(xml_file), str(image_file))
                except Exception as e:
                    self.logger.error(f"Error processing {xml_file}: {str(e)}")
                    continue


# # Example usage
# if __name__ == "__main__":
#     processor = ALTOProcessor()

#     # Process single file
#     lines = processor.process_alto_file(
#         "evaluate/0f8062d2ef23.xml",
#         "evaluate/0f8062d2ef23.jpeg"
#     )

#     print(f"Processed {len(lines)} lines")
#     for line in lines:
#         print(f"Line {line.line_id}: {line.word_count} words, "
#               f"size: {line.line_width}x{line.line_height}")
