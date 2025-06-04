"""
Unit tests for ALTO processor.
"""

import numpy as np

from llm_ocr.processors.alto import ALTOLine, ALTOProcessor


class TestALTOProcessor:
    """Tests for ALTOProcessor class."""

    def test_initialization(self):
        """Test ALTOProcessor initialization."""
        processor = ALTOProcessor()
        assert processor.ns == {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}

    def test_extract_line_image(self):
        """Test line image extraction."""
        processor = ALTOProcessor()

        # Create a simple test image
        image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        points = np.array([[10, 10], [190, 10], [190, 30], [10, 30]], np.int32)
        points = points.reshape((-1, 1, 2))

        line_image = processor.extract_line_image(image, points)
        assert line_image is not None
        assert line_image.shape[0] > 0  # Has height
        assert line_image.shape[1] > 0  # Has width

    def test_extract_line_image_invalid_points(self):
        """Test line image extraction with invalid points."""
        processor = ALTOProcessor()

        image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        invalid_points = np.array([[]], np.int32)

        line_image = processor.extract_line_image(image, invalid_points)
        assert line_image.size == 0


class TestALTOLine:
    """Tests for ALTOLine dataclass."""

    def test_alto_line_creation(self):
        """Test ALTOLine creation."""
        image = np.ones((20, 100, 3), dtype=np.uint8) * 255
        points = np.array([[0, 0], [100, 0], [100, 20], [0, 20]], np.int32)

        line = ALTOLine(
            text="Test line",
            points=points,
            image=image,
            label="test",
            line_id="line_001",
            block_id="block_001",
            page_id="page_001",
        )

        assert line.text == "Test line"
        assert line.line_id == "line_001"
        assert line.block_id == "block_001"
        assert line.page_id == "page_001"

    def test_get_base64_image(self):
        """Test base64 image conversion."""
        image = np.ones((20, 100, 3), dtype=np.uint8) * 255
        points = np.array([[0, 0], [100, 0], [100, 20], [0, 20]], np.int32)

        line = ALTOLine(
            text="Test line",
            points=points,
            image=image,
            label=None,
            line_id="line_001",
            block_id="block_001",
            page_id="page_001",
        )

        base64_img = line.get_base64_image()
        assert isinstance(base64_img, str)
        assert len(base64_img) > 0

    def test_get_dimensions(self):
        """Test getting line image dimensions."""
        image = np.ones((20, 100, 3), dtype=np.uint8) * 255
        points = np.array([[0, 0], [100, 0], [100, 20], [0, 20]], np.int32)

        line = ALTOLine(
            text="Test line",
            points=points,
            image=image,
            label=None,
            line_id="line_001",
            block_id="block_001",
            page_id="page_001",
        )

        height, width = line.get_dimensions()
        assert height == 20
        assert width == 100

    def test_get_word_count(self):
        """Test word count calculation."""
        image = np.ones((20, 100, 3), dtype=np.uint8) * 255
        points = np.array([[0, 0], [100, 0], [100, 20], [0, 20]], np.int32)

        line = ALTOLine(
            text="This is a test line",
            points=points,
            image=image,
            label=None,
            line_id="line_001",
            block_id="block_001",
            page_id="page_001",
        )

        word_count = line.get_word_count()
        assert word_count == 5

        # Test empty text
        line.text = ""
        assert line.get_word_count() == 0

        # Test None text
        line.text = None
        assert line.get_word_count() == 0
