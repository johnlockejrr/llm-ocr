import base64
import io
import logging
import os
from typing import Any, Dict, Tuple

from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def get_image_info(image_path: str) -> Dict[str, Any]:
    """
    Get image information including DPI, dimensions, and file size.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary with image information
    """
    try:
        with Image.open(image_path) as img:
            # Get DPI info
            dpi = img.info.get("dpi", (72, 72))  # Default to 72 DPI if not specified
            if isinstance(dpi, tuple):
                dpi_x, dpi_y = dpi
            else:
                dpi_x = dpi_y = dpi

            # Get file size
            file_size = os.path.getsize(image_path)

            return {
                "width": img.width,
                "height": img.height,
                "dpi_x": dpi_x,
                "dpi_y": dpi_y,
                "format": img.format,
                "mode": img.mode,
                "file_size_bytes": file_size,
                "file_size_mb": file_size / (1024 * 1024),
                "total_pixels": img.width * img.height,
            }
    except Exception as e:
        logging.error(f"Error getting image info: {e}")
        return {}


def resize_image_to_dpi(
    image_path: str, target_dpi: int = 150, max_pixels: int = 2000000
) -> Tuple[str, Dict[str, Any]]:
    """
    Resize image based on target DPI or pixel limit, whichever is more restrictive.
    This helps optimize images for LLM processing by reducing token usage.

    Args:
        image_path: Path to the original image
        target_dpi: Target DPI to resize to (default: 150)
        max_pixels: Maximum total pixels (width × height) allowed (default: 2M pixels)

    Returns:
        Tuple of (base64_string, resize_info)
    """
    try:
        # Get original image info
        original_info = get_image_info(image_path)
        logging.info(
            f"Original image: {original_info['width']}×{original_info['height']} ({original_info['total_pixels']:,} pixels) at {original_info['dpi_x']}×{original_info['dpi_y']} DPI, {original_info['file_size_mb']:.2f} MB"
        )

        with Image.open(image_path) as img:
            current_dpi = original_info.get("dpi_x", 72)
            current_pixels = original_info["total_pixels"]

            # Determine if resizing is needed based on DPI or pixel count
            dpi_resize_needed = current_dpi > target_dpi
            pixel_resize_needed = current_pixels > max_pixels

            if not dpi_resize_needed and not pixel_resize_needed:
                logging.info(
                    f"Image DPI ({current_dpi}) ≤ target ({target_dpi}) and pixels ({current_pixels:,}) ≤ max ({max_pixels:,}). No resizing needed."
                )
                # Return original image as base64
                with open(image_path, "rb") as image_file:
                    image_str = base64.b64encode(image_file.read()).decode("utf-8")
                return image_str, {
                    "resized": False,
                    "original_info": original_info,
                    "final_info": original_info,
                    "reason": "no_resize_needed",
                }

            # Calculate scale factors for both constraints
            dpi_scale_factor = target_dpi / current_dpi if dpi_resize_needed else 1.0
            pixel_scale_factor = (
                (max_pixels / current_pixels) ** 0.5 if pixel_resize_needed else 1.0
            )

            # Use the more restrictive (smaller) scale factor
            scale_factor = min(dpi_scale_factor, pixel_scale_factor)
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            new_pixels = new_width * new_height

            # Log the reason for resizing
            if dpi_resize_needed and pixel_resize_needed:
                if dpi_scale_factor < pixel_scale_factor:
                    resize_reason = f"DPI constraint (DPI: {current_dpi} → {target_dpi})"
                else:
                    resize_reason = (
                        f"Pixel constraint (pixels: {current_pixels:,} → {new_pixels:,})"
                    )
            elif dpi_resize_needed:
                resize_reason = f"DPI constraint (DPI: {current_dpi} → {target_dpi})"
            else:
                resize_reason = f"Pixel constraint (pixels: {current_pixels:,} → {new_pixels:,})"

            logging.info(f"Resizing image: {resize_reason}")
            logging.info(
                f"Dimensions: {img.width}×{img.height} → {new_width}×{new_height} (scale: {scale_factor:.3f})"
            )

            # Resize image
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save to memory buffer
            buffer = io.BytesIO()
            # Preserve original format or use JPEG for efficiency
            save_format = img.format if img.format in ["JPEG", "PNG"] else "JPEG"

            if save_format == "JPEG":
                # Ensure RGB mode for JPEG
                if resized_img.mode in ("RGBA", "LA", "P"):
                    resized_img = resized_img.convert("RGB")
                resized_img.save(
                    buffer, format=save_format, quality=95, dpi=(target_dpi, target_dpi)
                )
            else:
                resized_img.save(buffer, format=save_format, dpi=(target_dpi, target_dpi))

            buffer.seek(0)

            # Convert to base64
            resized_image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Get final image info
            buffer.seek(0)
            final_img = Image.open(buffer)
            final_info = {
                "width": final_img.width,
                "height": final_img.height,
                "dpi_x": target_dpi,
                "dpi_y": target_dpi,
                "format": save_format,
                "mode": final_img.mode,
                "file_size_bytes": len(buffer.getvalue()),
                "file_size_mb": len(buffer.getvalue()) / (1024 * 1024),
                "total_pixels": final_img.width * final_img.height,
            }

            size_reduction = (1 - final_info["file_size_mb"] / original_info["file_size_mb"]) * 100
            pixel_reduction = (1 - final_info["total_pixels"] / original_info["total_pixels"]) * 100

            logging.info(f"Image resized successfully:")
            logging.info(
                f"  File size: {final_info['file_size_mb']:.2f} MB ({size_reduction:.1f}% reduction)"
            )
            logging.info(
                f"  Pixels: {final_info['total_pixels']:,} ({pixel_reduction:.1f}% reduction)"
            )

            return resized_image_str, {
                "resized": True,
                "original_info": original_info,
                "final_info": final_info,
                "scale_factor": scale_factor,
                "size_reduction_percent": size_reduction,
                "pixel_reduction_percent": pixel_reduction,
                "resize_reason": resize_reason,
                "constraints_checked": {
                    "dpi_constraint": f"{current_dpi} → {target_dpi}",
                    "pixel_constraint": f"{current_pixels:,} → {max_pixels:,}",
                    "dpi_resize_needed": dpi_resize_needed,
                    "pixel_resize_needed": pixel_resize_needed,
                    "limiting_factor": "dpi" if dpi_scale_factor < pixel_scale_factor else "pixels",
                },
            }

    except Exception as e:
        logging.error(f"Error resizing image: {e}")
        # Fallback to original image
        with open(image_path, "rb") as image_file:
            image_str = base64.b64encode(image_file.read()).decode("utf-8")
        return image_str, {
            "resized": False,
            "error": str(e),
            "original_info": get_image_info(image_path),
        }


def resize_image_for_llm(
    image_path: str, target_size: str = "medium"
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to resize images for LLM processing with predefined sizes.

    Args:
        image_path: Path to the original image
        target_size: Size preset - "small", "medium", "large", or "xlarge"

    Returns:
        Tuple of (base64_string, resize_info)
    """
    size_presets = {
        "small": {"target_dpi": 100, "max_pixels": 1000000},  # 1MP, ~1000x1000
        "medium": {"target_dpi": 150, "max_pixels": 2000000},  # 2MP, ~1400x1400
        "large": {"target_dpi": 200, "max_pixels": 4000000},  # 4MP, ~2000x2000
        "xlarge": {"target_dpi": 300, "max_pixels": 8000000},  # 8MP, ~2800x2800
    }

    if target_size not in size_presets:
        raise ValueError(f"target_size must be one of: {list(size_presets.keys())}")

    preset = size_presets[target_size]
    logging.info(
        f"Using '{target_size}' preset: max {preset['max_pixels']:,} pixels, {preset['target_dpi']} DPI"
    )

    return resize_image_to_dpi(
        image_path=image_path, target_dpi=preset["target_dpi"], max_pixels=preset["max_pixels"]
    )
