"""
OCR fallback for PDFs where text extraction fails.
Requires: pytesseract (pip install pytesseract) and tesseract-ocr system package.

On macOS: brew install tesseract
On Ubuntu: sudo apt-get install tesseract-ocr
On Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
"""

import io
from PIL import Image, ImageFilter, ImageOps
import numpy as np

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    from pdf2image import convert_from_path, convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


def preprocess_image_for_ocr(image: Image.Image, aggressive: bool = True) -> Image.Image:
    """
    Preprocess image for better OCR accuracy.
    - Convert to grayscale
    - Binarize (threshold) for clearer text
    - Increase contrast
    - Sharpen
    - Optional: Denoise

    Optimized for both text-heavy and image-heavy pages.
    """
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')

    # Apply bilateral filter for denoising while preserving edges (if aggressive)
    if aggressive:
        # Use ImageFilter for simple operations
        image = image.filter(ImageFilter.MedianFilter(size=3))

    # Convert to numpy for binarization
    img_array = np.array(image)

    # Adaptive threshold binarization (black and white)
    # Using percentile 35 instead of 40 to catch fainter text
    threshold = np.percentile(img_array, 35)
    img_array = (img_array > threshold).astype(np.uint8) * 255

    image = Image.fromarray(img_array)

    # Increase contrast
    image = ImageOps.autocontrast(image, cutoff=1)

    # Sharpen once (was sharpening twice, which could over-process)
    image = image.filter(ImageFilter.SHARPEN)

    return image


def extract_text_from_image(image: Image.Image, lang: str = "nld+eng") -> str:
    """
    Extract text from image using Tesseract OCR.
    lang: "nld+eng" for Dutch+English, "nld" for Dutch only
    """
    if not PYTESSERACT_AVAILABLE:
        return ""

    try:
        # Preprocess image aggressively
        processed = preprocess_image_for_ocr(image, aggressive=True)

        # Tesseract config for better accuracy
        # --psm 3: Automatic page segmentation with sparse text (good for documents)
        # --oem 3: Use both legacy and LSTM OCR engine modes
        custom_config = r'--psm 3 --oem 3 -c preserve_interword_spaces=1'

        # Extract text with optimized config
        text = pytesseract.image_to_string(processed, lang=lang, config=custom_config)
        return text.strip()
    except Exception as e:
        print(f"OCR error: {e}")
        return ""


def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 300) -> list:
    """
    Convert PDF bytes to images using pdf2image (better quality than pdfplumber).
    Returns list of PIL Images, one per page.
    """
    if not PDF2IMAGE_AVAILABLE:
        return []

    try:
        images = convert_from_bytes(pdf_bytes, dpi=dpi)
        return images
    except Exception as e:
        print(f"PDF to image conversion failed: {e}")
        return []


def ocr_pdf_pages(pdf_pages_as_images: list[Image.Image], lang: str = "nld+eng") -> str:
    """
    Extract text from a list of page images using OCR.
    Returns concatenated text from all pages.
    """
    if not PYTESSERACT_AVAILABLE:
        return ""

    full_text = ""
    for page_num, image in enumerate(pdf_pages_as_images, 1):
        page_text = extract_text_from_image(image, lang=lang)
        full_text += f"\n\n--- PAGINA {page_num} ---\n{page_text}"

    return full_text


def check_ocr_available() -> tuple[bool, str]:
    """
    Check if OCR is available and return status message.
    Returns: (is_available, status_message)
    """
    if not PYTESSERACT_AVAILABLE:
        return False, "pytesseract not installed (pip install pytesseract)"

    try:
        # Test if tesseract is installed
        pytesseract.get_tesseract_version()
        return True, "OCR ready"
    except pytesseract.TesseractNotFoundError:
        return False, "Tesseract not installed. macOS: 'brew install tesseract'"
    except Exception as e:
        return False, f"OCR check failed: {e}"
