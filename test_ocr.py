"""
Diagnostic tool to test OCR extraction on a PDF.
Usage: python test_ocr.py <pdf_path>
"""

import sys
import pdfplumber
import pdf_ocr

def test_pdf_extraction(pdf_path: str):
    """Test text and OCR extraction on a PDF."""
    print(f"\n{'='*60}")
    print(f"Testing: {pdf_path}")
    print(f"{'='*60}\n")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Total pages: {len(pdf.pages)}\n")

            # Convert to images using pdf2image (better quality)
            if pdf_ocr.PDF2IMAGE_AVAILABLE:
                print("Using pdf2image for conversion (better quality)...\n")
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                pdf_images = pdf_ocr.pdf_bytes_to_images(pdf_bytes, dpi=300)
            else:
                print("pdf2image not available, falling back to pdfplumber\n")
                pdf_images = []

            total_text_extracted = 0
            total_ocr_extracted = 0

            for page_num, page in enumerate(pdf.pages, 1):
                print(f"\n--- PAGE {page_num} ---")

                # Try standard text extraction
                text = page.extract_text()
                text_chars = len(text) if text else 0
                total_text_extracted += text_chars
                print(f"Standard extraction: {text_chars} chars")
                if text:
                    print(f"  Sample: {text[:150]}...")

                # Try OCR with pdf2image
                try:
                    # Prefer pdf2image-converted image
                    if pdf_images and page_num - 1 < len(pdf_images):
                        pil_image = pdf_images[page_num - 1]
                        print(f"  Using pdf2image conversion")
                    else:
                        page_image = page.to_image(resolution=300)
                        pil_image = page_image.original
                        print(f"  Using pdfplumber conversion")

                    print(f"  Image size: {pil_image.size}")

                    ocr_text = pdf_ocr.extract_text_from_image(pil_image, lang="nld+eng")
                    ocr_chars = len(ocr_text) if ocr_text else 0
                    total_ocr_extracted += ocr_chars
                    print(f"OCR extraction: {ocr_chars} chars")
                    if ocr_text:
                        print(f"  Sample: {ocr_text[:150]}...")
                    else:
                        print(f"  ⚠️ OCR returned no text")

                except Exception as e:
                    print(f"❌ OCR failed: {e}")

                # Count images
                images = len(page.images)
                print(f"Embedded images: {images}")

            print(f"\n{'='*60}")
            print(f"SUMMARY")
            print(f"{'='*60}")
            print(f"Total chars (standard): {total_text_extracted}")
            print(f"Total chars (OCR):      {total_ocr_extracted}")
            print(f"Better method: {'OCR' if total_ocr_extracted > total_text_extracted else 'Standard'}")
            print(f"{'='*60}\n")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_ocr.py <pdf_path>")
        print("Example: python test_ocr.py './Hoofdstuk 1.pdf'")
        sys.exit(1)

    pdf_path = sys.argv[1]
    test_pdf_extraction(pdf_path)
