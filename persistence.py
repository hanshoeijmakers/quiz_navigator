import json
import os
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")

def ensure_data_dir(pdf_name: str, chapter: int):
    """Ensure the data directory structure exists."""
    pdf_dir = DATA_DIR / pdf_name
    pdf_dir.mkdir(parents=True, exist_ok=True)
    return pdf_dir

def get_chapter_file(pdf_name: str, chapter: int) -> Path:
    """Get the path to a chapter's JSON file."""
    pdf_dir = ensure_data_dir(pdf_name, chapter)
    return pdf_dir / f"chapter_{chapter}.json"

def load_chapter_data(pdf_name: str, chapter: int) -> dict:
    """Load all answers and AI suggestions for a chapter."""
    file_path = get_chapter_file(pdf_name, chapter)
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {"questions": {}}
    return {"questions": {}}

def load_all_chapters(pdf_name: str) -> dict:
    """Load all chapters for a PDF into session state structure."""
    pdf_dir = DATA_DIR / pdf_name
    if not pdf_dir.exists():
        return {}

    all_data = {"answers": {}, "ai_suggestions": {}}
    for chapter_file in pdf_dir.glob("chapter_*.json"):
        try:
            chapter_data = json.loads(chapter_file.read_text(encoding="utf-8"))
            for q_key, q_data in chapter_data.get("questions", {}).items():
                if q_data.get("answer"):
                    all_data["answers"][q_key] = q_data["answer"]
                if q_data.get("ai_suggestion"):
                    all_data["ai_suggestions"][q_key] = q_data["ai_suggestion"]
        except Exception as e:
            print(f"Error loading {chapter_file}: {e}")
    return all_data

def save_answer(pdf_name: str, chapter: int, question_key: str, answer: str):
    """Save an answer for a question and write to disk."""
    file_path = get_chapter_file(pdf_name, chapter)
    data = load_chapter_data(pdf_name, chapter)

    if question_key not in data["questions"]:
        data["questions"][question_key] = {}

    data["questions"][question_key]["answer"] = answer

    _write_chapter_file(file_path, data)

def save_ai_suggestion(pdf_name: str, chapter: int, question_key: str, suggestion: str):
    """Save an AI suggestion for a question and write to disk."""
    file_path = get_chapter_file(pdf_name, chapter)
    data = load_chapter_data(pdf_name, chapter)

    if question_key not in data["questions"]:
        data["questions"][question_key] = {}

    data["questions"][question_key]["ai_suggestion"] = suggestion

    _write_chapter_file(file_path, data)

def _write_chapter_file(file_path: Path, data: dict):
    """Write chapter data to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_answer(pdf_name: str, chapter: int, question_key: str) -> str:
    """Get an answer for a question."""
    data = load_chapter_data(pdf_name, chapter)
    return data.get("questions", {}).get(question_key, {}).get("answer", "")

def get_ai_suggestion(pdf_name: str, chapter: int, question_key: str) -> str:
    """Get an AI suggestion for a question."""
    data = load_chapter_data(pdf_name, chapter)
    return data.get("questions", {}).get(question_key, {}).get("ai_suggestion", "")

def delete_pdf_data(pdf_name: str):
    """Delete all data for a PDF."""
    import shutil
    pdf_dir = DATA_DIR / pdf_name
    if pdf_dir.exists():
        shutil.rmtree(pdf_dir)
    metadata_file = DATA_DIR / f"{pdf_name}_metadata.json"
    if metadata_file.exists():
        metadata_file.unlink()

def export_all_data(pdf_name: str) -> dict:
    """Export all data for a PDF as a single dict."""
    return load_all_chapters(pdf_name)

# ===================== PDF METADATA & ANALYSIS RESULTS =====================

def get_pdf_metadata_file(pdf_name: str) -> Path:
    """Get the path to a PDF's metadata file."""
    pdf_dir = ensure_data_dir(pdf_name, 1)  # Use chapter 1 to create dir
    return pdf_dir.parent / f"{pdf_name}_metadata.json"

def save_pdf_analysis(pdf_name: str, raw_text: str, images: list, structured: dict, preprocessing_info: dict):
    """Save the raw PDF extraction, preprocessing results, and LLM analysis."""
    metadata_file = get_pdf_metadata_file(pdf_name)

    metadata = {
        "pdf_name": pdf_name,
        "raw_text": raw_text,
        "images": images,  # Save images as base64 for persistence
        "preprocessing_info": preprocessing_info,
        "structured": structured,
        "saved_at": str(datetime.now())
    }

    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def load_pdf_analysis(pdf_name: str) -> dict:
    """Load saved PDF analysis results."""
    metadata_file = get_pdf_metadata_file(pdf_name)
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading PDF analysis for {pdf_name}: {e}")
    return None

def has_pdf_analysis(pdf_name: str) -> bool:
    """Check if PDF has been analyzed and saved."""
    return get_pdf_metadata_file(pdf_name).exists()
