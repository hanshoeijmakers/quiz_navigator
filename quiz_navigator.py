import streamlit as st
import streamlit.components.v1 as components
import pdfplumber
from PIL import Image
import io
import json
import os
from datetime import datetime
import base64
import requests  # voor xAI fallback indien nodig
from pathlib import Path
from dotenv import load_dotenv
from pdf_preprocessing import PDFPreprocessor
import pdf_ocr
from persistence import (
    load_all_chapters, save_answer, save_ai_suggestion,
    get_answer, get_ai_suggestion, save_note, get_note,
    delete_pdf_data, export_all_data,
    load_chapter_data, save_pdf_analysis, load_pdf_analysis, has_pdf_analysis
)

# Load environment variables from .env
load_dotenv()

# ===================== CONFIG =====================
st.set_page_config(
    page_title="Quiz Navigator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===================== SESSION STATE =====================
# All data is persisted to disk:
# - raw_text, images, preprocessing_info, structured → data/{pdf_name}/{pdf_name}_metadata.json
# - answers, ai_suggestions (per question) → data/{pdf_name}/chapter_N.json
# Session state only holds current session UI state, not user data
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = {}          # {filename: {"raw_text": str, "images": list[dict], "structured": dict}}
    # Load all previously analyzed PDFs from disk
    data_dir = Path("data")
    if data_dir.exists():
        for metadata_file in sorted(data_dir.glob("*_metadata.json")):
            pdf_name = metadata_file.name.replace("_metadata.json", "")
            analysis = load_pdf_analysis(pdf_name)
            if analysis:
                preprocessing_info = analysis.get("preprocessing_info", {})
                st.session_state.pdf_data[pdf_name] = {
                    "raw_text": analysis.get("raw_text", ""),
                    "images": analysis.get("images", []),
                    "structured": analysis.get("structured"),
                    "preprocessing_info": preprocessing_info,
                    "debug_log": preprocessing_info.get("debug_log"),
                }

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
if "page" not in st.session_state:
    st.session_state.page = "home"          # "home" | "timeline" | "navigator"
if "nav_chapter" not in st.session_state:
    st.session_state.nav_chapter = 0        # selected chapter index
if "nav_question" not in st.session_state:
    st.session_state.nav_question = 0       # selected question index within chapter
if "config" not in st.session_state:
    st.session_state.config = {
        "provider": "xai",
        "xai_key": os.getenv("XAI_API_KEY", ""),
        "xai_model": "grok-4-1-fast-reasoning",
        "openai_key": "",
        "openai_model": "gpt-4o"
    }

# ===================== SIDEBAR - CONFIG & NAVIGATION =====================
with st.sidebar:
    st.header("⚙️ Configuratie")

    # Check OCR availability for scanned PDFs
    if not pdf_ocr.PYTESSERACT_AVAILABLE:
        st.error("⚠️ OCR not available - install with: `pip install pytesseract` + system tesseract")
        with st.expander("📖 OCR Setup Instructions"):
            st.markdown("""
            Your PDFs are scanned, so OCR is required:

            **macOS:**
            ```bash
            brew install tesseract
            pip install pytesseract
            ```

            **Ubuntu/Debian:**
            ```bash
            sudo apt-get install tesseract-ocr
            pip install pytesseract
            ```

            **Windows:**
            1. Download installer: https://github.com/UB-Mannheim/tesseract/wiki
            2. Install (note the path)
            3. `pip install pytesseract`
            4. In Python: `pytesseract.pytesseract.pytesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'`
            """)
    provider = st.selectbox("AI Provider", ["xAI Grok", "OpenAI", "Geen AI (alleen handmatig)"],
                            index=0 if st.session_state.config["provider"] == "xai" else 1 if st.session_state.config["provider"] == "openai" else 2)

    if provider == "xAI Grok":
        st.session_state.config["provider"] = "xai"
        st.session_state.config["xai_model"] = st.selectbox("Model", ["grok-4-1-fast-reasoning", "grok-4.20-reasoning"],
                                                            index=0)
        if os.getenv("XAI_API_KEY"):
            st.caption("ℹ️ API key loaded from .env file")
        else:
            st.session_state.config["xai_key"] = st.text_input("xAI API Key", value=st.session_state.config["xai_key"], type="password")
    elif provider == "OpenAI":
        st.session_state.config["provider"] = "openai"
        st.session_state.config["openai_key"] = st.text_input("OpenAI API Key", value=st.session_state.config["openai_key"], type="password")
        st.session_state.config["openai_model"] = st.selectbox("Model", ["gpt-4o", "o1"], index=0)
    else:
        st.session_state.config["provider"] = "none"

    st.divider()

    # Main navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
    with col2:
        if st.button("⏰ Tijdlijn", use_container_width=True):
            st.session_state.page = "timeline"
            st.rerun()

    # Chapter/Question navigator - show all analyzed PDFs with expandable chapters
    analyzed_pdfs = {fname: data for fname, data in st.session_state.pdf_data.items() if data.get("structured")}

    if analyzed_pdfs:
        st.divider()
        st.header("📖 Navigator")

        # Show all PDFs with their chapters as expandable sections
        for pdf_name in sorted(analyzed_pdfs.keys()):
            data = analyzed_pdfs[pdf_name]
            questions = data["structured"].get("questions", [])

            if questions:
                # Group by chapter
                chapters = {}
                for q in questions:
                    ch = q["chapter"]
                    if ch not in chapters:
                        chapters[ch] = []
                    chapters[ch].append(q)

                # Show PDF name as collapsible section
                with st.expander(f"📄 {pdf_name}", expanded=False):
                    for ch in sorted(chapters.keys()):
                        with st.expander(f"📚 Hoofdstuk {ch}", expanded=False):
                            for q in chapters[ch]:
                                key = f"{q['chapter']}-{q['num']}"
                                pdf_key = pdf_name.replace(" ", "_").replace(".", "_")
                                if st.button(f"Vraag {q['num']}: {q['title']}", key=f"nav_{pdf_key}_{key}", use_container_width=True):
                                    st.session_state.page = "navigator"
                                    st.session_state.current_pdf = pdf_name
                                    st.session_state.nav_chapter = ch
                                    st.session_state.nav_question = q['num']
                                    st.rerun()

# ===================== LLM HELPER =====================
def call_llm(prompt: str, images: list[str] = None, temperature: float = 0.7) -> str:
    """
    Call LLM with optional image support.

    Args:
        prompt: Text prompt to send to LLM
        images: Optional list of base64 image strings (for vision models)
        temperature: Temperature for response generation

    Returns:
        LLM response text
    """
    provider = st.session_state.config["provider"]
    if provider == "none":
        return "Geen AI ingeschakeld. Bewerk handmatig."

    try:
        if provider == "xai":
            # xAI is OpenAI-compatible, supports vision
            from openai import OpenAI
            client = OpenAI(
                api_key=st.session_state.config["xai_key"],
                base_url="https://api.x.ai/v1"
            )

            # Build message content with text and optional images
            content = [{"type": "text", "text": prompt}]

            if images:
                for img_base64 in images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    })

            response = client.chat.completions.create(
                model=st.session_state.config["xai_model"],
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_tokens=4000
            )
            return response.choices[0].message.content.strip()

        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=st.session_state.config["openai_key"])

            # Build message content with text and optional images
            content = [{"type": "text", "text": prompt}]

            if images:
                for img_base64 in images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    })

            response = client.chat.completions.create(
                model=st.session_state.config["openai_model"],
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_tokens=4000
            )
            return response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"LLM fout: {e}")
        return f"LLM fout: {str(e)}"
    return ""

# ===================== VISION-BASED EXTRACTION =====================
@st.cache_data(show_spinner=False)
def _render_pdf_pages(filename: str, page_start: int, page_end: int) -> list[str]:
    """
    Render specific pages of the saved PDF as base64 JPEG strings.
    JPEG at quality=85 is ~5-10x smaller than PNG with minimal quality loss,
    which speeds up both browser display and LLM API uploads.
    Cached so repeated navigation doesn't re-render the same pages.
    """
    import io as _io
    pdf_path = Path("data") / filename / "original.pdf"
    if not pdf_path.exists():
        return []
    try:
        result = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num in range(page_start, min(page_end + 1, len(pdf.pages) + 1)):
                page = pdf.pages[page_num - 1]
                page_image = page.to_image(resolution=150)
                img = page_image.original.convert("RGB")
                buf = _io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                result.append(base64.b64encode(buf.getvalue()).decode())
        return result
    except Exception:
        return []


def _get_page_screenshots(filename: str) -> list[dict]:
    """
    Get full-page screenshots of a PDF for vision analysis.
    Uses the saved original PDF file (stored during upload).
    Returns list of {page: N, base64: "..."} dicts.
    """
    import io as _io
    pdf_path = Path("data") / filename / "original.pdf"
    if not pdf_path.exists():
        return []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            screenshots = []
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_image = page.to_image(resolution=150)
                    buf = _io.BytesIO()
                    page_image.original.save(buf, format="PNG")
                    screenshots.append({
                        "page": page_num,
                        "base64": base64.b64encode(buf.getvalue()).decode(),
                    })
                except Exception:
                    pass
            return screenshots
    except Exception as e:
        st.warning(f"⚠️ Could not load PDF for vision analysis: {str(e)[:100]}")
        return []


def extract_questions_with_vision(filename: str, preprocessed: dict) -> dict:
    """
    Use Grok's vision to extract questions from PDF page screenshots.

    Converts each PDF page to an image and sends it to the LLM, which is much
    more reliable than sending embedded images (photos/illustrations) which
    don't contain question text.

    Args:
        filename: PDF filename
        data: Session data with raw_text and images
        preprocessed: Preprocessing results (questions_detected, extraction_summary)

    Returns:
        Dict with vision_questions and merged_questions
    """
    page_screenshots = _get_page_screenshots(filename)
    if not page_screenshots:
        st.warning("⚠️ Vision analysis skipped: original PDF not found on disk. Re-upload the PDF to enable vision detection.")
        return {"vision_questions": [], "merged_questions": []}

    detected_q_summary = "\n".join([
        f"  - Vraag {q['num']}: {q['text'][:80]}..."
        for q in preprocessed.get("questions_detected", [])
    ])

    vision_questions = []

    with st.spinner(f"📸 Grok analyseert pagina's van {filename}..."):
        for screenshot in page_screenshots:
            page_num = screenshot["page"]
            img_base64 = screenshot["base64"]

            prompt = f"""Je bent een perfecte quiz-assistent. Analyseer deze PDF pagina-afbeelding en extracteer ALLE vragen.

CONTEXT - Vragen die al via OCR gevonden zijn (mogelijk incompleet):
{detected_q_summary}

TAAK:
1. Kijk naar de afbeelding
2. Identificeer ALLE vragen op deze pagina (volledig getal "Vraag 1", "Vraag 2", etc.)
3. Voor ELKE vraag die je ziet: geef nummer en volledige tekst
4. Als er geen vragen op deze pagina staan, geef een lege lijst
5. Geef resultaat als JSON:

{{
  "questions": [
    {{"num": 1, "full_text": "volledige vraagtekst"}},
    {{"num": 2, "full_text": "..."}}
  ]
}}

BELANGRIJK: Retourneer ALLEEN valide JSON, geen extra tekst.
"""

            try:
                result = call_llm(prompt, images=[img_base64], temperature=0.1)

                # Parse JSON
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    result = result.split("```")[1].split("```")[0].strip()
                vision_result = json.loads(result)

                for q in vision_result.get("questions", []):
                    if q.get("num") and q.get("full_text"):
                        vision_questions.append({
                            "num": q["num"],
                            "page": page_num,
                            "full_text": q.get("full_text", "")[:1000],
                        })
                        st.caption(f"  Pagina {page_num}: gevonden Vraag {q['num']}")
            except Exception as e:
                st.warning(f"⚠️ Vision analysis failed for page {page_num}: {str(e)[:100]}")

    # Merge vision results with detected questions
    merged_questions = {}

    # First, add all detected questions
    for q in preprocessed.get("questions_detected", []):
        merged_questions[q["num"]] = q

    # Then, enhance with vision results
    for vq in vision_questions:
        if vq["num"] in merged_questions:
            # Update with fuller text from vision
            merged_questions[vq["num"]]["text"] = vq["full_text"]
            merged_questions[vq["num"]]["vision_enhanced"] = True
        else:
            # Add new question found only by vision
            # Infer chapter from existing questions
            chapter = 1
            if preprocessed.get("questions_detected"):
                chapter = preprocessed["questions_detected"][0].get("chapter", 1)

            merged_questions[vq["num"]] = {
                "num": vq["num"],
                "chapter": chapter,
                "text": vq["full_text"],
                "vision_found": True
            }

    return {
        "vision_questions": vision_questions,
        "merged_questions": sorted(merged_questions.values(), key=lambda x: x["num"])
    }


# ===================== ANALYSEER PDF =====================
def analyze_pdf(filename: str):
    data = st.session_state.pdf_data[filename]
    if data["structured"] is not None:
        return

    # Step 1: Preprocess the text
    with st.spinner(f"Voorverwerking van {filename}..."):
        preprocessor = PDFPreprocessor()
        preprocessed = preprocessor.preprocess(data["raw_text"])

        # Store preprocessing info for debugging
        data["preprocessing_info"] = {
            "questions_detected": len(preprocessed["questions_detected"]),
            "timeline_sections": preprocessed["timeline_sections_detected"],
            "task_sections": preprocessed["task_sections_detected"]
        }

    # Store preprocessed info for debugging
    st.write(f"📊 Preprocessing detecties: {preprocessed['questions_detected'].__len__()} vragen, {preprocessed['timeline_sections_detected']} timeline, {preprocessed['task_sections_detected']} taken")

    # Show sample of what we detected
    with st.expander("🔍 Preprocessing details", expanded=False):
        st.write("**Gedetecteerde vragen:**")
        for q in preprocessed["questions_detected"][:5]:
            st.write(f"- Vraag {q['num']} (H{q['chapter']}): {q['text'][:100]}...")
        if len(preprocessed["questions_detected"]) > 5:
            st.write(f"... en nog {len(preprocessed['questions_detected']) - 5}")

        st.write("\n**Extraction summary (voor LLM):**")
        st.text(preprocessed["extraction_summary"][:1000])

    # Step 2: Use vision to enhance question extraction (fill gaps from OCR)
    vision_result = extract_questions_with_vision(filename, preprocessed)

    # Use merged questions for LLM structuring
    merged_questions_for_llm = vision_result["merged_questions"] if vision_result["merged_questions"] else preprocessed["questions_detected"]
    merged_questions_summary = "\n".join([
        f"  - Vraag {q['num']} (Chapter {q.get('chapter', 1)}, page_start={q.get('page', q.get('full_position', '?'))}): {q['text'][:400]}..."
        for q in merged_questions_for_llm
    ])

    # Build debug log entry for vision step
    debug_log = {
        "ocr_detected": [{"num": q["num"], "text": q["text"][:200]} for q in preprocessed.get("questions_detected", [])],
        "vision_per_page": [{"page": vq["page"], "num": vq["num"], "text": vq["full_text"][:200]} for vq in vision_result["vision_questions"]],
        "bevestigde_vragen": [
            {"num": q["num"], "source": "vision" if q.get("vision_found") else ("vision+ocr" if q.get("vision_enhanced") else "ocr"), "text": q.get("text", "")[:200]}
            for q in merged_questions_for_llm
        ],
        "safety_net_recovered": [],
    }

    # Step 3: Send preprocessed text to LLM for final structuring
    prompt = f"""Je bent een perfecte quiz-assistent. Analyseer de volgende Nederlandse quiz-PDF tekst en geef ALLEEN een valide JSON terug (geen extra tekst).

De tekst is voorverwerkt met gedetecteerde structuur markers:
- [CONTAINS_TIME_INFO] = regel bevat tijdinformatie
- [CONTAINS_TASK_INFO] = regel bevat actie/doe-opdracht
- [EXTRACTION_SUMMARY] = samenvatting van gedetecteerde structuur

BEVESTIGDE VRAGEN (visueel geverifieerd door directe analyse van de PDF-pagina's):
{merged_questions_summary}

KRITISCH: De bovenstaande lijst is de autoritatieve bron voor welke vragen bestaan. Elke vraag in deze lijst MOET in de JSON output verschijnen, ook als de tekst hieronder de vraagtekst niet goed weergeeft (OCR-fouten). Gebruik de vraagtekst uit de lijst als de tekst hieronder niet leesbaar is.

Tekst (mogelijk onvolledig door OCR-fouten in gescande pagina's):
{preprocessed['full_text_for_llm']}

JSON structuur (exact):
{{
  "timeline": [
    {{"time": "20:30-22:00", "description": "Golden Gate Bridge activity - fietsen of rijden tussen deze tijden", "question_ref": "Vraag 5"}},
    {{"time": "23:00", "description": "Uitchecken bij organisatie", "question_ref": "Vraag 5"}},
    ...
  ],
  "doe_opdrachten": [
    {{"vraag": "Vraag 1", "beschrijving": "Maak twee Route 30 foto's...", "hoofdstuk": 1}},
    ...
  ],
  "questions": [
    {{"chapter": 1, "num": 1, "title": "Vraag 1", "full_text": "Volledige vraagtekst inclusief subvragen", "type": "doe|kennis|foto|actie", "page_start": 3, "page_end": 4}},
    ...
  ]
}}

BELANGRIJK voor timeline:
- Let op [CONTAINS_TIME_INFO] markers - deze regels zijn waarschijnlijk relevant
- Zoek naar: uren vermeldingen (bijv "20.30 uur", "14:00", "tussen X en Y"), tijdsduuren
- Dit zijn ALLEEN activiteiten die deel uitmaken van de quiz zelf (wat studenten moeten doen), niet historische achtergrond
- Format tijdstip als "HH:MM" of "HH:MM-HH:MM" als het een bereik is
- "question_ref" moet de vraagcode zijn waar de activiteit staat

Voor doe-opdrachten:
- Let op [CONTAINS_TASK_INFO] markers - deze regels zijn waarschijnlijk relevant
- Alles wat een fysieke actie vraagt (foto maken, knutselen, video maken, etc.)

BELANGRIJK voor vragen:
- Neem ALLE vragen uit de BEVESTIGDE VRAGEN lijst op — sla er geen over
- De [EXTRACTION_SUMMARY] toont voor elke vraag een "page_start=N" waarde — gebruik die waarde exact als die beschikbaar is
- Als je een vraag vindt die NIET in de EXTRACTION_SUMMARY staat, gebruik dan de dichtstbijzijnde [PAGINA N] marker vóór die vraag in de tekst als page_start
- Verzin geen paginanummers; gebruik altijd de [PAGINA N] markers als bron
"""

    with st.spinner(f"AI analyseert {filename}..."):
        result = call_llm(prompt, temperature=0.3)
        try:
            # Probeer JSON te parsen (soms zit er markdown omheen)
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            structured = json.loads(result)
            # Safety net: if the LLM still dropped a vision-confirmed question, inject it
            # using the text Grok read from the page image (not OCR preprocessing)
            llm_q_nums = {q["num"] for q in structured.get("questions", [])}
            debug_log["llm_returned"] = sorted(llm_q_nums)
            for vq in merged_questions_for_llm:
                if vq["num"] not in llm_q_nums:
                    vision_text = vq.get("text", "")
                    structured.setdefault("questions", []).append({
                        "chapter": vq.get("chapter", 1),
                        "num": vq["num"],
                        "title": f"Vraag {vq['num']}",
                        "full_text": vision_text,
                        "type": "doe",
                        "page_start": vq.get("page", 0),
                        "page_end": vq.get("page", 0),
                        "vision_recovered": True,
                    })
                    debug_log["safety_net_recovered"].append(vq["num"])
            structured["questions"].sort(key=lambda q: (q.get("chapter", 1), q["num"]))

            data["structured"] = structured
            data["debug_log"] = debug_log
            timeline_count = len(structured.get('timeline', []))
            q_count = len(structured.get('questions', []))
            st.success(f"✅ {filename} geanalyseerd! ⏰ Tijdlijn: {timeline_count} | Vragen: {q_count}")

            # Save analysis results to disk (debug_log stored inside preprocessing_info)
            full_preprocessing_info = {**data.get("preprocessing_info", {}), "debug_log": debug_log}
            save_pdf_analysis(filename, data["raw_text"], data["images"], structured, full_preprocessing_info)

        except Exception as e:
            st.error(f"JSON parse fout: {e}")
            st.text_area("Raw LLM output (voor debug)", result, height=300)

            # Show what preprocessing detected
            with st.expander("📊 Preprocessing info"):
                st.json(data["preprocessing_info"])

# ===================== PAGE ROUTING =====================
if st.session_state.page == "home":
    # HOME PAGE - PDF Upload and Management
    st.title("📄 Quiz Navigator")
    st.markdown("""
**Houd het overzicht op 1 plaats**
• Overzicht van alle doe-opdrachten
• Chronologische tijdlijn van alle events
• Navigator per hoofdstuk & vraag met AI-suggesties/antwoorden
• Handmatig bewerken + opnieuw genereren met extra input
• Werkt met grote quizzen (veel hoofdstukken/vragen)
• API key vereist voor AI analyse en suggesties
""")
    st.header("📤 Upload je Quiz PDF's")
    st.markdown("""
    Sleep je quiz-PDF's hier naartoe. Je kunt meerdere PDF's tegelijk uploaden.
    """)

    uploaded_files = st.file_uploader("Sleep je quiz-PDF's hier (meerdere toegestaan)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        # Process one file at a time to avoid blocking the UI
        new_files = [f for f in uploaded_files if f.name not in st.session_state.pdf_data]

        if new_files:
            # Process only the first new file in this render
            file = new_files[0]
            progress_container = st.container()

            with progress_container:
                with st.spinner(f"Verwerken {file.name}..."):
                    try:
                        pdf_bytes = file.read()
                        # Save original PDF to disk for later vision analysis
                        pdf_dir = Path("data") / file.name
                        pdf_dir.mkdir(parents=True, exist_ok=True)
                        (pdf_dir / "original.pdf").write_bytes(pdf_bytes)

                        import io as _io
                        with pdfplumber.open(_io.BytesIO(pdf_bytes)) as pdf:
                            raw_text = ""
                            images = []
                            total_pages = len(pdf.pages)

                            # Use OCR for scanned PDFs (convert pages to images and extract text)
                            # Skip upfront pdf2image conversion - convert pages on-demand only if needed for OCR
                            progress_bar = st.progress(0, text=f"Pagina 0/{total_pages}")
                            for page_num, page in enumerate(pdf.pages, 1):
                                # Try regular text extraction first (fast)
                                text = page.extract_text() or ""

                                # If no text or very little text, use OCR
                                if not text or len(text.strip()) < 10:
                                    try:
                                        # Convert this page to image only if needed (lower DPI for speed)
                                        page_image = page.to_image(resolution=150)
                                        pil_image = page_image.original

                                        ocr_text = pdf_ocr.extract_text_from_image(pil_image, lang="nld+eng")
                                        text = ocr_text
                                    except Exception as ocr_err:
                                        st.warning(f"⚠️ OCR failed for page {page_num}: {str(ocr_err)[:100]}")

                                raw_text += f"\n\n--- PAGINA {page_num} ---\n{text}"

                                # Afbeeldingen extraheren (embedded images in PDF)
                                for img in page.images:
                                    try:
                                        img_data = img["stream"].get_data()
                                        pil_img = Image.open(io.BytesIO(img_data))
                                        buf = io.BytesIO()
                                        pil_img.save(buf, format="PNG")
                                        images.append({
                                            "page": page_num,
                                            "base64": base64.b64encode(buf.getvalue()).decode(),
                                            "width": pil_img.width,
                                            "height": pil_img.height
                                        })
                                    except:
                                        pass

                                # Update progress
                                progress_bar.progress(page_num / total_pages, text=f"Pagina {page_num}/{total_pages}")

                            st.session_state.pdf_data[file.name] = {
                                "raw_text": raw_text,
                                "images": images,
                                "structured": None
                                # answers and ai_suggestions are persisted to disk, loaded on demand
                            }

                            # Save raw extraction for future use (no need to re-extract)
                            save_pdf_analysis(file.name, raw_text, images, None, {})

                        st.success(f"✅ {file.name} geladen ({len(raw_text)} tekens, {len(images)} afbeeldingen)")
                        # Rerun to process next file
                        st.rerun()
                    except Exception as e:
                        st.error(f"Fout bij {file.name}: {e}")
                        import traceback
                        st.text(traceback.format_exc())

    # Show uploaded PDFs
    if st.session_state.pdf_data:
        st.divider()
        st.subheader("Geüploade PDF's")

        # Load previously analyzed PDFs from disk
        for fname in list(st.session_state.pdf_data.keys()):
            if st.session_state.pdf_data[fname]["structured"] is None and has_pdf_analysis(fname):
                saved_analysis = load_pdf_analysis(fname)
                if saved_analysis:
                    st.session_state.pdf_data[fname]["structured"] = saved_analysis.get("structured")
                    st.session_state.pdf_data[fname]["preprocessing_info"] = saved_analysis.get("preprocessing_info", {})

        for fname, data in st.session_state.pdf_data.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"📄 **{fname}**")
                if data["structured"]:
                    q_count = len(data["structured"].get("questions", []))
                    st.caption(f"✅ Geanalyseerd ({q_count} vragen)")
                else:
                    st.caption("⏳ Nog niet geanalyseerd")
            with col2:
                if not data["structured"]:
                    if st.button("📊 Analyseer", key=f"analyze_{fname}"):
                        st.session_state.current_pdf = fname
                        analyze_pdf(fname)
                        st.rerun()
                else:
                    if st.button("🔄 Heranalyseer", key=f"reanalyze_{fname}"):
                        data["structured"] = None
                        data["debug_log"] = None
                        st.session_state.current_pdf = fname
                        analyze_pdf(fname)
                        st.rerun()
            with col3:
                if st.button("🗑", key=f"delete_{fname}"):
                    st.session_state[f"confirm_delete_{fname}"] = True
                if st.session_state.get(f"confirm_delete_{fname}"):
                    st.warning("Weet je zeker dat je dit hoofdstuk met alle vragen en antwoorden wil verwijderen?")
                    confirm_col1, confirm_col2 = st.columns(2)
                    with confirm_col1:
                        if st.button("Ja, verwijder", key=f"confirm_yes_{fname}", type="primary"):
                            del st.session_state.pdf_data[fname]
                            delete_pdf_data(fname)  # Also delete persisted data
                            if st.session_state.current_pdf == fname:
                                st.session_state.current_pdf = None
                            del st.session_state[f"confirm_delete_{fname}"]
                            st.rerun()
                    with confirm_col2:
                        if st.button("Annuleer", key=f"confirm_no_{fname}"):
                            del st.session_state[f"confirm_delete_{fname}"]
                            st.rerun()

            # Show preprocessing info upfront if available
            if "preprocessing_info" in data and data["preprocessing_info"]:
                st.write("**📊 Preprocessing detecties:**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Vragen gedetecteerd", data["preprocessing_info"].get("questions_detected", "—"))
                with col_b:
                    st.metric("Timeline secties", data["preprocessing_info"].get("timeline_sections", "—"))
                with col_c:
                    st.metric("Task secties", data["preprocessing_info"].get("task_sections", "—"))

            # Debug: Show extraction details
            with st.expander(f"🔍 Debug: {fname}"):
                st.write(f"**Karakters geëxtraheerd:** {len(data['raw_text'])}")
                st.write(f"**Afbeeldingen gevonden:** {len(data['images'])}")

                if data.get("debug_log"):
                    dl = data["debug_log"]
                    st.write("**OCR gedetecteerd:**", [q["num"] for q in dl.get("ocr_detected", [])])
                    st.write("**Vision per pagina:**")
                    for vp in dl.get("vision_per_page", []):
                        st.caption(f"  Pagina {vp['page']}: Vraag {vp['num']} — {vp['text'][:80]}")
                    st.write("**BEVESTIGDE VRAGEN (naar LLM):**")
                    for bv in dl.get("bevestigde_vragen", []):
                        st.caption(f"  Vraag {bv['num']} [{bv['source']}]: {bv['text'][:100]}")
                    st.write("**LLM returneerde vragen:**", dl.get("llm_returned", []))
                    if dl.get("safety_net_recovered"):
                        st.warning(f"Safety net hersteld: {dl['safety_net_recovered']}")

                with st.expander("Raw text (first 2000 chars)", expanded=False):
                    st.text_area("Extracted text preview", value=data["raw_text"][:2000], height=300, disabled=True, key=f"raw_{fname}")

elif st.session_state.page == "timeline":
    # TIMELINE PAGE - Merged timeline + doe-opdrachten
    st.header("⏰ Quiz Tijdlijn & Opdrachten")

    if not st.session_state.pdf_data:
        st.info("Upload eerst PDF's op de Home pagina")
    else:
        # Collect all events
        all_events = []
        for fname, data in st.session_state.pdf_data.items():
            if data.get("structured"):
                for ev in data["structured"].get("timeline", []):
                    all_events.append({**ev, "pdf": fname})

        if all_events:
            st.subheader("⏰ Tijdlijn Events")
            for ev in sorted(all_events, key=lambda x: x.get('time', '')):
                st.markdown(f"**🕐 {ev['time']}** — {ev['description']}  \n<small>Vraag: {ev.get('question_ref', 'onbekend')} | {ev['pdf']}</small>")
        else:
            st.info("Geen timeline events gevonden")

        st.divider()

        # Doe-opdrachten
        st.subheader("🛠 Doe-Opdrachten")
        any_tasks = False
        for fname, data in st.session_state.pdf_data.items():
            if data.get("structured"):
                tasks = data["structured"].get("doe_opdrachten", [])
                if tasks:
                    any_tasks = True
                    st.write(f"**{fname}**")
                    for d in tasks:
                        st.markdown(f"• **{d['vraag']}** — {d['beschrijving']}")

        if not any_tasks:
            st.info("Geen doe-opdrachten gevonden")

        st.divider()

        # Download answers
        if st.button("📥 Download alles als JSON"):
            all_data = {}
            for fname in st.session_state.pdf_data.keys():
                all_data[fname] = export_all_data(fname)
            st.download_button("Download quiz_antwoorden.json",
                               data=json.dumps(all_data, indent=2, ensure_ascii=False),
                               file_name="quiz_antwoorden.json",
                               mime="application/json")

elif st.session_state.page == "navigator":
    # NAVIGATOR PAGE - Question with integrated images and answers
    if not st.session_state.current_pdf:
        st.error("Geen PDF geselecteerd")
    else:
        data = st.session_state.pdf_data.get(st.session_state.current_pdf)
        if not data or not data.get("structured"):
            st.error("PDF nog niet geanalyseerd")
        else:
            questions = data["structured"].get("questions", [])

            # Find the selected question
            selected_q = None
            for q in questions:
                if q["chapter"] == st.session_state.nav_chapter and q["num"] == st.session_state.nav_question:
                    selected_q = q
                    break

            if selected_q is None:
                st.error("Vraag niet gevonden")
            else:
                key = f"{selected_q['chapter']}-{selected_q['num']}"

                # Display question text (no header for navigator page)
                st.markdown(selected_q["full_text"])

                st.divider()

                page_start = selected_q.get("page_start", selected_q.get("page", 1))
                # Derive page_end: use next question's page_start - 1 (questions are in order)
                q_index = questions.index(selected_q)
                if q_index + 1 < len(questions):
                    next_q_page = questions[q_index + 1].get("page_start", page_start + 1)
                    page_end = max(page_start, next_q_page - 1)
                else:
                    page_end = selected_q.get("page_end", page_start)

                page_images = _render_pdf_pages(
                    st.session_state.current_pdf, page_start, page_end
                )

                if page_images:
                    st.subheader("📸 Afbeeldingen")

                    parts = []
                    for i, b64 in enumerate(page_images):
                        page_num = page_start + i
                        parts.append(
                            '<img src="data:image/jpeg;base64,' + b64 + '" '
                            'onclick="openLightbox(this.src)" title="Pagina ' + str(page_num) + '" />'
                        )
                    thumbs_html = "".join(parts)
                    num_rows = (len(page_images) + 2) // 3
                    thumb_height = num_rows * 180 + 20

                    lightbox_html = f"""
<style>
  body {{ margin: 0; background: transparent; }}
  .thumb-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }}
  .thumb-grid img {{
    width: 100%;
    cursor: zoom-in;
    border-radius: 6px;
    border: 1px solid #ddd;
    transition: opacity 0.15s;
  }}
  .thumb-grid img:hover {{ opacity: 0.75; }}
</style>
<div class="thumb-grid">{thumbs_html}</div>
<script>
function openLightbox(src) {{
  var doc = window.parent.document;
  var existing = doc.getElementById('qn-lightbox');
  if (existing) existing.remove();

  var overlay = doc.createElement('div');
  overlay.id = 'qn-lightbox';
  overlay.style.cssText = [
    'position:fixed', 'inset:0', 'background:rgba(0,0,0,0.88)',
    'z-index:999999', 'display:flex', 'align-items:center',
    'justify-content:center', 'cursor:zoom-out'
  ].join(';');

  var img = doc.createElement('img');
  img.src = src;
  img.style.cssText = 'max-width:92vw;max-height:92vh;object-fit:contain;border-radius:6px;box-shadow:0 4px 32px rgba(0,0,0,0.5)';

  overlay.appendChild(img);
  overlay.onclick = function() {{ overlay.remove(); }};
  doc.body.appendChild(overlay);

  function onKey(e) {{
    if (e.key === 'Escape') {{ overlay.remove(); window.parent.removeEventListener('keydown', onKey); }}
  }}
  window.parent.addEventListener('keydown', onKey);
}}
</script>
"""
                    components.html(lightbox_html, height=thumb_height)

                st.divider()

                # Answer section
                st.subheader("💾 Antwoord & Suggesties")

                # Retrieve both custom answer and AI suggestion from disk
                current_answer = get_answer(st.session_state.current_pdf, selected_q["chapter"], key)
                current_ai_suggestion = get_ai_suggestion(st.session_state.current_pdf, selected_q["chapter"], key)

                # Display AI suggestion if available
                if current_ai_suggestion:
                    with st.expander("🤖 AI Suggestie", expanded=True):
                        st.markdown(current_ai_suggestion)
                        if st.button("📋 Kopieer naar antwoord", key=f"copy_{key}"):
                            save_answer(st.session_state.current_pdf, selected_q["chapter"], key, current_ai_suggestion)
                            st.session_state[f"answer_{key}"] = current_ai_suggestion
                            st.toast("✅ AI suggestie gekopieerd naar antwoord!")

                # Custom answer input - use updated value after copy
                current_answer = get_answer(st.session_state.current_pdf, selected_q["chapter"], key)
                new_answer = st.text_area("Jouw Antwoord", value=current_answer, height=200, key=f"answer_{key}")

                generating_key = f"generating_suggestion_{key}"
                extra_input_key = f"extra_input_{key}"
                if generating_key not in st.session_state:
                    st.session_state[generating_key] = False

                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.text_input("Extra context voor AI (optioneel)", placeholder="Bijv. teamnummer, extra info", key=extra_input_key)
                with col2:
                    if st.button("🤖 Genereer AI Suggestie", type="primary", disabled=st.session_state[generating_key]):
                        st.session_state[generating_key] = True
                        st.rerun()
                with col3:
                    if st.button("💾 Opslaan"):
                        save_answer(st.session_state.current_pdf, selected_q["chapter"], key, new_answer)
                        st.toast("✅ Antwoord opgeslagen!")

                st.divider()

                # Notes section
                st.subheader("📝 Notities")
                current_note = get_note(st.session_state.current_pdf, selected_q["chapter"], key)
                new_note = st.text_area("Notities", value=current_note, height=120, key=f"note_{key}", label_visibility="collapsed")
                if st.button("💾 Notitie opslaan", key=f"save_note_{key}"):
                    save_note(st.session_state.current_pdf, selected_q["chapter"], key, new_note)
                    st.toast("✅ Notitie opgeslagen!")

                if st.session_state[generating_key]:
                    with st.spinner("🤖 AI suggestie genereren..."):
                        extra_input_val = st.session_state.get(extra_input_key, "")
                        prompt = f"""Geef een volledig, correct en creatief antwoord/suggestie voor deze quizvraag (in natuurlijk Nederlands):

Vraag:
{selected_q['full_text']}

Extra context van gebruiker:
{extra_input_val or 'geen'}"""
                        suggestion_images = page_images if page_images else None
                        suggestion = call_llm(prompt, images=suggestion_images)
                        save_ai_suggestion(st.session_state.current_pdf, selected_q["chapter"], key, suggestion)
                        st.session_state[generating_key] = False
                    st.rerun()

# ===================== FOOTER =====================
st.caption("© 2026 Quiz Navigator • Hans Hoeijmakers")
