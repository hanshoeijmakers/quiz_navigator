import streamlit as st
import pdfplumber
from PIL import Image
import io
import json
import os
from datetime import datetime
import base64
import requests  # voor xAI fallback indien nodig
from dotenv import load_dotenv
from pdf_preprocessing import PDFPreprocessor
import pdf_ocr
from persistence import (
    load_all_chapters, save_answer, save_ai_suggestion,
    get_answer, get_ai_suggestion, delete_pdf_data, export_all_data,
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

st.title("📄 Quiz Navigator")
st.markdown("""
**Houd het overzicht op 1 plaats**  
• Overzicht van alle doe-opdrachten    
• Chronologische tijdlijn van alle events
• Navigator per hoofdstuk & vraag met AI-suggesties/antwoorden  
• Handmatig bewerken + opnieuw genereren met extra input  
• Werkt met grote quizzen (veel hoofdstukken/vragen)  
• Volledig lokaal (behalve AI)
""")

# ===================== SESSION STATE =====================
# All data is persisted to disk:
# - raw_text, images, preprocessing_info, structured → data/{pdf_name}/{pdf_name}_metadata.json
# - answers, ai_suggestions (per question) → data/{pdf_name}/chapter_N.json
# Session state only holds current session UI state, not user data
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = {}          # {filename: {"raw_text": str, "images": list[dict], "structured": dict}}
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
        "openai_model": "gpt-4o",
        "ollama_model": "llama3.2"
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
    provider = st.selectbox("AI Provider", ["xAI Grok", "OpenAI", "Ollama (lokaal)", "Geen AI (alleen handmatig)"],
                            index=0 if st.session_state.config["provider"] == "xai" else 1 if st.session_state.config["provider"] == "openai" else 2)

    if provider == "xAI Grok":
        st.session_state.config["provider"] = "xai"
        st.session_state.config["xai_model"] = st.selectbox("Model", ["grok-4-1-fast-reasoning", "grok-4.20-reasoning"],
                                                            index=0)
        st.caption("ℹ️ API key loaded from .env file")
    elif provider == "OpenAI":
        st.session_state.config["provider"] = "openai"
        st.session_state.config["openai_key"] = st.text_input("OpenAI API Key", value=st.session_state.config["openai_key"], type="password")
        st.session_state.config["openai_model"] = st.selectbox("Model", ["gpt-4o", "o1"], index=0)
    elif provider == "Ollama (lokaal)":
        st.session_state.config["provider"] = "ollama"
        st.session_state.config["ollama_model"] = st.text_input("Ollama model", value=st.session_state.config["ollama_model"])
        st.caption("Ollama moet draaien op http://localhost:11434")
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

    # Chapter/Question navigator (only if PDF analyzed)
    if st.session_state.current_pdf and st.session_state.pdf_data.get(st.session_state.current_pdf, {}).get("structured"):
        st.divider()
        st.header("📖 Navigator")

        data = st.session_state.pdf_data[st.session_state.current_pdf]
        questions = data["structured"].get("questions", [])

        if questions:
            # Group by chapter
            chapters = {}
            for q in questions:
                ch = q["chapter"]
                if ch not in chapters:
                    chapters[ch] = []
                chapters[ch].append(q)

            for ch in sorted(chapters.keys()):
                with st.expander(f"📚 Hoofdstuk {ch}"):
                    for q in chapters[ch]:
                        key = f"{q['chapter']}-{q['num']}"
                        if st.button(f"Vraag {q['num']}: {q['title']}", key=f"nav_{key}", use_container_width=True):
                            st.session_state.page = "navigator"
                            st.session_state.nav_chapter = ch
                            st.session_state.nav_question = q['num']
                            st.rerun()

# ===================== LLM HELPER =====================
def call_llm(prompt: str, temperature: float = 0.7) -> str:
    provider = st.session_state.config["provider"]
    if provider == "none":
        return "Geen AI ingeschakeld. Bewerk handmatig."
    
    try:
        if provider == "xai":
            # xAI is OpenAI-compatible
            from openai import OpenAI
            client = OpenAI(
                api_key=st.session_state.config["xai_key"],
                base_url="https://api.x.ai/v1"
            )
            response = client.chat.completions.create(
                model=st.session_state.config["xai_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=4000
            )
            return response.choices[0].message.content.strip()
            
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=st.session_state.config["openai_key"])
            response = client.chat.completions.create(
                model=st.session_state.config["openai_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=4000
            )
            return response.choices[0].message.content.strip()
            
        elif provider == "ollama":
            import ollama
            response = ollama.chat(
                model=st.session_state.config["ollama_model"],
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"].strip()
            
    except Exception as e:
        st.error(f"LLM fout: {e}")
        return f"LLM fout: {str(e)}"
    return ""

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

    # Step 2: Send preprocessed text to LLM
    prompt = f"""Je bent een perfecte quiz-assistent. Analyseer de volgende Nederlandse quiz-PDF tekst en geef ALLEEN een valide JSON terug (geen extra tekst).

De tekst is voorverwerkt met gedetecteerde structuur markers:
- [CONTAINS_TIME_INFO] = regel bevat tijdinformatie
- [CONTAINS_TASK_INFO] = regel bevat actie/doe-opdracht
- [EXTRACTION_SUMMARY] = samenvatting van gedetecteerde structuur

Tekst:
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
- De [EXTRACTION_SUMMARY] toont hoeveel vragen gedetecteerd zijn
- "page_start" en "page_end" zijn paginanummers waar de vraag zich bevindt (bijv 3-4)
- Dit wordt gebruikt om afbeeldingen aan de juiste vragen te koppelen
"""

    with st.spinner(f"AI analyseert {filename}..."):
        result = call_llm(prompt, temperature=0.3)
        try:
            # Probeer JSON te parsen (soms zit er markdown omheen)
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            structured = json.loads(result)
            data["structured"] = structured
            timeline_count = len(structured.get('timeline', []))
            q_count = len(structured.get('questions', []))
            st.success(f"✅ {filename} geanalyseerd! ⏰ Tijdlijn: {timeline_count} | Vragen: {q_count}")

            # Show preprocessing info as debug hint
            with st.expander("📊 Preprocessing debug info"):
                st.json(data["preprocessing_info"])
            # Save analysis results to disk
            save_pdf_analysis(filename, data["raw_text"], data["images"], structured, data.get("preprocessing_info", {}))

        except Exception as e:
            st.error(f"JSON parse fout: {e}")
            st.text_area("Raw LLM output (voor debug)", result, height=300)

            # Show what preprocessing detected
            with st.expander("📊 Preprocessing info"):
                st.json(data["preprocessing_info"])

# ===================== PAGE ROUTING =====================
if st.session_state.page == "home":
    # HOME PAGE - PDF Upload and Management
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
                        with pdfplumber.open(file) as pdf:
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
                    st.button("✅ Klaar", disabled=True, key=f"done_{fname}")
            with col3:
                if st.button("🗑", key=f"delete_{fname}"):
                    del st.session_state.pdf_data[fname]
                    delete_pdf_data(fname)  # Also delete persisted data
                    if st.session_state.current_pdf == fname:
                        st.session_state.current_pdf = None
                    st.rerun()

            # Debug: Show extraction details
            with st.expander(f"🔍 Debug: {fname}"):
                st.write(f"**Karakters geëxtraheerd:** {len(data['raw_text'])}")
                st.write(f"**Afbeeldingen gevonden:** {len(data['images'])}")

                # Show raw extracted text
                with st.expander("Raw text (first 2000 chars)", expanded=False):
                    st.text_area("Extracted text preview", value=data["raw_text"][:2000], height=300, disabled=True, key=f"raw_{fname}")

                # Show preprocessing detection
                if "preprocessing_info" in data:
                    st.write("**Preprocessing detecties:**")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Vragen gedetecteerd", data["preprocessing_info"]["questions_detected"])
                    with col_b:
                        st.metric("Timeline secties", data["preprocessing_info"]["timeline_sections"])
                    with col_c:
                        st.metric("Task secties", data["preprocessing_info"]["task_sections"])

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

                # Question header
                st.header(f"Hoofdstuk {selected_q['chapter']} — {selected_q['title']}")
                st.markdown(selected_q["full_text"])

                st.divider()

                # Images for this question (matched by page range)
                page_start = selected_q.get("page_start", selected_q.get("page", 1))
                page_end = selected_q.get("page_end", selected_q.get("page", 1))

                matched_images = [img for img in data["images"] if page_start <= img["page"] <= page_end]

                if matched_images:
                    st.subheader("📸 Afbeeldingen")
                    cols = st.columns(3)
                    for i, img in enumerate(matched_images):
                        with cols[i % 3]:
                            st.image(f"data:image/png;base64,{img['base64']}", use_container_width=True, caption=f"Pagina {img['page']}")

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
                            st.success("✅ AI suggestie gekopieerd naar antwoord!")
                            st.rerun()

                # Custom answer input
                new_answer = st.text_area("Jouw Antwoord", value=current_answer, height=200, key=f"answer_{key}")

                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    extra_input = st.text_input("Extra context voor AI (optioneel)", placeholder="Bijv. teamnummer, extra info")
                with col2:
                    if st.button("🤖 Genereer AI Suggestie", type="primary"):
                        prompt = f"""Geef een volledig, correct en creatief antwoord/suggestie voor deze quizvraag (in natuurlijk Nederlands):

Vraag:
{selected_q['full_text']}

Extra context van gebruiker:
{extra_input or 'geen'}"""
                        suggestion = call_llm(prompt)
                        save_ai_suggestion(st.session_state.current_pdf, selected_q["chapter"], key, suggestion)
                        st.rerun()
                with col3:
                    if st.button("💾 Opslaan"):
                        save_answer(st.session_state.current_pdf, selected_q["chapter"], key, new_answer)
                        st.success("✅ Antwoord opgeslagen!")

# ===================== FOOTER =====================
st.caption("© 2026 Quiz Navigator • Hans Hoeijmakers")
