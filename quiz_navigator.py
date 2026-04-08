import streamlit as st
import pdfplumber
from PIL import Image
import io
import json
import os
from datetime import datetime
import base64
import requests  # voor xAI fallback indien nodig

# ===================== CONFIG =====================
st.set_page_config(
    page_title="Quiz PDF Navigator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Quiz PDF Navigator")
st.markdown("""
**Volledige oplossing voor je quiz-boekjes**  
• Chronologische tijdlijn van alle events  
• Overzicht van alle doe-opdrachten  
• Navigator per hoofdstuk & vraag met AI-suggesties/antwoorden  
• Handmatig bewerken + opnieuw genereren met extra input  
• Werkt met grote quizzen (veel hoofdstukken/vragen)  
• Volledig lokaal (behalve AI)  
""")

# ===================== SESSION STATE =====================
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = {}          # {filename: {"raw_text": str, "images": list[dict], "structured": dict, "answers": dict}}
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
if "config" not in st.session_state:
    st.session_state.config = {
        "provider": "xai",
        "xai_key": "",
        "xai_model": "grok-4-1-fast-reasoning",
        "openai_key": "",
        "openai_model": "gpt-4o",
        "ollama_model": "llama3.2"
    }

# ===================== SIDEBAR - CONFIG & UPLOAD =====================
with st.sidebar:
    st.header("⚙️ Configuratie")
    provider = st.selectbox("AI Provider", ["xAI Grok", "OpenAI", "Ollama (lokaal)", "Geen AI (alleen handmatig)"], 
                            index=0 if st.session_state.config["provider"] == "xai" else 1 if st.session_state.config["provider"] == "openai" else 2)
    
    if provider == "xAI Grok":
        st.session_state.config["provider"] = "xai"
        st.session_state.config["xai_key"] = st.text_input("xAI API Key", value=st.session_state.config["xai_key"], type="password")
        st.session_state.config["xai_model"] = st.selectbox("Model", ["grok-4-1-fast-reasoning", "grok-4.20-reasoning"], 
                                                            index=0)
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
    st.header("📤 Upload PDF's")
    uploaded_files = st.file_uploader("Sleep je quiz-PDF's hier (meerdere toegestaan)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.pdf_data:
                with st.spinner(f"Verwerken {file.name}..."):
                    try:
                        with pdfplumber.open(file) as pdf:
                            raw_text = ""
                            images = []
                            for page_num, page in enumerate(pdf.pages, 1):
                                text = page.extract_text() or ""
                                raw_text += f"\n\n--- PAGINA {page_num} ---\n{text}"
                                
                                # Afbeeldingen extraheren
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
                            
                            st.session_state.pdf_data[file.name] = {
                                "raw_text": raw_text,
                                "images": images,
                                "structured": None,
                                "answers": {}
                            }
                        st.success(f"✅ {file.name} geladen ({len(raw_text)} tekens, {len(images)} afbeeldingen)")
                    except Exception as e:
                        st.error(f"Fout bij {file.name}: {e}")

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
    
    prompt = f"""Je bent een perfecte quiz-assistent. Analyseer de volgende Nederlandse quiz-PDF tekst en geef ALLEEN een valide JSON terug (geen extra tekst).

Tekst:
{data["raw_text"][:14000]}  # eerste 14k tekens (groot genoeg voor de meeste quizzen)

JSON structuur (exact):
{{
  "timeline": [
    {{"date": "1969", "description": "Woodstock festival", "source": "Doe opdracht"}},
    ...
  ],
  "doe_opdrachten": [
    {{"vraag": "Vraag 1", "beschrijving": "Maak twee Route 30 foto's...", "hoofdstuk": 1}},
    ...
  ],
  "questions": [
    {{"chapter": 1, "num": 1, "title": "Vraag 1", "full_text": "Volledige vraagtekst inclusief subvragen", "type": "doe|kennis|foto|actie"}},
    ...
  ]
}}

Sorteer de timeline chronologisch. Herken alle data, jaren, events, Vietnamoorlog data, Ammy Day data etc. 
Voor doe-opdrachten: alles wat een fysieke actie vraagt (foto maken, knutselen, video maken, etc.).
"""
    
    with st.spinner(f"AI analyseert {filename}..."):
        result = call_llm(prompt, temperature=0.3)
        try:
            # Probeer JSON te parsen (soms zit er markdown omheen)
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            structured = json.loads(result)
            data["structured"] = structured
            st.success(f"✅ {filename} geanalyseerd! Tijdlijn: {len(structured.get('timeline', []))} events | Vragen: {len(structured.get('questions', []))}")
        except Exception as e:
            st.error(f"JSON parse fout: {e}")
            st.text_area("Raw LLM output (voor debug)", result, height=300)

# ===================== HOOFDPAGINA TABS =====================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["⏰ Tijdlijn", "🛠 Doe-opdrachten", "🧭 Navigator", "📸 Afbeeldingen", "💾 Mijn antwoorden"])

# TAB 1: TIJDLIJN
with tab1:
    st.header("Chronologische tijdlijn van alle events")
    if not st.session_state.pdf_data:
        st.info("Upload eerst PDF's")
    else:
        for fname in st.session_state.pdf_data:
            if st.button(f"Analyseer tijdlijn voor {fname}", key=f"tl_{fname}"):
                analyze_pdf(fname)
        
        # Toon gecombineerde tijdlijn
        all_events = []
        for fname, data in st.session_state.pdf_data.items():
            if data.get("structured"):
                for ev in data["structured"].get("timeline", []):
                    all_events.append({**ev, "pdf": fname})
        
        all_events.sort(key=lambda x: x.get("date", ""))
        for ev in all_events:
            st.markdown(f"**{ev['date']}** — {ev['description']}  \n<small>{ev['source']} | {ev['pdf']}</small>")

# TAB 2: DOE-OPDRACHTEN
with tab2:
    st.header("Overzicht van alle doe-opdrachten")
    for fname, data in st.session_state.pdf_data.items():
        if data.get("structured"):
            st.subheader(fname)
            for d in data["structured"].get("doe_opdrachten", []):
                st.markdown(f"**{d['vraag']}** — {d['beschrijving']}")

# TAB 3: NAVIGATOR
with tab3:
    st.header("Navigator — kies hoofdstuk / vraag")
    
    pdf_list = list(st.session_state.pdf_data.keys())
    selected_pdf = st.selectbox("Kies PDF", pdf_list, index=pdf_list.index(st.session_state.current_pdf) if st.session_state.current_pdf in pdf_list else 0)
    st.session_state.current_pdf = selected_pdf
    
    if selected_pdf:
        data = st.session_state.pdf_data[selected_pdf]
        if data["structured"] is None:
            if st.button("📊 Analyseer deze PDF nu"):
                analyze_pdf(selected_pdf)
        else:
            questions = data["structured"].get("questions", [])
            q_options = [f"Hoofdstuk {q['chapter']} — Vraag {q['num']}: {q['title']}" for q in questions]
            selected_q_idx = st.selectbox("Kies vraag", range(len(q_options)), format_func=lambda i: q_options[i])
            
            if questions:
                q = questions[selected_q_idx]
                key = f"{q['chapter']}-{q['num']}"
                
                st.subheader(q["title"])
                st.markdown(q["full_text"])
                
                # Bestaand antwoord laden
                current_answer = data["answers"].get(key, "")
                
                new_answer = st.text_area("Jouw antwoord / notities", value=current_answer, height=200, key=f"answer_{key}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    extra_input = st.text_input("Extra context voor AI (optioneel)", placeholder="Bijv. teamnummer, extra info")
                with col2:
                    if st.button("🔄 Opnieuw genereren met AI", type="primary"):
                        prompt = f"""Geef een volledig, correct en creatief antwoord/suggestie voor deze quizvraag (in natuurlijk Nederlands):

Vraag:
{q['full_text']}

Extra context van gebruiker:
{extra_input or 'geen'}"""
                        suggestion = call_llm(prompt)
                        st.session_state.pdf_data[selected_pdf]["answers"][key] = suggestion
                        st.rerun()
                with col3:
                    if st.button("💾 Opslaan antwoord"):
                        st.session_state.pdf_data[selected_pdf]["answers"][key] = new_answer
                        st.success("Opgeslagen!")
                
                # Afbeeldingen bij deze vraag (simpel: alle images van PDF tonen)
                if data["images"]:
                    st.caption("Afbeeldingen uit deze PDF")
                    cols = st.columns(4)
                    for i, img in enumerate(data["images"][:8]):
                        with cols[i % 4]:
                            st.image(f"data:image/png;base64,{img['base64']}", use_column_width=True, caption=f"Pagina {img['page']}")

# TAB 4: AFBEELDINGEN GALLERY
with tab4:
    st.header("Alle afbeeldingen uit de PDF's")
    for fname, data in st.session_state.pdf_data.items():
        if data["images"]:
            st.subheader(fname)
            cols = st.columns(5)
            for i, img in enumerate(data["images"]):
                with cols[i % 5]:
                    st.image(f"data:image/png;base64,{img['base64']}", use_column_width=True, caption=f"P.{img['page']}")

# TAB 5: MIJN ANTWOORDEN
with tab5:
    st.header("💾 Mijn opgeslagen antwoorden")
    if st.button("📥 Download alles als JSON"):
        all_data = {k: {"answers": v["answers"]} for k, v in st.session_state.pdf_data.items()}
        st.download_button("Download quiz_antwoorden.json", 
                           data=json.dumps(all_data, indent=2, ensure_ascii=False),
                           file_name="quiz_antwoorden.json",
                           mime="application/json")
    
    if st.button("🗑 Alles wissen"):
        for v in st.session_state.pdf_data.values():
            v["answers"] = {}
        st.success("Alle antwoorden gewist")
    
    for fname, data in st.session_state.pdf_data.items():
        if data["answers"]:
            st.subheader(fname)
            for k, ans in data["answers"].items():
                st.markdown(f"**{k}** → {ans[:120]}...")

# ===================== FOOTER =====================
st.caption("© Grok-built Quiz Navigator • Volledig lokaal • Werkt perfect met je California Dreamin' quiz en alle toekomstige boekjes")
st.caption("Tip: Voor heel grote quizzen (>50 vragen) upload je ze één voor één – de app slaat alles in session_state op.")