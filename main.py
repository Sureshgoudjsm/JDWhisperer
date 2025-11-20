# app.py  -- JD Whisperer Streamlit app (full)
import os
import io
import json
import re
import csv
import time
import shutil
import datetime
from dotenv import load_dotenv

import pandas as pd
import streamlit as st

# AI lib (Gemini)
import google.generativeai as genai

# PDF/DOCX parsing
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

# Optional Google Sheets support (only if you enable it and add service account)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSHEETS_LIBS = True
except Exception:
    HAS_GSHEETS_LIBS = False

# ---------------------------
# Basic page config + load env
# ---------------------------
st.set_page_config(page_title="JD Whisperer", layout="wide", page_icon="🤫")
load_dotenv()

# ---------------------------
# LOGO paths (local preferred)
# Developer note: using the uploaded file path from your history:
# '/mnt/data/jd_whisperer_logo.png'
# Fallback is the assets/logo.png you created inside the repo.
# ---------------------------
PREFERRED_LOGO = "/mnt/data/jd_whisperer_logo.png"
ASSETS_LOGO = "assets/logo.png"  # your created folder
if os.path.exists(PREFERRED_LOGO):
    LOGO_PATH = PREFERRED_LOGO
elif os.path.exists(ASSETS_LOGO):
    LOGO_PATH = ASSETS_LOGO
else:
    LOGO_PATH = None  # no logo available

# ---------------------------
# AI configuration
# ---------------------------
API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)
if not API_KEY:
    st.warning("Warning: GOOGLE_API_KEY not found in env or Streamlit secrets. AI will not work until set.")
else:
    genai.configure(api_key=API_KEY)

@st.cache_resource
def _get_model():
    return genai.GenerativeModel("gemini-2.5-flash")

# ---------------------------
# Google Sheets helpers (optional)
# ---------------------------
FIELD_ORDER = [
    "date_contacted","hr_name","phone_number","email_id","role_position",
    "recruiter_company","client_company","location","job_type","mode_of_contact",
    "interview_mode","interview_scheduled_date","round_1_details","round_2_details",
    "ctc_offered_expected","status","next_follow_up_date","review_notes",
    "extracted_keywords","match_score","skill_gap_analysis","prep_hint"
]

def _get_gsheets_creds_and_id():
    try:
        sheet_id = st.secrets.get("GOOGLE_SHEET_ID", None)
        sa_info = st.secrets.get("GOOGLE_SERVICE_ACCOUNT", None)
        if not sheet_id or not sa_info:
            return None, None
        creds_dict = json.loads(sa_info) if isinstance(sa_info, str) else dict(sa_info)
        return creds_dict, sheet_id
    except Exception:
        return None, None

@st.cache_resource
def _get_gsheets_worksheet():
    if not HAS_GSHEETS_LIBS:
        return None
    creds_dict, sheet_id = _get_gsheets_creds_and_id()
    if not creds_dict or not sheet_id:
        return None
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    sh = client.open_by_key(sheet_id)
    ws = sh.sheet1
    try:
        if not ws.get_all_values():
            ws.append_row(["timestamp_utc"] + FIELD_ORDER)
    except Exception:
        pass
    return ws

def save_history_to_gsheets(result: dict):
    try:
        ws = _get_gsheets_worksheet()
        if ws is None:
            return
        row = [datetime.datetime.utcnow().isoformat()] + [result.get(k, "") for k in FIELD_ORDER]
        ws.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        st.warning(f"Could not save to Google Sheets: {e}")

# ---------------------------
# Session state initialization (clear comments to find them)
# - app_state: main flow + inputs + result
# - history: in-memory history for session
# - mode: 'Analyze' or 'History'
# - saved_resume_text: persistent uploaded resume in session (optional)
# ---------------------------
def reset_app_state():
    st.session_state['app_state'] = {
        "current_view": "start",
        "profile_data": "",
        "job_description": "",
        "skills_data": "",
        "analysis_result": None
    }

if 'app_state' not in st.session_state:
    reset_app_state()

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Analyze'

# saved resume text (persist if user wants)
if 'saved_resume_text' not in st.session_state:
    st.session_state['saved_resume_text'] = ""

# ---------------------------
# Prompt and helper functions
# ---------------------------
EXTRACTION_PROMPT_BASE = """
You are an expert data extraction assistant for job seekers. Analyze the combined input:
- Applicant Profile / Resume excerpt
- Job Description / Recruiter email / Call notes
Return a single, valid JSON object (no extra text) with the keys:
{keys}

Rules:
- If a value can't be found, return "Not specified".
- interview_scheduled_date and next_follow_up_date: YYYY-MM-DD if present.
- match_score: numeric percentage 0-100.
- skill_gap_analysis: VERY SHORT (2-3 sentences max).
- prep_hint: 1-2 short sentences.

Input (do not ignore any section):
***
{body}
***
Return only the JSON object.
""".strip()

EXTRACTION_KEYS = FIELD_ORDER + ["match_score","skill_gap_analysis","prep_hint"]

def _build_prompt(applicant_text: str, job_text: str, extra_notes: str) -> str:
    # Add a tiny timestamp token to avoid accidental reuse/caching
    unique_token = datetime.datetime.utcnow().isoformat()
    body = (
        f"--- APPLICANT (Resume/Profile) ---\n{applicant_text}\n\n"
        f"--- JOB DETAILS / RECRUITER ---\n{job_text}\n\n"
        f"--- EXTRA NOTES ---\n{extra_notes}\n\n"
        f"--- TOKEN ---\n{unique_token}"
    )
    return EXTRACTION_PROMPT_BASE.format(keys=", ".join(EXTRACTION_KEYS), body=body)

def safe_json_from_response(text: str) -> dict:
    cleaned = (text or "").strip().replace("```json", "").replace("```", "")
    # grab first {...}
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if not m:
        # fallback: try to parse raw cleaned
        try:
            return json.loads(cleaned)
        except Exception:
            raise ValueError("AI output did not contain valid JSON.")
    return json.loads(m.group(0))

def keep_first_sentences(text: str, max_sentences: int = 3) -> str:
    if not isinstance(text, str):
        return text
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s for s in sents if s]
    return " ".join(sents[:max_sentences])

def process_recruiter_text(text_to_process: str) -> dict:
    """
    Calls the Gemini model. Adds safety around parsing.
    """
    if not API_KEY:
        return {"error": "AI key not configured."}

    model = _get_model()
    try:
        resp = model.generate_content(text_to_process)
        parsed = safe_json_from_response(resp.text)
        # keep brevity for two keys
        for key in ("skill_gap_analysis","prep_hint"):
            if key in parsed and isinstance(parsed[key], str):
                parsed[key] = keep_first_sentences(parsed[key], max_sentences=3)
        return parsed
    except Exception as e:
        return {"error": f"AI call or parsing error: {e}"}

# ---------------------------
# File parsing helpers: PDF / DOCX
# ---------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    if pdfplumber is None:
        return ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(pages)
    except Exception:
        return ""

def extract_text_from_docx(file_bytes: bytes) -> str:
    if docx is None:
        return ""
    try:
        tmp_path = "/tmp/tmp_docx.docx"
        with open(tmp_path, "wb") as fh:
            fh.write(file_bytes)
        doc = docx.Document(tmp_path)
        paragraphs = [p.text for p in doc.paragraphs]
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return "\n".join(paragraphs)
    except Exception:
        return ""

# ---------------------------
# Dataframe sanitizer for Streamlit (avoid pyarrow errors)
# ---------------------------
def sanitize_df_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].map(lambda x: isinstance(x, (dict,list,set,tuple))).any():
            df[col] = df[col].map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict,list,set,tuple)) else x)
    return df

# ---------------------------
# CSS / Brand styling (kept compact)
# ---------------------------
def load_brand_css():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Manrope:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
    :root{--bg:#0B1E37;--text:#F5F9FF;--muted:#9EACBE;--a1:#4C8CFF;--a2:#6A5CFF;--a3:#00D4D0}
    body{background:var(--bg);color:var(--text);font-family:Inter,Manrope,sans-serif}
    .block-container{padding:2rem !important}
    .whisper-divider{height:6px;border-radius:6px;background:linear-gradient(90deg,var(--a1),var(--a2),var(--a3));margin:16px 0}
    .brand-btn .stButton>button{background:linear-gradient(90deg,var(--a1),var(--a2))!important;color:white!important;border-radius:8px!important}
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# UI rendering functions
# ---------------------------

def try_show_logo(width=72):
    if LOGO_PATH:
        try:
            st.image(LOGO_PATH, width=width)
            return
        except Exception:
            pass
    # else show textual brand
    st.markdown("<h3 style='margin:0;color:var(--text)'>JD Whisperer</h3>", unsafe_allow_html=True)

def draw_sidebar():
    with st.sidebar:
        # show logo + small tagline
        st.markdown("<div style='display:flex;align-items:center;gap:12px'>", unsafe_allow_html=True)
        try_show_logo(width=48)
        st.markdown("<div><strong>JD Whisperer</strong><br><small style='color:var(--muted)'>Decode any job description</small></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Mode")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Analyze", use_container_width=True, key="mode_analyze"):
                st.session_state['mode'] = "Analyze"
                st.rerun()
        with col2:
            if st.button("History", use_container_width=True, key="mode_history"):
                st.session_state['mode'] = "History"
                st.rerun()
        st.caption(f"Current mode: **{st.session_state['mode']}**")

        st.markdown("---")
        st.markdown("**Resume Upload (optional)**")
        st.markdown("Upload a PDF or DOCX resume to auto-fill your profile. Or keep typing — both work.")
        st.markdown("If you upload once and choose 'Save resume for session', it will be reused until you clear it.")

        # Option: show saved resume present
        if st.session_state.get('saved_resume_text'):
            st.markdown("**Saved resume available for this session.**")
            if st.button("Clear saved resume"):
                st.session_state['saved_resume_text'] = ""
                st.success("Saved resume cleared.")

        st.markdown("---")
        st.markdown("**Persistence**")
        if HAS_GSHEETS_LIBS:
            st.success("Google Sheets support available (configure secrets).")
        else:
            st.info("Google Sheets disabled (gspread not installed).")

def draw_start_view():
    st.markdown("<div style='display:flex;align-items:center;gap:12px'>", unsafe_allow_html=True)
    try_show_logo(width=96)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<h1 style='margin:8px 0;font-family:Manrope'>JD Whisperer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--muted)'>Paste a Job Description and your Profile, or upload a resume to get a fast match & short prep hints.</p>", unsafe_allow_html=True)
    _, ccol, _ = st.columns([1,2,1])
    with ccol:
        if st.button("🚀 Start Mapping", key="start_map"):
            st.session_state['app_state']['current_view'] = 'map'
            st.rerun()

def draw_map_view():
    st.markdown("<h2>Build Your Career Mind Map</h2>", unsafe_allow_html=True)

    # Left column: inputs
    col1, col2 = st.columns([2,1], gap="large")
    with col1:
        st.markdown("### Your Profile / Resume (optional)")
        profile_text = st.text_area("Profile / Resume excerpt", value=st.session_state['app_state']['profile_data'], key="profile_input", height=180, label_visibility="visible")
        st.session_state['app_state']['profile_data'] = profile_text

        st.markdown("### Job Description / Recruiter Notes")
        job_text = st.text_area("Job Description / Recruiter details", value=st.session_state['app_state']['job_description'], key="jd_input", height=200)
        st.session_state['app_state']['job_description'] = job_text

        st.markdown("### Extra Notes (CTC, notice, call summary)")
        extra_text = st.text_area("Extra notes", value=st.session_state['app_state']['skills_data'], key="extra_input", height=120)
        st.session_state['app_state']['skills_data'] = extra_text

        st.markdown('<div class="whisper-divider"></div>', unsafe_allow_html=True)

        # Resume upload and options
        st.markdown("### Resume Upload")
        uploaded_file = st.file_uploader("Upload PDF or DOCX resume (optional)", type=['pdf','docx'])
        save_resume_for_session = st.checkbox("Save resume for this session (reuse for future analyses)", value=bool(st.session_state.get('saved_resume_text')))
        if uploaded_file is not None:
            fb = uploaded_file.read()
            extracted = ""
            if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
                if pdfplumber is None:
                    st.warning("pdfplumber not installed; resume extraction not available. Add pdfplumber to requirements.")
                else:
                    extracted = extract_text_from_pdf(fb)
            elif uploaded_file.name.lower().endswith(".docx"):
                if docx is None:
                    st.warning("python-docx not installed; docx extraction not available. Add python-docx to requirements.")
                else:
                    extracted = extract_text_from_docx(fb)
            if extracted:
                st.success("Resume text extracted (first 800 chars shown).")
                st.text_area("Resume preview", value=extracted[:800], height=120)
                # Give option to fill profile from resume
                if st.button("Use this resume as profile text"):
                    st.session_state['app_state']['profile_data'] = extracted
                    st.experimental_rerun()
                if save_resume_for_session:
                    st.session_state['saved_resume_text'] = extracted
            else:
                st.info("No text extracted from resume.")

        # Option to reuse saved resume
        if st.session_state.get('saved_resume_text') and not st.session_state['app_state']['profile_data'].strip():
            if st.button("Use saved resume as profile"):
                st.session_state['app_state']['profile_data'] = st.session_state['saved_resume_text']
                st.experimental_rerun()

        # Option to reuse last analysis inputs
        if st.session_state['history']:
            if st.checkbox("Pre-fill inputs from last analysis", value=False):
                last = st.session_state['history'][-1]
                # pick commonly useful fields
                st.session_state['app_state']['profile_data'] = last.get('review_notes', '') or last.get('extracted_keywords', '') or st.session_state['app_state']['profile_data']
                st.session_state['app_state']['job_description'] = last.get('role_position','') or st.session_state['app_state']['job_description']
                st.experimental_rerun()

    with col2:
        st.markdown("### Quick tips")
        st.markdown("- Paste full JD for best results\n- Keep resume/profile concise (1-2 paragraphs)\n- Use saved resume to avoid re-uploads")
        st.markdown("---")
        st.markdown("### Controls")
        is_ready = bool(st.session_state['app_state']['profile_data'].strip() and st.session_state['app_state']['job_description'].strip())
        if not is_ready:
            st.info("Fill Profile and Job Description to enable analysis.")
        generate_disabled = not is_ready or (not API_KEY)
        if st.button("✨ Generate Analysis", disabled=generate_disabled):
            combined_prompt = _build_prompt(
                st.session_state['app_state']['profile_data'],
                st.session_state['app_state']['job_description'],
                st.session_state['app_state']['skills_data']
            )
            with st.spinner("🧠 JD Whisperer analyzing..."):
                result = process_recruiter_text(combined_prompt)
                st.session_state['app_state']['analysis_result'] = result
                if "error" not in result:
                    st.session_state['history'].append(result)
                    # optional: save to sheets (silent)
                    try:
                        save_history_to_gsheets(result)
                    except Exception:
                        pass
                st.session_state['app_state']['current_view'] = 'results'
                st.experimental_rerun()

        # Small debug / force refresh option
        if st.button("🔁 Re-run last analysis (force new)"):
            last_inputs = (
                st.session_state['app_state']['profile_data'],
                st.session_state['app_state']['job_description'],
                st.session_state['app_state']['skills_data']
            )
            combined_prompt = _build_prompt(*last_inputs)
            with st.spinner("🧠 Forcing a fresh AI run..."):
                result = process_recruiter_text(combined_prompt)
                st.session_state['app_state']['analysis_result'] = result
                if "error" not in result:
                    st.session_state['history'].append(result)
                    try:
                        save_history_to_gsheets(result)
                    except Exception:
                        pass
                st.session_state['app_state']['current_view'] = 'results'
                st.experimental_rerun()

def draw_results_view():
    st.markdown("<h2>Job Match Analysis</h2>", unsafe_allow_html=True)
    result = st.session_state['app_state'].get('analysis_result')
    if not result:
        st.error("No analysis to show. Please generate one.")
        if st.button("⬅️ Go Back"):
            st.session_state['app_state']['current_view'] = 'map'
            st.experimental_rerun()
        return
    if "error" in result:
        st.error(result.get("error"))
        if st.button("⬅️ Go Back"):
            st.session_state['app_state']['current_view'] = 'map'
            st.experimental_rerun()
        return

    # Top: score & summary
    try:
        score_val = int(str(result.get('match_score','N/A')).replace('%','').strip())
        score_display = f"{score_val}%"
    except Exception:
        score_display = str(result.get('match_score','N/A'))

    colA, colB = st.columns([1,2], gap="large")
    with colA:
        st.metric("Overall Match Score", score_display)
        st.markdown(f"**Status:** {result.get('status','Not specified')}")
        st.markdown(f"**Next follow-up:** {result.get('next_follow_up_date','Not specified')}")
    with colB:
        st.subheader("Summary")
        st.markdown(f"- **Role:** {result.get('role_position','Not specified')}")
        st.markdown(f"- **Company:** {result.get('client_company','Not specified')}")
        st.markdown(f"- **Location:** {result.get('location','Not specified')}")
        st.markdown(f"- **Mode:** {result.get('mode_of_contact','Not specified')}")

    st.markdown('<div class="whisper-divider"></div>', unsafe_allow_html=True)

    st.subheader("🎯 Prep & Gap (short)")
    st.markdown(f"**Skill Gap:** {result.get('skill_gap_analysis','Not identified.')}")
    st.markdown(f"**Prep Hint:** {result.get('prep_hint','No hint available.')}")

    st.markdown('---')

    # Full extracted data (sanitized for DataFrame)
    st.subheader("📋 Full Extracted Data")
    df_display = pd.DataFrame([result]).T
    df_display.columns = ["Extracted Value"]
    df_display = sanitize_df_for_streamlit(df_display)
    st.dataframe(df_display, use_container_width=True)

    # Downloads
    st.subheader("💾 Downloads")
    csv_out = io.StringIO()
    writer = csv.DictWriter(csv_out, fieldnames=result.keys())
    writer.writeheader()
    writer.writerow(result)
    st.download_button("📄 Download CSV", csv_out.getvalue(), file_name="job_details.csv", mime="text/csv")

    ics = create_ics_file(result) if 'create_ics_file' in globals() else ""
    if ics:
        st.download_button("📅 Download .ics", ics, file_name="interview.ics", mime="text/calendar")

    # show session history
    if st.session_state.get('history'):
        with st.expander("View session history"):
            hist_df = pd.DataFrame(st.session_state['history'])
            hist_df = sanitize_df_for_streamlit(hist_df)
            st.dataframe(hist_df, use_container_width=True)

    # action buttons
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Edit Inputs"):
            st.session_state['app_state']['current_view'] = 'map'
            st.experimental_rerun()
    with c2:
        if st.button("🔄 New Analysis"):
            reset_app_state()
            st.experimental_rerun()

def draw_history_view():
    st.markdown("<h2>History Dashboard</h2>", unsafe_allow_html=True)
    df = None
    try:
        df = _get_gsheets_worksheet()
    except Exception:
        pass
    if df is None:
        # fallback to session history
        df = pd.DataFrame(st.session_state['history'])
    if df.empty:
        st.info("No history yet.")
        return

    # optional parsing of timestamps if present
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")

    # show filters + table
    st.dataframe(sanitize_df_for_streamlit(df), use_container_width=True)

# ---------------------------
# Small helper: create_ics_file (same as previous versions)
# ---------------------------
def create_ics_file(details: dict) -> str:
    date_str = details.get("interview_scheduled_date")
    if not date_str or date_str == "Not specified":
        return ""
    try:
        start = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(hour=10, minute=0)
        end = start + datetime.timedelta(hours=1)
        fmt = "%Y%m%dT%H%M%S"
        summary = f"Interview: {details.get('role_position','Job')} @ {details.get('client_company','Client')}"
        desc = f"Role: {details.get('role_position','N/A')}\\nCompany: {details.get('client_company','N/A')}\\nNotes: {details.get('review_notes','')}"
        ics = ("BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//JD Whisperer//EN\nBEGIN:VEVENT\n"
               f"UID:{datetime.datetime.now().strftime(fmt)}-{hash(summary)}\n"
               f"DTSTAMP:{datetime.datetime.now().strftime(fmt)}\n"
               f"DTSTART:{start.strftime(fmt)}\nDTEND:{end.strftime(fmt)}\n"
               f"SUMMARY:{summary}\nDESCRIPTION:{desc}\nEND:VEVENT\nEND:VCALENDAR")
        return ics
    except Exception:
        return ""

# ---------------------------
# Router & main
# ---------------------------
def main():
    load_brand_css()
    draw_sidebar()

    mode = st.session_state.get('mode','Analyze')
    if mode == "History":
        draw_history_view()
        return

    view = st.session_state['app_state'].get('current_view','start')
    if view == 'start':
        draw_start_view()
    elif view == 'map':
        draw_map_view()
    elif view == 'results':
        draw_results_view()
    else:
        reset_app_state()
        draw_start_view()

if __name__ == "__main__":
    main()
