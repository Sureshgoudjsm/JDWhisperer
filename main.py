# app.py
import os
import io
import json
import csv
import re
import shutil
import datetime
from dotenv import load_dotenv

import pandas as pd
import streamlit as st

# Optional libs used in your original app (may be missing in environment)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSHEETS_LIBS = True
except Exception:
    HAS_GSHEETS_LIBS = False

# ----------------------------------------
# PAGE / BRAND CONFIG
# ----------------------------------------
st.set_page_config(page_title="JD Whisperer", layout="wide", page_icon="🤫")

# ----------- Logo paths -----------
# Primary: local project asset (recommended)
LOCAL_ASSET_LOGO = os.path.join("assets", "logo.png")

# Secondary fallback: previously uploaded path on some hosts
FALLBACK_UPLOADED_LOGO = "/mnt/data/jd_whisperer_logo.png"

# ----------------------------------------
# ENV & AI CONFIG
# ----------------------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)

if not API_KEY and GENAI_AVAILABLE:
    st.warning("GOOGLE_API_KEY not found. AI calls will fail until you set the key in env or Streamlit secrets.")

if GENAI_AVAILABLE and API_KEY:
    genai.configure(api_key=API_KEY)

@st.cache_resource
def get_model():
    """Cached model accessor (if genai is installed)."""
    if not GENAI_AVAILABLE:
        return None
    return genai.GenerativeModel("gemini-2.5-flash")

# ----------------------------------------
# GOOGLE SHEETS (optional)
# ----------------------------------------
FIELD_ORDER = [
    "date_contacted","hr_name","phone_number","email_id","role_position",
    "recruiter_company","client_company","location","job_type","mode_of_contact",
    "interview_mode","interview_scheduled_date","round_1_details","round_2_details",
    "ctc_offered_expected","status","next_follow_up_date","review_notes",
    "extracted_keywords","match_score","skill_gap_analysis","prep_hint"
]

def _get_gsheets_creds_and_id():
    """
    Reads GOOGLE_SHEET_ID and GOOGLE_SERVICE_ACCOUNT from Streamlit secrets.
    Returns (creds_dict, sheet_id) or (None, None).
    """
    try:
        sheet_id = st.secrets.get("GOOGLE_SHEET_ID", None)
        sa_info = st.secrets.get("GOOGLE_SERVICE_ACCOUNT", None)
        if not sheet_id or not sa_info:
            return None, None
        if isinstance(sa_info, str):
            creds_dict = json.loads(sa_info)
        else:
            creds_dict = dict(sa_info)
        return creds_dict, sheet_id
    except Exception:
        return None, None

@st.cache_resource
def get_gsheets_worksheet():
    """Return a gspread worksheet object if configured, else None."""
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
    # ensure header row
    try:
        vals = ws.get_all_values()
        if not vals:
            headers = ["timestamp_utc"] + FIELD_ORDER
            ws.append_row(headers, value_input_option="USER_ENTERED")
    except Exception:
        pass
    return ws

def save_history_to_gsheets(result: dict):
    """Append a row to Google Sheets if configured (silently skip on error)."""
    try:
        ws = get_gsheets_worksheet()
        if ws is None:
            return
        ts = datetime.datetime.utcnow().isoformat()
        row = [ts] + [result.get(k, "") for k in FIELD_ORDER]
        ws.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        st.warning(f"Could not save to Google Sheets: {e}")

def load_history_dataframe() -> pd.DataFrame:
    """Load history from Google Sheets (preferred) or from session history fallback."""
    df = None
    try:
        ws = get_gsheets_worksheet()
    except Exception:
        ws = None
    if ws is not None:
        try:
            records = ws.get_all_records()
            if records:
                df = pd.DataFrame(records)
        except Exception as e:
            st.warning(f"Could not load history from Google Sheets: {e}")
    if (df is None or df.empty) and st.session_state.get("history"):
        df = pd.DataFrame(st.session_state.get("history"))
    if df is None:
        df = pd.DataFrame()
    return df

# ----------------------------------------
# SESSION STATE (clear comments / locations)
# ----------------------------------------
def reset_app_state():
    """
    Reset core app_state in session_state.
    - current_view: 'start' / 'map' / 'results'
    - profile_data, job_description, skills_data: user inputs
    - analysis_result: AI output (dict)
    """
    st.session_state.app_state = {
        "current_view": "start",
        "profile_data": "",
        "job_description": "",
        "skills_data": "",
        "analysis_result": None
    }

# initialize session keys with comments so you can find them quickly later
if "app_state" not in st.session_state:
    reset_app_state()   # <-- core app flow state

if "history" not in st.session_state:
    st.session_state.history = []   # <-- in-session run history (temporary)

if "mode" not in st.session_state:
    st.session_state.mode = "Analyze"  # <-- UI mode (Analyze / History)

# ----------------------------------------
# AI PROMPT + processing helpers
# ----------------------------------------
EXTRACTION_PROMPT = """
You are an expert data extraction assistant for job seekers. Your task is to analyze the provided texts:
1) Job Details (JD, email, call notes)
2) Applicant Skills (Resume/Summary)

You MUST return a single valid JSON object (no text outside JSON). Use "Not specified" when unknown.

Ensure:
- interview_scheduled_date: YYYY-MM-DD format if present
- skill_gap_analysis: VERY SHORT (2-3 sentences max)
- prep_hint: 1-2 short sentences

Return keys (all must be present):
""" + ", ".join(FIELD_ORDER + ["match_score","skill_gap_analysis","prep_hint"]) + """

Input:
***
{text_input}
***
"""

def safe_json_from_response(text: str) -> dict:
    """Extract JSON object from model response text robustly."""
    cleaned = (text or "").strip().replace("```json", "").replace("```", "")
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    # try naive load
    return json.loads(cleaned)

def keep_first_sentences(text: str, max_sentences: int = 3) -> str:
    """Keep only the first `max_sentences` sentences (used for prep/gap brevity)."""
    if not isinstance(text, str):
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]
    if not sentences:
        return text
    return " ".join(sentences[:max_sentences])

def process_recruiter_text(text_to_process: str) -> dict:
    """
    Call the model and return parsed JSON. If genai not available, returns a mocked example.
    """
    if not GENAI_AVAILABLE or not API_KEY:
        # Mock response for offline/dev environments
        return {
            "date_contacted": "Not specified",
            "hr_name": "Not specified",
            "phone_number": "Not specified",
            "email_id": "Not specified",
            "role_position": "Python Fullstack Developer",
            "recruiter_company": "Not specified",
            "client_company": "Not specified",
            "location": "Not specified",
            "job_type": "Not specified",
            "mode_of_contact": "Not specified",
            "interview_mode": "Not specified",
            "interview_scheduled_date": "Not specified",
            "round_1_details": "Not specified",
            "round_2_details": "Not specified",
            "ctc_offered_expected": "Not specified",
            "status": "Not specified",
            "next_follow_up_date": "Not specified",
            "review_notes": "Not specified",
            "extracted_keywords": "Python, AWS, SQL",
            "match_score": "85",
            "skill_gap_analysis": "Lacks advanced cloud infra examples; add Azure/AWS deployment samples.",
            "prep_hint": "Prepare concise examples of cloud infra and fullstack projects."
        }

    model = get_model()
    prompt_with_input = EXTRACTION_PROMPT.format(text_input=text_to_process)
    try:
        response = model.generate_content(prompt_with_input)
        parsed = safe_json_from_response(response.text)
        # ensure brevity on two fields
        for k in ("skill_gap_analysis", "prep_hint"):
            if k in parsed and isinstance(parsed[k], str):
                parsed[k] = keep_first_sentences(parsed[k], max_sentences=3)
        # ensure all keys exist
        for key in FIELD_ORDER + ["match_score", "skill_gap_analysis", "prep_hint"]:
            if key not in parsed:
                parsed[key] = "Not specified"
        return parsed
    except json.JSONDecodeError:
        return {"error": f"AI returned invalid JSON. Raw: {response.text if 'response' in locals() else 'N/A'}"}
    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------
# Utility helpers
# ----------------------------------------
def create_ics_file(details: dict) -> str:
    """Creates an .ics calendar string if interview_scheduled_date present (YYYY-MM-DD)."""
    date_str = details.get("interview_scheduled_date")
    if not date_str or date_str == "Not specified":
        return ""
    try:
        start_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(hour=10, minute=0)
        end_date = start_date + datetime.timedelta(hours=1)
        dt_format = "%Y%m%dT%H%M%S"
        summary = f"Interview: {details.get('role_position','Job')} @ {details.get('client_company','Client')}"
        description = f"Role: {details.get('role_position','N/A')}\\nCompany: {details.get('client_company','N/A')}\\nNotes: {details.get('review_notes','')}"
        ics = (
            "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//JD Whisperer//EN\nBEGIN:VEVENT\n"
            f"UID:{datetime.datetime.now().strftime(dt_format)}-{hash(summary)}\n"
            f"DTSTAMP:{datetime.datetime.now().strftime(dt_format)}\n"
            f"DTSTART:{start_date.strftime(dt_format)}\n"
            f"DTEND:{end_date.strftime(dt_format)}\n"
            f"SUMMARY:{summary}\nDESCRIPTION:{description}\nEND:VEVENT\nEND:VCALENDAR"
        )
        return ics
    except Exception:
        return ""

def sanitize_df_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """Convert complex column types to JSON strings so Streamlit/pyarrow can render them."""
    df = df.copy()
    for col in df.columns:
        if df[col].map(lambda x: isinstance(x, (dict, list, set, tuple))).any():
            df[col] = df[col].map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list, set, tuple)) else x)
    return df

# ----------------------------------------
# Styling & Logo rendering
# ----------------------------------------
def load_brand_css():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Manrope:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
    :root{
        --bg:#0B1E37; --surface:#122642; --muted:#9EACBE; --text:#F5F9FF;
        --accent-1:#4C8CFF; --accent-2:#6A5CFF; --accent-3:#00D4D0;
    }
    body{background:var(--bg); color:var(--text); font-family:Inter, sans-serif;}
    .block-container{ padding: 2rem !important; }
    .main-header h1{
        font-family:Manrope, sans-serif;
        font-weight:800;
        font-size:36px;
        background:linear-gradient(90deg,var(--accent-1),var(--accent-2),var(--accent-3));
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        margin:0;
    }
    .main-header p{ color:var(--muted); margin:4px 0 0 0; }
    .stButton > button{
        background: linear-gradient(90deg,var(--accent-1),var(--accent-2)) !important;
        color: white !important; border-radius:8px !important; font-weight:700 !important;
    }
    .whisper-divider{ height:6px; border-radius:8px; background: linear-gradient(90deg,var(--accent-1),var(--accent-2),var(--accent-3)); margin:16px 0; }
    .mind-map-card{ background: linear-gradient(180deg, rgba(22,38,60,0.55), rgba(17,30,45,0.35)); padding:16px; border-radius:12px; border:1px solid rgba(76,140,255,0.08); }
    </style>
    """, unsafe_allow_html=True)

def try_show_logo(width=72):
    """
    Try local asset first (assets/logo.png). If not found, try fallback uploaded path.
    This is the recommended method for bundling logos in your repo (assets folder).
    """
    if os.path.exists(LOCAL_ASSET_LOGO):
        try:
            st.image(LOCAL_ASSET_LOGO, width=width)
            return
        except Exception:
            pass
    if os.path.exists(FALLBACK_UPLOADED_LOGO):
        try:
            st.image(FALLBACK_UPLOADED_LOGO, width=width)
            return
        except Exception:
            pass
    # if nothing available, do nothing (no broken icons)
    return

# ----------------------------------------
# UI: Start / Map / Results / History
# ----------------------------------------
def draw_start_view():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    try_show_logo(width=72)
    st.markdown("<div style='display:inline-block;vertical-align:middle;margin-left:12px'><h1>JD Whisperer</h1><p>Decode job descriptions/Calls & chat summary into clear candidate insights — skills, gaps, match score & follow-ups.</p></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")
    _, center_col, _ = st.columns([1,1,1])
    with center_col:
        if st.button("🚀 Start Mapping"):
            st.session_state.app_state["current_view"] = "map"
            st.rerun()

def draw_map_view():
    st.markdown("<h2>Build Your Career Mind Map</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown('<div class="mind-map-card"><h3>Your Profile</h3><p>Paste resume summary or LinkedIn about.</p></div>', unsafe_allow_html=True)
        profile_text = st.text_area("Your Profile", height=200, key="profile_input", label_visibility="collapsed", placeholder="e.g., 5+ years in Python, AWS, SQL...")
        st.session_state.app_state['profile_data'] = profile_text

    with col2:
        st.markdown('<div class="mind-map-card"><h3>Job Description</h3><p>Paste the JD, recruiter email, or call notes.</p></div>', unsafe_allow_html=True)
        jd_text = st.text_area("Job Description", height=200, key="jd_input", label_visibility="collapsed", placeholder="Paste JD here...")
        st.session_state.app_state['job_description'] = jd_text

    with col3:
        st.markdown('<div class="mind-map-card"><h3>Extra Notes</h3><p>CTC, notice period, call summary.</p></div>', unsafe_allow_html=True)
        skills_text = st.text_area("Skill Assessment", height=200, key="skills_input", label_visibility="collapsed", placeholder="Additional notes...")
        st.session_state.app_state['skills_data'] = skills_text

    st.markdown('<div class="whisper-divider"></div>', unsafe_allow_html=True)

    is_ready = bool(st.session_state.app_state['profile_data'].strip() and st.session_state.app_state['job_description'].strip())
    if not is_ready:
        st.info("Please fill at least Your Profile and Job Description to run the analysis.")

    if st.button("✨ Generate Analysis", disabled=not is_ready):
        combined = (
            f"--- APPLICANT SKILLS ---\n{st.session_state.app_state['profile_data']}\n\n"
            f"--- JOB DETAILS ---\n{st.session_state.app_state['job_description']}\n\n"
            f"--- ADDITIONAL NOTES ---\n{st.session_state.app_state['skills_data']}"
        )
        with st.spinner("🧠 JD Whisperer is analyzing..."):
            result = process_recruiter_text(combined)
            st.session_state.app_state['analysis_result'] = result
            if "error" not in result:
                # append to session history and attempt persistence
                st.session_state.history.append(result)
                save_history_to_gsheets(result)
            st.session_state.app_state['current_view'] = 'results'
            st.rerun()

def draw_results_view():
    st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'>", unsafe_allow_html=True)
    left, right = st.columns([1,4])
    with left:
        try_show_logo(width=64)
    with right:
        st.markdown("<h2>Job Match & Tracker Analysis</h2><p style='color:#9EACBE;margin-top:0'>An at-a-glance analysis of your profile against the recruiter/job details.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    result = st.session_state.app_state.get('analysis_result')
    if not result:
        st.error("No analysis found. Please run an analysis.")
        if st.button("⬅️ Go Back"):
            st.session_state.app_state['current_view'] = 'map'
            st.rerun()
        return

    if "error" in result:
        st.error(result.get("error"))
        if st.button("⬅️ Go Back"):
            st.session_state.app_state['current_view'] = 'map'
            st.rerun()
        return

    raw_score = result.get('match_score', 'N/A')
    try:
        match_score_display = f"{int(str(raw_score).replace('%','').strip())}%"
    except Exception:
        match_score_display = str(raw_score)

    col1, col2 = st.columns([1,2], gap="large")
    with col1:
        st.metric("Overall Match Score", match_score_display)
        st.markdown(f"**Status:** {result.get('status','Not specified')}")
        st.markdown(f"**Next Follow-up:** {result.get('next_follow_up_date','Not specified')}")
    with col2:
        st.subheader("🧩 Summary")
        st.markdown(f"- **Role:** {result.get('role_position','Not specified')}")
        st.markdown(f"- **Company:** {result.get('client_company','Not specified')}")
        st.markdown(f"- **Location:** {result.get('location','Not specified')}")
        st.markdown(f"- **Mode:** {result.get('mode_of_contact','Not specified')}")

    st.markdown('<div class="whisper-divider"></div>', unsafe_allow_html=True)

    st.subheader("🎯 Prep & Gap")
    st.markdown(f"**Skill Gap (short):** {result.get('skill_gap_analysis','Not identified.')}")
    st.markdown(f"**Prep Hint:** {result.get('prep_hint','No hint available.')}")

    st.markdown("---")
    cL, cR = st.columns([2,1], gap="large")
    with cL:
        st.subheader("📋 Full Extracted Data")
        df_display = pd.DataFrame([result]).T
        df_display.columns = ["Extracted Value"]
        df_display = sanitize_df_for_streamlit(df_display)
        st.dataframe(df_display, use_container_width=True)
    with cR:
        st.subheader("💾 Downloads")
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)
        csv_data = buf.getvalue()
        st.download_button("📄 Download Job Tracker (.csv)", data=csv_data, file_name="job_details.csv", mime="text/csv", use_container_width=True)
        ics_data = create_ics_file(result)
        if ics_data:
            st.download_button("📅 Download Calendar Event (.ics)", data=ics_data, file_name="interview.ics", mime="text/calendar", use_container_width=True)
        else:
            st.caption("No valid interview date found to create a calendar event (YYYY-MM-DD expected).")

    st.markdown("---")
    if st.session_state.get('history'):
        with st.expander("🧾 View this session's history"):
            hist_df = pd.DataFrame(st.session_state['history'])
            hist_df = sanitize_df_for_streamlit(hist_df)
            st.dataframe(hist_df, use_container_width=True)

    back_col, new_col = st.columns(2)
    with back_col:
        if st.button("⬅️ Edit Inputs"):
            st.session_state.app_state['current_view'] = 'map'
            st.rerun()
    with new_col:
        if st.button("🔄 Start New Analysis"):
            reset_app_state()
            st.rerun()

def draw_history_view():
    st.markdown("<h2>📊 History Dashboard</h2>", unsafe_allow_html=True)
    df = load_history_dataframe()
    if df.empty:
        st.info("No history found yet.")
        return
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Records", len(df))
    if "role_position" in df.columns:
        c2.metric("Unique Roles", df["role_position"].fillna("").nunique())
    else:
        c2.metric("Unique Roles", "-")
    if "status" in df.columns:
        c3.metric("Statuses", df["status"].fillna("").nunique())
    else:
        c3.metric("Statuses", "-")
    st.markdown("---")
    st.subheader("🔍 Filters")
    left, right = st.columns([3,1])
    with left:
        text_query = st.text_input("Search role / recruiter / client", placeholder="e.g., Python, Accenture")
    with right:
        if "status" in df.columns:
            statuses = sorted([s for s in df["status"].dropna().unique() if s])
        else:
            statuses = []
        status_filter = st.multiselect("Status", options=statuses, default=statuses)
    date_range = None
    if "timestamp_utc" in df.columns and df["timestamp_utc"].notna().any():
        min_d = df["timestamp_utc"].min().date()
        max_d = df["timestamp_utc"].max().date()
        date_range = st.slider("Date range", min_value=min_d, max_value=max_d, value=(min_d, max_d))
    mask = pd.Series(True, index=df.index)
    if text_query:
        q = text_query.lower()
        cols_to_search = []
        for name in ["role_position","recruiter_company","client_company"]:
            if name in df.columns:
                cols_to_search.append(df[name].fillna("").str.lower())
        if cols_to_search:
            combined = cols_to_search[0].str.contains(q)
            for extra in cols_to_search[1:]:
                combined = combined | extra.str.contains(q)
            mask &= combined
    if status_filter and "status" in df.columns:
        mask &= df["status"].fillna("").isin(status_filter)
    if date_range and "timestamp_utc" in df.columns:
        s,e = date_range
        mask &= (df["timestamp_utc"].dt.date >= s) & (df["timestamp_utc"].dt.date <= e)
    df_filtered = df[mask].copy()
    df_filtered = sanitize_df_for_streamlit(df_filtered)
    st.markdown(f"Showing **{len(df_filtered)}** records")
    st.markdown("---")
    st.dataframe(df_filtered, use_container_width=True, hide_index=True)

# ----------------------------------------
# MAIN APP ROUTER
# ----------------------------------------
def main():
    load_brand_css()
    # Sidebar (logo + mode buttons)
    with st.sidebar:
        st.markdown("<div style='display:flex;gap:12px;align-items:center'>", unsafe_allow_html=True)
        try_show_logo(width=48)
        st.markdown("<div><strong>JD Whisperer</strong><br><small style='color:#9EACBE'>Decode any job description</small></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Mode")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Analyze", use_container_width=True, key="mode_analyze_btn"):
                st.session_state.mode = "Analyze"
                st.rerun()
        with col_b:
            if st.button("History", use_container_width=True, key="mode_history_btn"):
                st.session_state.mode = "History"
                st.rerun()
        st.caption(f"Current mode: **{st.session_state.mode}**")
        st.markdown("---")
        st.markdown("**Steps (Analyze mode):**\n1. Paste profile & JD\n2. Click Generate Analysis\n3. Download CSV / Calendar")
        creds_dict, sheet_id = _get_gsheets_creds_and_id()
        if HAS_GSHEETS_LIBS and creds_dict and sheet_id:
            st.success("Google Sheets logging: ON")
        else:
            st.info("Google Sheets logging: OFF (configure secrets)")

    # Mode routing
    if st.session_state.mode == "History":
        draw_history_view()
        return

    view = st.session_state.app_state.get("current_view", "start")
    if view == "start":
        draw_start_view()
    elif view == "map":
        draw_map_view()
    elif view == "results":
        draw_results_view()
    else:
        reset_app_state()
        draw_start_view()

if __name__ == "__main__":
    main()
