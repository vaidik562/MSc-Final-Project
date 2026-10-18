from pathlib import Path
from datetime import datetime
from io import BytesIO
import base64
import html

from PIL import Image, ImageFile, UnidentifiedImageError
import streamlit as st

from modules.database_module import init_db, save_scan, stats as database_stats
from modules.auth_module import (
    validate_registration,
    validate_password,
    register_user,
    login_user,
    reset_user_password,
)
from modules.care_advisory_module import PLANTS, plant_diseases, advisory
from modules.detection_module import predict, model_available, any_model_available
from modules.stress_detection_module import resolve_condition
from modules.validation_module import validate_analysis, leaf_likeness
from modules.severity_confidence_module import severity
from modules.smart_watering_module import smart_watering
from modules.weather_risk_module import weather_risk
from modules.scan_history_module import history_df
from modules.export_module import csv_bytes
from modules.report_generation_module import make_pdf

BASE = Path(__file__).resolve().parent
ASSETS = BASE / "assets"
LOGO_PATH = ASSETS / "greenguardian_logo.png"
AUTH_BG_PATH = ASSETS / "auth_background.webp"
APP_BG_PATH = ASSETS / "app_background.webp"

# Accept mildly truncated JPEGs that are still decodable. This is common with
# camera/stock-image exports and avoids rejecting otherwise valid leaf photos.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _data_uri(path: Path, mime: str) -> str:
    """Return a local image as a data URI so the packaged app stays portable."""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _open_uploaded_image(uploaded_file):
    """Decode a Streamlit UploadedFile/CameraFile without relying on its cursor position."""
    if uploaded_file is None:
        return None

    # Streamlit may hand the same in-memory file object to multiple widgets/reruns.
    # Reading from getvalue() is cursor-independent and is therefore much more reliable
    # than passing the file object directly to PIL.Image.open().
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    if not raw:
        raise UnidentifiedImageError("The selected file is empty.")

    last_error = None
    for attempt in range(2):
        try:
            buffer = BytesIO(raw)
            img = Image.open(buffer)
            img.load()
            return img.convert("RGB")
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                # Some JPEG exports contain harmless trailing data. Pillow with
                # LOAD_TRUNCATED_IMAGES=True usually handles them on a second decode.
                continue

    raise UnidentifiedImageError(str(last_error) if last_error else "Invalid image")


def inject_auth_theme():
    bg = _data_uri(AUTH_BG_PATH, "image/webp")
    st.markdown(
        f"""
        <style>
        :root {{
            --gg-green: #0ca85a;
            --gg-green-dark: #0a3c27;
            --gg-green-soft: #eef7f1;
            --gg-ink: #19352b;
            --gg-muted: #6f7975;
            --gg-border: #dde3e0;
        }}

        #MainMenu, footer, header[data-testid="stHeader"] {{ display: none !important; }}

        .stApp {{
            min-height: 100vh;
            background:
                linear-gradient(rgba(255,255,255,.18), rgba(255,255,255,.18)),
                url("{bg}") center center / cover fixed no-repeat;
        }}

        .block-container, div[data-testid="stMainBlockContainer"] {{
            max-width: 1280px !important;
            min-height: 100vh;
            padding: 2rem 1.2rem !important;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

        div[data-testid="stColumn"]:has(.auth-card-marker) {{
            max-width: 510px;
            margin: 0 auto;
            background: rgba(255,255,255,.94);
            border: 1px solid rgba(255,255,255,.98);
            border-radius: 30px;
            padding: 1rem 1.25rem .95rem;
            box-shadow: 0 18px 55px rgba(36, 72, 50, .16);
            backdrop-filter: blur(10px) saturate(118%);
            -webkit-backdrop-filter: blur(10px) saturate(118%);
        }}

        .auth-card-marker {{ height: 0; overflow: hidden; }}
        .auth-logo-wrap {{ text-align: center; margin: -.1rem 0 .25rem; }}
        .auth-logo {{
            width: 76px;
            height: 76px;
            object-fit: contain;
            filter: drop-shadow(0 10px 16px rgba(8, 92, 49, .12));
        }}
        .auth-card-head {{ text-align: center; margin-bottom: .9rem; }}
        .auth-card-head h1 {{
            margin: 0;
            color: var(--gg-green-dark);
            font-size: 1.58rem;
            font-weight: 860;
            letter-spacing: -.038em;
        }}
        .auth-card-head p {{
            margin: .28rem auto 0;
            color: var(--gg-muted);
            font-size: .88rem;
            line-height: 1.4;
            max-width: 460px;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            display: grid !important;
            grid-template-columns: 1fr 1fr;
            gap: 0 !important;
            background: #f1f5f3;
            border: 1px solid #edf1ef;
            border-radius: 999px;
            padding: .28rem;
            margin: .05rem 0 .78rem;
            box-shadow: inset 0 1px 2px rgba(0,0,0,.03);
        }}
        .stTabs [data-baseweb="tab"] {{
            width: 100% !important;
            justify-content: center !important;
            height: 40px;
            border-radius: 999px;
            color: #41534b;
            font-weight: 820;
            font-size: .94rem;
            background: transparent;
        }}
        .stTabs [aria-selected="true"] {{
            color: white !important;
            background: linear-gradient(135deg, #14994f, #1ebb61) !important;
            box-shadow: 0 7px 18px rgba(16, 153, 79, .24);
        }}
        .stTabs [data-baseweb="tab-highlight"] {{ display: none !important; }}

        /* AUTH INPUTS: style the real HTML input itself. This is intentionally
           independent of Streamlit/BaseWeb's internal border wrappers so fields
           stay rounded and consistent across Chrome/Edge and Streamlit updates. */
        .stTextInput {{
            margin-bottom: .28rem;
        }}
        .stTextInput label p {{
            color: #122d23;
            font-weight: 820;
            font-size: .82rem;
            margin-bottom: .05rem;
        }}

        /* Neutralise BaseWeb shells so they cannot create square/transparent
           layers around our rounded input. */
        .stTextInput div[data-baseweb="input"],
        .stTextInput div[data-baseweb="base-input"] {{
            position: relative !important;
            width: 100% !important;
            min-height: 44px !important;
            height: 44px !important;
            margin: 0 !important;
            padding: 0 !important;
            border: 0 !important;
            border-radius: 15px !important;
            background: transparent !important;
            box-shadow: none !important;
            overflow: visible !important;
        }}

        /* The visible field is always the input itself. */
        .stTextInput input {{
            box-sizing: border-box !important;
            display: block !important;
            width: 100% !important;
            min-width: 0 !important;
            min-height: 44px !important;
            height: 44px !important;
            margin: 0 !important;
            padding: 0 46px 0 46px !important;
            border: 1.5px solid #cfd9d4 !important;
            border-radius: 15px !important;
            outline: none !important;
            background-color: rgba(255,255,255,.98) !important;
            color: var(--gg-ink) !important;
            font-size: .88rem !important;
            line-height: 44px !important;
            box-shadow: 0 3px 10px rgba(27,74,50,.035) !important;
            background-repeat: no-repeat !important;
            background-position: 15px 50% !important;
            background-size: 19px 19px !important;
            transition: border-color .14s ease, box-shadow .14s ease !important;
        }}
        .stTextInput input:focus {{
            border-color: #24ae60 !important;
            box-shadow: 0 0 0 3px rgba(36,174,96,.11) !important;
        }}
        .stTextInput input::placeholder {{
            color: #a0aaa5 !important;
            opacity: 1 !important;
        }}

        /* Edge/Windows can add its own native password reveal icon.
           Streamlit already provides the eye button, so hide the browser
           reveal/clear controls to prevent duplicate eye icons. */
        .stTextInput input[type="password"]::-ms-reveal,
        .stTextInput input[type="password"]::-ms-clear {{
            display: none !important;
            width: 0 !important;
            height: 0 !important;
        }}
        .stTextInput input::-webkit-credentials-auto-fill-button,
        .stTextInput input::-webkit-contacts-auto-fill-button {{
            visibility: hidden !important;
            display: none !important;
            pointer-events: none !important;
            position: absolute !important;
            right: 0 !important;
        }}

        /* Put each left icon into the input background, not in a floating
           pseudo-element. This completely prevents icon/text overlap. */
        .stTextInput input[aria-label="Email address"] {{
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%235d6b65' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='5' width='18' height='14' rx='2'/%3E%3Cpath d='m3 7 9 6 9-6'/%3E%3C/svg%3E") !important;
        }}
        .stTextInput input[aria-label="Full name"] {{
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%235d6b65' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='8' r='4'/%3E%3Cpath d='M4.5 21a7.5 7.5 0 0 1 15 0'/%3E%3C/svg%3E") !important;
        }}
        .stTextInput input[aria-label="Password"],
        .stTextInput input[aria-label="Confirm password"],
        .stTextInput input[aria-label="New password"],
        .stTextInput input[aria-label="Confirm new password"] {{
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%235d6b65' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5' y='10' width='14' height='10' rx='2'/%3E%3Cpath d='M8 10V7a4 4 0 0 1 8 0v3'/%3E%3C/svg%3E") !important;
            padding-right: 48px !important;
        }}

        /* Streamlit's password visibility button. Keep it absolutely centred
           inside the 44px field. No hover bubble, no jumping after toggle. */
        .stTextInput div[data-baseweb="input"] button,
        .stTextInput div[data-baseweb="base-input"] button {{
            position: absolute !important;
            right: 8px !important;
            top: 50% !important;
            bottom: auto !important;
            transform: translateY(-50%) !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 30px !important;
            min-width: 30px !important;
            max-width: 30px !important;
            height: 30px !important;
            min-height: 30px !important;
            max-height: 30px !important;
            margin: 0 !important;
            padding: 0 !important;
            border: 0 !important;
            border-radius: 8px !important;
            background: transparent !important;
            color: #56665f !important;
            box-shadow: none !important;
            line-height: 1 !important;
            z-index: 20 !important;
        }}
        .stTextInput div[data-baseweb="input"] button:hover,
        .stTextInput div[data-baseweb="input"] button:focus,
        .stTextInput div[data-baseweb="base-input"] button:hover,
        .stTextInput div[data-baseweb="base-input"] button:focus {{
            background: transparent !important;
            color: #0b8c4c !important;
            box-shadow: none !important;
            outline: none !important;
        }}
        .stTextInput div[data-baseweb="input"] button svg,
        .stTextInput div[data-baseweb="base-input"] button svg {{
            display: block !important;
            width: 18px !important;
            height: 18px !important;
            margin: 0 !important;
        }}

        /* Chrome/Edge autofill must not wash out the field or hide the icon. */
        .stTextInput input:-webkit-autofill,
        .stTextInput input:-webkit-autofill:hover,
        .stTextInput input:-webkit-autofill:focus {{
            -webkit-text-fill-color: var(--gg-ink) !important;
            -webkit-box-shadow: 0 0 0 1000px #ffffff inset !important;
            caret-color: var(--gg-ink) !important;
            border: 1.5px solid #cfd9d4 !important;
            border-radius: 15px !important;
        }}

        .reset-card-kicker {{
            text-align: center;
            color: #0b8b4b;
            font-size: .72rem;
            font-weight: 850;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin: .05rem 0 .22rem;
        }}
        .reset-page-title {{
            text-align: center;
            color: var(--gg-green-dark);
            font-size: 1.42rem;
            font-weight: 860;
            margin: 0 0 .2rem;
        }}
        .reset-page-copy {{
            text-align: center;
            color: #6f7f78;
            font-size: .84rem;
            line-height: 1.42;
            margin: 0 auto .85rem;
            max-width: 400px;
        }}

        .forgot-password {{
            display: flex;
            justify-content: flex-end;
            margin: -.12rem .12rem .42rem 0;
        }}
        .forgot-password a {{
            color: #087743 !important;
            font-size: .9rem;
            font-weight: 780;
            text-decoration: none !important;
        }}
        .forgot-password a:hover {{
            color: #0ba358 !important;
            text-decoration: underline !important;
        }}
        .password-help {{
            margin: -.15rem 0 .85rem;
            padding: .8rem .95rem;
            border-radius: 14px;
            background: rgba(239, 249, 243, .94);
            border: 1px solid rgba(24, 151, 82, .13);
            color: #476157;
            font-size: .86rem;
            line-height: 1.45;
        }}
        .password-help strong {{ color: #0a6038; }}
        .reset-title {{
            text-align: center;
            color: var(--gg-green-dark);
            font-size: 1.1rem;
            font-weight: 850;
            margin: -.15rem 0 .18rem;
        }}
        .reset-copy {{
            text-align: center;
            color: #6f7f78;
            font-size: .82rem;
            line-height: 1.4;
            margin: 0 0 .8rem;
        }}
        .reset-back {{
            display: flex;
            justify-content: center;
            margin-top: .55rem;
        }}
        .reset-back a {{
            color: #087743 !important;
            font-size: .86rem;
            font-weight: 780;
            text-decoration: none !important;
        }}
        .reset-back a:hover {{ text-decoration: underline !important; }}

        .stFormSubmitButton > button {{
            width: 100%;
            min-height: 44px;
            margin-top: .18rem;
            border: 0 !important;
            border-radius: 999px !important;
            color: white !important;
            font-weight: 860 !important;
            font-size: .96rem !important;
            background: linear-gradient(135deg, #15964f, #29bf67) !important;
            box-shadow: 0 12px 28px rgba(14, 150, 78, .20);
            transition: transform .16s ease, box-shadow .16s ease;
        }}
        .stFormSubmitButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 16px 32px rgba(14, 150, 78, .26);
        }}

        .auth-secure {{
            text-align: center;
            margin-top: .7rem;
            color: #6e7f78;
            font-size: .82rem;
            padding-bottom: .1rem;
        }}
        .auth-secure span {{ color: #48695b; font-weight: 700; }}
        .register-note {{
            text-align: center;
            color: #73837b;
            font-size: .78rem;
            margin: -.05rem 0 .25rem;
        }}
        div[data-testid="stAlert"] {{ border-radius: 16px; }}

        @media (max-width: 760px) {{
            .block-container, div[data-testid="stMainBlockContainer"] {{
                padding: .9rem !important;
                justify-content: center;
            }}
            div[data-testid="stColumn"]:has(.auth-card-marker) {{
                max-width: 100%;
                padding: 1.15rem 1rem 1rem;
                border-radius: 26px;
            }}
            .auth-logo {{ width: 82px; height: 82px; }}
            .auth-card-head h1 {{ font-size: 1.65rem; }}
            .auth-card-head p {{ font-size: .82rem; }}
            .stTabs [data-baseweb="tab"] {{ height: 44px; font-size: .9rem; }}
            .stTextInput input {{ min-height: 44px !important; height: 44px !important; }}
            .stFormSubmitButton > button {{ min-height: 48px; font-size: .94rem !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_app_theme():
    bg = _data_uri(APP_BG_PATH, "image/webp")
    st.markdown(
        f"""
        <style>
        :root {{
            --gg-green:#0b7a43;
            --gg-green-dark:#075933;
            --gg-ink:#12372a;
            --gg-muted:#60786d;
            --gg-card:rgba(255,255,255,.86);
            --gg-border:rgba(16,113,65,.10);
        }}

        #MainMenu, footer {{ visibility:hidden; }}
        header[data-testid="stHeader"] {{ background:transparent; }}

        /* Hide Streamlit's automatic permalink/link icon beside headings. */
        [data-testid="stHeadingWithActionElements"] a,
        [data-testid="stHeadingWithActionElements"] button,
        [data-testid="stMarkdownContainer"] h1 > a,
        [data-testid="stMarkdownContainer"] h2 > a,
        [data-testid="stMarkdownContainer"] h3 > a,
        [data-testid="stMarkdownContainer"] h4 > a,
        [data-testid="stMarkdownContainer"] h5 > a,
        [data-testid="stMarkdownContainer"] h6 > a,
        [data-testid="stMarkdownContainer"] h1 a[href^="#"],
        [data-testid="stMarkdownContainer"] h2 a[href^="#"],
        [data-testid="stMarkdownContainer"] h3 a[href^="#"],
        [data-testid="stMarkdownContainer"] h4 a[href^="#"],
        [data-testid="stMarkdownContainer"] h5 a[href^="#"],
        [data-testid="stMarkdownContainer"] h6 a[href^="#"] {{
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            min-width: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }}


        /* Keep every post-login element inside the available browser width. */
        .stApp,
        div[data-testid="stAppViewContainer"],
        section[data-testid="stMain"],
        div[data-testid="stMainBlockContainer"] {{
            overflow-x: hidden !important;
        }}
        div[data-testid="stMainBlockContainer"],
        div[data-testid="stMainBlockContainer"] *,
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] * {{
            box-sizing: border-box !important;
        }}

        .stApp {{
            background:
                linear-gradient(rgba(247,253,249,.62), rgba(241,250,245,.76)),
                url("{bg}") center center / cover fixed no-repeat;
        }}

        /* Main glass panel: responsive width, centred and never wider than its parent. */
        .block-container,
        div[data-testid="stMainBlockContainer"] {{
            width: calc(100% - 1.6rem) !important;
            max-width: 1380px !important;
            min-width: 0 !important;
            padding: 1.35rem 1.35rem 2rem !important;
            margin: 1rem auto 1.35rem !important;
            background: rgba(255,255,255,.76);
            border: 1px solid rgba(255,255,255,.78);
            border-radius: 26px;
            backdrop-filter: blur(18px) saturate(125%);
            -webkit-backdrop-filter: blur(18px) saturate(125%);
            box-shadow: 0 20px 56px rgba(26,76,52,.12);
        }}

        /* Streamlit columns can otherwise keep a minimum width and push cards outside. */
        div[data-testid="stHorizontalBlock"] {{
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            gap: .9rem !important;
            align-items: stretch !important;
        }}
        div[data-testid="stColumn"] {{
            min-width: 0 !important;
            max-width: 100% !important;
        }}
        div[data-testid="stColumn"] > div {{
            min-width: 0 !important;
            max-width: 100% !important;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: rgba(246,252,248,.92);
            border-right: 1px solid rgba(16,113,65,.08);
            backdrop-filter: blur(20px) saturate(120%);
            -webkit-backdrop-filter: blur(20px) saturate(120%);
        }}
        section[data-testid="stSidebar"] > div {{
            padding-top: 1rem !important;
        }}
        .sidebar-logo-wrap {{
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: .1rem 0 .25rem;
        }}
        .sidebar-logo {{
            display: block;
            width: 92px;
            height: 92px;
            object-fit: contain;
            margin: 0 auto;
            filter: drop-shadow(0 9px 14px rgba(3,85,44,.14));
        }}
        .sidebar-brand {{ text-align:center; margin:0 0 1rem; }}
        .sidebar-brand h2 {{ color:var(--gg-green-dark); margin:.05rem 0 0; font-size:1.22rem; }}
        .sidebar-brand p {{ color:#6a8175; margin:.18rem auto 0; font-size:.76rem; line-height:1.35; max-width:220px; }}
        .user-chip {{
            width: 100%;
            padding: .78rem .86rem;
            border-radius: 16px;
            background: rgba(255,255,255,.76);
            border: 1px solid rgba(16,124,69,.08);
            box-shadow: 0 8px 20px rgba(29,87,58,.05);
            margin-bottom: .7rem;
            overflow: hidden;
        }}
        .user-chip strong {{ color:#163b2c; display:block; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
        .user-chip small {{ color:#718379; display:block; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}

        section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap:.34rem; width:100%; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] {{
            width: 100% !important;
            min-width: 0 !important;
            border-radius: 13px;
            padding: .62rem .72rem;
            transition: .18s ease;
            margin: 0;
        }}
        /* Hide the default radio circle: menu labels already have meaningful icons. */
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {{
            display: none !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] p {{
            width:100%;
            margin:0 !important;
            font-weight:690;
            color:#314d40;
            white-space:nowrap;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"]:hover {{
            background: rgba(20,135,75,.07);
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {{
            background: linear-gradient(135deg,#24b45a,#08723e);
            color:white;
            box-shadow:0 8px 18px rgba(11,122,67,.17);
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) p {{
            color:white !important;
            font-weight:780;
        }}

        h1,h2,h3 {{ color:#143d2d; letter-spacing:-.025em; }}
        h1 {{ font-weight:800; }}
        p, li {{ color:#39584a; }}
        .page-marker {{ display:none !important; height:0 !important; margin:0 !important; padding:0 !important; }}
        .page-kicker {{
            text-transform:uppercase;
            letter-spacing:.15em;
            color:#14804a;
            font-size:.72rem;
            font-weight:800;
            margin-bottom:.12rem;
        }}

        .dashboard-hero {{
            width:100%;
            max-width:100%;
            position:relative;
            overflow:hidden;
            border-radius:22px;
            padding:1.2rem 1.3rem;
            margin:.1rem 0 .9rem;
            background:linear-gradient(135deg,rgba(255,255,255,.90),rgba(239,255,243,.76));
            border:1px solid rgba(17,116,65,.08);
            box-shadow:0 10px 26px rgba(22,87,55,.07);
        }}
        .dashboard-hero::after {{
            content:"🌿";
            position:absolute;
            right:1rem;
            bottom:-1rem;
            font-size:5.6rem;
            opacity:.075;
            transform:rotate(-10deg);
        }}
        .dashboard-hero h1 {{
            margin:.12rem 0 .4rem;
            font-size:clamp(1.8rem,2.65vw,2.75rem);
            line-height:1.08;
            overflow-wrap:anywhere;
        }}
        .dashboard-hero p {{ max-width:820px; margin:0; font-size:.96rem; color:#536d61; }}

        .gg-stat-grid {{
            width:100%;
            max-width:100%;
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:.8rem;
            margin:.8rem 0 1.1rem;
        }}
        .gg-stat {{
            min-width:0;
            overflow:hidden;
            background:rgba(255,255,255,.86);
            border:1px solid rgba(17,116,65,.08);
            border-radius:18px;
            padding:.9rem .95rem;
            box-shadow:0 8px 22px rgba(23,78,50,.06);
        }}
        .gg-stat .icon {{
            width:38px; height:38px; border-radius:12px; display:grid; place-items:center;
            background:#e7f8eb; font-size:1.08rem; margin-bottom:.55rem;
        }}
        .gg-stat .label {{ color:#668075; font-size:.75rem; font-weight:750; }}
        .gg-stat .value {{ color:#153a2c; font-size:1.65rem; font-weight:850; margin-top:.06rem; overflow-wrap:anywhere; }}
        .gg-stat .detail {{ color:#789087; font-size:.73rem; margin-top:.08rem; overflow-wrap:anywhere; }}

        /* FIX: Streamlit's bordered-container wrapper was creating the dark top line.
           Remove its native border completely and draw one clean card on the inner block. */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            width:100% !important;
            max-width:100% !important;
            min-width:0 !important;
            margin:0 !important;
            padding:0 !important;
            border:0 !important;
            outline:0 !important;
            border-radius:18px !important;
            background:transparent !important;
            box-shadow:none !important;
            overflow:hidden !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]::before,
        div[data-testid="stVerticalBlockBorderWrapper"]::after {{
            content:none !important;
            display:none !important;
            border:0 !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            width:100% !important;
            max-width:100% !important;
            min-width:0 !important;
            margin:0 !important;
            padding:.92rem !important;
            border:1px solid rgba(17,116,65,.10) !important;
            border-radius:18px !important;
            outline:0 !important;
            background:rgba(255,255,255,.84) !important;
            box-shadow:0 8px 22px rgba(23,78,50,.055) !important;
            overflow:hidden !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] > div > div {{
            max-width:100% !important;
            min-width:0 !important;
        }}

        /* Make every widget/card respect its column width. */
        .stButton,
        .stDownloadButton,
        .stTextInput,
        .stNumberInput,
        .stSelectbox,
        .stFileUploader,
        .stDataFrame,
        div[data-testid="stMetric"],
        div[data-testid="stAlert"],
        div[data-testid="stImage"] {{
            width:100% !important;
            max-width:100% !important;
            min-width:0 !important;
        }}
        .stButton > button,
        .stDownloadButton > button {{
            width:100% !important;
            max-width:100% !important;
            min-width:0 !important;
            min-height:44px;
            border-radius:12px;
            border:1px solid rgba(13,114,63,.12);
            font-weight:750;
            color:#0b6e3d;
            background:rgba(245,255,248,.94);
            transition:.18s ease;
            white-space:normal !important;
            overflow-wrap:anywhere;
        }}
        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            color:white;
            border-color:transparent;
            background:linear-gradient(135deg,#26b95a,#08723e);
            transform:translateY(-1px);
            box-shadow:0 9px 20px rgba(10,117,64,.15);
        }}
        .stButton > button[kind="primary"] {{
            color:white;
            background:linear-gradient(135deg,#27b957,#08723e);
            border:0;
        }}

        .stTextInput input,
        .stNumberInput input,
        div[data-baseweb="select"] > div {{
            max-width:100% !important;
            border-radius:12px !important;
            border-color:rgba(17,110,63,.13) !important;
            background:rgba(255,255,255,.86) !important;
        }}
        .stFileUploader section {{
            width:100% !important;
            max-width:100% !important;
            border-radius:16px;
            border:1.5px dashed rgba(11,122,67,.30);
            background:rgba(246,255,249,.70);
        }}
        div[data-testid="stAlert"] {{ border-radius:14px; border-width:1px; overflow:hidden; }}
        div[data-testid="stMetric"] {{
            background:rgba(255,255,255,.76);
            border:1px solid rgba(17,116,65,.08);
            padding:.82rem;
            border-radius:15px;
            overflow:hidden;
        }}
        div[data-testid="stDataFrame"] {{
            border-radius:15px;
            overflow:hidden !important;
            box-shadow:0 7px 20px rgba(25,78,51,.05);
        }}
        div[data-testid="stImage"] img {{ max-width:100% !important; height:auto !important; }}
        .stExpander {{
            max-width:100% !important;
            border-radius:14px !important;
            overflow:hidden !important;
            background:rgba(255,255,255,.58);
        }}
        .section-title {{
            display:flex;
            align-items:center;
            gap:.5rem;
            margin:.15rem 0 .55rem;
            font-weight:800;
            color:#173f2f;
            font-size:1.08rem;
        }}

        /* FINAL POST-LOGIN CARD SYSTEM
           Keep every section inside its parent and give the entire app the same
           soft rounded-card language. Auth CSS above is intentionally untouched. */
        div[data-testid="stMainBlockContainer"] {{
            overflow: visible !important;
        }}
        section[data-testid="stMain"] {{
            overflow-x: clip !important;
        }}
        div[data-testid="stMainBlockContainer"] > div,
        div[data-testid="stMainBlockContainer"] > div > div,
        div[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {{
            min-width: 0 !important;
            max-width: 100% !important;
        }}

        /* No hard/black separators anywhere in the signed-in app. */
        div[data-testid="stMainBlockContainer"] hr,
        section[data-testid="stSidebar"] hr {{
            border: 0 !important;
            height: 1px !important;
            background: rgba(31,118,75,.10) !important;
            margin: .85rem 0 !important;
        }}

        /* Rounded Streamlit containers, including nested containers. */
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stVerticalBlockBorderWrapper"] > div,
        div[data-testid="stVerticalBlockBorderWrapper"] > div > div {{
            border-radius: 22px !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            overflow: visible !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            overflow: hidden !important;
            background: linear-gradient(145deg, rgba(255,255,255,.94), rgba(244,251,247,.90)) !important;
            border: 1px solid rgba(22,113,66,.11) !important;
            box-shadow: 0 9px 24px rgba(22,78,51,.055) !important;
        }}

        /* All status/info blocks get a real rounded coloured panel, not a square strip. */
        div[data-testid="stAlert"] {{
            border-radius: 18px !important;
            overflow: hidden !important;
            border: 1px solid rgba(56,128,165,.10) !important;
            background: rgba(226,243,250,.78) !important;
        }}
        div[data-testid="stAlert"] > div,
        div[data-testid="stAlert"] > div > div {{
            border-radius: inherit !important;
        }}

        /* Page-specific cards. Only post-login pages carry these marker classes. */
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker)
        div[data-testid="stColumn"]:has(h3),
        div[data-testid="stMainBlockContainer"]:has(.reports-page-marker)
        div[data-testid="stColumn"],
        div[data-testid="stMainBlockContainer"]:has(.about-page-marker)
        div[data-testid="stColumn"],
        div[data-testid="stMainBlockContainer"]:has(.medicines-page-marker)
        div[data-testid="stColumn"] {{
            border-radius: 22px !important;
            background: linear-gradient(145deg, rgba(255,255,255,.90), rgba(244,250,247,.84));
            border: 1px solid rgba(19,111,64,.095);
            box-shadow: 0 8px 22px rgba(21,76,49,.045);
            padding: .9rem !important;
            overflow: hidden !important;
        }}

        /* Nested form columns should remain simple fields, not cards. */
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker)
        div[data-testid="stColumn"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            padding: 0 !important;
            border-radius: 0 !important;
            overflow: visible !important;
        }}

        /* Scan history gets one clean rounded content surface. */
        div[data-testid="stMainBlockContainer"]:has(.scan-page-marker)
        div[data-testid="stDataFrame"],
        div[data-testid="stMainBlockContainer"]:has(.scan-page-marker)
        div[data-testid="stAlert"] {{
            border-radius: 20px !important;
        }}

        /* Inputs, selectors, uploaders, dataframes and action buttons all share rounded edges. */
        .stTextInput input,
        .stNumberInput input,
        div[data-baseweb="select"] > div,
        .stFileUploader section,
        div[data-testid="stDataFrame"],
        div[data-testid="stMetric"],
        .stExpander,
        .stButton > button,
        .stDownloadButton > button {{
            border-radius: 16px !important;
        }}
        .stFileUploader section,
        div[data-testid="stDataFrame"],
        .stExpander {{
            overflow: hidden !important;
        }}
        div[data-testid="stDataFrame"] > div {{
            max-width: 100% !important;
        }}
        iframe, canvas, video, img {{
            max-width: 100% !important;
        }}

        /* Dashboard: keep the three quick-action cards and lower cards perfectly rounded. */
        div[data-testid="stMainBlockContainer"]:has(.dashboard-page-marker)
        div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            border-radius: 22px !important;
        }}
        .dashboard-hero,
        .gg-stat {{
            border-radius: 22px !important;
        }}

        /* Sidebar gets the same rounded visual system. */
        section[data-testid="stSidebar"] {{
            overflow-x: hidden !important;
        }}
        section[data-testid="stSidebar"] > div {{
            padding-left: .85rem !important;
            padding-right: .85rem !important;
        }}
        .sidebar-logo-wrap {{
            justify-content: center !important;
            text-align: center !important;
        }}
        .sidebar-logo {{
            margin-left: auto !important;
            margin-right: auto !important;
        }}
        .user-chip,
        section[data-testid="stSidebar"] div[data-testid="stAlert"],
        section[data-testid="stSidebar"] .stButton > button,
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] {{
            border-radius: 16px !important;
        }}

        /* Keep long copy and labels from ever pushing a card outside its column. */
        div[data-testid="stMainBlockContainer"] p,
        div[data-testid="stMainBlockContainer"] h1,
        div[data-testid="stMainBlockContainer"] h2,
        div[data-testid="stMainBlockContainer"] h3,
        div[data-testid="stMainBlockContainer"] label,
        div[data-testid="stMainBlockContainer"] span {{
            overflow-wrap: anywhere;
            word-break: normal;
        }}


        /* ===== UNIFIED POST-LOGIN ROUND CARD SYSTEM =====
           This block is intentionally last so it wins over Streamlit defaults.
           It applies only after login because it lives inside inject_app_theme(). */
        .page-intro-card {{
            width: 100%;
            max-width: 100%;
            margin: 0 0 1rem 0;
            padding: 1.15rem 1.25rem;
            border-radius: 26px !important;
            background: linear-gradient(145deg, rgba(255,255,255,.94), rgba(246,252,248,.90));
            border: 1px solid rgba(20,116,66,.10);
            box-shadow: 0 10px 28px rgba(22,78,51,.065);
            overflow: hidden;
        }}
        .page-intro-card h1 {{
            margin: .12rem 0 .32rem !important;
            font-size: clamp(1.9rem, 3vw, 2.8rem) !important;
            line-height: 1.08 !important;
            color: #143d2d !important;
        }}
        .page-intro-card p {{
            margin: 0 !important;
            color: #5b7167 !important;
            font-size: .95rem !important;
            line-height: 1.5 !important;
        }}
        .wrap-copy {{
            width:100%;
            max-width:100%;
            color:#48675a;
            line-height:1.55;
            overflow-wrap:anywhere;
            word-break:break-word;
            margin:.15rem 0 .7rem;
        }}

        /* The Streamlit bordered container itself is the card.  Earlier versions
           styled an inner child, which is why some cards still looked square. */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            height: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            border: 1px solid rgba(20,116,66,.11) !important;
            border-radius: 26px !important;
            outline: none !important;
            background: linear-gradient(145deg, rgba(255,255,255,.95), rgba(246,252,248,.91)) !important;
            box-shadow: 0 9px 25px rgba(22,78,51,.06) !important;
            overflow: hidden !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] > div,
        div[data-testid="stVerticalBlockBorderWrapper"] > div > div {{
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            border: 0 !important;
            outline: 0 !important;
            box-shadow: none !important;
            background: transparent !important;
            border-radius: inherit !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            padding: 1rem !important;
            overflow: hidden !important;
        }}

        /* Cards inside columns must never protrude past the main glass panel. */
        div[data-testid="stHorizontalBlock"],
        div[data-testid="stColumn"],
        div[data-testid="stColumn"] > div,
        div[data-testid="stVerticalBlock"] {{
            min-width: 0 !important;
            max-width: 100% !important;
        }}
        div[data-testid="stColumn"] {{
            overflow: visible !important;
        }}
        div[data-testid="stMainBlockContainer"] {{
            overflow-x: clip !important;
            overflow-y: visible !important;
        }}

        /* Dashboard custom cards use the exact same shape as every other card. */
        .dashboard-hero,
        .gg-stat {{
            border-radius: 26px !important;
            border: 1px solid rgba(20,116,66,.10) !important;
            background: linear-gradient(145deg, rgba(255,255,255,.95), rgba(246,252,248,.91)) !important;
            box-shadow: 0 9px 25px rgba(22,78,51,.06) !important;
            overflow: hidden !important;
        }}
        .gg-stat .icon {{ border-radius: 15px !important; }}

        /* Blue/green status panels must also have rounded backs, never square strips. */
        div[data-testid="stAlert"] {{
            width: 100% !important;
            max-width: 100% !important;
            border-radius: 20px !important;
            overflow: hidden !important;
            border: 1px solid rgba(56,128,165,.11) !important;
            box-shadow: none !important;
        }}
        div[data-testid="stAlert"] > div,
        div[data-testid="stAlert"] > div > div {{
            border-radius: inherit !important;
            overflow: hidden !important;
        }}

        /* Controls also follow the same rounded language. */
        .stButton > button,
        .stDownloadButton > button {{
            border-radius: 18px !important;
            overflow: hidden !important;
        }}
        .stTextInput input,
        .stNumberInput input,
        div[data-baseweb="select"] > div {{
            border-radius: 18px !important;
        }}
        .stExpander,
        details[data-testid="stExpander"],
        div[data-testid="stExpander"] {{
            border-radius: 20px !important;
            overflow: hidden !important;
        }}

        /* File uploader: round the complete drop zone AND its Browse button. */
        .stFileUploader,
        div[data-testid="stFileUploader"] {{
            width:100% !important;
            max-width:100% !important;
            overflow:hidden !important;
            border-radius:22px !important;
        }}
        .stFileUploader section,
        div[data-testid="stFileUploader"] section {{
            width:100% !important;
            max-width:100% !important;
            border-radius:22px !important;
            overflow:hidden !important;
            background:rgba(248,253,250,.88) !important;
            border:1.5px dashed rgba(11,122,67,.27) !important;
        }}
        .stFileUploader section button,
        div[data-testid="stFileUploader"] section button {{
            border-radius:16px !important;
            border:1px solid rgba(17,110,63,.12) !important;
            background:rgba(237,249,241,.95) !important;
        }}

        /* Tables, metrics, images and camera/upload surfaces. */
        div[data-testid="stMetric"],
        div[data-testid="stDataFrame"],
        div[data-testid="stCameraInput"],
        div[data-testid="stImage"] img {{
            border-radius: 20px !important;
            overflow: hidden !important;
        }}
        div[data-testid="stMetric"] {{
            border:1px solid rgba(20,116,66,.10) !important;
            background:rgba(255,255,255,.90) !important;
        }}
        div[data-testid="stDataFrame"] {{
            border:1px solid rgba(20,116,66,.09) !important;
            background:rgba(255,255,255,.90) !important;
        }}

        /* Remove dark lines that can appear from native Streamlit wrappers/dividers. */
        div[data-testid="stMainBlockContainer"] hr,
        section[data-testid="stSidebar"] hr {{
            border:0 !important;
            height:1px !important;
            background:rgba(20,116,66,.08) !important;
            box-shadow:none !important;
        }}

        /* Sidebar matches the same rounded-card system. */
        section[data-testid="stSidebar"] {{
            overflow-x:hidden !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stImage"] {{
            display:flex !important;
            justify-content:center !important;
            width:100% !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stImage"] img {{
            margin-left:auto !important;
            margin-right:auto !important;
        }}
        .user-chip,
        section[data-testid="stSidebar"] div[data-testid="stAlert"],
        section[data-testid="stSidebar"] .stButton > button,
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] {{
            border-radius: 18px !important;
            overflow: hidden !important;
        }}

        /* Long disease names / report text must wrap instead of leaving the card. */
        div[data-testid="stMainBlockContainer"] p,
        div[data-testid="stMainBlockContainer"] span,
        div[data-testid="stMainBlockContainer"] label,
        div[data-testid="stMainBlockContainer"] div {{
            overflow-wrap: anywhere;
        }}
        @media (max-width: 1180px) {{
            .gg-stat-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
            .block-container, div[data-testid="stMainBlockContainer"] {{
                width:calc(100% - 1rem) !important;
                padding:1.1rem !important;
            }}
        }}
        @media (max-width: 860px) {{
            div[data-testid="stHorizontalBlock"] {{
                flex-wrap:wrap !important;
            }}
            div[data-testid="stColumn"] {{
                flex:1 1 100% !important;
                width:100% !important;
            }}
        }}
        @media (max-width: 700px) {{
            .block-container, div[data-testid="stMainBlockContainer"] {{
                width:calc(100% - .7rem) !important;
                margin:.35rem auto .7rem !important;
                padding:.9rem !important;
                border-radius:19px;
            }}
            .gg-stat-grid {{ grid-template-columns:1fr; }}
            .dashboard-hero {{ padding:1rem; border-radius:18px; }}
            .dashboard-hero h1 {{ font-size:1.75rem; }}
        }}

        /* ===== FINAL VISUAL SYSTEM: ONE CARD PER SECTION =====
           This deliberately overrides earlier experimental styles. */
        .block-container,
        div[data-testid="stMainBlockContainer"] {{
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
        }}

        /* Columns are layout only. Never render a column itself as a card. */
        div[data-testid="stMainBlockContainer"] div[data-testid="stColumn"] {{
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            border-radius: 0 !important;
            padding: 0 !important;
            overflow: visible !important;
        }}

        /* One visible rounded surface for every explicit bordered container. */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            width:100% !important;
            max-width:100% !important;
            min-width:0 !important;
            margin:0 !important;
            padding:0 !important;
            border:1px solid rgba(18,92,56,.11) !important;
            border-radius:24px !important;
            background:rgba(255,255,255,.94) !important;
            box-shadow:0 10px 28px rgba(28,78,52,.07) !important;
            overflow:hidden !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            padding:1.08rem !important;
            border:0 !important;
            outline:0 !important;
            border-radius:inherit !important;
            background:transparent !important;
            box-shadow:none !important;
            overflow:visible !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] > div > div {{
            border:0 !important;
            outline:0 !important;
            background:transparent !important;
            box-shadow:none !important;
        }}
        /* If a bordered container ever ends up nested, the nested wrapper is not another card. */
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] {{
            border:0 !important;
            background:transparent !important;
            box-shadow:none !important;
            border-radius:0 !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            padding:.25rem 0 !important;
        }}

        /* Modern page heading used by Plant Analysis. */
        .plant-topbar {{
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:1rem;
            margin:.05rem 0 .9rem;
        }}
        .plant-title-wrap {{ display:flex; align-items:center; gap:.75rem; min-width:0; }}
        .plant-title-icon {{
            width:48px; height:48px; border-radius:15px; display:grid; place-items:center;
            background:#e8f8ed; font-size:1.55rem; flex:0 0 48px;
        }}
        .plant-topbar h1 {{ margin:0 !important; font-size:2rem !important; line-height:1.05 !important; color:#102f25 !important; }}
        .plant-topbar p {{ margin:.2rem 0 0 !important; color:#708079 !important; font-size:.91rem !important; }}
        .ai-health-badge {{
            flex:0 0 auto; display:flex; align-items:center; gap:.55rem; max-width:360px;
            padding:.72rem .9rem; border-radius:16px; background:linear-gradient(135deg,#edf9f0,#e3f7e9);
            border:1px solid rgba(27,141,76,.09); color:#1d6841;
        }}
        .ai-health-badge .badge-icon {{ font-size:1.25rem; }}
        .ai-health-badge strong {{ display:block; font-size:.82rem; color:#1a5d3b; }}
        .ai-health-badge small {{ display:block; margin-top:.04rem; color:#658074; font-size:.72rem; }}

        .card-heading {{ display:flex; gap:.68rem; align-items:flex-start; margin-bottom:.8rem; }}
        .card-heading-icon {{
            width:42px; height:42px; border-radius:13px; display:grid; place-items:center;
            background:#e9f8ed; flex:0 0 42px; font-size:1.3rem;
        }}
        .card-heading-icon.blue {{ background:#e9f3ff; }}
        .card-heading h3 {{ margin:0 !important; color:#143a2b !important; font-size:1.18rem !important; }}
        .card-heading p {{ margin:.16rem 0 0 !important; color:#74847c !important; font-size:.79rem !important; line-height:1.35 !important; }}
        .soft-divider {{ height:1px; background:rgba(33,104,67,.10); margin:.95rem 0; }}
        .smart-care-head {{ display:flex; align-items:flex-start; gap:.6rem; margin-bottom:.65rem; }}
        .smart-care-head .mini-icon {{
            width:38px; height:38px; border-radius:12px; display:grid; place-items:center;
            background:#e7f7ec; font-size:1.15rem; flex:0 0 38px;
        }}
        .smart-care-head strong {{ display:block; color:#173d2e; font-size:.98rem; }}
        .smart-care-head span {{ display:block; color:#7b8982; font-size:.75rem; margin-top:.1rem; }}
        .configured-label {{ color:#75857c; font-size:.78rem; font-weight:700; margin:.45rem 0 .12rem; }}
        .configured-copy {{ color:#64786e; font-size:.75rem; line-height:1.42; margin:0 0 .25rem; }}

        /* Plant page fields match the sample: compact, round, clean. */
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stSelectbox > div > div,
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stNumberInput input,
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stTextInput input {{
            min-height:46px !important;
            border-radius:13px !important;
            border:1px solid rgba(41,89,65,.16) !important;
            background:#fff !important;
            box-shadow:none !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) label p {{
            color:#314d40 !important; font-size:.80rem !important; font-weight:760 !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stFileUploader section {{
            min-height:210px !important;
            border-radius:18px !important;
            border:1.5px dashed rgba(16,133,72,.28) !important;
            background:rgba(250,253,251,.92) !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stFileUploader section button {{
            border-radius:12px !important;
            background:#f1faf4 !important;
            color:#0d7643 !important;
            border:1px solid rgba(14,121,67,.18) !important;
        }}
        .analyse-bar .stButton > button,
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stButton > button[kind="primary"] {{
            min-height:52px !important;
            border-radius:14px !important;
            color:#fff !important;
            font-size:1rem !important;
            font-weight:800 !important;
            border:0 !important;
            background:linear-gradient(90deg,#21b85b,#0c8b4c) !important;
            box-shadow:0 10px 22px rgba(16,137,74,.18) !important;
        }}

        .results-card {{
            width:100%; padding:1.05rem; border-radius:24px; background:rgba(255,255,255,.94);
            border:1px solid rgba(18,92,56,.11); box-shadow:0 10px 28px rgba(28,78,52,.07); margin-top:1rem;
        }}
        .results-head {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:.85rem; }}
        .results-head h3 {{ margin:0; font-size:1.12rem; color:#173d2e; }}
        .results-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.7rem; }}
        .result-tile {{ border-radius:16px; padding:.8rem; min-width:0; }}
        .result-tile.problem {{ background:#fff0f0; }}
        .result-tile.severity {{ background:#fff6e9; }}
        .result-tile.confidence {{ background:#edf7ff; }}
        .result-tile.method {{ background:#f5efff; }}
        .result-tile .r-label {{ font-size:.7rem; font-weight:800; color:#6d7a74; text-transform:uppercase; letter-spacing:.04em; }}
        .result-tile .r-value {{ margin-top:.25rem; color:#17382b; font-size:1.05rem; font-weight:850; overflow-wrap:anywhere; }}

        /* Sidebar closer to the supplied reference. */
        section[data-testid="stSidebar"] {{
            background:rgba(255,255,255,.95) !important;
            border-right:1px solid rgba(20,91,56,.08) !important;
        }}
        .sidebar-brand-row {{ display:flex; align-items:center; gap:.65rem; padding:.15rem .1rem .8rem; border-bottom:1px solid rgba(23,82,52,.09); }}
        .sidebar-brand-row img {{ width:58px; height:58px; object-fit:contain; flex:0 0 58px; }}
        .sidebar-brand-row strong {{ color:#12372a; font-size:1.05rem; line-height:1.15; }}
        .sidebar-profile {{ display:flex; align-items:center; gap:.7rem; padding:.8rem .1rem; border-bottom:1px solid rgba(23,82,52,.09); }}
        .profile-avatar {{ width:44px; height:44px; border-radius:50%; background:#e9f0ed; display:grid; place-items:center; font-size:1.4rem; flex:0 0 44px; }}
        .sidebar-profile strong {{ display:block; color:#163a2c; font-size:.91rem; }}
        .sidebar-profile small {{ display:block; color:#74857d; font-size:.75rem; max-width:165px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
        .sidebar-nav-title {{ color:#607269; font-size:.76rem; font-weight:800; margin:.78rem .15rem .4rem; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] {{
            position:relative; min-height:46px; border-radius:14px !important; padding:.65rem .78rem !important; background:transparent !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {{
            color:#0b6038 !important; background:linear-gradient(90deg,#e5f7e9,#effaf2) !important; box-shadow:none !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked)::before {{
            content:""; position:absolute; left:0; top:8px; bottom:8px; width:4px; border-radius:0 4px 4px 0; background:#0e8a4d;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) p {{ color:#0b6038 !important; }}
        .sidebar-model-status {{ margin:.55rem 0 .15rem; padding:.52rem .65rem; border-radius:12px; background:#eff8fb; color:#58747e; font-size:.75rem; }}
        section[data-testid="stSidebar"] .stButton > button {{
            justify-content:flex-start !important; padding-left:.8rem !important; border:0 !important; background:transparent !important; color:#2e493d !important; box-shadow:none !important;
        }}
        section[data-testid="stSidebar"] .stButton > button:hover {{ background:#f2f7f4 !important; color:#0b6d3e !important; transform:none !important; }}

        @media (max-width: 1000px) {{
            .plant-topbar {{ align-items:flex-start; flex-direction:column; }}
            .ai-health-badge {{ max-width:100%; width:100%; }}
            .results-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
        }}
        @media (max-width: 700px) {{
            .results-grid {{ grid-template-columns:1fr; }}
        }}
        

        /* ===== PLANT ANALYSIS POLISH OVERRIDES ===== */
        section[data-testid="stSidebar"] {{
            width: 286px !important;
            min-width: 286px !important;
            max-width: 286px !important;
        }}
        section[data-testid="stSidebar"] > div,
        section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {{
            width: 286px !important;
            max-width: 286px !important;
        }}
        section[data-testid="stSidebar"] > div {{
            padding: 1rem .9rem 1.15rem !important;
        }}
        .sidebar-brand-row {{
            gap:.7rem !important;
            padding:.2rem .15rem .9rem !important;
        }}
        .sidebar-brand-row img {{ width:52px !important; height:52px !important; flex-basis:52px !important; }}
        .sidebar-brand-row strong {{ font-size:1.02rem !important; white-space:nowrap; }}
        .sidebar-profile {{ padding:.85rem .15rem !important; }}
        .sidebar-nav-title {{ margin:.9rem .2rem .42rem !important; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap:.18rem !important; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] {{
            min-height:44px !important;
            padding:.58rem .72rem !important;
            border-radius:13px !important;
        }}

        div[data-testid="stMainBlockContainer"] {{
            max-width: 1480px !important;
            padding: 1.15rem 1.35rem 2rem !important;
        }}

        /* Columns are layout only. The one real card is the bordered Streamlit container. */
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) div[data-testid="stColumn"] {{
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            padding: 0 !important;
            border-radius: 0 !important;
            overflow: visible !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker)
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border: 1px solid rgba(31,103,67,.11) !important;
            border-radius: 22px !important;
            background: rgba(255,255,255,.95) !important;
            box-shadow: 0 10px 26px rgba(25,76,51,.055) !important;
            overflow: hidden !important;
            margin: 0 !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker)
        div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            padding: 1.05rem 1.08rem !important;
            border: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
            border-radius: inherit !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker)
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] {{
            border:0 !important;
            background:transparent !important;
            box-shadow:none !important;
            border-radius:0 !important;
        }}

        .plant-topbar-polished {{
            padding:.15rem .05rem .05rem;
            margin-bottom:.85rem !important;
        }}
        .plant-topbar-polished .plant-title-icon {{
            width:46px !important; height:46px !important; flex-basis:46px !important;
            border-radius:14px !important; background:#e8f7ed !important;
        }}
        .plant-topbar-polished h1 {{ font-size:1.85rem !important; }}
        .plant-topbar-polished p {{ font-size:.86rem !important; max-width:680px; }}
        .plant-topbar-polished .ai-health-badge {{
            max-width:340px !important; padding:.62rem .78rem !important; border-radius:15px !important;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.plant-primary-card) {{
            min-height: 445px !important;
        }}
        .plant-primary-card {{ height:0; overflow:hidden; margin:0; padding:0; }}
        .plant-section-heading {{ margin-bottom:.68rem !important; }}
        .plant-section-heading .card-heading-icon {{ width:40px !important; height:40px !important; flex-basis:40px !important; }}
        .plant-section-heading h3 {{ font-size:1.12rem !important; }}
        .plant-section-heading p {{ font-size:.76rem !important; }}

        .plant-model-status {{
            display:flex; align-items:center; gap:.35rem;
            min-height:34px; margin:.42rem 0 .34rem; padding:.42rem .66rem;
            border-radius:11px; font-size:.76rem; line-height:1.2;
        }}
        .plant-model-status span {{ font-size:.66rem; }}
        .plant-model-status strong {{ margin-right:.08rem; }}
        .plant-model-status.ready {{ background:#eaf8ef; color:#17633e; }}
        .plant-model-status.pending {{ background:#edf7fb; color:#4e6f7b; }}
        .optional-tag {{
            display:inline-block !important; margin-left:.2rem; padding:.08rem .38rem; border-radius:999px;
            background:#eef5f1; color:#698077; font-size:.67rem !important; font-weight:700 !important;
        }}
        .soft-divider.compact {{ margin:.65rem 0 .72rem !important; }}
        .smart-care-head {{ margin-bottom:.5rem !important; }}
        .smart-care-head .mini-icon {{ width:34px !important; height:34px !important; flex-basis:34px !important; font-size:1rem !important; }}
        .smart-care-head strong {{ font-size:.9rem !important; }}
        .smart-care-head span {{ font-size:.71rem !important; line-height:1.3; }}

        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stSelectbox,
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stNumberInput,
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stRadio {{
            margin-bottom:.18rem !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stSelectbox > div > div,
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stNumberInput input {{
            min-height:42px !important;
            height:42px !important;
            border-radius:12px !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) label p {{
            font-size:.75rem !important;
            margin-bottom:.12rem !important;
        }}

        /* Source choice becomes a tidy segmented control. */
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stRadio div[role="radiogroup"] {{
            display:flex !important; gap:.35rem !important; padding:.25rem !important;
            border-radius:12px !important; background:#f4f8f6 !important;
            border:1px solid rgba(31,103,67,.08) !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stRadio label[data-baseweb="radio"] {{
            flex:1 1 0 !important; justify-content:center !important; min-height:38px !important;
            padding:.42rem .55rem !important; border-radius:10px !important; margin:0 !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stRadio label[data-baseweb="radio"]:has(input:checked) {{
            background:#e6f6eb !important; color:#0b7140 !important;
        }}

        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stFileUploader section {{
            min-height: 190px !important;
            border-radius:17px !important;
            border:1.5px dashed rgba(14,128,70,.28) !important;
            background:linear-gradient(180deg,#fbfefc,#f6fbf8) !important;
            padding:.85rem !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stFileUploader section button {{
            min-height:40px !important; border-radius:11px !important;
        }}
        .upload-guidance {{
            margin:.4rem 0 0; color:#7a8982; font-size:.71rem; text-align:center; line-height:1.3;
        }}

        .analyse-bar {{ height:.2rem; }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stButton > button[kind="primary"] {{
            min-height:50px !important; border-radius:15px !important; font-size:.98rem !important;
            margin:.45rem 0 .15rem !important;
        }}

        .results-card {{ margin-top:.9rem !important; border-radius:22px !important; }}
        .results-head span {{ display:block; margin-top:.08rem; color:#7c8a83; font-size:.73rem; }}
        .results-head small {{ color:#7d8b84; font-size:.72rem; }}
        .result-tile {{ min-height:92px; display:flex; flex-direction:column; justify-content:center; }}

        @media (max-width: 1150px) {{
            section[data-testid="stSidebar"] {{ width:260px !important; min-width:260px !important; max-width:260px !important; }}
            section[data-testid="stSidebar"] > div,
            section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {{ width:260px !important; max-width:260px !important; }}
            .plant-topbar-polished .ai-health-badge {{ max-width:300px !important; }}
        }}
        @media (max-width: 900px) {{
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.plant-primary-card) {{ min-height:0 !important; }}
            .plant-topbar-polished {{ flex-direction:column !important; align-items:flex-start !important; }}
            .plant-topbar-polished .ai-health-badge {{ width:100% !important; max-width:none !important; }}
        }}

        /* ===== EXACT PREVIEW MATCH: POST-LOGIN / PLANT ANALYSIS ===== */
        section[data-testid="stSidebar"] {{
            width: 272px !important; min-width:272px !important; max-width:272px !important;
            background: rgba(255,255,255,.97) !important;
            border-right: 1px solid #e8eeea !important;
            box-shadow: 12px 0 32px rgba(29,73,49,.04) !important;
        }}
        section[data-testid="stSidebar"] > div,
        section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {{
            width:272px !important; max-width:272px !important;
        }}
        section[data-testid="stSidebar"] > div {{ padding: 1.05rem .95rem 1rem !important; }}
        .sidebar-brand-row {{ padding:.15rem .15rem .95rem !important; gap:.7rem !important; }}
        .sidebar-brand-row img {{ width:54px !important; height:54px !important; }}
        .sidebar-brand-row strong {{ color:#0a3c27 !important; font-size:1.12rem !important; }}
        .sidebar-profile {{ padding:.85rem .15rem !important; }}
        .profile-avatar {{ width:46px !important; height:46px !important; flex-basis:46px !important; }}
        .sidebar-nav-title {{ margin:.9rem .2rem .48rem !important; text-transform:uppercase; letter-spacing:.08em; font-size:.68rem !important; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] {{
            min-height:46px !important; border-radius:14px !important; padding:.62rem .78rem !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {{
            background:linear-gradient(90deg,#def4e5 0%,#edf9f1 100%) !important;
            box-shadow:none !important; color:#075c38 !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked)::before {{
            top:7px !important; bottom:7px !important; width:4px !important; background:#0a8d4c !important;
        }}

        div[data-testid="stMainBlockContainer"] {{
            max-width: 1500px !important;
            padding: 1.05rem 1.2rem 1.8rem !important;
            background: rgba(248,252,249,.72) !important;
            border-radius: 28px !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) {{
            background: rgba(247,251,248,.80) !important;
        }}
        .plant-topbar-polished {{
            margin:0 0 .85rem !important; padding:.12rem .1rem .08rem !important;
            align-items:center !important;
        }}
        .plant-topbar-polished .plant-title-icon {{
            width:48px !important; height:48px !important; flex-basis:48px !important;
            border-radius:14px !important; background:#e8f8ed !important;
        }}
        .plant-topbar-polished h1 {{ font-size:1.95rem !important; color:#12382b !important; }}
        .plant-topbar-polished p {{ font-size:.88rem !important; color:#708078 !important; }}
        .plant-topbar-polished .ai-health-badge {{
            padding:.72rem .9rem !important; border-radius:16px !important; max-width:360px !important;
            background:linear-gradient(135deg,#eaf8ee,#e1f6e7) !important;
            border:1px solid #d8eddf !important; box-shadow:none !important;
        }}

        /* One single rounded card only. No nested visible borders. */
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) div[data-testid="stVerticalBlockBorderWrapper"] {{
            border:1px solid #e1e9e4 !important;
            border-radius:24px !important;
            background:rgba(255,255,255,.97) !important;
            box-shadow:0 10px 26px rgba(32,76,53,.055) !important;
            overflow:hidden !important;
            padding:0 !important; margin:0 !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            border:0 !important; outline:0 !important; box-shadow:none !important;
            background:transparent !important; border-radius:0 !important;
            padding:1.08rem 1.12rem !important; overflow:visible !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) div[data-testid="stVerticalBlockBorderWrapper"] > div > div {{
            border:0 !important; outline:0 !important; box-shadow:none !important; background:transparent !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] {{
            border:0 !important; border-radius:0 !important; background:transparent !important; box-shadow:none !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            padding:0 !important;
        }}

        .plant-section-heading {{ margin-bottom:.78rem !important; }}
        .plant-section-heading .card-heading-icon {{
            width:42px !important; height:42px !important; flex-basis:42px !important; border-radius:13px !important;
        }}
        .plant-section-heading h3 {{ font-size:1.18rem !important; }}
        .plant-section-heading p {{ font-size:.78rem !important; color:#7a8982 !important; }}
        .plant-model-status {{
            min-height:36px !important; border-radius:12px !important; padding:.46rem .68rem !important; margin:.48rem 0 .4rem !important;
        }}
        .soft-divider.compact {{ margin:.72rem 0 .78rem !important; }}
        .smart-care-head .mini-icon {{ width:38px !important; height:38px !important; flex-basis:38px !important; border-radius:12px !important; }}

        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stSelectbox > div > div,
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stNumberInput input {{
            min-height:44px !important; height:44px !important; border-radius:13px !important;
            border:1px solid #dfe7e2 !important; background:#fff !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stRadio div[role="radiogroup"] {{
            background:#f5f8f6 !important; border:1px solid #e7ede9 !important; border-radius:12px !important; padding:.25rem !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stRadio label[data-baseweb="radio"] {{
            border-radius:10px !important; min-height:38px !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stFileUploader section {{
            min-height:205px !important; border-radius:20px !important;
            border:1.5px dashed #cfded5 !important; background:#fbfdfc !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stFileUploader section button {{
            border-radius:12px !important; border:1px solid #bcd9c8 !important; background:#f2faf5 !important;
        }}
        div[data-testid="stMainBlockContainer"]:has(.plant-page-marker) .stButton > button[kind="primary"] {{
            min-height:52px !important; border-radius:14px !important;
            background:linear-gradient(90deg,#20b65a,#0a8b4b) !important;
            box-shadow:0 8px 20px rgba(12,137,73,.16) !important;
        }}
        .results-card {{
            background:rgba(255,255,255,.97) !important; border:1px solid #e1e9e4 !important;
            border-radius:24px !important; box-shadow:0 10px 26px rgba(32,76,53,.055) !important;
        }}
        .result-tile {{ border-radius:16px !important; }}

        @media (max-width: 1050px) {{
            section[data-testid="stSidebar"] {{ width:250px !important; min-width:250px !important; max-width:250px !important; }}
            section[data-testid="stSidebar"] > div, section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {{ width:250px !important; max-width:250px !important; }}
        }}
        @media (max-width: 850px) {{
            .plant-topbar-polished {{ flex-direction:column !important; align-items:flex-start !important; }}
            .plant-topbar-polished .ai-health-badge {{ width:100% !important; max-width:none !important; }}
        }}

        /* ===== FINAL LAYOUT SAFETY + UNIFIED SIDEBAR =====
           Last override on purpose. Keeps every signed-in page inside the viewport,
           rounds every visible card once, and prevents labels/uploader copy from clipping. */
        html, body, .stApp, div[data-testid="stAppViewContainer"] {{
            max-width:100vw !important;
            overflow-x:hidden !important;
        }}
        section[data-testid="stMain"] {{
            min-width:0 !important;
            max-width:100% !important;
            overflow-x:hidden !important;
        }}
        div[data-testid="stMainBlockContainer"] {{
            width:100% !important;
            max-width:none !important;
            min-width:0 !important;
            margin:0 !important;
            padding:1rem 1rem 1.6rem !important;
            overflow-x:hidden !important;
            overflow-y:visible !important;
        }}
        div[data-testid="stMainBlockContainer"] > div,
        div[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"],
        div[data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"],
        div[data-testid="stMainBlockContainer"] [data-testid="stColumn"] {{
            min-width:0 !important;
            max-width:100% !important;
        }}
        div[data-testid="stHorizontalBlock"] {{
            width:100% !important;
            max-width:100% !important;
            gap:.85rem !important;
        }}
        div[data-testid="stColumn"] {{
            min-width:0 !important;
            overflow:visible !important;
        }}

        /* One visible rounded surface per section. */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            width:100% !important;
            max-width:100% !important;
            min-width:0 !important;
            border:1px solid #e1e9e4 !important;
            border-radius:24px !important;
            background:rgba(255,255,255,.96) !important;
            box-shadow:0 10px 26px rgba(31,76,52,.055) !important;
            overflow:hidden !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            width:100% !important;
            max-width:100% !important;
            min-width:0 !important;
            padding:1rem !important;
            border:0 !important;
            border-radius:inherit !important;
            background:transparent !important;
            box-shadow:none !important;
            overflow:visible !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] {{
            border:0 !important;
            border-radius:0 !important;
            background:transparent !important;
            box-shadow:none !important;
        }}
        .dashboard-hero, .gg-stat, .results-card, .page-intro-card {{
            border-radius:24px !important;
            overflow:hidden !important;
        }}

        /* Never let copy, controls or uploader content leave their cards. */
        div[data-testid="stMainBlockContainer"] h1,
        div[data-testid="stMainBlockContainer"] h2,
        div[data-testid="stMainBlockContainer"] h3,
        div[data-testid="stMainBlockContainer"] p,
        div[data-testid="stMainBlockContainer"] span,
        div[data-testid="stMainBlockContainer"] label,
        div[data-testid="stMainBlockContainer"] small {{
            max-width:100% !important;
            overflow-wrap:anywhere !important;
            word-break:normal !important;
        }}
        .stSelectbox, .stNumberInput, .stTextInput, .stRadio,
        .stFileUploader, div[data-testid="stFileUploader"],
        div[data-testid="stCameraInput"], .stButton, .stDownloadButton,
        div[data-testid="stAlert"], div[data-testid="stDataFrame"] {{
            width:100% !important;
            max-width:100% !important;
            min-width:0 !important;
        }}
        div[data-baseweb="select"] > div,
        .stNumberInput input, .stTextInput input,
        .stButton > button, .stDownloadButton > button {{
            border-radius:16px !important;
        }}
        .stButton > button, .stDownloadButton > button {{
            white-space:normal !important;
            overflow-wrap:anywhere !important;
        }}

        /* File uploader and camera blocks: complete round shape, no clipped words. */
        .stFileUploader, div[data-testid="stFileUploader"] {{
            overflow:visible !important;
        }}
        .stFileUploader section,
        div[data-testid="stFileUploader"] section {{
            width:100% !important;
            max-width:100% !important;
            min-width:0 !important;
            border-radius:20px !important;
            overflow:hidden !important;
            padding:.9rem !important;
        }}
        .stFileUploader section > div,
        div[data-testid="stFileUploader"] section > div {{
            min-width:0 !important;
            max-width:100% !important;
            flex:1 1 auto !important;
        }}
        .stFileUploader section p,
        .stFileUploader section span,
        .stFileUploader section small,
        div[data-testid="stFileUploader"] section p,
        div[data-testid="stFileUploader"] section span,
        div[data-testid="stFileUploader"] section small {{
            white-space:normal !important;
            overflow:visible !important;
            text-overflow:clip !important;
            overflow-wrap:anywhere !important;
        }}
        .stFileUploader section button,
        div[data-testid="stFileUploader"] section button {{
            flex:0 0 auto !important;
            border-radius:14px !important;
        }}
        div[data-testid="stCameraInput"] {{
            border-radius:20px !important;
            overflow:hidden !important;
        }}

        /* Alerts/status strips keep a rounded coloured background. */
        div[data-testid="stAlert"] {{
            border-radius:18px !important;
            overflow:hidden !important;
        }}

        /* Sidebar: centred logo, clean profile, consistent icons and active state. */
        section[data-testid="stSidebar"] {{
            width:248px !important;
            min-width:248px !important;
            max-width:248px !important;
            overflow-x:hidden !important;
            background:rgba(255,255,255,.97) !important;
            border-right:1px solid #e6eee9 !important;
        }}
        section[data-testid="stSidebar"] > div,
        section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {{
            width:248px !important;
            max-width:248px !important;
            padding-left:.82rem !important;
            padding-right:.82rem !important;
        }}
        .sidebar-brand-center {{
            width:100%;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            text-align:center;
            padding:.1rem .25rem .85rem;
            border-bottom:1px solid rgba(23,82,52,.09);
            overflow:visible;
        }}
        .sidebar-brand-center img {{
            display:block;
            width:82px !important;
            height:82px !important;
            max-width:82px !important;
            object-fit:contain !important;
            object-position:center !important;
            margin:0 auto .35rem !important;
            filter:drop-shadow(0 8px 14px rgba(3,85,44,.13));
        }}
        .sidebar-brand-center strong {{
            display:block;
            color:#113b2c;
            font-size:1.12rem;
            line-height:1.2;
        }}
        .sidebar-brand-center small {{
            display:block;
            color:#76877f;
            font-size:.72rem;
            margin-top:.12rem;
        }}
        .sidebar-profile {{
            width:100%;
            min-width:0;
            padding:.78rem .2rem !important;
        }}
        .sidebar-profile > div:last-child {{ min-width:0 !important; }}
        .sidebar-profile strong,
        .sidebar-profile small {{
            max-width:155px !important;
            white-space:nowrap !important;
            overflow:hidden !important;
            text-overflow:ellipsis !important;
        }}
        .sidebar-nav-title {{
            margin:.8rem .2rem .42rem !important;
            font-size:.70rem !important;
            letter-spacing:.09em !important;
            text-transform:uppercase !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] {{
            width:100% !important;
            gap:.2rem !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] {{
            width:100% !important;
            min-width:0 !important;
            min-height:45px !important;
            padding:.6rem .7rem !important;
            border-radius:14px !important;
            overflow:hidden !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] p {{
            font-size:.88rem !important;
            line-height:1.2 !important;
            white-space:nowrap !important;
            overflow:hidden !important;
            text-overflow:ellipsis !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {{
            background:linear-gradient(90deg,#dff4e5 0%,#eef9f1 100%) !important;
            color:#075c38 !important;
            box-shadow:none !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked)::before {{
            content:"";
            position:absolute;
            left:0;
            top:7px;
            bottom:7px;
            width:4px;
            border-radius:0 5px 5px 0;
            background:#0b8e4d;
        }}
        .sidebar-model-status {{ border-radius:14px !important; }}
        section[data-testid="stSidebar"] .stButton > button {{
            border-radius:14px !important;
            min-height:44px !important;
        }}

        /* Responsive protection: when the usable main area gets tight, stack cards instead of cutting them. */
        @media (max-width: 1180px) {{
            section[data-testid="stSidebar"] {{ width:228px !important; min-width:228px !important; max-width:228px !important; }}
            section[data-testid="stSidebar"] > div,
            section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {{ width:228px !important; max-width:228px !important; }}
            div[data-testid="stMainBlockContainer"] {{ padding:.85rem !important; }}
            .plant-topbar-polished {{ flex-direction:column !important; align-items:flex-start !important; }}
            .plant-topbar-polished .ai-health-badge {{ width:100% !important; max-width:none !important; }}
        }}
        @media (max-width: 980px) {{
            div[data-testid="stMainBlockContainer"] div[data-testid="stHorizontalBlock"] {{
                flex-wrap:wrap !important;
            }}
            div[data-testid="stMainBlockContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
                flex:1 1 100% !important;
                width:100% !important;
            }}
            .gg-stat-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)) !important; }}
        }}
        @media (max-width: 720px) {{
            .gg-stat-grid, .results-grid {{ grid-template-columns:1fr !important; }}
            div[data-testid="stMainBlockContainer"] {{ padding:.65rem !important; }}
        }}
</style>
        """,
        unsafe_allow_html=True,
    )


    # FINAL SIDEBAR / VIEWPORT FIX
    # Keep the main page scrollable, let the collapsed sidebar reserve zero space,
    # and keep Streamlit's native reopen control fully clickable.
    st.markdown(
        """
        <style>
        html, body, #root, .stApp, div[data-testid="stAppViewContainer"] {
            width: 100% !important;
            max-width: 100% !important;
            min-height: 100% !important;
        }
        .stApp, div[data-testid="stAppViewContainer"] {
            min-height: 100vh !important;
            min-height: 100dvh !important;
        }

        /* Main area remains the vertical scroller on every page. */
        section[data-testid="stMain"] {
            flex: 1 1 auto !important;
            min-width: 0 !important;
            width: auto !important;
            height: 100vh !important;
            height: 100dvh !important;
            max-height: 100vh !important;
            max-height: 100dvh !important;
            overflow-x: hidden !important;
            overflow-y: auto !important;
            -webkit-overflow-scrolling: touch !important;
            overscroll-behavior-y: contain !important;
        }
        section[data-testid="stMain"] > div,
        div[data-testid="stMainBlockContainer"],
        .block-container {
            min-width: 0 !important;
            height: auto !important;
            min-height: max-content !important;
            overflow: visible !important;
        }
        div[data-testid="stMainBlockContainer"], .block-container {
            width: 100% !important;
            max-width: none !important;
            box-sizing: border-box !important;
            padding-bottom: 4rem !important;
        }

        /* Expanded sidebar. Only style the sidebar while it is actually open. */
        section[data-testid="stSidebar"][aria-expanded="true"] {
            display: block !important;
            flex: 0 0 272px !important;
            width: 272px !important;
            min-width: 272px !important;
            max-width: 272px !important;
            height: 100vh !important;
            height: 100dvh !important;
            overflow: hidden !important;
            background: rgba(255,255,255,.97) !important;
            border-right: 1px solid #e6eee9 !important;
        }
        section[data-testid="stSidebar"][aria-expanded="true"] > div,
        section[data-testid="stSidebar"][aria-expanded="true"] div[data-testid="stSidebarContent"] {
            width: 272px !important;
            min-width: 272px !important;
            max-width: 272px !important;
            height: 100vh !important;
            height: 100dvh !important;
            overflow-x: hidden !important;
            overflow-y: auto !important;
            padding-left: .9rem !important;
            padding-right: .9rem !important;
            padding-bottom: 2rem !important;
        }

        /* IMPORTANT: do not keep a hidden 248/272px shell after collapse. */
        section[data-testid="stSidebar"][aria-expanded="false"] {
            display: none !important;
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            flex: 0 0 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            border: 0 !important;
        }

        /* Full-width content while the sidebar is collapsed. */
        body:has(section[data-testid="stSidebar"][aria-expanded="false"]) section[data-testid="stMain"] {
            width: 100vw !important;
            max-width: 100vw !important;
            flex: 1 1 100vw !important;
            margin-left: 0 !important;
        }
        body:has(section[data-testid="stSidebar"][aria-expanded="false"]) div[data-testid="stMainBlockContainer"],
        body:has(section[data-testid="stSidebar"][aria-expanded="false"]) .block-container {
            width: calc(100% - 1.5rem) !important;
            max-width: none !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        /* Reopen arrow: show it ONLY when the sidebar is collapsed.
           When the sidebar is open, Streamlit already shows its own collapse arrow,
           so hiding this control prevents the duplicate two-arrow UI. */
        div[data-testid="stSidebarCollapsedControl"] {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            pointer-events: none !important;
            position: fixed !important;
            top: .7rem !important;
            left: .7rem !important;
            z-index: 2147483647 !important;
            width: 44px !important;
            height: 44px !important;
            align-items: center !important;
            justify-content: center !important;
        }
        body:has(section[data-testid="stSidebar"][aria-expanded="false"]) div[data-testid="stSidebarCollapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
        }
        body:has(section[data-testid="stSidebar"][aria-expanded="true"]) div[data-testid="stSidebarCollapsedControl"] {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }
        div[data-testid="stSidebarCollapsedControl"] *,
        div[data-testid="stSidebarCollapsedControl"] button,
        button[data-testid="stSidebarCollapseButton"] {
            pointer-events: auto !important;
            cursor: pointer !important;
        }
        div[data-testid="stSidebarCollapsedControl"] button {
            width: 42px !important;
            height: 42px !important;
            min-width: 42px !important;
            min-height: 42px !important;
            border-radius: 12px !important;
        }

        /* Keep the sidebar brand centred and unclipped. */
        .sidebar-brand-center {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            overflow: visible !important;
            padding: .25rem .25rem 1rem !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
        }
        .sidebar-brand-center img {
            display: block !important;
            width: 78px !important;
            height: 78px !important;
            max-width: 78px !important;
            max-height: 78px !important;
            object-fit: contain !important;
            object-position: center center !important;
            margin: .1rem auto .45rem !important;
            flex: 0 0 auto !important;
        }
        .sidebar-brand-center strong,
        .sidebar-brand-center small {
            display: block !important;
            width: 100% !important;
            max-width: 100% !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: normal !important;
            word-break: normal !important;
            text-align: center !important;
        }
        .sidebar-brand-center strong { line-height: 1.2 !important; }

        .sidebar-profile { width: 100% !important; min-width: 0 !important; }
        .sidebar-profile > div:last-child { min-width: 0 !important; flex: 1 1 auto !important; }
        .sidebar-profile strong, .sidebar-profile small {
            display: block !important;
            width: 100% !important;
            max-width: 100% !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }

        @media (max-width: 1180px) {
            section[data-testid="stSidebar"][aria-expanded="true"],
            section[data-testid="stSidebar"][aria-expanded="true"] > div,
            section[data-testid="stSidebar"][aria-expanded="true"] div[data-testid="stSidebarContent"] {
                width: 248px !important;
                min-width: 248px !important;
                max-width: 248px !important;
                flex-basis: 248px !important;
            }
            section[data-testid="stMain"] { overflow-y: auto !important; }
        }
        

        /* FINAL RESULT-CARD POLISH: keep the four post-analysis panels visually grouped. */
        .result-detail-marker { height: 0 !important; margin: 0 !important; padding: 0 !important; overflow: hidden !important; }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.result-detail-marker) {
            border: 1px solid rgba(26,104,67,.13) !important;
            border-radius: 24px !important;
            background: rgba(255,255,255,.92) !important;
            box-shadow: 0 12px 28px rgba(35,83,58,.08) !important;
            overflow: hidden !important;
            min-height: 100% !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.result-detail-marker) > div {
            padding: 1.2rem 1.25rem 1.25rem !important;
            min-width: 0 !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.care-detail-card) {
            background: linear-gradient(145deg, rgba(247,253,249,.98), rgba(255,255,255,.96)) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.medicine-detail-card) {
            background: linear-gradient(145deg, rgba(255,251,248,.98), rgba(255,255,255,.96)) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.watering-detail-card) {
            background: linear-gradient(145deg, rgba(244,251,255,.98), rgba(255,255,255,.96)) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.weather-detail-card) {
            background: linear-gradient(145deg, rgba(247,250,255,.98), rgba(255,255,255,.96)) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.result-detail-marker) h3 {
            margin-top: 0 !important;
            margin-bottom: .8rem !important;
            color: #153b2d !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.result-detail-marker) p,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.result-detail-marker) div,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.result-detail-marker) span {
            overflow-wrap: anywhere !important;
            word-break: normal !important;
        }

        /* Uploader must never clip the selected filename, browse control or helper text. */
        div[data-testid="stFileUploader"],
        div[data-testid="stFileUploader"] > div,
        div[data-testid="stFileUploaderDropzone"] {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            box-sizing: border-box !important;
        }
        div[data-testid="stFileUploaderDropzone"] {
            border-radius: 20px !important;
            overflow: hidden !important;
            min-height: 150px !important;
            padding: 1rem !important;
        }
        div[data-testid="stFileUploader"] button {
            border-radius: 14px !important;
            max-width: 100% !important;
        }
        div[data-testid="stFileUploader"] small,
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] p {
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            word-break: break-word !important;
        }

        @media (max-width: 980px) {
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.result-detail-marker) > div {
                padding: 1rem !important;
            }
        }
</style>
        """,
        unsafe_allow_html=True,
    )


def page_intro(kicker: str, title: str, text: str):
    st.markdown(
        f"""
        <div class="page-intro-card">
          <div class="page-kicker">{html.escape(kicker)}</div>
          <h1>{html.escape(title)}</h1>
          <p>{html.escape(text)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def go_to(page_name: str):
    st.session_state.nav_page = page_name

st.set_page_config(
    page_title="GreenGuardian AI",
    page_icon=Image.open(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

if "user" not in st.session_state:
    st.session_state.user = None

if "last_record" not in st.session_state:
    st.session_state.last_record = None

if "auth_message" not in st.session_state:
    st.session_state.auth_message = None


def auth_page():
    inject_auth_theme()
    logo = _data_uri(LOGO_PATH, "image/png")
    auth_view = str(st.query_params.get("auth", "login")).strip().lower()

    left_space, auth_col, right_space = st.columns([1.1, 1.15, 1.1], gap="medium")

    with auth_col:
        st.markdown('<div class="auth-card-marker"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="auth-logo-wrap">
              <img class="auth-logo" src="{logo}" alt="GreenGuardian AI logo">
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Forgot Password is a separate auth page. It intentionally does not render
        # the Login/Create Account tabs so it feels like an independent page.
        if auth_view == "reset":
            st.markdown(
                """
                <div class="reset-card-kicker">Account Recovery</div>
                <div class="reset-page-title">Reset your password</div>
                <div class="reset-page-copy">Enter your registered email address and choose a new password.</div>
                """,
                unsafe_allow_html=True,
            )

            with st.form("reset_password_form", clear_on_submit=False, border=False):
                reset_email = st.text_input(
                    "Email address",
                    key="reset_email",
                    placeholder="you@example.com",
                    autocomplete="email",
                )
                new_password = st.text_input(
                    "New password",
                    type="password",
                    key="reset_new_password",
                    placeholder="Create a new password",
                    autocomplete="new-password",
                )
                confirm_new_password = st.text_input(
                    "Confirm new password",
                    type="password",
                    key="reset_confirm_password",
                    placeholder="Repeat your new password",
                    autocomplete="new-password",
                )
                st.markdown(
                    '<div class="register-note">Use at least 8 characters with one letter and one number.</div>',
                    unsafe_allow_html=True,
                )
                reset_clicked = st.form_submit_button(
                    "Reset Password  →",
                    type="primary",
                    use_container_width=True,
                )

            if reset_clicked:
                reset_errors = []
                if not reset_email.strip():
                    reset_errors.append("Email address is required.")
                reset_errors.extend(validate_password(new_password, confirm_new_password))

                if reset_errors:
                    for error in reset_errors:
                        st.error(error)
                else:
                    ok, message = reset_user_password(reset_email, new_password)
                    if ok:
                        st.session_state.auth_message = message
                        st.query_params.clear()
                        st.rerun()
                    else:
                        st.error(message)

            st.markdown(
                '<div class="reset-back"><a href="?" target="_self">← Back to Login</a></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="auth-secure">🛡️ <span>Your new password is stored securely as a salted hash.</span></div>',
                unsafe_allow_html=True,
            )
            return

        st.markdown(
            """
            <div class="auth-card-head">
              <h1>GreenGuardian AI</h1>
              <p>Smart Plant Disease Detection and Care Advisory System</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.auth_message:
            st.success(st.session_state.auth_message)
            st.session_state.auth_message = None

        login_tab, register_tab = st.tabs(["Login", "Create Account"])

        with login_tab:
            with st.form("login_form", clear_on_submit=False, border=False):
                email = st.text_input(
                    "Email address",
                    key="login_email",
                    placeholder="you@example.com",
                    autocomplete="email",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    key="login_password",
                    placeholder="Enter your password",
                    autocomplete="current-password",
                )

                st.markdown(
                    '<div class="forgot-password"><a href="?auth=reset" target="_self">Forgot password?</a></div>',
                    unsafe_allow_html=True,
                )

                login_clicked = st.form_submit_button(
                    "Login to GreenGuardian AI  →",
                    type="primary",
                    use_container_width=True,
                )

            if login_clicked:
                if not email.strip() or not password:
                    st.error("Email and password are required.")
                else:
                    account = login_user(email, password)
                    if account:
                        st.query_params.clear()
                        st.session_state.user = account
                        st.rerun()
                    else:
                        st.error("Incorrect email or password.")

            st.markdown(
                '<div class="auth-secure">🛡️ <span>Your data is securely stored and protected.</span></div>',
                unsafe_allow_html=True,
            )

        with register_tab:
            with st.form("register_form", clear_on_submit=False, border=False):
                name = st.text_input(
                    "Full name",
                    key="reg_name",
                    placeholder="Your full name",
                    autocomplete="name",
                )
                reg_email = st.text_input(
                    "Email address",
                    key="reg_email",
                    placeholder="you@example.com",
                    autocomplete="email",
                )
                reg_password = st.text_input(
                    "Password",
                    type="password",
                    key="reg_password",
                    placeholder="Create a password",
                    autocomplete="new-password",
                )
                confirm = st.text_input(
                    "Confirm password",
                    type="password",
                    key="reg_confirm",
                    placeholder="Repeat your password",
                    autocomplete="new-password",
                )

                st.markdown(
                    '<div class="register-note">Use at least 8 characters with one letter and one number.</div>',
                    unsafe_allow_html=True,
                )

                register_clicked = st.form_submit_button(
                    "Create GreenGuardian Account  →",
                    type="primary",
                    use_container_width=True,
                )

            if register_clicked:
                errors = validate_registration(name, reg_email, reg_password, confirm)
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    ok, message = register_user(name, reg_email, reg_password)
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)

            st.markdown(
                '<div class="auth-secure">🌿 <span>One account keeps your scans, reports and care guidance together.</span></div>',
                unsafe_allow_html=True,
            )


if st.session_state.user is None:
    auth_page()
    st.stop()

inject_app_theme()

user = st.session_state.user

with st.sidebar:
    sidebar_logo = _data_uri(LOGO_PATH, "image/png")
    st.markdown(
        f"""
        <div class="sidebar-brand-center">
          <img src="{sidebar_logo}" alt="GreenGuardian AI logo">
          <strong>GreenGuardian AI</strong>
          <small>Smart Plant Care</small>
        </div>
        <div class="sidebar-profile">
          <div class="profile-avatar">👤</div>
          <div>
            <strong>{html.escape(user['full_name'])}</strong>
            <small>{html.escape(user['email'])}</small>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if any_model_available():
        st.markdown('<div class="sidebar-model-status">✅ CNN model available</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sidebar-model-status">ℹ️ CNN training pending</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-nav-title">Navigation</div>', unsafe_allow_html=True)

    nav_items = [
        "Dashboard",
        "Plant Analysis",
        "Scan History",
        "Medicines",
        "Reports",
        "About Project",
    ]
    nav_labels = {
        "Dashboard": "🏠  Dashboard",
        "Plant Analysis": "🌿  Plant Analysis",
        "Scan History": "🕘  Scan History",
        "Medicines": "💊  Medicines",
        "Reports": "📊  Reports",
        "About Project": "ℹ️  About Project",
    }
    if "nav_page" not in st.session_state or st.session_state.nav_page not in nav_items:
        st.session_state.nav_page = "Dashboard"

    page = st.radio(
        "Navigation",
        nav_items,
        key="nav_page",
        label_visibility="collapsed",
        format_func=lambda item: nav_labels[item],
    )

    st.divider()

    if st.button("↪️  Logout", use_container_width=True):
        st.session_state.user = None
        st.session_state.last_record = None
        st.rerun()



def show_about_project():
    """Project-information page only. No changes to navigation or diagnosis logic."""
    about_assets = ASSETS / "about"
    image_sources = {
        name: _data_uri(about_assets / filename, "image/jpeg")
        for name, filename in {
            "HERO": "hero_leaf.jpg",
            "TOMATO": "tomato.jpg",
            "APPLE": "apple.jpg",
            "ROSE": "rose.jpg",
            "GRAPE": "grape.jpg",
            "CORN": "corn.jpg",
        }.items()
    }
    image_sources["LOGO"] = _data_uri(LOGO_PATH, "image/png")

    # All CSS is scoped to .gg-about, keeping every other screen unchanged.
    markup = r'''
<style>
.gg-about, .gg-about * {box-sizing:border-box;}
.gg-about {color:#172b43;font-family:inherit;width:100%;max-width:1536px;margin:0 auto;padding:0 0 24px;}
.gg-about .gg-hero {min-height:208px;position:relative;overflow:hidden;isolation:isolate;border:1px solid #dfefe9;border-radius:20px;background:linear-gradient(114deg,#eaf8ff 0%,#edfbf4 52%,#def5e7 100%);padding:34px 35px;box-shadow:0 10px 25px rgba(31,91,70,.07);display:flex;align-items:center;}
.gg-about .gg-hero-copy {position:relative;z-index:2;max-width:72%;}
.gg-about .gg-hero-heading {display:flex;align-items:center;gap:17px;}
.gg-about .gg-hero-logo {width:68px;height:68px;object-fit:contain;filter:drop-shadow(0 4px 6px #8cb4a3);}
.gg-about h1 {margin:0!important;color:#08713e!important;letter-spacing:-.04em;font-size:clamp(27px,3.15vw,44px)!important;line-height:1.14!important;font-weight:800!important;}
.gg-about h1 span{color:#162d43;}
.gg-about .gg-tagline {font-weight:750;font-size:clamp(13px,1.45vw,19px);margin:7px 0 8px 85px;color:#162b44;line-height:1.4;}
.gg-about .gg-description {font-size:clamp(12px,1.08vw,15px);line-height:1.6;color:#324e63;margin:0 0 0 85px;max-width:710px;}
.gg-about .gg-hero-photo {position:absolute;z-index:1;right:0;top:0;bottom:0;width:35%;background-image:url('__HERO__');background-position:center;background-size:cover;clip-path:ellipse(94% 110% at 100% 50%);}
.gg-about .gg-hero-photo:before {content:'';position:absolute;left:0;top:0;bottom:0;width:19%;background:linear-gradient(90deg,rgba(181,245,189,.45),transparent);}
.gg-about .gg-grid {display:grid;gap:14px;margin-top:14px;}
.gg-about .gg-grid.two {grid-template-columns:1fr 1.08fr;}
.gg-about .gg-grid.bottom {grid-template-columns:1fr 1fr;}
.gg-about .gg-card {min-width:0;background:rgba(255,255,255,.97);border:1px solid #dce8f1;border-radius:17px;padding:18px 19px;box-shadow:0 4px 18px rgba(38,76,95,.035);}
.gg-about .gg-card-title {display:flex;gap:13px;align-items:center;margin:0 0 13px;}
.gg-about .gg-card-title strong {font-weight:750;font-size:clamp(15px,1.48vw,20px);letter-spacing:-.02em;color:#14263e;line-height:1.3;}
.gg-about .gg-icon {width:44px;height:44px;border-radius:50%;background:#e9f3ff;display:inline-flex;align-items:center;justify-content:center;font-size:23px;flex:0 0 44px;}
.gg-about .gg-card p {font-size:clamp(12px,1.05vw,15px);line-height:1.65;margin:0;color:#21344a;}
.gg-about .gg-scope-sub{margin:-8px 0 11px 57px!important;color:#516479!important;font-size:13px!important;}
.gg-about .gg-plants {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px;}
.gg-about .gg-plant {text-align:center;min-width:0;}
.gg-about .gg-plant img {display:block;width:100%;height:86px;object-fit:cover;border-radius:12px;}
.gg-about .gg-plant b {font-size:12px;display:block;color:#1d2b40;margin-top:5px;}
.gg-about .gg-models {border-radius:12px;background:linear-gradient(120deg,#edfbed,#edfbf7);padding:15px 15px 10px;display:grid;grid-template-columns:1fr 1fr 1.25fr;}
.gg-about .gg-model-cell {min-width:0;text-align:center;padding:5px 10px;border-right:1px solid #a9e7d1;}
.gg-about .gg-model-cell:last-child {border:none;text-align:left;}
.gg-about .gg-model-icon {display:block;font-size:25px;margin-bottom:5px;}
.gg-about .gg-model-number {font-size:31px;line-height:1.05;color:#08733e;font-weight:850;}
.gg-about .gg-model-label {font-size:13px;font-weight:750;margin:4px 0 3px;display:block;}
.gg-about .gg-model-cell small {display:block;font-size:11px;line-height:1.5;color:#304c55;}
.gg-about .gg-model-cell p {font-size:11px;line-height:1.55;}
.gg-about .gg-accuracies {display:flex;flex-wrap:wrap;gap:6px;margin-top:11px;}
.gg-about .gg-accuracies span {background:#f0f9f4;border:1px solid #d5eee1;border-radius:20px;padding:4px 8px;font-size:10px;color:#185d3e;white-space:nowrap;font-weight:650;}
.gg-about .gg-features {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;}
.gg-about .gg-feature {border:1px solid #e0eaf5;border-radius:13px;background:#fbfdff;display:flex;align-items:flex-start;gap:8px;padding:11px 8px;min-height:85px;}
.gg-about .gg-feature-icon {flex:0 0 28px;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:19px;background:#eef8ff;}
.gg-about .gg-feature b {display:block;font-size:11px;margin:1px 0 4px;color:#1d314c;line-height:1.3;}
.gg-about .gg-feature span {display:block;font-size:10px;line-height:1.45;color:#41556c;}
.gg-about .gg-tech-list {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:7px;border-top:1px solid #e7eff5;margin-top:9px;padding-top:15px;}
.gg-about .gg-tech {min-width:0;text-align:center;}
.gg-about .gg-tech-symbol {display:block;font-size:30px;line-height:1.3;margin:0 0 4px;}
.gg-about .gg-tech b {display:block;font-size:11px;color:#1a2d46;}
.gg-about .gg-tech small {display:block;font-size:10px;line-height:1.5;color:#41586f;}
.gg-about .gg-notice {background:linear-gradient(105deg,#fff8e7,#fff7e8);border-radius:12px;padding:13px 15px;font-size:12px;line-height:1.68;color:#394958;}
.gg-about .gg-footnote {margin-top:9px!important;font-size:11px!important;color:#69788a!important;}
@media(max-width:1150px){.gg-about .gg-hero{padding:25px 22px;}.gg-about .gg-hero-logo{width:54px;height:54px;}.gg-about .gg-tagline,.gg-about .gg-description{margin-left:71px;}.gg-about .gg-hero-copy{max-width:78%;}.gg-about .gg-card{padding:16px 14px;}.gg-about .gg-feature{gap:5px;padding:8px 6px;}.gg-about .gg-feature-icon{width:22px;height:22px;flex-basis:22px;font-size:16px;}}
@media(max-width:900px){.gg-about .gg-grid.two,.gg-about .gg-grid.bottom{grid-template-columns:1fr;}.gg-about .gg-hero-photo{opacity:.35;width:48%;}.gg-about .gg-hero-copy{max-width:100%;}.gg-about .gg-features{grid-template-columns:repeat(3,minmax(0,1fr));}}
@media(max-width:550px){.gg-about .gg-hero{padding:23px 17px;min-height:210px;}.gg-about .gg-hero-heading{gap:8px;}.gg-about .gg-hero-logo{width:44px;height:44px;}.gg-about .gg-tagline,.gg-about .gg-description{margin-left:0;}.gg-about .gg-hero-photo{opacity:.13;width:90%;}.gg-about .gg-card-title{gap:7px;}.gg-about .gg-icon{width:37px;height:37px;flex-basis:37px;}.gg-about .gg-plants{gap:5px;}.gg-about .gg-plant img{height:56px;border-radius:7px;}.gg-about .gg-plant b{font-size:10px;}.gg-about .gg-scope-sub{margin-left:0!important;}.gg-about .gg-features{grid-template-columns:1fr 1fr;}.gg-about .gg-models{padding:10px 5px;}.gg-about .gg-model-cell{padding:5px;}.gg-about .gg-model-cell p,.gg-about .gg-model-cell small{font-size:10px;}.gg-about .gg-tech-symbol{font-size:22px;}.gg-about .gg-tech b{font-size:9px;}.gg-about .gg-tech small{font-size:9px;}}
</style>
<div class="gg-about">
  <section class="gg-hero">
    <div class="gg-hero-copy">
      <div class="gg-hero-heading"><img class="gg-hero-logo" src="__LOGO__" alt="GreenGuardian AI logo" /><h1><span>About</span> GreenGuardian AI</h1></div>
      <div class="gg-tagline">Smart Plant Disease Detection and Care Advisory System</div>
      <div class="gg-description">An AI-powered system to detect plant leaf diseases, provide smart care recommendations and support healthy crop growth.</div>
    </div><div class="gg-hero-photo" aria-hidden="true"></div>
  </section>
  <div class="gg-grid two">
    <section class="gg-card">
      <div class="gg-card-title"><div class="gg-icon">📄</div><strong>Project Overview</strong></div>
      <p>GreenGuardian AI is a deep learning-based plant disease detection and care advisory system. It helps identify supported leaf diseases in Tomato, Apple, Rose, Grape and Corn using plant-specific trained CNN models and provides personalized recommendations for treatment, smart watering and weather-risk guidance.</p>
    </section>
    <section class="gg-card">
      <div class="gg-card-title"><div class="gg-icon">🎯</div><strong>Project Scope</strong></div>
      <p class="gg-scope-sub">This project focuses on 5 commonly grown plants:</p>
      <div class="gg-plants">
        <div class="gg-plant"><img src="__TOMATO__" alt="Tomato"/><b>Tomato</b></div>
        <div class="gg-plant"><img src="__APPLE__" alt="Apple"/><b>Apple</b></div>
        <div class="gg-plant"><img src="__ROSE__" alt="Rose"/><b>Rose</b></div>
        <div class="gg-plant"><img src="__GRAPE__" alt="Grape"/><b>Grape</b></div>
        <div class="gg-plant"><img src="__CORN__" alt="Corn"/><b>Corn</b></div>
      </div>
    </section>
  </div>
  <div class="gg-grid two">
    <section class="gg-card">
      <div class="gg-card-title"><div class="gg-icon">🔗</div><strong>AI Model Information</strong></div>
      <div class="gg-models">
        <div class="gg-model-cell"><span class="gg-model-icon">🍃</span><div class="gg-model-number">5</div><b class="gg-model-label">Plant Types</b><small>Tomato, Apple, Rose, Grape, Corn</small></div>
        <div class="gg-model-cell"><span class="gg-model-icon">⚙️</span><div class="gg-model-number">38</div><b class="gg-model-label">CNN Classes</b><small>Supported disease &amp; healthy categories</small></div>
        <div class="gg-model-cell"><span class="gg-model-icon">📊</span><b class="gg-model-label">Trained CNN Models</b><p>Architecture: MobileNetV2<br/>Framework: TensorFlow<br/>Input: 224 × 224 pixels</p></div>
      </div>
      <div class="gg-accuracies" aria-label="CNN test accuracies">
        <span>Tomato 92.53%</span><span>Apple 91.22%</span><span>Rose 95.05%</span><span>Grape 98.56%</span><span>Corn 94.16%</span>
      </div>
    </section>
    <section class="gg-card">
      <div class="gg-card-title"><div class="gg-icon">⭐</div><strong>Key Features</strong></div>
      <div class="gg-features">
        <div class="gg-feature"><div class="gg-feature-icon">🔎</div><div><b>Disease Detection</b><span>Identify supported leaf diseases using CNN models</span></div></div>
        <div class="gg-feature"><div class="gg-feature-icon">💧</div><div><b>Smart Watering</b><span>Provide watering recommendations</span></div></div>
        <div class="gg-feature"><div class="gg-feature-icon">💊</div><div><b>Medicine Suggestion</b><span>Care options with indicative costs</span></div></div>
        <div class="gg-feature"><div class="gg-feature-icon">☁️</div><div><b>Weather Risk Analysis</b><span>Assess supplied environmental conditions</span></div></div>
        <div class="gg-feature"><div class="gg-feature-icon">📗</div><div><b>PDF Report</b><span>Generate downloadable analysis reports</span></div></div>
        <div class="gg-feature"><div class="gg-feature-icon">📋</div><div><b>History &amp; CSV Export</b><span>Save scans and export records</span></div></div>
      </div>
    </section>
  </div>
  <div class="gg-grid bottom">
    <section class="gg-card">
      <div class="gg-card-title"><div class="gg-icon">⚙️</div><strong>Technology Stack</strong></div>
      <div class="gg-tech-list">
        <div class="gg-tech"><span class="gg-tech-symbol">🐍</span><b>Python</b><small>Programming</small></div>
        <div class="gg-tech"><span class="gg-tech-symbol">🖥️</span><b>Streamlit</b><small>Web framework</small></div>
        <div class="gg-tech"><span class="gg-tech-symbol">🔶</span><b>TensorFlow</b><small>Deep learning</small></div>
        <div class="gg-tech"><span class="gg-tech-symbol">🧠</span><b>MobileNetV2</b><small>CNN backbone</small></div>
        <div class="gg-tech"><span class="gg-tech-symbol">🗄️</span><b>SQLite</b><small>Database</small></div>
      </div>
    </section>
    <section class="gg-card">
      <div class="gg-card-title"><div class="gg-icon">🛡️</div><strong>Responsible Use</strong></div>
      <div class="gg-notice">GreenGuardian AI is a prototype advisory tool and is not a replacement for professional horticultural diagnosis. Predictions may be incorrect or uncertain, especially for unfamiliar images. Seek qualified advice before important treatment decisions.</div>
      <p class="gg-footnote">Scan history is stored locally in SQLite; account passwords are hashed.</p>
    </section>
  </div>
</div>
'''
    for name, uri in image_sources.items():
        markup = markup.replace("__" + name + "__", uri)
    st.markdown(markup, unsafe_allow_html=True)


if page == "Dashboard":
    st.markdown('<div class="page-marker dashboard-page-marker"></div>', unsafe_allow_html=True)
    total, healthy, high_alerts, latest_problem = database_stats(user["id"])
    today_label = datetime.now().strftime("%A, %d %B %Y")

    st.markdown(
        f"""
        <div class="dashboard-hero">
          <div class="page-kicker">Welcome back · {html.escape(today_label)}</div>
          <h1>Let’s keep your plants healthy and thriving.</h1>
          <p>Good to see you, {html.escape(user['full_name'])}. Use GreenGuardian AI to analyse leaves,
          review care guidance and keep your plant-health records organised.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="gg-stat-grid">
          <div class="gg-stat"><div class="icon">🍃</div><div class="label">TOTAL SCANS</div><div class="value">{total}</div><div class="detail">Saved to your account</div></div>
          <div class="gg-stat"><div class="icon">✅</div><div class="label">HEALTHY RESULTS</div><div class="value">{healthy}</div><div class="detail">Healthy Plant diagnoses</div></div>
          <div class="gg-stat"><div class="icon">⚠️</div><div class="label">HIGH-SEVERITY ALERTS</div><div class="value">{high_alerts}</div><div class="detail">Needs closer attention</div></div>
          <div class="gg-stat"><div class="icon">📄</div><div class="label">LATEST RESULT</div><div class="value" style="font-size:1.18rem;padding-top:.35rem;">{html.escape(str(latest_problem))}</div><div class="detail">Most recent saved diagnosis</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>Quick Actions</div>", unsafe_allow_html=True)
    q1, q2, q3 = st.columns(3)
    with q1:
        with st.container(border=True):
            st.subheader("🍃 Plant Analysis")
            st.caption("Upload a leaf photo or use your camera to run disease analysis.")
            st.button("Open Plant Analysis", use_container_width=True, on_click=go_to, args=("Plant Analysis",))
    with q2:
        with st.container(border=True):
            st.subheader("🕘 Scan History")
            st.caption("Review, search and export analyses saved under your account.")
            st.button("View Scan History", use_container_width=True, on_click=go_to, args=("Scan History",))
    with q3:
        with st.container(border=True):
            st.subheader("📊 Reports")
            st.caption("Download your CSV history and the latest PDF plant report.")
            st.button("Open Reports", use_container_width=True, on_click=go_to, args=("Reports",))

    recent = history_df(user["id"])
    left_dash, right_dash = st.columns([1.35, 1])
    with left_dash:
        st.markdown("<div class='section-title'>Recent Scan History</div>", unsafe_allow_html=True)
        with st.container(border=True):
            if recent.empty:
                st.info("No scans yet. Start with Plant Analysis to build your history.")
            else:
                recent_view = recent[["Date", "Plant", "Problem", "Severity", "Confidence"]].head(5)
                st.dataframe(recent_view, use_container_width=True, hide_index=True, height=235)

    with right_dash:
        st.markdown("<div class='section-title'>Care Focus</div>", unsafe_allow_html=True)
        with st.container(border=True):
            latest_care = st.session_state.get("last_record")

            if latest_care:
                latest_plant = html.escape(str(latest_care.get("plant_type", "Plant")))
                latest_problem = html.escape(str(latest_care.get("detected_problem", "No recent condition")))
                latest_severity = html.escape(str(latest_care.get("severity", "—")))
                latest_water_status = html.escape(str(latest_care.get("smart_watering_status", "No watering status")))
                latest_weather = html.escape(str(latest_care.get("weather_alert", "No weather-risk message")))

                st.subheader("💧 Latest smart-care summary")
                st.markdown(
                    f"""
                    **Plant:** {latest_plant}  
                    **Condition:** {latest_problem}  
                    **Severity:** {latest_severity}
                    """
                )
                st.success(f"Smart Watering: {latest_water_status}")
                st.info(f"Weather Risk: {latest_weather}")
                st.caption("This section updates automatically after each successful plant analysis.")
            else:
                st.subheader("💧 Smart care, in context")
                st.write(
                    "Watering, temperature, humidity, disease and plant-stress guidance are "
                    "combined after each analysis so you can act on the result, not just read a label."
                )
                st.info("For best results, upload a sharp, well-lit close-up of the affected leaf.")



elif page == "Plant Analysis":
    st.markdown('<div class="page-marker plant-page-marker"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="plant-topbar plant-topbar-polished">
          <div class="plant-title-wrap">
            <div class="plant-title-icon">🌱</div>
            <div>
              <h1>Plant Analysis</h1>
              <p>Upload a leaf image, add the plant conditions and run a focused disease analysis.</p>
            </div>
          </div>
          <div class="ai-health-badge">
            <div class="badge-icon">🌿</div>
            <div>
              <strong>AI-Powered Plant Health Detection</strong>
              <small>Clear input. Focused diagnosis. Practical care guidance.</small>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    image = None
    uploaded = None

    left, right = st.columns([1, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown('<div class="plant-primary-card plant-selection-card"></div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="card-heading plant-section-heading">
                  <div class="card-heading-icon">🌱</div>
                  <div>
                    <h3>Plant Selection</h3>
                    <p>Choose the plant and add optional growing conditions for a better care recommendation.</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            plant = st.selectbox("🌿  Plant type", PLANTS)

            if model_available(plant):
                st.markdown(
                    f'<div class="plant-model-status ready"><span>●</span><strong>{html.escape(plant)}</strong> CNN model ready</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="plant-model-status pending"><span>●</span><strong>{html.escape(plant)}</strong> CNN training pending</div>',
                    unsafe_allow_html=True,
                )

            with st.expander("Supported disease & stress conditions", expanded=False):
                st.caption(", ".join(plant_diseases(plant)))

            st.markdown('<div class="soft-divider compact"></div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="smart-care-head">
                  <div class="mini-icon">⚙️</div>
                  <div>
                    <strong>Smart Care Details <span class="optional-tag">Optional</span></strong>
                    <span>Current conditions help personalise watering and risk guidance.</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            a, b = st.columns(2, gap="medium")
            temp = a.number_input(
                "🌡️  Temperature (°C)",
                min_value=-5.0,
                max_value=50.0,
                value=25.0,
                step=1.0,
            )
            humidity = b.number_input(
                "💧  Air Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=1.0,
            )

            c, d = st.columns(2, gap="medium")
            soil = c.number_input(
                "🌱  Soil Moisture (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
                help="Enter the current soil-moisture percentage. This is used by Smart Watering and stress detection.",
            )
            last_days = d.number_input(
                "📅  Days Since Last Watering",
                min_value=0,
                max_value=30,
                value=3,
                step=1,
                help="Enter how many days have passed since the plant was last watered.",
            )

    with right:
        with st.container(border=True):
            st.markdown('<div class="plant-primary-card leaf-image-card"></div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="card-heading plant-section-heading">
                  <div class="card-heading-icon blue">📷</div>
                  <div>
                    <h3>Leaf Image</h3>
                    <p>Use one clear, close-up leaf photo. Keep the leaf centred and well lit.</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            source = st.radio(
                "Image source",
                ["Upload Image", "Use Camera"],
                horizontal=True,
            )

            if source == "Upload Image":
                uploaded = st.file_uploader(
                    "Upload JPG, JPEG or PNG",
                    type=["jpg", "jpeg", "png"],
                    help="For best results use a sharp, well-lit close-up image.",
                )
                st.markdown(
                    '<div class="upload-guidance">JPG, JPEG or PNG · Max 200 MB · Use a sharp close-up of one leaf</div>',
                    unsafe_allow_html=True,
                )
            else:
                uploaded = st.camera_input("Capture leaf image")
                st.markdown(
                    '<div class="upload-guidance">Hold the camera steady and fill most of the frame with the affected leaf.</div>',
                    unsafe_allow_html=True,
                )

            if uploaded is not None:
                try:
                    image = _open_uploaded_image(uploaded)
                    st.image(image, caption="Selected leaf image", use_column_width=True)
                except (UnidentifiedImageError, OSError, ValueError):
                    image = None
                    st.error("This file could not be decoded as an image. Please choose a JPG, JPEG or PNG file, or take a new camera photo.")

    st.markdown('<div class="analyse-bar"></div>', unsafe_allow_html=True)
    analyse = st.button(
        "✨  Analyse Plant  →",
        type="primary",
        use_container_width=True,
    )

    if analyse:
        errors = validate_analysis(
            plant,
            image,
            float(temp),
            float(humidity),
            float(soil),
            int(last_days),
        )

        if errors:
            for error in errors:
                st.error(error)
        else:
            likeness = leaf_likeness(image)

            if likeness < 0.32:
                st.session_state.last_record = None
                st.error("This image does not strongly resemble plant foliage.")
                st.info(
                    "Please upload a clearer close-up image of Tomato, Apple, Rose, "
                    "Grape or Corn foliage."
                )
            else:
                problem, confidence, method = predict(
                    image,
                    getattr(uploaded, "name", "camera_photo.jpg"),
                    plant,
                    float(humidity),
                    float(temp),
                    float(soil),
                )

                # Proposal-complete hybrid assessment: keep supported disease
                # diagnoses, but screen Healthy/Uncertain results for Nutrient
                # Deficiency, Overwatering Stress and General Plant Stress using
                # the supplied environmental values plus transparent image cues.
                problem, confidence, method = resolve_condition(
                    image=image,
                    base_problem=problem,
                    base_confidence=confidence,
                    base_method=method,
                    plant=plant,
                    soil=float(soil),
                    last_days=int(last_days),
                    humidity=float(humidity),
                    temp=float(temp),
                )

                if problem == "__UNCERTAIN__":
                    st.session_state.last_record = None
                    st.error("Unable to diagnose reliably from this image.")
                    st.info(
                        "Try a clearer image. Unsupported or low-confidence patterns "
                        "are not forced into a false Healthy result."
                    )
                else:
                    info = advisory(problem, plant)
                    level = severity(problem, confidence, float(humidity), float(temp), float(soil))
                    water_status, water_reason, ideal = smart_watering(
                        plant, problem, float(soil), int(last_days), float(humidity), float(temp)
                    )
                    weather = weather_risk(problem, float(humidity), float(temp))

                    record = {
                        "scan_date": datetime.now().strftime("%d %B %Y, %H:%M"),
                        "plant_type": plant,
                        "image_source": source,
                        "temperature": float(temp),
                        "humidity": float(humidity),
                        "soil_moisture": float(soil),
                        "last_watered_days": int(last_days),
                        "detected_problem": problem,
                        "severity": level,
                        "confidence": confidence,
                        "symptoms": info["symptoms"],
                        "treatment": info["treatment"],
                        "medicine_name": info["medicine"],
                        "medicine_cost": info["cost"],
                        "medicine_image": info["medicine_image"],
                        "watering_advice": info["watering"],
                        "sunlight_advice": info["sunlight"],
                        "smart_watering_status": water_status,
                        "smart_watering_reason": water_reason,
                        "weather_alert": weather,
                        "detection_method": method,
                    }
                    save_scan(user["id"], record)
                    st.session_state.last_record = record
                    st.success("Analysis completed and saved.")

    record = st.session_state.last_record

    if record:
        safe_problem = html.escape(str(record["detected_problem"]))
        safe_severity = html.escape(str(record["severity"]))
        safe_method = html.escape(str(record["detection_method"]))
        safe_date = html.escape(str(record["scan_date"]))
        st.markdown(
            f"""
            <div class="results-card">
              <div class="results-head">
                <div><h3>📊 Analysis Results</h3><span>AI diagnosis and care summary</span></div>
                <small>Analysed on {safe_date}</small>
              </div>
              <div class="results-grid">
                <div class="result-tile problem"><div class="r-label">Detected Problem</div><div class="r-value">{safe_problem}</div></div>
                <div class="result-tile severity"><div class="r-label">Severity</div><div class="r-value">{safe_severity}</div></div>
                <div class="result-tile confidence"><div class="r-label">Confidence</div><div class="r-value">{record['confidence']:.1f}%</div></div>
                <div class="result-tile method"><div class="r-label">Detection Method</div><div class="r-value">{safe_method}</div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        care_col, medicine_col = st.columns(2, gap="large")
        with care_col:
            with st.container(border=True):
                st.markdown('<div class="result-detail-marker care-detail-card"></div>', unsafe_allow_html=True)
                st.subheader("🌿 Care Advisory")
                st.write("**Symptoms**")
                st.write(record["symptoms"])
                st.write("**Treatment**")
                st.write(record["treatment"])
                st.write("**Watering Advice**")
                st.write(record["watering_advice"])
                st.write("**Sunlight Advice**")
                st.write(record["sunlight_advice"])

        with medicine_col:
            with st.container(border=True):
                st.markdown('<div class="result-detail-marker medicine-detail-card"></div>', unsafe_allow_html=True)
                st.subheader("💊 Medicine Suggestion")
                if record["medicine_image"]:
                    medicine_path = Path(record["medicine_image"])
                    if medicine_path.exists():
                        st.image(str(medicine_path), use_column_width=True)
                    st.write(f"**Medicine:** {record['medicine_name']}")
                    st.write(f"**Approximate cost:** {record['medicine_cost']}")
                    st.caption("Illustrative project image; not a branded product endorsement.")
                else:
                    st.success("No medicine required / suggested.")
                    st.write("Continue appropriate care and monitoring.")
                    st.write(f"Approximate cost: {record['medicine_cost']}")

        water_col, weather_col = st.columns(2, gap="large")
        with water_col:
            with st.container(border=True):
                st.markdown('<div class="result-detail-marker watering-detail-card"></div>', unsafe_allow_html=True)
                st.subheader("💧 Smart Watering")
                st.write(f"**Status:** {record['smart_watering_status']}")
                st.write(record["smart_watering_reason"])
                _, _, ideal = smart_watering(
                    record["plant_type"], record["detected_problem"], record["soil_moisture"],
                    record["last_watered_days"], record["humidity"], record["temperature"]
                )
                st.write(f"Current soil moisture: {record['soil_moisture']:.0f}%")
                st.write(f"Preferred range: {ideal}")

        with weather_col:
            with st.container(border=True):
                st.markdown('<div class="result-detail-marker weather-detail-card"></div>', unsafe_allow_html=True)
                st.subheader("🌦️ Weather Risk Advisory")
                st.info(record["weather_alert"])
                st.write(f"Temperature: {record['temperature']:.0f}°C")
                st.write(f"Humidity: {record['humidity']:.0f}%")

        st.download_button(
            "Download PDF Report",
            make_pdf(user["full_name"], record),
            file_name=f"GreenGuardian_AI_{record['plant_type']}_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

elif page == "Scan History":
    st.markdown('<div class="page-marker scan-page-marker"></div>', unsafe_allow_html=True)
    page_intro(
        "Your plant records",
        "Scan History",
        "Search and filter analyses saved under your registered account.",
    )

    df = history_df(user["id"])

    with st.container(border=True):
        if df.empty:
            st.info("No scan history yet.")
        else:
            q1, q2 = st.columns([2, 1])

            query = q1.text_input(
                "Search",
                placeholder="e.g. Tomato, mildew, copper...",
            )
            severity_filter = q2.selectbox(
                "Severity",
                ["All", "Low", "Medium", "High"],
            )

            view = df.copy()

            if query.strip():
                mask = view.astype(str).apply(
                    lambda col: col.str.contains(
                        query.strip(),
                        case=False,
                        na=False,
                    )
                ).any(axis=1)
                view = view[mask]

            if severity_filter != "All":
                view = view[view["Severity"] == severity_filter]

            st.dataframe(
                view,
                use_container_width=True,
                hide_index=True,
                height=430,
            )

            st.caption(f"Showing {len(view)} of {len(df)} saved records.")

            st.download_button(
                "Download Filtered CSV",
                csv_bytes(view, user["full_name"]),
                file_name=(
                    f"GreenGuardian_AI_"
                    f"{user['full_name'].replace(' ', '_')}_History.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )


elif page == "Medicines":
    st.markdown('<div class="page-marker medicines-page-marker"></div>', unsafe_allow_html=True)
    page_intro(
        "Treatment references",
        "Medicine Library",
        "Illustrative treatment references used by the care-advisory catalogue.",
    )

    with st.container(border=True):
        st.info(
            "Medicine images are illustrative project graphics. "
            "Cost ranges are approximate and not product endorsements."
        )

    meds = []
    seen = set()

    for plant_name in PLANTS:
        for disease in plant_diseases(plant_name):
            info = advisory(disease, plant_name)

            if info["medicine_image"] and info["medicine"] not in seen:
                seen.add(info["medicine"])
                meds.append(
                    (
                        info["medicine"],
                        info["cost"],
                        info["medicine_image"],
                    )
                )

    for i in range(0, len(meds), 3):
        cols = st.columns(3)

        for col, (name, cost, img) in zip(cols, meds[i:i + 3]):
            with col:
                with st.container(border=True):
                    p = Path(img)

                    if p.exists():
                        st.image(str(p), use_column_width=True)

                    st.write(f"**{name}**")
                    st.write(cost)
                    st.caption("Illustrative medicine reference")


elif page == "Reports":
    st.markdown('<div class="page-marker reports-page-marker"></div>', unsafe_allow_html=True)
    page_intro(
        "Data & documents",
        "Reports & Export",
        "Download user-specific history and the latest plant analysis report.",
    )

    df = history_df(user["id"])
    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.subheader("CSV History Export")

            if df.empty:
                st.info("Nothing to export. Complete a plant analysis first.")
            else:
                st.write(f"{len(df)} saved record(s) are ready to export.")

                st.download_button(
                    "Download Complete Scan History CSV",
                    csv_bytes(df, user["full_name"]),
                    file_name=(
                        f"GreenGuardian_AI_"
                        f"{user['full_name'].replace(' ', '_')}_History.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True,
                )

    with right:
        with st.container(border=True):
            st.subheader("Latest PDF Report")

            if st.session_state.last_record:
                latest = st.session_state.last_record

                st.write(f"**Plant:** {latest['plant_type']}")
                st.write(f"**Diagnosis:** {latest['detected_problem']}")
                st.write(f"**Confidence:** {latest['confidence']}%")

                st.download_button(
                    "Download Latest PDF Report",
                    make_pdf(user["full_name"], latest),
                    file_name=(
                        f"GreenGuardian_AI_{latest['plant_type']}_Report.pdf"
                    ),
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.info("No active report. Run Plant Analysis first.")


else:
    st.markdown('<div class="page-marker about-page-marker"></div>', unsafe_allow_html=True)
    show_about_project()
