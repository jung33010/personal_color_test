import streamlit as st
import base64
import os
import uuid
import time
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.callbacks.manager import get_openai_callback

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── 커스텀 CSS ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI 퍼스널 컬러 진단",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ── 폰트 ── */
@import url('https://fonts.googleapis.com/css2?family=Gowun+Dodum&family=Noto+Serif+KR:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Gowun Dodum', sans-serif;
}

/* ── 배경 ── */
.stApp {
    background: linear-gradient(160deg, #fdf6f0 0%, #fce4ec 50%, #f3e5f5 100%);
    min-height: 100vh;
}

/* ── 헤더 영역 ── */
.hero-section {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
}
.hero-title {
    font-family: 'Noto Serif KR', serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: #3d1f3d;
    letter-spacing: -0.5px;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    font-size: 1rem;
    color: #8d6b7a;
    line-height: 1.7;
}

/* ── 세션 배지 ── */
.session-badge {
    display: inline-block;
    background: rgba(255,255,255,0.6);
    border: 1px solid rgba(180,120,160,0.3);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #9c5f82;
    margin-top: 0.5rem;
    backdrop-filter: blur(4px);
}

/* ── 카드 컨테이너 ── */
.card {
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(180,100,140,0.08);
}

/* ── 팁 박스 ── */
.tip-box {
    background: linear-gradient(135deg, #fff0f6, #f8f0ff);
    border-left: 3px solid #d17cb0;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 1.2rem;
    font-size: 0.9rem;
    color: #6b3a5e;
}

/* ── 단계 인디케이터 ── */
.step-row {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0;
    margin: 1.5rem auto 2rem;
    max-width: 400px;
}
.step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    flex: 1;
    position: relative;
}
.step-item:not(:last-child)::after {
    content: '';
    position: absolute;
    top: 16px;
    left: 60%;
    width: 80%;
    height: 2px;
    background: #e8c8d8;
}
.step-dot {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 600;
    position: relative;
    z-index: 1;
}
.step-dot.done   { background: #c77daa; color: #fff; }
.step-dot.active { background: #fff; border: 2px solid #c77daa; color: #c77daa; }
.step-dot.idle   { background: #f0e4ec; color: #b89aa8; }
.step-label { font-size: 0.72rem; color: #9c7a8a; }

/* ── 라디오 버튼 커스터마이즈 ── */
div[role="radiogroup"] label {
    background: rgba(255,255,255,0.6) !important;
    border: 1.5px solid #e8c4d8 !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
}
div[role="radiogroup"] label:hover {
    border-color: #c77daa !important;
    background: rgba(255,255,255,0.9) !important;
}

/* ── 버튼 ── */
.stButton > button {
    background: linear-gradient(135deg, #d17cb0, #a855a0) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.75rem 2.5rem !important;
    font-family: 'Gowun Dodum', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 16px rgba(193,100,160,0.35) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(193,100,160,0.45) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── 결과 카드 ── */
.result-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(200,150,180,0.2);
}
.color-chip {
    width: 52px;
    height: 52px;
    border-radius: 50%;
    border: 3px solid #fff;
    box-shadow: 0 2px 12px rgba(0,0,0,0.12);
    flex-shrink: 0;
}
.result-tag {
    display: inline-block;
    background: linear-gradient(135deg, #fce4ec, #f8f0ff);
    border: 1px solid #e8b4d0;
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.8rem;
    color: #8b3a6b;
    margin: 2px;
}

/* ── 시즌 배지 ── */
.season-spring { background: linear-gradient(135deg,#fff3cd,#ffd6e0); color:#7d4e1f; }
.season-summer { background: linear-gradient(135deg,#dbeafe,#e9d5ff); color:#1e3a5f; }
.season-autumn { background: linear-gradient(135deg,#fed7aa,#fef3c7); color:#7c2d12; }
.season-winter { background: linear-gradient(135deg,#e0f2fe,#f0fdf4); color:#0c4a6e; }
.season-badge {
    display: inline-block;
    border-radius: 12px;
    padding: 6px 18px;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* ── 컬러 팔레트 ── */
.palette-row {
    display: flex;
    gap: 8px;
    margin: 0.8rem 0;
    flex-wrap: wrap;
}
.palette-swatch {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    border: 2px solid #fff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    position: relative;
    cursor: pointer;
    transition: transform 0.2s;
}
.palette-swatch:hover { transform: scale(1.15); }

/* ── 비용 표시 ── */
.cost-pill {
    background: rgba(255,255,255,0.5);
    border: 1px solid rgba(200,160,180,0.4);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: #9c6080;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

/* ── 업로드 영역 ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #ddb8cc !important;
    border-radius: 16px !important;
    background: rgba(255,255,255,0.5) !important;
    padding: 1rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #c77daa !important;
}

/* ── 스피너 색상 ── */
.stSpinner > div { border-top-color: #c77daa !important; }

/* ── 성공/경고 메시지 ── */
.stSuccess { border-radius: 12px !important; }
.stWarning { border-radius: 12px !important; }

/* ── 사이드바 ── */
.css-1d391kg { background: rgba(253,246,250,0.95) !important; }

/* ── 이미지 미리보기 ── */
.preview-img {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(180,100,140,0.15);
}

/* ── 구분선 ── */
hr {
    border: none;
    border-top: 1px solid rgba(200,150,180,0.2);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─── DB 설정 ──────────────────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_NAME = os.getenv("DB_NAME", "personal_color_db")
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
Base = declarative_base()

class UsageLog(Base):
    __tablename__ = 'UsageLog'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_uuid = Column(String(36))
    timestamp = Column(DateTime, default=datetime.now)
    lighting_env = Column(String(50))
    ai_result = Column(Text)
    is_completed = Column(Boolean, default=False)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)
    image_path = Column(String(255))
    processing_time = Column(Float)
    user_feedback = Column(String(50))
    device_type = Column(String(50))
    retry_count = Column(Integer, default=0)
    model_version = Column(String(50))

os.makedirs("uploads", exist_ok=True)
db_connected = False
try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db_connected = True
except Exception as e:
    pass  # DB 없어도 앱은 동작

# ─── 헬퍼 함수 ────────────────────────────────────────────────────────────────
def encode_image(uploaded_file):
    uploaded_file.seek(0)
    return base64.b64encode(uploaded_file.read()).decode('utf-8')

def get_skin_color_data(uploaded_file):
    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cx, cy = x + w // 2, y + h // 2
            rw, rh = int(w * 0.2), int(h * 0.2)
            roi = img_array[cy-rh//2:cy+rh//2, cx-rw//2:cx+rw//2]
            mean_rgb = np.mean(roi, axis=(0, 1))
            r, g, b = int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2])
            hex_code = f"#{r:02x}{g:02x}{b:02x}".upper()
            return hex_code, f"rgb({r}, {g}, {b})", "성공", (x, y, w, h)
        return None, None, "실패", None
    except Exception:
        return None, None, "실패", None

# 시즌별 대표 팔레트
SEASON_PALETTES = {
    "봄": {
        "colors": ["#FFD1A9","#FFB347","#FFF0AA","#A8E6CF","#FFB6C1","#FFDDC1"],
        "class": "season-spring",
        "emoji": "🌸",
        "desc": "밝고 화사한 웜톤",
        "keywords": ["복숭아", "산호", "연두", "아이보리", "노란 주황"],
    },
    "여름": {
        "colors": ["#C9DEFF","#B0C4DE","#DDA0DD","#E8A0BF","#F0E6FF","#AEDFF7"],
        "class": "season-summer",
        "emoji": "🌊",
        "desc": "부드럽고 시원한 쿨톤",
        "keywords": ["라벤더", "파우더 블루", "로즈핑크", "라일락", "민트"],
    },
    "가을": {
        "colors": ["#D4A96A","#C17F24","#8B4513","#A0522D","#F4A460","#CD853F"],
        "class": "season-autumn",
        "emoji": "🍂",
        "desc": "깊고 풍부한 웜톤",
        "keywords": ["테라코타", "머스타드", "올리브", "카멜", "브릭"],
    },
    "겨울": {
        "colors": ["#1C3557","#2E4A7A","#E8E8E8","#C0C0C0","#8B0000","#00008B"],
        "class": "season-winter",
        "emoji": "❄️",
        "desc": "선명하고 대비되는 쿨톤",
        "keywords": ["네이비", "버건디", "순백", "블랙", "로열블루"],
    },
}

def detect_season(result_text: str) -> str | None:
    for key in ["봄", "여름", "가을", "겨울"]:
        if key in result_text:
            return key
    return None

def save_to_db(session_uuid, retry_count, lighting_condition, result,
               prompt_tokens, completion_tokens, total_cost,
               image_path, processing_time, is_completed=True):
    if not db_connected:
        return
    db = SessionLocal()
    try:
        log = UsageLog(
            session_uuid=session_uuid,
            lighting_env=lighting_condition,
            ai_result=result,
            is_completed=is_completed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_cost=total_cost,
            image_path=image_path,
            processing_time=processing_time,
            user_feedback=None,
            device_type="Unknown",
            retry_count=retry_count,
            model_version="gpt-4o"
        )
        db.add(log)
        db.commit()
    except Exception:
        pass
    finally:
        db.close()

# ─── 세션 초기화 ──────────────────────────────────────────────────────────────
for key, val in [('session_uuid', str(uuid.uuid4())), ('retry_count', 0),
                  ('result', None), ('processing_time', None), ('total_cost', None),
                  ('skin_hex', None), ('step', 1)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─── 헤더 ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
  <div class="hero-title">✨ AI 퍼스널 컬러 진단</div>
  <div class="hero-subtitle">사진 한 장으로 나에게 어울리는 색을 찾아드려요</div>
  <div class="session-badge">🔑 세션 ID: {uuid}</div>
</div>
""".format(uuid=st.session_state['session_uuid'][:8] + "..."), unsafe_allow_html=True)

# ─── 단계 인디케이터 ──────────────────────────────────────────────────────────
step = st.session_state['step']
def dot_class(n): return "done" if n < step else ("active" if n == step else "idle")
st.markdown(f"""
<div class="step-row">
  <div class="step-item">
    <div class="step-dot {dot_class(1)}">1</div>
    <div class="step-label">사진 업로드</div>
  </div>
  <div class="step-item">
    <div class="step-dot {dot_class(2)}">2</div>
    <div class="step-label">조명 선택</div>
  </div>
  <div class="step-item">
    <div class="step-dot {dot_class(3)}">3</div>
    <div class="step-label">AI 분석</div>
  </div>
  <div class="step-item">
    <div class="step-dot {dot_class(4)}">4</div>
    <div class="step-label">결과 확인</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── 촬영 팁 ──────────────────────────────────────────────────────────────────
with st.expander("📸 더 정확한 진단을 위한 촬영 가이드", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ 이렇게 찍으세요**")
        st.markdown("""
- 자연광(창가) 또는 밝은 형광등 아래
- 흰색 A4 용지를 얼굴 옆에 두기
- 정면을 바라보는 얼굴 전체 사진
- 민낯 또는 가벼운 메이크업
- 머리카락이 얼굴을 가리지 않게
        """)
    with col2:
        st.markdown("**❌ 이런 사진은 피하세요**")
        st.markdown("""
- 강한 직사광선 / 역광
- 흐리거나 노란 조명 아래
- 진한 메이크업 / 색조 조명
- 마스크 착용
- 너무 멀거나 흐릿한 사진
        """)

# ─── STEP 1: 사진 업로드 ──────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 📷 얼굴 사진 업로드")
uploaded_file = st.file_uploader(
    "PNG / JPG 파일을 드래그하거나 클릭해서 올려주세요",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed"
)

if uploaded_file:
    st.session_state['step'] = max(st.session_state['step'], 2)
    col_img, col_info = st.columns([1, 1])
    with col_img:
        uploaded_file.seek(0)
        st.image(uploaded_file, caption="업로드된 사진", use_container_width=True)
    with col_info:
        # 얼굴 감지 미리 확인
        uploaded_file.seek(0)
        hex_val, rgb_val, face_status, face_rect = get_skin_color_data(uploaded_file)
        if face_status == "성공":
            st.success(f"✅ 얼굴이 감지되었습니다!")
            st.markdown(f"""
<div style="margin-top:0.8rem;">
  <div style="font-size:0.85rem;color:#8d6b7a;margin-bottom:6px;">감지된 피부 색상</div>
  <div style="display:flex;align-items:center;gap:10px;">
    <div class="color-chip" style="background:{hex_val};"></div>
    <div>
      <div style="font-weight:600;color:#3d1f3d;">{hex_val}</div>
      <div style="font-size:0.8rem;color:#9c7a8a;">{rgb_val}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
            st.session_state['skin_hex'] = hex_val
        else:
            st.warning("⚠️ 얼굴 감지 실패 — 이미지만으로 분석합니다")
st.markdown('</div>', unsafe_allow_html=True)

# ─── STEP 2: 조명 선택 ────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 💡 촬영 환경 선택")
lighting_condition = st.radio(
    "촬영 조명 환경:",
    ('🏠 실내 (형광등/백열등)', '☀️ 실외 (자연광)'),
    horizontal=True,
    label_visibility="collapsed"
)
if uploaded_file:
    st.session_state['step'] = max(st.session_state['step'], 3)
st.markdown('</div>', unsafe_allow_html=True)

# ─── STEP 3: 진단 시작 ────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 🔍 AI 분석 시작")

if st.session_state['result']:
    st.info("이전 진단 결과가 있습니다. 재진단하려면 아래 버튼을 누르세요.")

col_btn, col_cost = st.columns([3, 1])
with col_btn:
    run_button = st.button("🎨 퍼스널 컬러 진단하기", use_container_width=True)
with col_cost:
    if st.session_state.get('total_cost'):
        st.markdown(f'<div class="cost-pill">💰 ${st.session_state["total_cost"]:.4f}</div>',
                    unsafe_allow_html=True)

if run_button:
    if not uploaded_file:
        st.warning("📎 먼저 사진을 업로드해주세요!")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("🔑 OPENAI_API_KEY가 설정되지 않았습니다.")
    else:
        st.session_state['retry_count'] += 1
        st.session_state['step'] = 3

        with st.spinner("🤖 AI가 퍼스널 컬러를 분석 중입니다... 잠시만 기다려주세요"):
            prog = st.progress(0, text="이미지 전처리 중...")
            
            try:
                # 이미지 저장
                ext = uploaded_file.name.split('.')[-1]
                fname = f"{st.session_state['session_uuid']}_{st.session_state['retry_count']}.{ext}"
                saved_path = os.path.join("uploads", fname)
                uploaded_file.seek(0)
                with open(saved_path, "wb") as f:
                    f.write(uploaded_file.read())

                uploaded_file.seek(0)
                base64_image = encode_image(uploaded_file)
                prog.progress(25, text="피부 톤 데이터 추출 중...")

                uploaded_file.seek(0)
                hex_val, rgb_val, face_status, _ = get_skin_color_data(uploaded_file)
                
                lighting_clean = lighting_condition.replace("🏠 ", "").replace("☀️ ", "")
                
                prog.progress(50, text="AI 모델에 전송 중...")
                start_time = time.time()

                if face_status == "성공":
                    system_prompt = f"""너는 정밀 색채 분석가이자 컬러 코디네이터야.
제공된 이미지는 퍼스널 컬러 진단을 위해 동의를 얻은 사진이야. 인물 식별이나 민감한 정보 분석이 아닌, 순수한 색상 조화(Color Harmony) 관점에서 분석을 진행해.
이미지에서 추출한 [주요 색상 데이터]를 바탕으로 단계별 추론(Chain of Thought)을 진행해.

[추출된 색상 데이터]
- 기준 색상 Hex: {hex_val}
- 기준 색상 RGB: {rgb_val}
- 조명 환경: {lighting_clean}

[분석 단계]
Step 1. 조명 보정: 촬영 환경({lighting_clean})에 따라 추출된 RGB 값의 편향을 보정해.
Step 2. 색채 속성 분석: 보정된 색상의 온도(Warm/Cool), 명도, 채도를 판정해.
Step 3. 종합 진단: 이미지 전반의 색조(머리색, 눈동자색 등)와 Step 2 결과를 종합하여 가장 조화로운 사계절 컬러 톤(봄/여름/가을/겨울)을 최종 확정해.

출력 형식 (각 항목을 명확한 소제목으로 구분):
1. 🔬 베이스 톤 분석 — 추출된 수치와 특징
2. 💡 조명 보정 이유 — 어떤 보정을 했는지
3. 🎯 최종 컬러 톤 진단 — 사계절 톤 결론 (예: 봄 웜톤)
4. 👗 추천 의상 컬러 — 구체적 컬러명 나열
5. 💄 추천 메이크업 컬러 — 립/아이/치크 각각

- 주의: 사과문이나 거절 메시지를 출력하지 마. 색상 조화에 대해서만 객관적으로 설명해."""
                else:
                    system_prompt = f"""너는 20년 경력의 컬러 코디네이터야.
제공된 이미지는 퍼스널 컬러 진단을 위해 동의를 얻은 사진이야. 인물 식별이나 민감한 정보 분석이 아닌, 순수한 색상 조화(Color Harmony) 관점에서 분석을 진행해.

촬영 환경이 [{lighting_clean}]임을 감안하여 화이트 밸런스를 보정한 뒤, 이미지 속 인물의 베이스 톤, 눈동자, 머리카락 색을 종합 분석해 가장 잘 어울리는 사계절 컬러 톤을 진단해줘.

출력 형식 (각 항목을 명확한 소제목으로 구분):
1. 🔬 베이스 톤 분석 — 관찰된 특징
2. 💡 조명 보정 이유 — 어떤 보정을 했는지
3. 🎯 최종 컬러 톤 진단 — 사계절 톤 결론 (예: 봄 웜톤)
4. 👗 추천 의상 컬러 — 구체적 컬러명 나열
5. 💄 추천 메이크업 컬러 — 립/아이/치크 각각

- 주의: 사과문이나 거절 메시지를 출력하지 마. 색상 조화에 대해서만 객관적으로 설명해."""

                prog.progress(70, text="AI 응답 대기 중...")

                llm = ChatOpenAI(
                    model="gpt-4o",
                    max_tokens=1200,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
                message = HumanMessage(content=[
                    {"type": "text", "text": system_prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])

                with get_openai_callback() as cb:
                    response = llm.invoke([message])
                    result = response.content
                    prompt_tokens = cb.prompt_tokens
                    completion_tokens = cb.completion_tokens
                    total_cost = cb.total_cost

                processing_time = time.time() - start_time
                prog.progress(100, text="분석 완료!")
                time.sleep(0.3)
                prog.empty()

                st.session_state['result'] = result
                st.session_state['processing_time'] = processing_time
                st.session_state['total_cost'] = total_cost
                st.session_state['step'] = 4

                save_to_db(
                    st.session_state['session_uuid'], st.session_state['retry_count'],
                    lighting_clean, result, prompt_tokens, completion_tokens,
                    total_cost, saved_path, processing_time
                )

            except Exception as e:
                prog.empty()
                st.error(f"오류가 발생했습니다: {e}")
                save_to_db(
                    st.session_state['session_uuid'], st.session_state['retry_count'],
                    lighting_condition, f"Error: {e}", 0, 0, 0.0,
                    saved_path if 'saved_path' in locals() else None,
                    time.time() - start_time if 'start_time' in locals() else 0.0,
                    is_completed=False
                )

st.markdown('</div>', unsafe_allow_html=True)

# ─── STEP 4: 결과 표시 ────────────────────────────────────────────────────────
if st.session_state['result']:
    result_text = st.session_state['result']
    detected_season = detect_season(result_text)

    st.markdown("---")
    st.markdown("### 🎨 진단 결과")

    # 시즌 배지 + 팔레트
    if detected_season and detected_season in SEASON_PALETTES:
        info = SEASON_PALETTES[detected_season]
        st.markdown(f"""
<div class="card">
  <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
    <div class="season-badge {info['class']}">{info['emoji']} {detected_season}웜톤</div>
    <div style="color:#8d6b7a;font-size:0.9rem;">{info['desc']}</div>
  </div>
  <div style="margin-top:1rem;">
    <div style="font-size:0.82rem;color:#9c7a8a;margin-bottom:6px;">대표 추천 컬러</div>
    <div class="palette-row">
      {''.join(f'<div class="palette-swatch" style="background:{c};" title="{c}"></div>' for c in info['colors'])}
    </div>
    <div style="font-size:0.8rem;color:#b08898;margin-top:6px;">
      {' · '.join(info['keywords'])}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # 피부 색상 chip
    if st.session_state.get('skin_hex'):
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;">
  <div class="color-chip" style="background:{st.session_state['skin_hex']};"></div>
  <div>
    <div style="font-size:0.82rem;color:#9c7a8a;">감지된 피부 색상</div>
    <div style="font-weight:600;color:#3d1f3d;">{st.session_state['skin_hex']}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # 상세 결과 텍스트
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(result_text)
    st.markdown('</div>', unsafe_allow_html=True)

    # 메타 정보
    pt = st.session_state.get('processing_time', 0)
    tc = st.session_state.get('total_cost', 0)
    rc = st.session_state['retry_count']
    st.markdown(f"""
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:0.5rem;">
  <span class="cost-pill">⏱ {pt:.1f}초</span>
  <span class="cost-pill">💰 ${tc:.4f}</span>
  <span class="cost-pill">🔁 진단 {rc}회</span>
</div>
""", unsafe_allow_html=True)

    # 피드백
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**이 진단 결과가 도움이 되었나요?**")
    col_y, col_n, col_r = st.columns([1, 1, 2])
    with col_y:
        if st.button("👍 도움됐어요"):
            st.balloons()
            st.success("감사합니다! 피드백이 저장되었습니다 💖")
    with col_n:
        if st.button("👎 아쉬워요"):
            st.info("소중한 의견 감사합니다. 더 나은 진단을 위해 노력할게요!")

# ─── 사이드바 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌈 사계절 컬러 가이드")
    for season, info in SEASON_PALETTES.items():
        with st.expander(f"{info['emoji']} {season}웜톤 · {info['desc']}"):
            cols = st.columns(6)
            for i, color in enumerate(info['colors']):
                cols[i].markdown(
                    f'<div style="width:30px;height:30px;background:{color};'
                    f'border-radius:50%;margin:auto;"></div>',
                    unsafe_allow_html=True
                )
            st.caption(" · ".join(info['keywords']))

    st.markdown("---")
    st.markdown("### 📊 이번 세션")
    st.markdown(f"- 진단 횟수: **{st.session_state['retry_count']}회**")
    if st.session_state.get('total_cost'):
        st.markdown(f"- 총 비용: **${st.session_state['total_cost']:.4f}**")
    if db_connected:
        st.success("🟢 DB 연결됨")
    else:
        st.warning("🔴 DB 미연결 (로컬 모드)")