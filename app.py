import os
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import numpy as np
import time
import re

# 추가 API 키 설정
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# OpenAI 클라이언트
client = OpenAI(api_key=OPENAI_API_KEY)

# Anthropic 클라이언트 (Claude)
try:
    import anthropic
    claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
except ImportError:
    claude_client = None

# Google 클라이언트 (Gemini)
try:
    import google.generativeai as genai
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-pro')
    else:
        gemini_model = None
except ImportError:
    gemini_model = None

# 모델 정보
MODELS = {
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo",
        "description": "빠르고 경제적인 기본 모델",
        "best_for": ["간단한 설명", "정의", "기본 질문"],
        "color": "model-gpt35"
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "description": "균형잡힌 성능과 비용",
        "best_for": ["요약", "분석", "중간 복잡도 질문"],
        "color": "model-gpt4mini"
    },
    "gpt-4o": {
        "name": "GPT-4o",
        "description": "최고 품질의 고급 모델",
        "best_for": ["복잡한 분석", "전략", "창의적 작업"],
        "color": "model-gpt4o"
    },
    "claude-3-5-sonnet": {
        "name": "Claude 3.5 Sonnet",
        "description": "Anthropic의 최신 모델",
        "best_for": ["창의적 글쓰기", "코드 생성", "상세한 분석"],
        "color": "model-claude"
    },
    "gemini-pro": {
        "name": "Gemini Pro",
        "description": "Google의 고성능 모델",
        "best_for": ["다양한 작업", "멀티모달", "실시간 정보"],
        "color": "model-gemini"
    }
}

# OpenAI API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    key_candidates = [
        os.path.join(os.path.dirname(__file__), "nocommit_key.txt"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "nocommit_key.txt"),
    ]
    for cand in key_candidates:
        if os.path.isfile(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    OPENAI_API_KEY = f.read().strip()
                break
            except Exception:
                pass

# 시각화 라이브러리 (선택적)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from collections import Counter

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# 페이지 설정
st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🤖",
    layout="wide"
)

# CSS 스타일
st.markdown("""
<style>
    /* 전체 배경 및 기본 스타일 */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* 헤더 스타일 */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* 사이드바 스타일 */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* 카드 스타일 */
    .upload-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .upload-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .answer-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #28a745;
        color: #333;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .answer-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .answer-card h4 {
        color: #333;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .answer-card p {
        color: #333;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .improved-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
        color: #333;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .improved-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .improved-card h4 {
        color: #333;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .improved-card p {
        color: #333;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    /* 배지 스타일 */
    .quality-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .quality-good { background: linear-gradient(45deg, #d4edda, #c3e6cb); color: #155724; }
    .quality-medium { background: linear-gradient(45deg, #fff3cd, #ffeaa7); color: #856404; }
    .quality-bad { background: linear-gradient(45deg, #f8d7da, #f5c6cb); color: #721c24; }
    
    .model-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .model-gpt35 { background: linear-gradient(45deg, #e3f2fd, #bbdefb); color: #1565c0; }
    .model-gpt4mini { background: linear-gradient(45deg, #f3e5f5, #e1bee7); color: #7b1fa2; }
    .model-gpt4o { background: linear-gradient(45deg, #e8f5e8, #c8e6c9); color: #2e7d32; }
    
    /* 스마트 선택 박스 */
    .smart-selection {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    .smart-selection h4 {
        color: #856404;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .smart-selection p {
        color: #856404;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    
    /* 섹션 헤더 */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* 입력 필드 스타일 */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* 파일 업로더 스타일 */
    .stFileUploader > div {
        border-radius: 15px;
        border: 2px dashed #667eea;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
    }
    
    /* 메트릭 카드 */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
    }
    
    /* 히스토리 스타일 */
    .history-item {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .history-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    }
    
    /* 프로그레스 바 스타일 */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* 스피너 스타일 */
    .stSpinner > div {
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* 경고 메시지 스타일 */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* 성공 메시지 스타일 */
    .stSuccess {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }
    
    /* 텍스트 색상 */
    .stMarkdown, .stText, .stTextInput, .stTextArea {
        color: #333 !important;
    }
    
    /* 사이드바 메트릭 */
    .sidebar-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* 다크모드 스타일 */
    .dark-mode {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    .dark-mode .stApp {
        background-color: #1a1a1a !important;
    }
    
    .dark-mode .stTextInput, .dark-mode .stTextArea, .dark-mode .stSelectbox {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border-color: #444444 !important;
    }
    
    .dark-mode .stButton > button {
        background-color: #4a4a4a !important;
        color: #ffffff !important;
        border-color: #666666 !important;
    }
    
    .dark-mode .stButton > button:hover {
        background-color: #5a5a5a !important;
    }
    
    .dark-mode .info-card, .dark-mode .answer-card, .dark-mode .improved-card {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border-color: #444444 !important;
    }
    
    .dark-mode .section-header {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown("""
<div class="main-header">
    <h1>🤖 AI PDF Assistant</h1>
    <p>스마트한 PDF 기반 질의응답 시스템</p>
</div>
""", unsafe_allow_html=True)

# 모델 설정
MODELS = {
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo",
        "description": "빠르고 경제적인 기본 모델",
        "best_for": ["간단한 설명", "정의", "기본 질문"],
        "color": "model-gpt35"
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "description": "균형잡힌 성능과 비용",
        "best_for": ["요약", "분석", "중간 복잡도 질문"],
        "color": "model-gpt4mini"
    },
    "gpt-4o": {
        "name": "GPT-4o",
        "description": "최고 품질의 고급 모델",
        "best_for": ["복잡한 분석", "전략", "창의적 작업"],
        "color": "model-gpt4o"
    }
}

def analyze_question_complexity(question: str) -> dict:
    """질문 복잡도 분석"""
    complexity_score = 0
    question_lower = question.lower()
    
    # 복잡한 키워드들
    complex_keywords = [
        "분석", "비교", "평가", "전략", "방안", "해결책", "대안", "장단점",
        "왜", "어떻게", "어떤", "가장", "최적", "효율", "효과", "영향",
        "관계", "연관", "차이", "유사", "특징", "장점", "단점"
    ]
    
    # 간단한 키워드들
    simple_keywords = [
        "정의", "설명", "뭐", "무엇", "어디", "언제", "누구", "개념",
        "의미", "용어", "기본", "간단", "요약", "정리"
    ]
    
    # 복잡도 계산
    for word in complex_keywords:
        if word in question_lower:
            complexity_score += 2
    
    for word in simple_keywords:
        if word in question_lower:
            complexity_score -= 1
    
    # 질문 길이 고려
    if len(question) > 50:
        complexity_score += 1
    if len(question) > 100:
        complexity_score += 2
    
    # 질문 유형 판별
    question_type = "기본"
    if complexity_score >= 4:
        question_type = "복잡"
    elif complexity_score >= 2:
        question_type = "중간"
    
    return {
        "score": complexity_score,
        "type": question_type,
        "complex_keywords": [w for w in complex_keywords if w in question_lower],
        "simple_keywords": [w for w in simple_keywords if w in question_lower]
    }

def select_model_automatically(question: str, context_length: int = 0) -> dict:
    """자동 모델 선택"""
    complexity = analyze_question_complexity(question)
    
    # 컨텍스트 길이 고려
    if context_length > 5000:
        complexity["score"] += 3
    elif context_length > 2000:
        complexity["score"] += 1
    
    # 모델 선택 로직
    if complexity["score"] >= 5:
        selected_model = "gpt-4o"
        reason = "복잡한 분석/전략 질문으로 판단됨"
    elif complexity["score"] >= 2:
        selected_model = "gpt-4o-mini"
        reason = "중간 복잡도 질문으로 판단됨"
    else:
        selected_model = "gpt-3.5-turbo"
        reason = "기본 질문으로 판단됨"
    
    return {
        "model": selected_model,
        "reason": reason,
        "complexity": complexity,
        "context_length": context_length
    }

# 사이드바 설정
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h3>⚙️ 설정</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # 모델 선택 방식
    model_selection_mode = st.radio(
        "🤖 AI 모델 선택 방식",
        ["자동 선택 (추천)", "수동 선택"],
        help="자동 선택: 질문을 분석해서 최적의 모델을 자동으로 선택합니다"
    )
    
    if model_selection_mode == "수동 선택":
        use_gpt4 = st.checkbox("GPT-4o 사용", value=False)
        use_gpt4mini = st.checkbox("GPT-4o-mini 사용", value=False)
        use_claude = st.checkbox("Claude 3.5 Sonnet 사용", value=False)
        use_gemini = st.checkbox("Gemini Pro 사용", value=False)
    else:
        use_gpt4 = False
        use_gpt4mini = False
        use_claude = False
        use_gemini = False
    
    st.markdown("---")
    
    rag_enabled = st.toggle("🔍 RAG 기능 활성화", value=True)
    use_hierarchical = st.checkbox("계층적 답변 개선", value=True)
    auto_improve = st.checkbox("자동 품질 개선", value=True)
    
    st.markdown("---")
    
    chunk_size = st.slider("청크 크기", 100, 500, 200)
    overlap_size = st.slider("겹침 크기", 10, 100, 50)
    quality_threshold = st.slider("품질 임계값", 0, 100, 60, help="이 점수 이하면 자동 개선")

# 세션 상태 초기화
if "docs" not in st.session_state:
    st.session_state.docs = None
    st.session_state.embs = None
    st.session_state.pdf_text = ""

if "history" not in st.session_state:
    st.session_state.history = []

# 대화 메모리 시스템
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = {
        "preferred_model": "gpt-4o",
        "auto_improve": True,
        "web_search": True,
        "response_length": "medium",
        "answer_style": "balanced",  # simple, detailed, professional, friendly
        "favorite_questions": [],
        "custom_prompts": {},
        "learning_profile": {
            "preferred_topics": [],
            "difficulty_level": "medium",
            "interaction_style": "conversational"
        }
    }

def read_pdf(file) -> str:
    """PDF 읽기"""
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50):
    """텍스트 청킹"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_context(question: str, docs: list, embs: list) -> str:
    """컨텍스트 생성"""
    if not docs or not embs:
        return ""
    
    # 간단한 키워드 매칭
    question_words = set(question.lower().split())
    best_chunks = []
    
    for i, doc in enumerate(docs[:5]):  # 최대 5개 청크만
        doc_words = set(doc.lower().split())
        overlap = len(question_words.intersection(doc_words))
        if overlap > 0:
            best_chunks.append(doc)
    
    return "\n\n".join(best_chunks[:3]) if best_chunks else docs[0] if docs else ""

def analyze_answer_quality(answer: str, question: str) -> dict:
    """답변 품질 분석"""
    if not answer or len(answer.strip()) < 10:
        return {'score': 0, 'issues': ['답변이 너무 짧습니다'], 'level': 'bad'}
    
    score = 0
    issues = []
    
    # 1. 길이 점수 (최대 25점)
    length_score = min(len(answer) / 100, 25)
    score += length_score
    
    # 2. 구체성 점수 (최대 25점)
    specific_words = ['예시', '구체적으로', '예를 들어', '첫째', '둘째', '셋째', '또한', '그러나', '따라서']
    specificity_count = sum(1 for word in specific_words if word in answer)
    specificity_score = min(specificity_count * 5, 25)
    score += specificity_score
    
    # 3. 불확실성 감소 점수 (최대 25점)
    uncertainty_words = ['모르겠습니다', '확실하지 않습니다', '추측', '아마도', '어쩌면']
    uncertainty_count = sum(1 for word in uncertainty_words if word in answer)
    uncertainty_score = max(0, 25 - uncertainty_count * 5)
    score += uncertainty_score
    
    # 4. 키워드 포함 점수 (최대 25점)
    question_words = set(re.findall(r'\w+', question.lower()))
    answer_words = set(re.findall(r'\w+', answer.lower()))
    keyword_overlap = len(question_words.intersection(answer_words))
    keyword_score = min(keyword_overlap * 3, 25)
    score += keyword_score
    
    # 이슈 식별
    if length_score < 10:
        issues.append('답변이 너무 짧습니다')
    if specificity_score < 10:
        issues.append('구체적인 예시가 부족합니다')
    if uncertainty_score < 15:
        issues.append('불확실한 표현이 많습니다')
    if keyword_score < 10:
        issues.append('질문과 관련성이 낮습니다')
    
    # 레벨 결정
    if score >= 80:
        level = 'good'
    elif score >= 60:
        level = 'medium'
    else:
        level = 'bad'
    
    return {
        'score': round(score, 1),
        'issues': issues,
        'level': level
    }

def generate_answer(question: str, context: str, model: str) -> str:
    """답변 생성"""
    try:
        if context:
            prompt = f"""다음 정보를 참고하여 질문에 답변하세요.

참고 정보:
{context}

질문: {question}

답변:"""
        else:
            prompt = question
        
        if model in ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        elif model == "claude-3-5-sonnet" and claude_client:
            response = claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        elif model == "gemini-pro" and gemini_model:
            response = gemini_model.generate_content(prompt)
            return response.text
        else:
            return f"지원하지 않는 모델이거나 API 키가 설정되지 않았습니다: {model}"
        
    except Exception as e:
        return f"오류 발생: {str(e)}"

def analyze_sentiment_and_tone(text: str) -> dict:
    """감정 및 톤 분석"""
    try:
        # 간단한 감정 분석
        positive_words = ['좋은', '훌륭한', '멋진', '유용한', '효과적인', '성공적인']
        negative_words = ['나쁜', '문제', '실패', '어려운', '복잡한', '불편한']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            sentiment = "긍정적"
            tone = "친근하고 격려적"
        elif negative_count > positive_count:
            sentiment = "부정적"
            tone = "우려스럽고 경계적"
        else:
            sentiment = "중립적"
            tone = "객관적이고 균형잡힌"
        
        return {
            'sentiment': sentiment,
            'tone': tone,
            'positive_score': positive_count,
            'negative_score': negative_count
        }
    except Exception as e:
        return {'sentiment': '분석 불가', 'tone': '분석 불가', 'positive_score': 0, 'negative_score': 0}

def classify_topic(text: str) -> str:
    """주제 분류"""
    try:
        topics = {
            '기술': ['프로그래밍', '코드', '알고리즘', '데이터베이스', 'API', '개발'],
            '비즈니스': ['경영', '전략', '마케팅', '수익', '고객', '서비스'],
            '교육': ['학습', '교육', '강의', '과정', '지식', '이해'],
            '의료': ['진단', '치료', '증상', '의학', '건강', '병원'],
            '법률': ['법률', '계약', '소송', '권리', '의무', '규정']
        }
        
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in topics.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            topic_scores[topic] = score
        
        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            return best_topic if topic_scores[best_topic] > 0 else "일반"
        else:
            return "일반"
    except Exception as e:
        return "일반"

def improve_math_readability(text: str) -> str:
    """수학 기호 가독성 개선"""
    try:
        # LaTeX 수식을 일반 텍스트로 변환
        math_replacements = {
            r'\frac{([^}]+)}{([^}]+)}': r'분수(\1/\2)',
            r'\sqrt{([^}]+)}': r'제곱근(\1)',
            r'\sum_{([^}]+)}': r'합계(\1)',
            r'\int_{([^}]+)}': r'적분(\1)',
            r'\alpha': '알파',
            r'\beta': '베타',
            r'\gamma': '감마',
            r'\delta': '델타',
            r'\epsilon': '엡실론',
            r'\theta': '세타',
            r'\lambda': '람다',
            r'\mu': '뮤',
            r'\pi': '파이',
            r'\sigma': '시그마',
            r'\phi': '파이',
            r'\omega': '오메가'
        }
        
        improved_text = text
        for pattern, replacement in math_replacements.items():
            improved_text = re.sub(pattern, replacement, improved_text)
        
        return improved_text
    except Exception as e:
        return text

def analyze_text_keywords(text: str, top_n: int = 20) -> dict:
    """텍스트에서 키워드 분석"""
    try:
        # 한국어 불용어
        stop_words = ['이', '그', '저', '것', '수', '등', '때', '곳', '말', '일', '뿐', '뒤', '앞', '밖', '안', '속', '사이', '중', '위', '아래', '앞', '뒤', '왼쪽', '오른쪽', '가운데', '옆', '반대', '같은', '다른', '모든', '어떤', '무슨', '어느', '몇', '얼마', '언제', '어디', '어떻게', '왜', '무엇', '누구', '어떤', '무슨', '어느', '몇', '얼마', '언제', '어디', '어떻게', '왜', '무엇', '누구']
        
        # 텍스트 정리
        text_clean = re.sub(r'[^\w\s]', ' ', text)
        words = text_clean.split()
        
        # 불용어 제거 및 길이 필터링
        filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
        
        # 빈도수 계산
        word_counts = Counter(filtered_words)
        
        # 상위 키워드 반환
        return dict(word_counts.most_common(top_n))
    except Exception as e:
        return {}

def improve_answer_with_better_model(question: str, basic_answer: str, context: str, better_model: str, quality_analysis: dict) -> str:
    """더 나은 모델로 답변 개선"""
    try:
        # 품질 분석 결과를 바탕으로 개선 방향 설정
        improvement_directions = []
        if quality_analysis['score'] < 60:
            improvement_directions.append("더 구체적이고 상세한 답변을 제공하세요")
        if '구체적인 예시가 부족합니다' in quality_analysis['issues']:
            improvement_directions.append("구체적인 예시를 포함하세요")
        if '불확실한 표현이 많습니다' in quality_analysis['issues']:
            improvement_directions.append("확실하고 명확한 표현을 사용하세요")
        
        improvement_text = " ".join(improvement_directions) if improvement_directions else "답변을 더 정확하고 유용하게 개선하세요"
        
        prompt = f"""다음 답변을 개선해주세요. 개선 방향: {improvement_text}

원본 질문: {question}
컨텍스트: {context}
현재 답변: {basic_answer}

개선된 답변:"""
        
        response = client.chat.completions.create(
            model=better_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.5
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"답변 개선 중 오류가 발생했습니다: {str(e)}"

# 메인 컨테이너
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="section-header">📄 PDF 업로드</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("PDF 파일을 선택하세요", type=['pdf'])
        
        if uploaded_file is not None:
            # 세련된 로딩 컨테이너
            with st.container():
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem;
                    border-radius: 15px;
                    text-align: center;
                    color: white;
                    margin: 1rem 0;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                ">
                    <div style="
                        width: 60px;
                        height: 60px;
                        border: 4px solid rgba(255,255,255,0.3);
                        border-top: 4px solid white;
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                        margin: 0 auto 1rem auto;
                    "></div>
                    <h3>📄 PDF 분석 중...</h3>
                    <p>텍스트 추출 및 AI 학습을 위한 준비를 하고 있습니다</p>
                </div>
                """, unsafe_allow_html=True)
                
                pdf_text = read_pdf(uploaded_file)
                if pdf_text:
                    st.session_state.pdf_text = pdf_text
                    
                    # 청킹 및 임베딩 생성
                    chunks = chunk_text(pdf_text, chunk_size, overlap_size)
                    
                    if chunks:
            # 임베딩 생성
                        embeddings = []
                        
                        # 진행률 표시
                        progress_container = st.container()
                        with progress_container:
                            st.markdown("""
                            <div style="
                                background: white;
                                padding: 1.5rem;
                                border-radius: 15px;
                                margin: 1rem 0;
                                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                            ">
                                <h4 style="color: #333; margin-bottom: 1rem;">🤖 AI 학습 진행률</h4>
                                <div style="
                                    background: #e9ecef;
                                    border-radius: 10px;
                                    height: 20px;
                                    overflow: hidden;
                                    margin-bottom: 0.5rem;
                                ">
                                    <div id="progress-bar" style="
                                        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                                        height: 100%;
                                        width: 0%;
                                        transition: width 0.3s ease;
                                        border-radius: 10px;
                                    "></div>
                                </div>
                                <p id="progress-text" style="color: #666; margin: 0; font-size: 0.9rem;">텍스트 청킹 중...</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        for i, chunk in enumerate(chunks):
                            try:
                                response = client.embeddings.create(
                                model="text-embedding-3-small",
                                    input=chunk
                                )
                                emb = np.array(response.data[0].embedding)
                                embeddings.append(emb)
                            except Exception:
                                embeddings.append(np.zeros(1536))
                            
                            # 진행률 업데이트
                            progress_percent = (i + 1) / len(chunks) * 100
                            progress_text = f"청크 {i+1}/{len(chunks)} 처리 중... ({progress_percent:.1f}%)"
                            
                            # JavaScript로 진행률 업데이트
                            st.markdown(f"""
                            <script>
                                document.getElementById('progress-bar').style.width = '{progress_percent}%';
                                document.getElementById('progress-text').textContent = '{progress_text}';
                            </script>
                            """, unsafe_allow_html=True)
                        
                        st.session_state.docs = chunks
                        st.session_state.embs = embeddings
                        
                        # 완료 메시지
        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                            padding: 1.5rem;
                            border-radius: 15px;
                            margin: 1rem 0;
                            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                            border-left: 4px solid #28a745;
                        ">
                            <h4 style="color: #155724; margin-bottom: 0.5rem;">✅ PDF 처리 완료!</h4>
                            <p style="color: #155724; margin: 0;">📊 {len(st.session_state.docs) if st.session_state.docs else 0}개 청크 생성 | 📄 {len(st.session_state.pdf_text) if st.session_state.pdf_text else 0:,} 문자 분석</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
with col2:
    st.markdown('<div class="section-header">📊 처리 정보</div>', unsafe_allow_html=True)
    if st.session_state.docs:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 청크 수</h4>
            <h2>{len(st.session_state.docs)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
        st.markdown(f"""
        <div class="metric-card">
            <h4>📄 PDF 크기</h4>
            <h2>{len(st.session_state.pdf_text):,}</h2>
            <p>문자</p>
        </div>
        """, unsafe_allow_html=True)

# 질문 입력
st.markdown('<div class="section-header">💬 질문하기</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    question = st.text_input("질문을 입력하세요:", placeholder="PDF 내용에 대해 질문해보세요!")
    
    if question and st.button("🔍 답변 생성", type="primary"):
        if not st.session_state.docs and rag_enabled:
            st.warning("먼저 PDF 파일을 업로드해주세요!")
            else:
            # 세련된 답변 생성 로딩
            with st.container():
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem;
                    border-radius: 15px;
                    text-align: center;
                    color: white;
                    margin: 1rem 0;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                ">
                    <div style="
                        width: 50px;
                        height: 50px;
                        border: 3px solid rgba(255,255,255,0.3);
                        border-top: 3px solid white;
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                        margin: 0 auto 1rem auto;
                    "></div>
                    <h3>🤖 AI 답변 생성 중...</h3>
                    <p>질문을 분석하고 최적의 답변을 찾고 있습니다</p>
                </div>
                """, unsafe_allow_html=True)
    
    # 컨텍스트 생성
                context = get_context(question, st.session_state.docs, st.session_state.embs) if rag_enabled else ""
                
                # 자동 모델 선택
                if model_selection_mode == "자동 선택 (추천)":
                    auto_selection = select_model_automatically(question, len(context))
                    selected_model = auto_selection["model"]
                    
                    # 스마트 선택 정보 표시
                    st.markdown(f"""
                    <div class="smart-selection">
                        <h4>🤖 스마트 모델 선택</h4>
                        <p><strong>선택된 모델:</strong> {MODELS[selected_model]['name']}</p>
                        <p><strong>선택 이유:</strong> {auto_selection['reason']}</p>
                        <p><strong>복잡도 점수:</strong> {auto_selection['complexity']['score']} (유형: {auto_selection['complexity']['type']})</p>
                        <p><strong>적합한 작업:</strong> {', '.join(MODELS[selected_model]['best_for'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 자동 선택된 모델로 답변 생성
                    auto_answer = generate_answer(question, context, selected_model)
                    auto_quality = analyze_answer_quality(auto_answer, question)
                    
                    # 자동 선택 답변 표시
                    with st.container():
                        st.markdown(f"""
                        <div class="answer-card">
                            <h4>🤖 {MODELS[selected_model]['name']} 답변 (자동 선택)</h4>
                            <p>{auto_answer}</p>
                            <div class="quality-badge quality-{auto_quality['level']}">
                                품질 점수: {auto_quality['score']}/100
                            </div>
                            <div class="model-badge {MODELS[selected_model]['color']}">
                                {MODELS[selected_model]['name']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # 기본 답변 생성 (GPT-3.5)
                basic_model = "gpt-3.5-turbo"
                basic_answer = generate_answer(question, context, basic_model)
        
        # 품질 분석
            quality_analysis = analyze_answer_quality(basic_answer, question)
            
                # 결과 표시
                st.markdown('<div class="section-header">📝 답변</div>', unsafe_allow_html=True)
                
                # 기본 답변 (GPT-3.5)
                with st.container():
        st.markdown(f"""
        <div class="answer-card">
                        <h4>🤖 GPT-3.5 답변</h4>
            <p>{basic_answer}</p>
                        <div class="quality-badge quality-{quality_analysis['level']}">
                            품질 점수: {quality_analysis['score']}/100
                        </div>
                        <div class="model-badge model-gpt35">
                            GPT-3.5 Turbo
            </div>
        </div>
        """, unsafe_allow_html=True)
        
                # 수동 선택 모델들
                if model_selection_mode == "수동 선택":
                    if use_gpt4mini:
                        with st.container():
                            st.markdown("""
                            <div style="
                                background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
                                padding: 1.5rem;
                                border-radius: 15px;
                                text-align: center;
                                color: #7b1fa2;
                                margin: 1rem 0;
                                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                            ">
                                <div style="
                                    width: 40px;
                                    height: 40px;
                                    border: 3px solid rgba(123, 31, 162, 0.3);
                                    border-top: 3px solid #7b1fa2;
                                    border-radius: 50%;
                                    animation: spin 1s linear infinite;
                                    margin: 0 auto 0.5rem auto;
                                "></div>
                                <p style="margin: 0; font-weight: 600;">🚀 GPT-4o-mini 분석 중...</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            gpt4mini_answer = generate_answer(question, context, "gpt-4o-mini")
                            gpt4mini_quality = analyze_answer_quality(gpt4mini_answer, question)
                            
                            with st.container():
                                st.markdown(f"""
                                <div class="answer-card">
                                    <h4>🚀 GPT-4o-mini 답변</h4>
                                    <p>{gpt4mini_answer}</p>
                                    <div class="quality-badge quality-{gpt4mini_quality['level']}">
                                        품질 점수: {gpt4mini_quality['score']}/100
                                    </div>
                                    <div class="model-badge model-gpt4mini">
                                        GPT-4o Mini
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    if use_gpt4:
                        with st.container():
                            st.markdown("""
                            <div style="
                                background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
                                padding: 1.5rem;
                                border-radius: 15px;
                                text-align: center;
                                color: #2e7d32;
                                margin: 1rem 0;
                                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                            ">
                                <div style="
                                    width: 40px;
                                    height: 40px;
                                    border: 3px solid rgba(46, 125, 50, 0.3);
                                    border-top: 3px solid #2e7d32;
                                    border-radius: 50%;
                                    animation: spin 1s linear infinite;
                                    margin: 0 auto 0.5rem auto;
                                "></div>
                                <p style="margin: 0; font-weight: 600;">🚀 GPT-4o 분석 중...</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            gpt4_answer = generate_answer(question, context, "gpt-4o")
                            gpt4_quality = analyze_answer_quality(gpt4_answer, question)
                            
                            with st.container():
                                st.markdown(f"""
                                <div class="answer-card">
                                    <h4>🚀 GPT-4o 답변</h4>
                                    <p>{gpt4_answer}</p>
                                    <div class="quality-badge quality-{gpt4_quality['level']}">
                                        품질 점수: {gpt4_quality['score']}/100
                                    </div>
                                    <div class="model-badge model-gpt4o">
                                        GPT-4o
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # 계층적 답변 개선 (선택적)
                improved_answer = None
                improved_quality = None
                
                if use_hierarchical and (use_gpt4 or use_gpt4mini or quality_analysis['score'] < quality_threshold):
                    with st.container():
                        st.markdown("""
                        <div style="
                            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                            padding: 1.5rem;
                            border-radius: 15px;
                            text-align: center;
                            color: #856404;
                            margin: 1rem 0;
                            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                        ">
                            <div style="
                                width: 40px;
                                height: 40px;
                                border: 3px solid rgba(133, 100, 4, 0.3);
                                border-top: 3px solid #856404;
                                border-radius: 50%;
                                animation: spin 1s linear infinite;
                                margin: 0 auto 0.5rem auto;
                            "></div>
                            <p style="margin: 0; font-weight: 600;">✨ 답변 품질 개선 중...</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        better_model = "gpt-4o"
                        improved_answer = improve_answer_with_better_model(
                    question, basic_answer, context, better_model, quality_analysis
                )
                improved_quality = analyze_answer_quality(improved_answer, question)
            
                with st.container():
            st.markdown(f"""
            <div class="improved-card">
                                <h4>✨ 개선된 답변 (GPT-4o)</h4>
                <p>{improved_answer}</p>
                                <div class="quality-badge quality-{improved_quality['level']}">
                                    개선된 품질 점수: {improved_quality['score']}/100
                                </div>
                                <div class="model-badge model-gpt4o">
                                    GPT-4o (개선)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
                # 품질 분석 상세 정보
                with st.expander("📊 품질 분석 상세"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**GPT-3.5 답변 분석**")
                        st.write(f"점수: {quality_analysis['score']}/100")
                        st.write(f"레벨: {quality_analysis['level']}")
                        if quality_analysis['issues']:
                            st.write("문제점:")
                            for issue in quality_analysis['issues']:
                                st.write(f"- {issue}")
                    
                    if model_selection_mode == "수동 선택":
                        if use_gpt4mini:
                            with col2:
                                st.markdown("**GPT-4o-mini 답변 분석**")
                                st.write(f"점수: {gpt4mini_quality['score']}/100")
                                st.write(f"레벨: {gpt4mini_quality['level']}")
                                if gpt4mini_quality['issues']:
                                    st.write("문제점:")
                                    for issue in gpt4mini_quality['issues']:
                                        st.write(f"- {issue}")
                        elif use_gpt4:
                            with col2:
                                st.markdown("**GPT-4o 답변 분석**")
                                st.write(f"점수: {gpt4_quality['score']}/100")
                                st.write(f"레벨: {gpt4_quality['level']}")
                                if gpt4_quality['issues']:
                                    st.write("문제점:")
                                    for issue in gpt4_quality['issues']:
                                        st.write(f"- {issue}")
                    else:
                        with col2:
                            st.markdown("**자동 선택 모델 분석**")
                            st.write(f"점수: {auto_quality['score']}/100")
                            st.write(f"레벨: {auto_quality['level']}")
                            if auto_quality['issues']:
                                st.write("문제점:")
                                for issue in auto_quality['issues']:
                                    st.write(f"- {issue}")
                
                # 컨텍스트 표시
                if context and rag_enabled:
                    with st.expander("📄 사용된 컨텍스트"):
                        st.text_area("컨텍스트", context, height=200, disabled=True)
        
        # 히스토리에 저장
                history_entry = {
                    'question': question,
                    'basic_answer': basic_answer,
                    'basic_quality': quality_analysis['score'],
                    'auto_answer': auto_answer if model_selection_mode == "자동 선택 (추천)" else None,
                    'auto_quality': auto_quality['score'] if model_selection_mode == "자동 선택 (추천)" else None,
                    'auto_model': selected_model if model_selection_mode == "자동 선택 (추천)" else None,
                    'gpt4mini_answer': gpt4mini_answer if model_selection_mode == "수동 선택" and use_gpt4mini else None,
                    'gpt4mini_quality': gpt4mini_quality['score'] if model_selection_mode == "수동 선택" and use_gpt4mini else None,
                    'gpt4_answer': gpt4_answer if model_selection_mode == "수동 선택" and use_gpt4 else None,
                    'gpt4_quality': gpt4_quality['score'] if model_selection_mode == "수동 선택" and use_gpt4 else None,
                    'improved_answer': improved_answer,
                    'improved_quality': improved_quality['score'] if improved_quality else None,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.history.append(history_entry)
    st.markdown('</div>', unsafe_allow_html=True)

# 대화 히스토리
if st.session_state.history:
    st.markdown('<div class="section-header">📚 대화 히스토리</div>', unsafe_allow_html=True)
    for i, entry in enumerate(reversed(st.session_state.history[-5:]), 1):
        with st.expander(f"질문 {i}: {entry['question'][:50]}..."):
            st.markdown(f"""
            <div class="history-item">
                <h4>💬 질문</h4>
                <p>{entry['question']}</p>
                
                <h4>🤖 GPT-3.5 답변</h4>
                <p>{entry['basic_answer']}</p>
                <div class="quality-badge quality-{'good' if entry['basic_quality'] >= 80 else 'medium' if entry['basic_quality'] >= 60 else 'bad'}">
                    품질: {entry['basic_quality']}/100
                </div>
            """, unsafe_allow_html=True)
            
            if entry['auto_answer']:
                st.markdown(f"""
                <h4>🚀 자동 선택 모델 ({entry['auto_model']})</h4>
                <p>{entry['auto_answer']}</p>
                <div class="quality-badge quality-{'good' if entry['auto_quality'] >= 80 else 'medium' if entry['auto_quality'] >= 60 else 'bad'}">
                    품질: {entry['auto_quality']}/100
                </div>
                """, unsafe_allow_html=True)
            
            if entry['gpt4mini_answer']:
                st.markdown(f"""
                <h4>🚀 GPT-4o-mini 답변</h4>
                <p>{entry['gpt4mini_answer']}</p>
                <div class="quality-badge quality-{'good' if entry['gpt4mini_quality'] >= 80 else 'medium' if entry['gpt4mini_quality'] >= 60 else 'bad'}">
                    품질: {entry['gpt4mini_quality']}/100
                </div>
                """, unsafe_allow_html=True)
            
            if entry['gpt4_answer']:
                st.markdown(f"""
                <h4>🚀 GPT-4o 답변</h4>
                <p>{entry['gpt4_answer']}</p>
                <div class="quality-badge quality-{'good' if entry['gpt4_quality'] >= 80 else 'medium' if entry['gpt4_quality'] >= 60 else 'bad'}">
                    품질: {entry['gpt4_quality']}/100
                </div>
                """, unsafe_allow_html=True)
            
            if entry['improved_answer']:
                st.markdown(f"""
                <h4>✨ 개선된 답변</h4>
                <p>{entry['improved_answer']}</p>
                <div class="quality-badge quality-{'good' if entry['improved_quality'] >= 80 else 'medium' if entry['improved_quality'] >= 60 else 'bad'}">
                    개선된 품질: {entry['improved_quality']}/100
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style="margin-top: 1rem; padding: 0.5rem; background: #f8f9fa; border-radius: 8px; text-align: center;">
                <small>⏰ {entry['timestamp']}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)


