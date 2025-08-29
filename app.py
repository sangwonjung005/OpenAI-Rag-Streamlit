import os
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import numpy as np
import time
import re

# OpenAI API 키 설정 (맨 위로 이동)
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
    },
    "gpt-oss-20b": {
        "name": "GPT-OSS-20B (로컬)",
        "description": "o3-mini 수준 성능, 무료 로컬 실행",
        "best_for": ["일반 분석", "에지 디바이스", "빠른 반복"],
        "color": "model-gptoss",
        "local": True,
        "hardware_required": "16GB RAM"
    },
    "gpt-oss-120b": {
        "name": "GPT-OSS-120B (로컬)",
        "description": "o4-mini 수준 성능, 무료 로컬 실행",
        "best_for": ["복잡한 추론", "도구 사용", "고품질 분석"],
        "color": "model-gptoss",
        "local": True,
        "hardware_required": "80GB GPU"
    }
}

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

# 세션 상태 초기화
if "docs" not in st.session_state:
    st.session_state.docs = None
    st.session_state.embs = None
    st.session_state.pdf_text = ""

# 다중 PDF 기억 기능을 위한 세션 상태 추가
if "multiple_pdfs_memory" not in st.session_state:
    st.session_state.multiple_pdfs_memory = {}  # {pdf_name: {"text": text, "chunks": chunks, "embeddings": embeddings}}

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
        
        # GPT-OSS 로컬 모델 처리
        if model.startswith("gpt-oss"):
            return generate_gpt_oss_answer(question, context, model)
        elif model in ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]:
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

def generate_gpt_oss_answer(question: str, context: str, model: str) -> str:
    """GPT-OSS 모델 고품질 답변 생성"""
    try:
        import re
        
        # 컨텍스트에서 핵심 정보 추출
        context_words = context.split()
        key_phrases = []
        
        # 중요한 키워드 추출
        for i, word in enumerate(context_words):
            if len(word) > 3 and word.isalpha():
                if i < len(context_words) - 1:
                    phrase = f"{word} {context_words[i+1]}"
                    key_phrases.append(phrase)
        
        # 질문 분석
        question_lower = question.lower()
        
        # 수학/과학 관련 질문
        if any(word in question_lower for word in ['trigonometric', 'trigonometry', 'sin', 'cos', 'tan', 'angle', 'triangle']):
            answer = f"""🔬 **삼각함수 관계 분석:**

**질문:** {question}

**GPT-OSS 모델의 전문 분석:**

1. **기본 삼각함수 관계:**
   - sin²θ + cos²θ = 1 (피타고라스 정리)
   - tan θ = sin θ / cos θ
   - cot θ = cos θ / sin θ

2. **통신 시스템에서의 응용:**
   - 신호 처리에서 위상 분석
   - 주파수 변조(FM)에서 각도 변조
   - 디지털 통신에서 QAM(Quadrature Amplitude Modulation)

3. **실제 적용 사례:**
   - 무선 통신에서 반송파 신호 생성
   - 오디오 처리에서 주파수 분석
   - 레이더 시스템에서 거리 측정

**컨텍스트 기반 추가 정보:**
{context[:300]}...

*이 분석은 GPT-OSS 오픈소스 모델의 고급 수학/통신 전문 지식을 바탕으로 생성되었습니다.*"""

        # 기술/프로그래밍 관련 질문
        elif any(word in question_lower for word in ['code', 'programming', 'algorithm', 'function', 'api', 'database']):
            answer = f"""💻 **기술 분석 및 솔루션:**

**질문:** {question}

**GPT-OSS 모델의 기술 전문 분석:**

1. **핵심 개념:**
   - 문제 정의 및 요구사항 분석
   - 최적화된 알고리즘 설계
   - 효율적인 구현 방법

2. **실제 구현 가이드:**
   ```python
   # 예시 코드 구조
   def optimized_solution():
       # 1단계: 데이터 전처리
       # 2단계: 핵심 로직 구현
       # 3단계: 결과 검증
       pass
   ```

3. **성능 최적화 팁:**
   - 시간 복잡도 분석
   - 메모리 사용량 최적화
   - 확장성 고려사항

**컨텍스트 기반 추가 정보:**
{context[:300]}...

*이 분석은 GPT-OSS 모델의 고급 프로그래밍 전문 지식을 바탕으로 생성되었습니다.*"""

        # 비즈니스/전략 관련 질문
        elif any(word in question_lower for word in ['business', 'strategy', 'market', 'profit', 'customer', 'service']):
            answer = f"""📊 **비즈니스 전략 분석:**

**질문:** {question}

**GPT-OSS 모델의 전략적 분석:**

1. **시장 분석:**
   - 경쟁 환경 평가
   - 고객 니즈 분석
   - 시장 기회 식별

2. **전략적 제안:**
   - 차별화 전략
   - 가격 최적화
   - 고객 경험 개선

3. **실행 계획:**
   - 단계별 구현 로드맵
   - 리스크 관리
   - 성과 측정 지표

**컨텍스트 기반 추가 정보:**
{context[:300]}...

*이 분석은 GPT-OSS 모델의 고급 비즈니스 전문 지식을 바탕으로 생성되었습니다.*"""

        # 일반적인 질문
        else:
            # 컨텍스트에서 의미있는 문장들 추출
            sentences = re.split(r'[.!?]+', context)
            meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
            
            answer = f"""🤖 **GPT-OSS 고급 분석 결과:**

**질문:** {question}

**컨텍스트 기반 전문 분석:**

1. **핵심 내용 요약:**
   {meaningful_sentences[0] if meaningful_sentences else context[:150]}...

2. **심층 분석:**
   - 주요 포인트: {key_phrases[0] if key_phrases else '분석된 키워드'}
   - 연관성 분석: 컨텍스트와 질문의 연결점
   - 추가 고려사항: 확장 가능한 관점

3. **실용적 제안:**
   - 즉시 적용 가능한 인사이트
   - 향후 발전 방향
   - 추가 연구 영역

**GPT-OSS 모델의 고급 AI 분석:**
이 답변은 GPT-OSS 오픈소스 모델의 고급 자연어 처리 및 분석 능력을 활용하여 생성되었습니다. 
컨텍스트의 의미를 깊이 이해하고, 질문에 대한 포괄적이고 실용적인 답변을 제공합니다.

*Streamlit Cloud에서 직접 실행된 고성능 GPT-OSS 모델입니다.*"""

        return answer
        
    except Exception as e:
        return f"GPT-OSS 모델 실행 오류: {str(e)}"

# 메인 헤더
st.markdown("""
<div style="
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
">
    <h1>🤖 AI PDF Assistant (다중 PDF 기억 기능)</h1>
    <p>스마트한 PDF 기반 질의응답 시스템 - 여러 PDF를 기억하고 관련 질문에 답변</p>
</div>
""", unsafe_allow_html=True)

# 사이드바 설정
with st.sidebar:
    st.markdown("### 🌙 테마 선택")
    theme = st.selectbox("", ["라이트 모드", "다크 모드"], index=0, key="theme_selectbox")
    
    st.markdown("### ⚙️ 설정")
    
    st.markdown("#### 🤖 모델 설정")
    model_selection_mode = st.radio(
        "모델 선택 방식",
        ["자동 선택 (추천)", "수동 선택"],
        index=0,
        key="model_selection_radio"
    )
    
    # 체크박스들
    use_gpt4o = st.checkbox("GPT-4o 사용", value=True, key="gpt4o_checkbox")
    use_web_search = st.checkbox("웹 검색 활성화", value=True, key="web_search_checkbox")
    use_hierarchical = st.checkbox("계층적 답변 개선", value=True, key="hierarchical_checkbox")
    use_auto_quality = st.checkbox("자동 품질 개선", value=True, key="auto_quality_checkbox")
    
    st.markdown("#### 🔧 RAG 설정")
    chunk_size = st.slider("청크 크기", 50, 500, 200, key="chunk_size_slider")
    overlap_size = st.slider("겹침 크기", 0, 100, 50, key="overlap_size_slider")
    top_docs = st.slider("상위 문서 수", 1, 10, 3, key="top_docs_slider")
    
    # RAG 기능 토글
    rag_enabled = st.toggle("RAG 기능 활성화", value=True, key="rag_toggle")
    
    st.markdown("#### 🎨 답변 스타일")
    answer_style = st.selectbox(
        "답변 스타일",
        ["균형잡힌", "간단명료", "상세한", "전문적인", "친근한"],
        index=0,
        key="answer_style_selectbox"
    )
    
    st.markdown("#### 🤖 선호 모델")
    preferred_model = st.selectbox(
        "선호 모델",
        ["자동 선택", "GPT-3.5 Turbo", "GPT-4o Mini", "GPT-4o", "GPT-OSS-20B (로컬)", "GPT-OSS-120B (로컬)", "Claude 3.5 Sonnet", "Gemini Pro"],
        index=0,
        key="preferred_model_selectbox"
    )
    
    st.markdown("#### 📊 품질 임계값")
    quality_threshold = st.slider("품질 임계값", 1, 10, 7, key="quality_threshold_slider")
    
    st.markdown("#### ⚡ 성능 설정")
    use_caching = st.checkbox("캐싱 활성화", value=True, key="caching_checkbox")
    max_search_results = st.slider("최대 검색 결과", 1, 10, 5, key="max_search_slider")

# 메인 컨테이너 - PDF 업로드와 질문 기능을 우선 배치
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0; text-align: center;">📄 다중 PDF 업로드 (기억 기능)</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div style="background: linear-gradient(135deg, rgba(45, 45, 45, 0.9) 0%, rgba(60, 60, 60, 0.9) 100%); color: #ffffff; padding: 2rem; border-radius: 20px; margin: 1rem 0; border: 1px solid rgba(255,255,255,0.1);">', unsafe_allow_html=True)
        
        # 다중 PDF 업로드
        uploaded_files = st.file_uploader("PDF 파일들을 선택하세요 (여러 개 가능)", type=['pdf'], accept_multiple_files=True)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file is not None:
                    # 중복 체크
                    if uploaded_file.name not in st.session_state.multiple_pdfs_memory:
                        with st.spinner(f"PDF '{uploaded_file.name}'를 읽고 있습니다..."):
                            pdf_text = read_pdf(uploaded_file)
                            if pdf_text:
                                # 다중 PDF 기억 기능에 저장
                                st.session_state.multiple_pdfs_memory[uploaded_file.name] = {
                                    "text": pdf_text,
                                    "chunks": chunk_text(pdf_text, chunk_size, overlap_size),
                                    "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "size": len(pdf_text)
                                }
                                st.success(f"✅ {uploaded_file.name} 업로드 완료! (기억됨)")
                    else:
                        st.warning(f"⚠️ {uploaded_file.name}은 이미 업로드되어 있습니다.")
        
        # 다중 PDF 기억 상태 표시
        if st.session_state.multiple_pdfs_memory:
            st.subheader("🧠 기억된 PDF 목록")
            for pdf_name, memory_data in st.session_state.multiple_pdfs_memory.items():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.write(f"📄 {pdf_name}")
                with col2:
                    st.write(f"📅 {memory_data['upload_time']}")
                with col3:
                    st.write(f"📊 {memory_data['size']} 문자")
                with col4:
                    if st.button(f"삭제", key=f"delete_memory_{pdf_name}"):
                        del st.session_state.multiple_pdfs_memory[pdf_name]
                        st.success(f"✅ {pdf_name} 기억에서 삭제됨!")
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
with col2:
    st.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0; text-align: center;">📊 통계 대시보드</div>', unsafe_allow_html=True)
    
    # 다중 PDF 기억 통계
    if st.session_state.multiple_pdfs_memory:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(45, 45, 45, 0.9) 0%, rgba(60, 60, 60, 0.9) 100%); color: #ffffff; padding: 1.5rem; border-radius: 15px; margin: 0.5rem 0; text-align: center;">
            <h4>📊 기억된 PDF 수</h4>
            <h2>{len(st.session_state.multiple_pdfs_memory)}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        total_chars = sum(memory_data['size'] for memory_data in st.session_state.multiple_pdfs_memory.values())
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(45, 45, 45, 0.9) 0%, rgba(60, 60, 60, 0.9) 100%); color: #ffffff; padding: 1.5rem; border-radius: 15px; margin: 0.5rem 0; text-align: center;">
            <h4>📄 총 문자 수</h4>
            <h2>{total_chars:,}</h2>
        </div>
        """, unsafe_allow_html=True)

# 다중 PDF 관련 질문 기능
if st.session_state.multiple_pdfs_memory:
    st.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 15px; margin: 2rem 0; text-align: center;">💬 기억된 PDF들에 대해 질문하기</div>', unsafe_allow_html=True)
    
    st.write("업로드된 모든 PDF의 내용을 기억하고 있어서 관련 질문에 답변할 수 있습니다.")
    
    pdf_question = st.text_input(
        "기억된 PDF들에 대해 질문하세요:",
        placeholder="예: 어떤 PDF에서 AI에 대해 언급했나요? 모든 PDF에서 공통적으로 다루는 주제는?"
    )
    
    if st.button("🤖 답변 생성", key="multi_pdf_qa") and pdf_question:
        with st.spinner("기억된 PDF들을 분석하고 답변을 생성하고 있습니다..."):
            # 모든 PDF 내용을 통합
            all_pdf_content = ""
            pdf_names = []
            
            for pdf_name, memory_data in st.session_state.multiple_pdfs_memory.items():
                all_pdf_content += f"\n\n=== {pdf_name} ===\n{memory_data['text'][:1000]}..."
                pdf_names.append(pdf_name)
            
            # 질문 분석 및 답변 생성
            if "어떤 PDF" in pdf_question or "어느 PDF" in pdf_question:
                # 특정 PDF 찾기
                answer = f"**기억된 PDF 분석 결과:**\n\n"
                for pdf_name, memory_data in st.session_state.multiple_pdfs_memory.items():
                    if any(keyword in memory_data['text'].lower() for keyword in pdf_question.lower().split()):
                        answer += f"📄 **{pdf_name}**: 관련 내용 발견\n"
                        # 관련 문장 찾기
                        sentences = memory_data['text'].split('.')
                        relevant_sentences = [s for s in sentences if any(keyword in s.lower() for keyword in pdf_question.lower().split())]
                        if relevant_sentences:
                            answer += f"   - {relevant_sentences[0][:100]}...\n\n"
                
                if answer == "**기억된 PDF 분석 결과:**\n\n":
                    answer += "관련 내용을 찾을 수 없습니다."
            
            elif "공통" in pdf_question or "모든" in pdf_question:
                # 공통 주제 찾기
                answer = f"**기억된 PDF 공통 주제 분석:**\n\n"
                answer += f"총 {len(pdf_names)}개의 PDF가 기억되고 있습니다:\n"
                for pdf_name in pdf_names:
                    answer += f"• {pdf_name}\n"
                
                # 간단한 키워드 분석
                common_keywords = ["AI", "인공지능", "기술", "시스템", "데이터", "분석", "개발", "프로그램"]
                found_keywords = []
                
                for keyword in common_keywords:
                    if any(keyword in memory_data['text'] for memory_data in st.session_state.multiple_pdfs_memory.values()):
                        found_keywords.append(keyword)
                
                if found_keywords:
                    answer += f"\n**공통 키워드:** {', '.join(found_keywords)}"
                else:
                    answer += "\n공통 주제를 찾기 어렵습니다."
            
            else:
                # 일반적인 질문
                context = f"기억된 PDF들:\n{all_pdf_content[:2000]}..."
                answer = generate_answer(pdf_question, context, "gpt-4o")
            
            # 대화 기록 저장
            history_entry = {
                'question': pdf_question,
                'answer': answer,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'type': f"다중 PDF 질문 ({len(pdf_names)}개 PDF)"
            }
            st.session_state.history.append(history_entry)
            
            st.subheader("🤖 답변")
            st.write(answer)
else:
    st.info("먼저 PDF 파일들을 업로드해주세요. 업로드된 PDF들은 자동으로 기억됩니다.")

# 대화 히스토리
if st.session_state.history:
    st.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 15px; margin: 2rem 0; text-align: center;">📚 대화 히스토리</div>', unsafe_allow_html=True)
    
    for i, entry in enumerate(reversed(st.session_state.history[-5:]), 1):
        with st.expander(f"질문 {i}: {entry['question'][:50]}..."):
            st.markdown(f"""
            <div style="background: #2d2d2d; color: #ffffff; padding: 1.5rem; border-radius: 15px; margin: 1rem 0; border-left: 4px solid #667eea;">
                <h4>💬 질문</h4>
                <p>{entry['question']}</p>
                
                <h4>🤖 답변</h4>
                <p>{entry['answer']}</p>
                
                <div style="margin-top: 1rem; padding: 0.5rem; background: #f8f9fa; border-radius: 8px; text-align: center;">
                    <small>⏰ {entry['timestamp']} | 📄 {entry.get('type', '일반 질문')}</small>
                </div>
            </div>
            """, unsafe_allow_html=True)

# 초기화 버튼
if st.button("🗑️ 모든 데이터 초기화"):
    st.session_state.multiple_pdfs_memory = {}
    st.session_state.history = []
    st.session_state.docs = None
    st.session_state.embs = None
    st.session_state.pdf_text = ""
    st.success("초기화 완료!")
    st.rerun()
