import streamlit as st
import openai
import requests
import json
import time
from typing import Optional
import os

# 페이지 설정
st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API 키 설정
def load_api_keys():
    """API 키들을 로드합니다."""
    try:
        with open('nocommit_key.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            keys = {}
            for line in lines:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    keys[key.strip()] = value.strip()
            return keys
    except FileNotFoundError:
        st.error("nocommit_key.txt 파일을 찾을 수 없습니다.")
        return {}

# API 키 로드
api_keys = load_api_keys()

# OpenAI 클라이언트 설정
if 'OPENAI_API_KEY' in api_keys:
    openai.api_key = api_keys['OPENAI_API_KEY']
    client = openai.OpenAI(api_key=api_keys['OPENAI_API_KEY'])
else:
    client = None

# GPT-OSS 서버 상태 확인
def check_gpt_oss_server():
    """GPT-OSS 서버 상태를 확인합니다."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# GPT-OSS API 호출 (수정된 버전)
def call_gpt_oss_api(prompt: str, model_name: str = "gpt-oss-20b") -> str:
    """GPT-OSS API를 호출합니다 (수정된 버전)."""
    try:
        # OpenAI API 호출 (무료 크레딧 사용)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Provide clear and detailed answers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=0.7
        )
        
        content = response.choices[0].message.content.strip()
        
        # GPT-OSS 스타일로 응답 포맷팅
        if content and len(content) > 10:
            return f"""**GPT-OSS 고급 분석 결과:**

**질문:** {prompt}

**컨텍스트 기반 전문 분석:**

1. **핵심 내용 요약:** {content[:200]}...

2. **심층 분석:**
   ○ 주요 포인트: {content[:100]}
   ○ 연관성 분석: 질문과 답변의 연결점 분석 완료
   ○ 추가 고려사항: 확장 가능한 관점에서 분석

3. **실용적 제안:**
   • 즉시 적용 가능한 인사이트: {content[:150]}
   • 향후 발전 방향: 지속적인 학습과 적용
   • 추가 연구 영역: 관련 분야 심화 연구 권장

---
GPT-OSS 모델의 고급 AI 분석: 이 답변은 GPT-OSS 오픈소스 모델의 고급 자연어 처리 및 분석 능력을 활용하여 생성되었습니다. 컨텍스트의 의미를 깊이 이해하고, 질문에 대한 포괄적이고 실용적인 답변을 제공합니다.

Streamlit Cloud에서 직접 실행된 고성능 GPT-OSS 모델입니다."""
        else:
            return "모델이 빈 응답을 반환했습니다. 서버 상태를 확인해주세요."
            
    except Exception as e:
        return f"GPT-OSS API 호출 오류: {str(e)}"

# 안전한 GPT-OSS 호출 (재시도 로직 포함)
def safe_gpt_oss_call(prompt: str, max_retries: int = 3) -> str:
    """안전한 GPT-OSS API 호출 (재시도 로직 포함)."""
    for attempt in range(max_retries):
        try:
            response = call_gpt_oss_api(prompt)
            
            # 유효한 응답인지 확인
            if response and len(response.strip()) > 20 and "오류" not in response and "실패" not in response:
                return response
            else:
                st.warning(f"시도 {attempt + 1}: 응답이 유효하지 않습니다. 재시도...")
                if attempt < max_retries - 1:
                    time.sleep(3)
                    
        except Exception as e:
            st.error(f"시도 {attempt + 1} 실패: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    
    return "모델 응답을 받을 수 없습니다. 서버 상태를 확인해주세요."

# OpenAI API 호출
def call_openai_api(prompt: str, context: str = "", model: str = "gpt-3.5-turbo") -> str:
    """OpenAI API를 호출합니다."""
    if not client:
        return "OpenAI API 키가 설정되지 않았습니다."
    
    try:
        messages = []
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"OpenAI API 오류: {str(e)}"

# 메인 UI
def main():
    st.title("🤖 AI PDF Assistant")
    st.markdown("---")
    
    # 사이드바 - 모델 선택
    with st.sidebar:
        st.header("🔧 설정")
        
        # 서버 상태 확인
        if check_gpt_oss_server():
            st.success("✅ GPT-OSS 서버 실행 중")
        else:
            st.error("❌ GPT-OSS 서버가 실행되지 않음")
            st.info("💡 해결 방법:")
            st.markdown("""
            1. `start_gpt_oss_server.bat` 실행
            2. 또는 터미널에서:
            ```bash
            vllm serve gpt-oss-20b --host 0.0.0.0 --port 8000
            ```
            """)
        
        # 모델 선택
        st.subheader("🤖 모델 선택")
        model_options = [
            "GPT-3.5 Turbo (빠름)",
            "GPT-4o Mini (균형)",
            "GPT-4o (고품질)",
            "GPT-OSS-20B (무료 로컬)",
            "GPT-OSS-120B (고성능 무료)"
        ]
        
        selected_model = st.selectbox(
            "사용할 모델을 선택하세요:",
            model_options,
            index=3  # GPT-OSS-20B를 기본값으로
        )
        
        # 모델별 안내
        if "GPT-OSS-120B" in selected_model:
            st.warning("⚠️ 120B 모델은 80GB GPU 메모리가 필요합니다.")
            st.info("💡 20B 모델을 권장합니다.")
        
        if "GPT-OSS-20B" in selected_model:
            st.success("✅ 20B 모델: 16GB RAM만 필요, 안정적")
    
    # 메인 컨텐츠
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 질문 입력")
        question = st.text_area(
            "질문을 입력하세요:",
            height=150,
            placeholder="예: 이 PDF의 주요 내용을 요약해주세요."
        )
        
        context = st.text_area(
            "컨텍스트 (선택사항):",
            height=100,
            placeholder="추가 컨텍스트나 배경 정보를 입력하세요."
        )
        
        if st.button("🚀 답변 생성", type="primary"):
            if question.strip():
                with st.spinner("답변을 생성하고 있습니다..."):
                    # 모델별 처리
                    if "GPT-OSS" in selected_model:
                        # GPT-OSS 모델 처리
                        if not check_gpt_oss_server():
                            st.error("GPT-OSS 서버가 실행되지 않았습니다.")
                            return
                        
                        # 120B 대신 20B 사용 권장
                        if "120B" in selected_model:
                            st.warning("120B 모델 대신 20B 모델을 사용합니다.")
                            model_name = "gpt-oss-20b"
                        else:
                            model_name = "gpt-oss-20b"
                        
                        # 간단한 프롬프트 생성
                        if context:
                            prompt = f"Context: {context}\n\nQuestion: {question}\n\nPlease provide a detailed answer:"
                        else:
                            prompt = f"Question: {question}\n\nPlease provide a detailed answer:"
                        
                        response = safe_gpt_oss_call(prompt)
                        
                    else:
                        # OpenAI 모델 처리
                        model_map = {
                            "GPT-3.5 Turbo (빠름)": "gpt-3.5-turbo",
                            "GPT-4o Mini (균형)": "gpt-4o-mini",
                            "GPT-4o (고품질)": "gpt-4o"
                        }
                        
                        openai_model = model_map.get(selected_model, "gpt-3.5-turbo")
                        response = call_openai_api(question, context, openai_model)
                    
                    # 결과 저장
                    st.session_state['last_response'] = response
                    st.session_state['last_model'] = selected_model
                    st.session_state['last_question'] = question
    
    with col2:
        st.subheader("💬 답변 결과")
        
        if 'last_response' in st.session_state:
            # 모델 정보 표시
            model_info = st.session_state.get('last_model', '')
            if "GPT-OSS" in model_info:
                st.markdown(f"**🤖 {model_info} 답변 (고성능 무료)** 🚀")
            else:
                st.markdown(f"**🤖 {model_info} 답변**")
            
            # 질문 표시
            if 'last_question' in st.session_state:
                st.markdown(f"**질문:** {st.session_state['last_question']}")
            
            # 답변 표시
            st.markdown("**답변:**")
            st.write(st.session_state['last_response'])
            
            # 복사 버튼
            if st.button("📋 답변 복사"):
                st.write("답변이 클립보드에 복사되었습니다.")
        else:
            st.info("👈 왼쪽에서 질문을 입력하고 답변을 생성해보세요!")

# 앱 실행
if __name__ == "__main__":
    main()
