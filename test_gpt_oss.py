import streamlit as st
import openai
import requests
import json
import time
from typing import Optional
import os

# 페이지 설정
st.set_page_config(
    page_title="GPT-OSS Test",
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

# 새로운 GPT-OSS API 호출 함수
def call_gpt_oss_api_new(user_question: str, context: str = "") -> str:
    """새로운 GPT-OSS API 호출 함수"""
    try:
        if not client:
            return "OpenAI API 키가 설정되지 않았습니다."
        
        # 간단한 프롬프트
        prompt = user_question
        
        # OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer questions directly and accurately."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        # 실제 AI 응답
        ai_response = response.choices[0].message.content.strip()
        
        # 새로운 포맷팅
        formatted_response = f"""**새로운 GPT-OSS 답변**

**질문:** {user_question}

**답변:** {ai_response}

---
이것은 새로운 테스트 버전입니다."""
        
        return formatted_response
            
    except Exception as e:
        return f"오류: {str(e)}"

# 메인 UI
def main():
    st.title("🚀 GPT-OSS 테스트")
    st.markdown("---")
    
    # 사이드바
    with st.sidebar:
        st.header("🔧 설정")
        
        if client:
            st.success("✅ OpenAI API 연결됨")
        else:
            st.error("❌ OpenAI API 키 없음")
    
    # 메인 컨텐츠
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 질문 입력")
        question = st.text_area(
            "질문을 입력하세요:",
            height=150,
            placeholder="예: What is the transistors?"
        )
        
        if st.button("🚀 답변 생성", type="primary"):
            if question.strip():
                with st.spinner("답변을 생성하고 있습니다..."):
                    response = call_gpt_oss_api_new(question)
                    
                    # 결과 저장
                    st.session_state['last_response'] = response
                    st.session_state['last_question'] = question
    
    with col2:
        st.subheader("💬 답변 결과")
        
        if 'last_response' in st.session_state:
            # 질문 표시
            if 'last_question' in st.session_state:
                st.markdown(f"**질문:** {st.session_state['last_question']}")
            
            # 답변 표시
            st.markdown("**답변:**")
            st.write(st.session_state['last_response'])
        else:
            st.info("👈 왼쪽에서 질문을 입력하고 답변을 생성해보세요!")

# 앱 실행
if __name__ == "__main__":
    main()
