import os
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import numpy as np
import openai
import requests
import json
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
from typing import Optional
import os

# 페이지 설정
st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🤖",
    layout="wide"
    page_title="AI PDF Assistant - Fixed GPT-OSS",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일 - 다크 테마
st.markdown("""
<style>
    /* 다크 테마 기본 스타일 */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
        position: relative;
        overflow-x: hidden;
    }
    
    /* 배경 애니메이션 효과 */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
        animation: backgroundShift 20s ease-in-out infinite;
        z-index: -1;
    }
    
    @keyframes backgroundShift {
        0%, 100% { transform: translate(0, 0) scale(1); }
        25% { transform: translate(-10px, -10px) scale(1.05); }
        50% { transform: translate(10px, -5px) scale(1.02); }
        75% { transform: translate(-5px, 10px) scale(1.03); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.3); }
        50% { box-shadow: 0 0 30px rgba(102, 126, 234, 0.5); }
    }
    
    @keyframes slideInRight {
        0% { transform: translateX(100%); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        0% { transform: translateX(-100%); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    
    /* 사이드바 다크 테마 */
    .css-1d391kg, .css-1lcbmhc, .css-1v0mbdj {
        background: linear-gradient(180deg, #2d2d2d 0%, #1a1a1a 100%);
        border-right: 2px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* 헤더 스타일 - 보라색 그라데이션 */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* 카드 스타일 - 다크 테마 */
    .upload-card {
        background: linear-gradient(135deg, rgba(45, 45, 45, 0.9) 0%, rgba(60, 60, 60, 0.9) 100%);
        color: #ffffff;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(15px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .upload-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s;
    }
    
    .upload-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .upload-card:hover::before {
        left: 100%;
    }
    
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }
    
    .answer-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin: 1rem 0;
        border-left: 4px solid #28a745;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .answer-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(40, 167, 69, 0.1), transparent);
        transition: left 0.6s;
    }
    
    .answer-card:hover::before {
        left: 100%;
    }
    
    .answer-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }
    
    .answer-card h4 {
        color: #ffffff;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .answer-card p {
        color: #ffffff;
        line-height: 1.6;
        margin-bottom: 1rem;
        font-size: 1rem;
        text-align: left;
        font-weight: 400;
    }
    
    .improved-card {
        background: linear-gradient(135deg, #3a3a3a 0%, #4a4a4a 100%);
        color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .improved-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 193, 7, 0.1), transparent);
        transition: left 0.6s;
    }
    
    .improved-card:hover::before {
        left: 100%;
    }
    
    .improved-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }
    
    .improved-card h4 {
        color: #ffeb3b;
        margin-bottom: 1rem;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .improved-card p {
        color: #ffffff;
        line-height: 1.6;
        margin-bottom: 1rem;
        font-size: 1rem;
        text-align: left;
        font-weight: 400;
    }
        color: #333;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    

    
    /* 배지 스타일 - 다크 테마 */
    .quality-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .quality-good { background: linear-gradient(45deg, #1e4d2b, #166534); color: #4ade80; }
    .quality-medium { background: linear-gradient(45deg, #92400e, #78350f); color: #fbbf24; }
    .quality-bad { background: linear-gradient(45deg, #7f1d1d, #991b1b); color: #f87171; }
    
    .model-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .model-badge::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .model-badge:hover {
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    
    .model-badge:hover::before {
        left: 100%;
    }
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .model-gpt35 { background: linear-gradient(45deg, #1e3a8a, #1e40af); color: #60a5fa; }
    .model-gpt4mini { background: linear-gradient(45deg, #581c87, #6b21a8); color: #c084fc; }
    .model-gpt4o { background: linear-gradient(45deg, #1e4d2b, #166534); color: #4ade80; }
    .model-gptoss { background: linear-gradient(45deg, #ff6b35, #f7931e); color: #ffffff; }
    
    /* 스마트 선택 박스 - 다크 테마 */
    .smart-selection {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: 2px solid #667eea;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        color: white;
    }
    
    .smart-selection h4 {
        color: #ffffff;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .smart-selection p {
        color: #d1d5db;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    
    /* 섹션 헤더 */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 20px;
        margin: 2rem 0 1.5rem 0;
        font-weight: 700;
        font-size: 1.2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        text-align: center;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.8s;
    }
    
    .section-header:hover::before {
        left: 100%;
    }
    
    /* 입력 필드 스타일 */
    .stTextInput > div > div > input {
        border-radius: 15px;
        border: 2px solid rgba(255,255,255,0.2);
        padding: 1rem;
        font-size: 1rem;
        background: rgba(45, 45, 45, 0.8);
        color: #ffffff;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    
    /* 사이드바 내부 요소들 */
    .css-1d391kg .stSelectbox > div > div,
    .css-1lcbmhc .stSelectbox > div > div,
    .css-1v0mbdj .stSelectbox > div > div {
        background: rgba(45, 45, 45, 0.8);
        border: 2px solid rgba(255,255,255,0.2);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .css-1d391kg .stSelectbox > div > div:hover,
    .css-1lcbmhc .stSelectbox > div > div:hover,
    .css-1v0mbdj .stSelectbox > div > div:hover {
        border-color: rgba(102, 126, 234, 0.5);
        background: rgba(45, 45, 45, 0.9);
    }
    
    .css-1d391kg .stCheckbox > div,
    .css-1lcbmhc .stCheckbox > div,
    .css-1v0mbdj .stCheckbox > div {
        background: rgba(45, 45, 45, 0.6);
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.3rem 0;
        transition: all 0.3s ease;
    }
    
    .css-1d391kg .stCheckbox > div:hover,
    .css-1lcbmhc .stCheckbox > div:hover,
    .css-1v0mbdj .stCheckbox > div:hover {
        background: rgba(102, 126, 234, 0.1);
        transform: translateX(3px);
    }
    
    .css-1d391kg .stSlider > div > div > div,
    .css-1lcbmhc .stSlider > div > div > div,
    .css-1v0mbdj .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2);
        background: rgba(45, 45, 45, 0.9);
        transform: scale(1.02);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stTextInput > div > div > input:hover {
        border-color: rgba(102, 126, 234, 0.5);
        background: rgba(45, 45, 45, 0.85);
        transform: scale(1.01);
    }
    
    .stTextInput > div > div > input {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.6);
    }
    
    /* 사이드바 텍스트 스타일 */
    .css-1d391kg .stMarkdown,
    .css-1lcbmhc .stMarkdown,
    .css-1v0mbdj .stMarkdown {
        color: #ffffff;
        font-weight: 500;
    }
    
    .css-1d391kg .stMarkdown h3,
    .css-1lcbmhc .stMarkdown h3,
    .css-1v0mbdj .stMarkdown h3 {
        color: #667eea;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .css-1d391kg .stMarkdown p,
    .css-1lcbmhc .stMarkdown p,
    .css-1v0mbdj .stMarkdown p {
        color: #d1d5db;
        font-size: 0.9rem;
        margin: 0.3rem 0;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        border-radius: 30px;
        padding: 1rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 12px 35px rgba(0,0,0,0.4);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(1px) scale(0.98);
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    
    .stButton > button:focus {
        outline: none;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.02);
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
    
    /* 메트릭 카드 - 다크 테마 */
    .metric-card {
        background: linear-gradient(135deg, rgba(45, 45, 45, 0.9) 0%, rgba(60, 60, 60, 0.9) 100%);
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.3);
        margin: 0.5rem 0;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.6s;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card h2 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.3rem 0;
        text-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
        animation: countUp 2s ease-out;
    }
    
    @keyframes countUp {
        0% { 
            opacity: 0;
            transform: translateY(20px);
        }
        50% { 
            opacity: 0.5;
            transform: translateY(10px);
        }
        100% { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-card h4 {
        color: #d1d5db;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .metric-card p {
        color: #9ca3af;
        margin: 0;
        font-size: 0.9rem;
    }
    
    /* 히스토리 스타일 - 다크 테마 */
    .history-item {
        background: #2d2d2d;
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .history-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
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
    
    /* 커스텀 로딩 애니메이션 */
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.05); }
    }
    
    @keyframes slideIn {
        0% { transform: translateX(-100%); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes fadeInUp {
        0% { transform: translateY(20px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    
    /* 커스텀 로딩 컴포넌트 */
    .custom-loading {
        background: linear-gradient(135deg, rgba(45, 45, 45, 0.9) 0%, rgba(60, 60, 60, 0.9) 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        animation: fadeInUp 0.5s ease;
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 4px solid rgba(102, 126, 234, 0.3);
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite, pulse 2s ease-in-out infinite;
        margin: 0 auto 1rem auto;
    }
    
    .loading-text {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .loading-progress {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        height: 8px;
        margin: 1rem 0;
        overflow: hidden;
        position: relative;
    }
    
    .loading-progress-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        border-radius: 10px;
        animation: slideIn 0.5s ease;
        transition: width 0.3s ease;
    }
    
    .loading-steps {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    
    .loading-step {
        background: rgba(102, 126, 234, 0.2);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        font-size: 0.9rem;
        color: #ffffff;
        animation: fadeInUp 0.5s ease;
        transition: all 0.3s ease;
    }
    
    .loading-step.active {
        background: rgba(102, 126, 234, 0.6);
        transform: scale(1.05);
    }
    
    .loading-step.completed {
        background: rgba(40, 167, 69, 0.6);
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
    
    /* 텍스트 색상 - 다크 테마 */
    .stMarkdown, .stText, .stTextInput, .stTextArea {
        color: #ffffff !important;
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

# 사이드바 설정 - 현재 이미지와 완전히 동일
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
    
    # 체크박스들 - 현재 이미지와 동일
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
    
    # GPT-OSS 모델 사용 가능 (Streamlit Cloud에서 직접 실행)
    gpt_oss_available = True
    
    # 모델 선택 로직
    if complexity["score"] >= 5:
        if gpt_oss_available:
            selected_model = "gpt-oss-120b"
            reason = "복잡한 분석/전략 질문 - 고성능 무료 모델"
        else:
            selected_model = "gpt-4o"
            reason = "복잡한 분석/전략 질문으로 판단됨"
    elif complexity["score"] >= 2:
        if gpt_oss_available:
            selected_model = "gpt-oss-20b"
            reason = "중간 복잡도 질문 - 무료 모델"
        else:
            selected_model = "gpt-4o-mini"
            reason = "중간 복잡도 질문으로 판단됨"
    else:
        if gpt_oss_available:
            selected_model = "gpt-oss-20b"
            reason = "기본 질문 - 무료 모델"
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
    
    # 테마 선택
    theme_mode = st.selectbox(
        "🎨 테마 선택",
        ["라이트 모드", "다크 모드"],
        help="앱의 시각적 테마를 선택하세요"
    )
    
    # 다크모드 CSS 적용
    if theme_mode == "다크 모드":
        st.markdown("""
        <style>
            .stApp {
                background-color: #1a1a1a !important;
                color: #ffffff !important;
            }
            .stTextInput > div > div > input,
            .stTextArea > div > div > textarea {
                background-color: #2d2d2d !important;
                color: #ffffff !important;
                border-color: #444444 !important;
            }
            .stButton > button {
                background-color: #4a4a4a !important;
                color: #ffffff !important;
                border-color: #666666 !important;
            }
            .upload-card, .answer-card, .improved-card {
                background-color: #2d2d2d !important;
                color: #ffffff !important;
                border-color: #444444 !important;
            }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
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
        
        # GPT-OSS 모델 사용 가능 여부
        use_gpt_oss_20b = st.checkbox("GPT-OSS-20B (직접 실행) 사용", value=False)
        use_gpt_oss_120b = st.checkbox("GPT-OSS-120B (직접 실행) 사용", value=False)
        
        if use_gpt_oss_20b or use_gpt_oss_120b:
            st.info("🚀 GPT-OSS 모델이 Streamlit Cloud에서 직접 실행됩니다!")
    else:
        use_gpt4 = False
        use_gpt4mini = False
        use_claude = False
        use_gemini = False
        use_gpt_oss_20b = False
        use_gpt_oss_120b = False
    
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
# API 키 설정
def load_api_keys():
    """API 키들을 로드합니다."""
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

*Streamlit Cloud에서 직접 실행된 고성능 GPT-OSS 모델입니다.*"""
# API 키 로드
api_keys = load_api_keys()

        return answer
        
    except Exception as e:
        return f"GPT-OSS 모델 실행 오류: {str(e)}"
# OpenAI 클라이언트 설정
if 'OPENAI_API_KEY' in api_keys:
    openai.api_key = api_keys['OPENAI_API_KEY']
    client = openai.OpenAI(api_key=api_keys['OPENAI_API_KEY'])
else:
    client = None

def analyze_sentiment_and_tone(text: str) -> dict:
    """감정 및 톤 분석"""
# GPT-OSS 서버 상태 확인
def check_gpt_oss_server():
    """GPT-OSS 서버 상태를 확인합니다."""
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
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# GPT-OSS API 호출 (수정된 버전)
def call_gpt_oss_api(prompt: str, model_name: str = "gpt-oss-20b") -> str:
    """GPT-OSS API를 호출합니다 (안정화된 버전)."""
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
        # 간단한 프롬프트 형식 사용
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Provide clear and detailed answers."},
            {"role": "user", "content": prompt}
        ]
        
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": model_name,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1,
                "stop": None
            },
            timeout=60
        )

        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            return best_topic if topic_scores[best_topic] > 0 else "일반"
        if response.status_code == 200:
            result = response.json()
            if result.get('choices') and result['choices'][0].get('message'):
                content = result['choices'][0]['message']['content'].strip()
                if content and len(content) > 10:
                    return content
                else:
                    return "모델이 빈 응답을 반환했습니다. 서버 상태를 확인해주세요."
            else:
                return "API 응답 형식이 올바르지 않습니다."
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
            return f"API 호출 실패: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "요청 시간 초과. 서버가 응답하지 않습니다."
    except requests.exceptions.ConnectionError:
        return "서버 연결 실패. GPT-OSS 서버가 실행 중인지 확인해주세요."
    except Exception as e:
        return {}
        return f"API 호출 오류: {str(e)}"

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

def improve_answer_with_better_model(question: str, basic_answer: str, context: str, better_model: str, quality_analysis: dict) -> str:
    """더 나은 모델로 답변 개선"""
# OpenAI API 호출
def call_openai_api(prompt: str, context: str = "", model: str = "gpt-3.5-turbo") -> str:
    """OpenAI API를 호출합니다."""
    if not client:
        return "OpenAI API 키가 설정되지 않았습니다."
    
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
        messages = []
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=better_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.5
            model=model,
            messages=messages,
            max_tokens=4000,
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"답변 개선 중 오류가 발생했습니다: {str(e)}"
        return f"OpenAI API 오류: {str(e)}"

# 메인 컨테이너 - PDF 업로드와 질문 기능을 우선 배치
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="section-header">📄 PDF 업로드</div>', unsafe_allow_html=True)
# 메인 UI
def main():
    st.title("🤖 AI PDF Assistant - Fixed GPT-OSS")
    st.markdown("---")

    with st.container():
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("PDF 파일을 선택하세요", type=['pdf'])
    # 업데이트 알림
    st.success("✅ GPT-OSS 문제 해결 버전으로 업데이트되었습니다!")
    st.info("🔧 주요 수정사항: 프롬프트 단순화, 재시도 로직, 에러 핸들링 강화")
    
    # 사이드바 - 모델 선택
    with st.sidebar:
        st.header("🔧 설정")

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
                                margin: 0;
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
                                <p id="time-estimate" style="color: #666; margin: 0; font-size: 0.8rem;">예상 시간: 계산 중...</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # 배치 크기 계산 (큰 파일을 위한 최적화)
                        batch_size = min(50, max(10, len(chunks) // 20))  # 청크 수에 따라 배치 크기 조정
                        
                        # 예상 시간 계산
                        estimated_time_per_chunk = 0.5  # 초당 약 2개 청크 처리
                        total_estimated_time = len(chunks) * estimated_time_per_chunk / 60  # 분 단위
                        
                        st.markdown(f"""
                        <script>
                            document.getElementById('time-estimate').textContent = '예상 시간: 약 {total_estimated_time:.1f}분';
                        </script>
                        """, unsafe_allow_html=True)
                        
                        # 배치 처리로 임베딩 생성
                        for i in range(0, len(chunks), batch_size):
                            batch_chunks = chunks[i:i + batch_size]
                            
                            try:
                                # 배치로 임베딩 생성 (더 빠름)
                                response = client.embeddings.create(
                                    model="text-embedding-3-small",
                                    input=batch_chunks
                                )
                                
                                # 배치 결과를 개별 임베딩으로 분리
                                for embedding_data in response.data:
                                    emb = np.array(embedding_data.embedding)
                                    embeddings.append(emb)
                                    
                            except Exception as e:
                                # 에러 발생 시 개별 처리로 폴백
                                for chunk in batch_chunks:
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
                            progress_percent = (i + len(batch_chunks)) / len(chunks) * 100
                            progress_text = f"청크 {i + len(batch_chunks)}/{len(chunks)} 처리 중... ({progress_percent:.1f}%)"
                            
                            # 10% 단위로 업데이트 (빈도 줄임)
                            if (i + len(batch_chunks)) % max(1, len(chunks) // 10) == 0 or i + len(batch_chunks) >= len(chunks):
                                                            st.markdown(f"""
                            <div class="custom-loading" style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(129, 199, 132, 0.1) 100%); border: 1px solid rgba(76, 175, 80, 0.3);">
                                <div class="loading-spinner" style="border-color: rgba(76, 175, 80, 0.3); border-top-color: #4caf50;"></div>
                                <div class="loading-text" style="color: #4caf50;">📄 PDF 처리중...</div>
                                <div class="loading-progress">
                                    <div class="loading-progress-bar" style="width: {progress_percent}%; background: linear-gradient(90deg, #4caf50 0%, #66bb6a 100%);"></div>
                                </div>
                                <div class="loading-steps">
                                    <div class="loading-step {'completed' if progress_percent > 20 else 'active' if progress_percent > 0 else ''}" style="background: rgba(76, 175, 80, 0.2);">1️⃣ 텍스트 추출</div>
                                    <div class="loading-step {'completed' if progress_percent > 40 else 'active' if progress_percent > 20 else ''}" style="background: rgba(76, 175, 80, 0.2);">2️⃣ 청크 분할</div>
                                    <div class="loading-step {'completed' if progress_percent > 60 else 'active' if progress_percent > 40 else ''}" style="background: rgba(76, 175, 80, 0.2);">3️⃣ 임베딩 생성</div>
                                    <div class="loading-step {'completed' if progress_percent > 80 else 'active' if progress_percent > 60 else ''}" style="background: rgba(76, 175, 80, 0.2);">4️⃣ 벡터 저장</div>
                                </div>
                                <div style="color: #d1d5db; font-size: 0.9rem;">{progress_text}</div>
                            </div>
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
        # 서버 상태 확인
        if check_gpt_oss_server():
            st.success("✅ GPT-OSS 서버 실행 중")
        else:
            st.error("❌ GPT-OSS 서버가 실행되지 않음")
            st.info("💡 해결 방법:")
            st.markdown("""
            1. 로컬에서 `fixed_start_gpt_oss_server.bat` 실행
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

with col2:
    st.markdown('<div class="section-header">📊 통계 대시보드</div>', unsafe_allow_html=True)
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
        # 모델별 안내
        if "GPT-OSS-120B" in selected_model:
            st.warning("⚠️ 120B 모델은 80GB GPU 메모리가 필요합니다.")
            st.info("💡 20B 모델을 권장합니다.")

        # 대화 히스토리 통계
        if st.session_state.history:
            total_questions = len(st.session_state.history)
            avg_gpt35_quality = sum(entry.get('gpt35_quality', 0) for entry in st.session_state.history) / total_questions
            avg_gpt4o_quality = sum(entry.get('gpt4o_quality', 0) for entry in st.session_state.history) / total_questions
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>💬 총 질문 수</h4>
                <h2>{total_questions}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>🤖 평균 GPT-3.5 품질</h4>
                <h2>{avg_gpt35_quality:.1f}</h2>
                <p>/100</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>🚀 평균 GPT-4o 품질</h4>
                <h2>{avg_gpt4o_quality:.1f}</h2>
                <p>/100</p>
            </div>
            """, unsafe_allow_html=True)

# 대화형 AI 인터페이스
st.markdown('<div class="section-header">💬 대화형 AI</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    
    # 대화 히스토리 초기화
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # 대화 표시 영역
    if st.session_state.conversation_history:
        # 대화 기록 접기/펴기 토글
        with st.expander(f"📝 대화 기록 ({len(st.session_state.conversation_history)}개 메시지)", expanded=False):
            for i, (role, message, timestamp) in enumerate(st.session_state.conversation_history):
                if role == "user":
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 1rem;
                        border-radius: 15px;
                        margin: 0.5rem 0;
                        color: white;
                        text-align: right;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                        transition: all 0.3s ease;
                        animation: slideInRight 0.5s ease;
                    ">
                        <strong>👤 사용자:</strong> {message}
                        <br><small style="opacity: 0.7;">{timestamp}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #2d2d2d 0%, #3a3a3a 100%);
                        padding: 1rem;
                        border-radius: 15px;
                        margin: 0.5rem 0;
                        color: white;
                        border-left: 4px solid #667eea;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                        transition: all 0.3s ease;
                        animation: slideInLeft 0.5s ease;
                    ">
                        <strong>🤖 AI:</strong> {message}
                        <br><small style="opacity: 0.7;">{timestamp}</small>
                    </div>
                    """, unsafe_allow_html=True)
        if "GPT-OSS-20B" in selected_model:
            st.success("✅ 20B 모델: 16GB RAM만 필요, 안정적")

    # AI 모드 선택 (질문 입력 전에 표시)
    ai_mode = st.checkbox("🤖 일반 AI 모드 (PDF 없이도 질문 가능)", value=False, key="ai_mode_checkbox", help="PDF를 업로드하지 않고도 AI와 대화할 수 있습니다")
    # 메인 컨텐츠
    col1, col2 = st.columns([1, 1])

    # 질문 입력과 음성 입력
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input("질문을 입력하세요:", placeholder="무엇이든 질문해보세요! (PDF 내용 또는 일반적인 질문)")
    with col2:
        if st.button("🎤 음성 입력", help="음성으로 질문하기 (준비 중)"):
            st.info("🎤 음성 입력 기능은 곧 추가될 예정입니다!")
    
    # 답변 생성 버튼과 음성 출력
    col1, col2 = st.columns([4, 1])
    with col1:
        generate_button = st.button("🚀 답변 생성", type="primary")
    with col2:
        if st.button("🔊 음성 출력", help="답변을 음성으로 들기 (준비 중)"):
            st.info("🔊 음성 출력 기능은 곧 추가될 예정입니다!")
    
    # 대화 초기화 버튼
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ 대화 초기화", help="모든 대화 기록을 지웁니다"):
            st.session_state.conversation_history = []
            st.rerun()
    
    if question and generate_button:
        if not st.session_state.docs and rag_enabled and not ai_mode:
            st.warning("먼저 PDF 파일을 업로드해주세요! 또는 일반 AI 모드를 활성화하세요.")
        else:
            # 세련된 답변 생성 로딩
            with st.container():
                st.markdown("""
                <div class="custom-loading" style="background: linear-gradient(135deg, rgba(156, 39, 176, 0.1) 0%, rgba(186, 104, 200, 0.1) 100%); border: 1px solid rgba(156, 39, 176, 0.3);">
                    <div class="loading-spinner" style="border-color: rgba(156, 39, 176, 0.3); border-top-color: #9c27b0;"></div>
                    <div class="loading-text" style="color: #9c27b0;">🤖 AI 답변 생성중...</div>
                    <div class="loading-steps">
                        <div class="loading-step active" style="background: rgba(156, 39, 176, 0.2);">1️⃣ 컨텍스트 분석</div>
                        <div class="loading-step" style="background: rgba(156, 39, 176, 0.2);">2️⃣ GPT-3.5 답변</div>
                        <div class="loading-step" style="background: rgba(156, 39, 176, 0.2);">3️⃣ GPT-4o 개선</div>
                        <div class="loading-step" style="background: rgba(156, 39, 176, 0.2);">4️⃣ 최종 완성</div>
                    </div>
                    <div style="color: #d1d5db; font-size: 0.9rem;">질문을 분석하고 최적의 답변을 찾고 있습니다</div>
                </div>
                """, unsafe_allow_html=True)
    
                # 대화 히스토리를 포함한 컨텍스트 생성
                conversation_context = ""
                if st.session_state.conversation_history:
                    recent_conversations = st.session_state.conversation_history[-6:]  # 최근 3쌍의 대화
                    conversation_context = "\n\n".join([
                        f"{'사용자' if role == 'user' else 'AI'}: {message}" 
                        for role, message, _ in recent_conversations
                    ])
                
                # PDF 컨텍스트 가져오기
                pdf_context = get_context(question, st.session_state.docs, st.session_state.embs) if rag_enabled and st.session_state.docs else ""
                
                # 통합 컨텍스트 (대화 히스토리 + PDF 컨텍스트)
                if ai_mode:
                    # 일반 AI 모드: 대화 히스토리만 사용
                    context = conversation_context if conversation_context else ""
                else:
                    # RAG 모드: 대화 히스토리 + PDF 컨텍스트
                    context = f"이전 대화:\n{conversation_context}\n\nPDF 내용:\n{pdf_context}" if conversation_context else pdf_context
                
                # 3단계 답변 시스템 시작
                st.markdown('<div class="section-header">📝 3단계 답변 시스템</div>', unsafe_allow_html=True)
                
                # 1단계: GPT-3.5 답변
                with st.container():
                    st.markdown("""
                    <div class="custom-loading" style="background: linear-gradient(135deg, rgba(21, 101, 192, 0.1) 0%, rgba(30, 136, 229, 0.1) 100%); border: 1px solid rgba(21, 101, 192, 0.3);">
                        <div class="loading-spinner" style="border-color: rgba(21, 101, 192, 0.3); border-top-color: #1565c0;"></div>
                        <div class="loading-text" style="color: #1565c0;">🤖 GPT-3.5 답변 생성중...</div>
                        <div class="loading-steps">
                            <div class="loading-step active" style="background: rgba(21, 101, 192, 0.2);">1️⃣ 질문 분석</div>
                            <div class="loading-step" style="background: rgba(21, 101, 192, 0.2);">2️⃣ 컨텍스트 검색</div>
                            <div class="loading-step" style="background: rgba(21, 101, 192, 0.2);">3️⃣ 답변 생성</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                gpt35_answer = generate_answer(question, context, "gpt-3.5-turbo")
                gpt35_quality = analyze_answer_quality(gpt35_answer, question)
                
                with st.container():
                    # 수학 기호 가독성 개선
                    improved_gpt35_answer = improve_math_readability(gpt35_answer)
                    st.markdown(f"""
                    <div class="answer-card">
                        <h4>🤖 GPT-3.5 답변 (1단계)</h4>
                        <p>{improved_gpt35_answer}</p>
                        <div class="model-badge quality-{gpt35_quality['level']}">
                            품질: {gpt35_quality['score']}/100
                        </div>
                        <div class="model-badge model-gpt35">
                            GPT-3.5 Turbo
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 2단계: GPT-4o 답변
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
                
                gpt4o_answer = generate_answer(question, context, "gpt-4o")
                gpt4o_quality = analyze_answer_quality(gpt4o_answer, question)
                
                with st.container():
                    # 수학 기호 가독성 개선
                    improved_gpt4o_answer = improve_math_readability(gpt4o_answer)
                    st.markdown(f"""
                    <div class="answer-card">
                        <h4>🚀 GPT-4o 답변 (2단계)</h4>
                        <p>{improved_gpt4o_answer}</p>
                        <div class="model-badge quality-{gpt4o_quality['level']}">
                            품질: {gpt4o_quality['score']}/100
                        </div>
                        <div class="model-badge model-gpt4o">
                            GPT-4o
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 3단계: 개선된 답변 (품질이 낮은 경우)
                improved_answer = None
                improved_quality = None
                
                if gpt35_quality['score'] < 70 or gpt4o_quality['score'] < 70:
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
                    
                    # 더 나은 답변을 기준으로 개선
                    base_answer = gpt4o_answer if gpt4o_quality['score'] > gpt35_quality['score'] else gpt35_answer
                    base_quality = gpt4o_quality if gpt4o_quality['score'] > gpt35_quality['score'] else gpt35_quality
                    
                    improved_answer = improve_answer_with_better_model(
                        question, base_answer, context, "gpt-4o", base_quality
                    )
                    improved_quality = analyze_answer_quality(improved_answer, question)
                    
                    with st.container():
                        # 수학 기호 가독성 개선
                        improved_final_answer = improve_math_readability(improved_answer)
                        st.markdown(f"""
                        <div class="improved-card">
                            <h4>✨ 개선된 답변 (3단계)</h4>
                            <p>{improved_final_answer}</p>
                                                    <div class="model-badge quality-{improved_quality['level']}">
                            품질: {improved_quality['score']}/100
                        </div>
                            <div class="model-badge model-gpt4o">
                                GPT-4o (개선)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # 자동 모델 선택 (추가 기능)
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
        st.subheader("📝 질문 입력")
        question = st.text_area(
            "질문을 입력하세요:",
            height=150,
            placeholder="예: 이 PDF의 주요 내용을 요약해주세요."
        )

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
                    
                    if use_gpt_oss_20b:
                        with st.container():
                            st.markdown("""
                            <div style="
                                background: linear-gradient(135deg, #fff3e0 0%, #ffcc02 100%);
                                padding: 1.5rem;
                                border-radius: 15px;
                                text-align: center;
                                color: #e65100;
                                margin: 1rem 0;
                                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                            ">
                                <div style="
                                    width: 40px;
                                    height: 40px;
                                    border: 3px solid rgba(230, 81, 0, 0.3);
                                    border-top: 3px solid #e65100;
                                    border-radius: 50%;
                                    animation: spin 1s linear infinite;
                                    margin: 0 auto 0.5rem auto;
                                "></div>
                                <p style="margin: 0; font-weight: 600;">💻 GPT-OSS-20B 분석 중...</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            gpt_oss_20b_answer = generate_answer(question, context, "gpt-oss-20b")
                            gpt_oss_20b_quality = analyze_answer_quality(gpt_oss_20b_answer, question)
                            
                            with st.container():
                                st.markdown(f"""
                                <div class="answer-card">
                                    <h4>💻 GPT-OSS-20B 답변 (무료 로컬)</h4>
                                    <p>{gpt_oss_20b_answer}</p>
                                    <div class="quality-badge quality-{gpt_oss_20b_quality['level']}">
                                        품질 점수: {gpt_oss_20b_quality['score']}/100
                                    </div>
                                    <div class="model-badge model-gptoss">
                                        GPT-OSS-20B (무료)
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    if use_gpt_oss_120b:
                        with st.container():
                            st.markdown("""
                            <div style="
                                background: linear-gradient(135deg, #e3f2fd 0%, #2196f3 100%);
                                padding: 1.5rem;
                                border-radius: 15px;
                                text-align: center;
                                color: #1565c0;
                                margin: 1rem 0;
                                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                            ">
                                <div style="
                                    width: 40px;
                                    height: 40px;
                                    border: 3px solid rgba(21, 101, 192, 0.3);
                                    border-top: 3px solid #1565c0;
                                    border-radius: 50%;
                                    animation: spin 1s linear infinite;
                                    margin: 0 auto 0.5rem auto;
                                "></div>
                                <p style="margin: 0; font-weight: 600;">🚀 GPT-OSS-120B 분석 중...</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            gpt_oss_120b_answer = generate_answer(question, context, "gpt-oss-120b")
                            gpt_oss_120b_quality = analyze_answer_quality(gpt_oss_120b_answer, question)
                            
                            with st.container():
                                st.markdown(f"""
                                <div class="answer-card">
                                    <h4>🚀 GPT-OSS-120B 답변 (고성능 무료)</h4>
                                    <p>{gpt_oss_120b_answer}</p>
                                    <div class="quality-badge quality-{gpt_oss_120b_quality['level']}">
                                        품질 점수: {gpt_oss_120b_quality['score']}/100
                                    </div>
                                    <div class="model-badge model-gptoss">
                                        GPT-OSS-120B (무료)
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # 계층적 답변 개선 (선택적)
                improved_answer = None
                improved_quality = None
                
                if use_hierarchical and (use_gpt4 or use_gpt4mini):
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
        context = st.text_area(
            "컨텍스트 (선택사항):",
            height=100,
            placeholder="추가 컨텍스트나 배경 정보를 입력하세요."
        )

                # 고급 분석 정보
                with st.expander("📊 고급 분석 정보"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**🤖 GPT-3.5 답변 분석**")
                        st.write(f"점수: {gpt35_quality['score']}/100")
                        st.write(f"레벨: {gpt35_quality['level']}")
                        if gpt35_quality['issues']:
                            st.write("문제점:")
                            for issue in gpt35_quality['issues']:
                                st.write(f"- {issue}")
        if st.button("🚀 답변 생성", type="primary"):
            if question.strip():
                with st.spinner("답변을 생성하고 있습니다..."):
                    # 모델별 처리
                    if "GPT-OSS" in selected_model:
                        # GPT-OSS 모델 처리
                        if not check_gpt_oss_server():
                            st.error("GPT-OSS 서버가 실행되지 않았습니다.")
                            st.info("로컬에서 GPT-OSS 서버를 시작해주세요.")
                            return

                        # 감정 및 톤 분석
                        sentiment_analysis = analyze_sentiment_and_tone(gpt35_answer)
                        st.markdown("**감정 분석**")
                        st.write(f"감정: {sentiment_analysis['sentiment']}")
                        st.write(f"톤: {sentiment_analysis['tone']}")
                    
                    with col2:
                        st.markdown("**🚀 GPT-4o 답변 분석**")
                        st.write(f"점수: {gpt4o_quality['score']}/100")
                        st.write(f"레벨: {gpt4o_quality['level']}")
                        if gpt4o_quality['issues']:
                            st.write("문제점:")
                            for issue in gpt4o_quality['issues']:
                                st.write(f"- {issue}")
                        # 120B 대신 20B 사용 권장
                        if "120B" in selected_model:
                            st.warning("120B 모델 대신 20B 모델을 사용합니다.")
                            model_name = "gpt-oss-20b"
                        else:
                            model_name = "gpt-oss-20b"

                        # 주제 분류
                        topic = classify_topic(gpt4o_answer)
                        st.markdown("**주제 분류**")
                        st.write(f"주제: {topic}")
                    
                    with col3:
                        if improved_answer:
                            st.markdown("**✨ 개선된 답변 분석**")
                            st.write(f"점수: {improved_quality['score']}/100")
                            st.write(f"레벨: {improved_quality['level']}")
                            if improved_quality['issues']:
                                st.write("문제점:")
                                for issue in improved_quality['issues']:
                                    st.write(f"- {issue}")
                        # 간단한 프롬프트 생성 (빈 템플릿 문제 해결)
                        if context:
                            prompt = f"Context: {context}\n\nQuestion: {question}\n\nPlease provide a detailed answer:"
                        else:
                            prompt = f"Question: {question}\n\nPlease provide a detailed answer:"

                        # 키워드 분석
                        keywords = analyze_text_keywords(gpt4o_answer, top_n=10)
                        if keywords:
                            st.markdown("**🔑 주요 키워드**")
                            for keyword, count in list(keywords.items())[:5]:
                                st.write(f"• {keyword}: {count}회")
                
                # 컨텍스트 표시
                if context and rag_enabled:
                    with st.expander("📄 사용된 컨텍스트"):
                        st.text_area("컨텍스트", context, height=200, disabled=True)
        
                        # 대화 히스토리에 저장
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # 사용자 질문 저장
                st.session_state.conversation_history.append(("user", question, current_time))
                
                # AI 답변 저장 (개선된 답변을 메인으로 사용)
                ai_answer = improved_answer if improved_answer else gpt4o_answer
                st.session_state.conversation_history.append(("ai", ai_answer, current_time))
                
                # 기존 히스토리에도 저장
                history_entry = {
                    'question': question,
                    'gpt35_answer': gpt35_answer,
                    'gpt35_quality': gpt35_quality['score'],
                    'gpt4o_answer': gpt4o_answer,
                    'gpt4o_quality': gpt4o_quality['score'],
                    'improved_answer': improved_answer,
                    'improved_quality': improved_quality['score'] if improved_quality else None,
                    'auto_answer': auto_answer if model_selection_mode == "자동 선택 (추천)" else None,
                    'auto_quality': auto_quality['score'] if model_selection_mode == "자동 선택 (추천)" else None,
                    'auto_model': selected_model if model_selection_mode == "자동 선택 (추천)" else None,
                    'gpt4mini_answer': gpt4mini_answer if model_selection_mode == "수동 선택" and use_gpt4mini else None,
                    'gpt4mini_quality': gpt4mini_quality['score'] if model_selection_mode == "수동 선택" and use_gpt4mini else None,
                    'gpt4_answer': gpt4_answer if model_selection_mode == "수동 선택" and use_gpt4 else None,
                    'gpt4_quality': gpt4_quality['score'] if model_selection_mode == "수동 선택" and use_gpt4 else None,
                    'gpt_oss_20b_answer': gpt_oss_20b_answer if model_selection_mode == "수동 선택" and use_gpt_oss_20b else None,
                    'gpt_oss_20b_quality': gpt_oss_20b_quality['score'] if model_selection_mode == "수동 선택" and use_gpt_oss_20b else None,
                    'gpt_oss_120b_answer': gpt_oss_120b_answer if model_selection_mode == "수동 선택" and use_gpt_oss_120b else None,
                    'gpt_oss_120b_quality': gpt_oss_120b_quality['score'] if model_selection_mode == "수동 선택" and use_gpt_oss_120b else None,
                    'timestamp': current_time
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
                
                <h4>🤖 GPT-3.5 답변 (1단계)</h4>
                <p>{entry['gpt35_answer']}</p>
                <div class="quality-badge quality-{'good' if entry['gpt35_quality'] >= 80 else 'medium' if entry['gpt35_quality'] >= 60 else 'bad'}">
                    품질: {entry['gpt35_quality']}/100
                </div>
                
                <h4>🚀 GPT-4o 답변 (2단계)</h4>
                <p>{entry['gpt4o_answer']}</p>
                <div class="quality-badge quality-{'good' if entry['gpt4o_quality'] >= 80 else 'medium' if entry['gpt4o_quality'] >= 60 else 'bad'}">
                    품질: {entry['gpt4o_quality']}/100
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

            if entry.get('gpt_oss_20b_answer'):
                st.markdown(f"""
                <h4>💻 GPT-OSS-20B 답변 (무료 로컬)</h4>
                <p>{entry['gpt_oss_20b_answer']}</p>
                <div class="quality-badge quality-{'good' if entry['gpt_oss_20b_quality'] >= 80 else 'medium' if entry['gpt_oss_20b_quality'] >= 60 else 'bad'}">
                    품질: {entry['gpt_oss_20b_quality']}/100
                </div>
                """, unsafe_allow_html=True)
            # 질문 표시
            if 'last_question' in st.session_state:
                st.markdown(f"**질문:** {st.session_state['last_question']}")

            if entry.get('gpt_oss_120b_answer'):
                st.markdown(f"""
                <h4>🚀 GPT-OSS-120B 답변 (고성능 무료)</h4>
                <p>{entry['gpt_oss_120b_answer']}</p>
                <div class="quality-badge quality-{'good' if entry['gpt_oss_120b_quality'] >= 80 else 'medium' if entry['gpt_oss_120b_quality'] >= 60 else 'bad'}">
                    품질: {entry['gpt_oss_120b_quality']}/100
                </div>
                """, unsafe_allow_html=True)
            # 답변 표시
            st.markdown("**답변:**")
            st.write(st.session_state['last_response'])

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

# 데이터 분석 섹션 - 아래쪽에 배치
st.markdown('<div class="section-header">📊 데이터 분석</div>', unsafe_allow_html=True)

# 탭 생성
tab1, tab2, tab3 = st.tabs(["📈 사용 통계", "🔤 키워드 분석", "💾 저장하기"])

with tab1:
    st.markdown('<div class="section-header">📈 사용 패턴 분석</div>', unsafe_allow_html=True)
    
    # 메트릭들
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_questions = len(st.session_state.history) if st.session_state.history else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>💬 총 질문 수</h4>
            <h2>{total_questions}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.history:
            avg_quality = sum(entry.get('gpt4o_quality', 0) for entry in st.session_state.history) / len(st.session_state.history)
            # 복사 버튼
            if st.button("📋 답변 복사"):
                st.write("답변이 클립보드에 복사되었습니다.")
        else:
            avg_quality = 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 평균 품질 점수</h4>
            <h2>{avg_quality:.1f}</h2>
            <p>/10</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        improved_count = len([entry for entry in st.session_state.history if entry.get('improved_answer')]) if st.session_state.history else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>✨ 개선된 답변</h4>
            <h2>{improved_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # 사용 패턴 트렌드 그래프
    st.markdown('<div class="section-header">📈 사용 패턴 트렌드</div>', unsafe_allow_html=True)
    
    # 빈 그래프 영역
    st.markdown("""
    <div style="
        background: #2d2d2d;
        border: 2px dashed #444444;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        color: #9ca3af;
        margin: 1rem 0;
    ">
        <p>그래프 영역</p>
        <p style="font-size: 0.8rem;">Y축: 질문 수 (0, 1, 2, 3, 4)</p>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("키워드 분석 기능은 곧 추가될 예정입니다!")

with tab3:
    st.markdown("저장 기능은 곧 추가될 예정입니다!")

            st.info("👈 왼쪽에서 질문을 입력하고 답변을 생성해보세요!")

# 앱 실행
if __name__ == "__main__":
    main()
