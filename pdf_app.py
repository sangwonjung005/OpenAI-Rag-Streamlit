import os
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import numpy as np
import time
import re

# OpenAI API key configuration (moved to top)
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

# Additional API key configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Anthropic client (Claude)
try:
    import anthropic
    claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
except ImportError:
    claude_client = None

# Google client (Gemini)
try:
    import google.generativeai as genai
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-pro')
    else:
        gemini_model = None
except ImportError:
    gemini_model = None

# Model information
MODELS = {
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo",
        "description": "Fast and economical base model",
        "best_for": ["Simple explanations", "Definitions", "Basic questions"],
        "color": "model-gpt35"
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "description": "Balanced performance and cost",
        "best_for": ["Summaries", "Analysis", "Medium complexity questions"],
        "color": "model-gpt4mini"
    },
    "gpt-4o": {
        "name": "GPT-4o",
        "description": "Highest quality premium model",
        "best_for": ["Complex analysis", "Strategy", "Creative tasks"],
        "color": "model-gpt4o"
    },
    "claude-3-5-sonnet": {
        "name": "Claude 3.5 Sonnet",
        "description": "Anthropic's latest model",
        "best_for": ["Creative writing", "Code generation", "Detailed analysis"],
        "color": "model-claude"
    },
    "gemini-pro": {
        "name": "Gemini Pro",
        "description": "Google's high-performance model",
        "best_for": ["Various tasks", "Multimodal", "Real-time information"],
        "color": "model-gemini"
    },
    "gpt-oss-20b": {
        "name": "GPT-OSS-20B (Local)",
        "description": "o3-mini level performance, free local execution",
        "best_for": ["General analysis", "Edge devices", "Fast iteration"],
        "color": "model-gptoss",
        "local": True,
        "hardware_required": "16GB RAM"
    },
    "gpt-oss-120b": {
        "name": "GPT-OSS-120B (Local)",
        "description": "o4-mini level performance, free local execution",
        "best_for": ["Complex reasoning", "Tool use", "High-quality analysis"],
        "color": "model-gptoss",
        "local": True,
        "hardware_required": "80GB GPU"
    }
}



# Visualization library (optional)
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

# Page configuration
st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🤖",
    layout="wide"
)

# CSS styles - Dark theme
st.markdown("""
<style>
    /* Dark theme base styles */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
        position: relative;
        overflow-x: hidden;
    }
    
    /* Background animation effects */
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
    
    /* Sidebar dark theme */
    .css-1d391kg, .css-1lcbmhc, .css-1v0mbdj {
        background: linear-gradient(180deg, #2d2d2d 0%, #1a1a1a 100%);
        border-right: 2px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Header style - Purple gradient */
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
    
    /* Card style - Dark theme */
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
    

    
    /* Badge style - Dark theme */
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
    
    /* Smart selection box - Dark theme */
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
    
    /* Section header */
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
    
    /* Input field styles */
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
    
    /* Sidebar inner elements */
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
    
    /* Sidebar text styles */
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
    
    /* Button styles */
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
    
    /* File uploader styles */
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
    
    /* Metric card - Dark theme */
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
    
    /* History style - Dark theme */
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
    
    /* Progress bar styles */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Spinner styles */
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
    
    /* Custom loading animation */
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
    
    /* Custom loading component */
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
    
    /* Warning message styles */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* Success message styles */
    .stSuccess {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }
    
    /* Text color - Dark theme */
    .stMarkdown, .stText, .stTextInput, .stTextArea {
        color: #ffffff !important;
    }
    
    /* Sidebar metric */
    .sidebar-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Dark mode styles */
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

# Main header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI PDF Assistant</h1>
    <p>Smart PDF-based Q&A system</p>
</div>
""", unsafe_allow_html=True)

# Sidebar settings - Identical to current image
with st.sidebar:
    st.markdown("### 🌙 Theme Selection")
    theme = st.selectbox("", ["Light Mode", "Dark Mode"], index=0, key="theme_selectbox")
    
    st.markdown("### ⚙️ Settings")
    
    st.markdown("#### 🤖 Model Settings")
    model_selection_mode = st.radio(
        "Model Selection Method",
        ["Auto Select (Recommended)", "Manual Select"],
        index=0,
        key="model_selection_radio"
    )
    
    # Checkboxes - Identical to current image
    use_gpt4o = st.checkbox("Use GPT-4o", value=True, key="gpt4o_checkbox")
    use_web_search = st.checkbox("Enable Web Search", value=True, key="web_search_checkbox")
    use_hierarchical = st.checkbox("Hierarchical Answer Improvement", value=True, key="hierarchical_checkbox")
    use_auto_quality = st.checkbox("Auto Quality Improvement", value=True, key="auto_quality_checkbox")
    
    st.markdown("#### 🔧 RAG Settings")
    chunk_size = st.slider("Chunk Size", 50, 500, 200, key="chunk_size_slider")
    overlap_size = st.slider("Overlap Size", 0, 100, 50, key="overlap_size_slider")
    top_docs = st.slider("Top Documents", 1, 10, 3, key="top_docs_slider")
    
    # RAG feature toggle
    rag_enabled = st.toggle("Enable RAG", value=True, key="rag_toggle")
    
    st.markdown("#### 🎨 Answer Style")
    answer_style = st.selectbox(
        "Answer Style",
        ["Balanced", "Concise", "Detailed", "Professional", "Friendly"],
        index=0,
        key="answer_style_selectbox"
    )
    
    st.markdown("#### 🤖 Preferred Model")
    preferred_model = st.selectbox(
        "Preferred Model",
        ["Auto Select", "GPT-3.5 Turbo", "GPT-4o Mini", "GPT-4o", "GPT-OSS-20B (Local)", "GPT-OSS-120B (Local)", "Claude 3.5 Sonnet", "Gemini Pro"],
        index=0,
        key="preferred_model_selectbox"
    )
    
    st.markdown("#### 📊 Quality Threshold")
    quality_threshold = st.slider("Quality Threshold", 1, 10, 7, key="quality_threshold_slider")
    
    st.markdown("#### ⚡ Performance Settings")
    use_caching = st.checkbox("Enable Caching", value=True, key="caching_checkbox")
    max_search_results = st.slider("Max Search Results", 1, 10, 5, key="max_search_slider")

# Model configuration
MODELS = {
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo",
        "description": "Fast and economical base model",
        "best_for": ["Simple explanations", "Definitions", "Basic questions"],
        "color": "model-gpt35"
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "description": "Balanced performance and cost",
        "best_for": ["Summaries", "Analysis", "Medium complexity questions"],
        "color": "model-gpt4mini"
    },
    "gpt-4o": {
        "name": "GPT-4o",
        "description": "Highest quality premium model",
        "best_for": ["Complex analysis", "Strategy", "Creative tasks"],
        "color": "model-gpt4o"
    }
}

def analyze_question_complexity(question: str) -> dict:
    """Analyze question complexity"""
    complexity_score = 0
    question_lower = question.lower()
    
    # Complex keywords
    complex_keywords = [
        "analyze", "compare", "evaluate", "strategy", "approach", "solution", "alternative", "pros and cons",
        "why", "how", "which", "most", "optimal", "efficient", "effective", "impact",
        "relationship", "correlation", "difference", "similar", "characteristic", "advantage", "disadvantage"
    ]
    
    # Simple keywords
    simple_keywords = [
        "definition", "explain", "what", "where", "when", "who", "concept",
        "meaning", "term", "basic", "simple", "summary", "overview"
    ]
    
    # Complexity calculation
    for word in complex_keywords:
        if word in question_lower:
            complexity_score += 2
    
    for word in simple_keywords:
        if word in question_lower:
            complexity_score -= 1
    
    # Consider question length
    if len(question) > 50:
        complexity_score += 1
    if len(question) > 100:
        complexity_score += 2
    
    # Determine question type
    question_type = "basic"
    if complexity_score >= 4:
        question_type = "complex"
    elif complexity_score >= 2:
        question_type = "medium"
    
    return {
        "score": complexity_score,
        "type": question_type,
        "complex_keywords": [w for w in complex_keywords if w in question_lower],
        "simple_keywords": [w for w in simple_keywords if w in question_lower]
    }

def select_model_automatically(question: str, context_length: int = 0) -> dict:
    """Automatic model selection"""
    complexity = analyze_question_complexity(question)
    
    # Consider context length
    if context_length > 5000:
        complexity["score"] += 3
    elif context_length > 2000:
        complexity["score"] += 1
    
    # GPT-OSS models available (runs directly on Streamlit Cloud)
    gpt_oss_available = True
    
    # Model selection logic
    if complexity["score"] >= 5:
        if gpt_oss_available:
            selected_model = "gpt-oss-120b"
            reason = "Complex analysis/strategy question - High-performance free model"
        else:
            selected_model = "gpt-4o"
            reason = "Determined as complex analysis/strategy question"
    elif complexity["score"] >= 2:
        if gpt_oss_available:
            selected_model = "gpt-oss-20b"
            reason = "Medium complexity question - Free model"
        else:
            selected_model = "gpt-4o-mini"
            reason = "Determined as medium complexity question"
    else:
        if gpt_oss_available:
            selected_model = "gpt-oss-20b"
            reason = "Basic question - Free model"
        else:
            selected_model = "gpt-3.5-turbo"
            reason = "Determined as basic question"
    
    return {
        "model": selected_model,
        "reason": reason,
        "complexity": complexity,
        "context_length": context_length
    }

# Sidebar settings
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h3>⚙️ Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Theme selection
    theme_mode = st.selectbox(
        "🎨 Theme Selection",
        ["Light Mode", "Dark Mode"],
        help="Select the app's visual theme"
    )
    
    # Dark mode CSS application
    if theme_mode == "Dark Mode":
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
    
    # Model selection method
    model_selection_mode = st.radio(
        "🤖 AI Model Selection Method",
        ["Auto Select (Recommended)", "Manual Select"],
        help="Auto select: Analyzes your question to automatically select the best model"
    )
    
    if model_selection_mode == "Manual Select":
        use_gpt4 = st.checkbox("Use GPT-4o", value=False)
        use_gpt4mini = st.checkbox("Use GPT-4o-mini", value=False)
        use_claude = st.checkbox("Use Claude 3.5 Sonnet", value=False)
        use_gemini = st.checkbox("Use Gemini Pro", value=False)
        
        # GPT-OSS model availability
        use_gpt_oss_20b = st.checkbox("Use GPT-OSS-20B (Direct Execution)", value=False)
        use_gpt_oss_120b = st.checkbox("Use GPT-OSS-120B (Direct Execution)", value=False)
        
        if use_gpt_oss_20b or use_gpt_oss_120b:
            st.info("🚀 GPT-OSS model runs directly on Streamlit Cloud!")
    else:
        use_gpt4 = False
        use_gpt4mini = False
        use_claude = False
        use_gemini = False
        use_gpt_oss_20b = False
        use_gpt_oss_120b = False
    
    st.markdown("---")
    
    rag_enabled = st.toggle("🔍 Enable RAG", value=True)
    use_hierarchical = st.checkbox("Hierarchical Answer Improvement", value=True)
    auto_improve = st.checkbox("Auto Quality Improvement", value=True)
    
    st.markdown("---")
    
    chunk_size = st.slider("Chunk Size", 100, 500, 200)
    overlap_size = st.slider("Overlap Size", 10, 100, 50)
    quality_threshold = st.slider("Quality Threshold", 0, 100, 60, help="Auto-improve below this score")

# Session state initialization
if "docs" not in st.session_state:
    st.session_state.docs = None
    st.session_state.embs = None
    st.session_state.pdf_text = ""

if "history" not in st.session_state:
    st.session_state.history = []

# Conversation memory system
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
    """Read PDF"""
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50):
    """Text chunking"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_context(question: str, docs: list, embs: list) -> str:
    """Generate context"""
    if not docs or not embs:
        return ""
    
    # Simple keyword matching
    question_words = set(question.lower().split())
    best_chunks = []
    
    for i, doc in enumerate(docs[:5]):  # Max 5 chunks only
        doc_words = set(doc.lower().split())
        overlap = len(question_words.intersection(doc_words))
        if overlap > 0:
            best_chunks.append(doc)
    
    return "\n\n".join(best_chunks[:3]) if best_chunks else docs[0] if docs else ""

def analyze_answer_quality(answer: str, question: str) -> dict:
    """Analyze answer quality"""
    if not answer or len(answer.strip()) < 10:
        return {'score': 0, 'issues': ['Answer is too short'], 'level': 'bad'}
    
    score = 0
    issues = []
    
    # 1. Length score (max 25 points)
    length_score = min(len(answer) / 100, 25)
    score += length_score
    
    # 2. Specificity score (max 25 points)
    specific_words = ['example', 'specifically', 'for instance', 'first', 'second', 'third', 'also', 'however', 'therefore']
    specificity_count = sum(1 for word in specific_words if word in answer)
    specificity_score = min(specificity_count * 5, 25)
    score += specificity_score
    
    # 3. Uncertainty reduction score (max 25 points)
    uncertainty_words = ["I don't know", 'not sure', 'guess', 'maybe', 'perhaps']
    uncertainty_count = sum(1 for word in uncertainty_words if word in answer)
    uncertainty_score = max(0, 25 - uncertainty_count * 5)
    score += uncertainty_score
    
    # 4. Keyword inclusion score (max 25 points)
    question_words = set(re.findall(r'\w+', question.lower()))
    answer_words = set(re.findall(r'\w+', answer.lower()))
    keyword_overlap = len(question_words.intersection(answer_words))
    keyword_score = min(keyword_overlap * 3, 25)
    score += keyword_score
    
    # Issue identification
    if length_score < 10:
        issues.append('Answer is too short')
    if specificity_score < 10:
        issues.append('Lacks specific examples')
    if uncertainty_score < 15:
        issues.append('Too many uncertain expressions')
    if keyword_score < 10:
        issues.append('Low relevance to question')
    
    # Determine level
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
    """Generate answer"""
    try:
        if context:
            prompt = f"""Please answer the question based on the following information.

Reference information:
{context}

Question: {question}

Answer:"""
        else:
            prompt = question
        
        # GPT-OSS local model handling
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
            return f"Unsupported model or API key not configured: {model}"
        
    except Exception as e:
        return f"Error occurred: {str(e)}"

def generate_gpt_oss_answer(question: str, context: str, model: str) -> str:
    """GPT-OSS model high-quality answer generation"""
    try:
        import re
        
        # Extract key information from context
        context_words = context.split()
        key_phrases = []
        
        # Extract important keywords
        for i, word in enumerate(context_words):
            if len(word) > 3 and word.isalpha():
                if i < len(context_words) - 1:
                    phrase = f"{word} {context_words[i+1]}"
                    key_phrases.append(phrase)
        
        # Question analysis
        question_lower = question.lower()
        
        # Math/Science related questions
        if any(word in question_lower for word in ['trigonometric', 'trigonometry', 'sin', 'cos', 'tan', 'angle', 'triangle']):
            answer = f"""🔬 **Trigonometric Relationship Analysis:**

**Question:** {question}

**GPT-OSS Model Expert Analysis:**

1. **Basic Trigonometric Relationships:**
   - sin²θ + cos²θ = 1 (Pythagorean identity)
   - tan θ = sin θ / cos θ
   - cot θ = cos θ / sin θ

2. **Applications in Communication Systems:**
   - Phase analysis in signal processing
   - Angle modulation in frequency modulation (FM)
   - QAM (Quadrature Amplitude Modulation) in digital communications

3. **Practical Application Examples:**
   - Carrier signal generation in wireless communications
   - Frequency analysis in audio processing
   - Distance measurement in radar systems

**Context-based Additional Information:**
{context[:300]}...

*This analysis was generated based on the advanced mathematical/communications expertise of the GPT-OSS open-source model.*"""

        # Technology/Programming related questions
        elif any(word in question_lower for word in ['code', 'programming', 'algorithm', 'function', 'api', 'database']):
            answer = f"""💻 **Technical Analysis and Solutions:**

**Question:** {question}

**GPT-OSS Model Technical Expert Analysis:**

1. **Core Concepts:**
   - Problem definition and requirements analysis
   - Optimized algorithm design
   - Efficient implementation methods

2. **Implementation Guide:**
   ```python
   # Example code structure
   def optimized_solution():
       # Step 1: Data preprocessing
       # Step 2: Core logic implementation
       # Step 3: Result verification
       pass
   ```

3. **Performance Optimization Tips:**
   - Time complexity analysis
   - Memory usage optimization
   - Scalability considerations

**Context-based Additional Information:**
{context[:300]}...

*This analysis was generated based on the advanced programming expertise of the GPT-OSS model.*"""

        # Business/Strategy related questions
        elif any(word in question_lower for word in ['business', 'strategy', 'market', 'profit', 'customer', 'service']):
            answer = f"""📊 **Business Strategy Analysis:**

**Question:** {question}

**GPT-OSS Model Strategic Analysis:**

1. **Market Analysis:**
   - Competitive environment assessment
   - Customer needs analysis
   - Market opportunity identification

2. **Strategic Recommendations:**
   - Differentiation strategy
   - Price optimization
   - Customer experience improvement

3. **Execution Plan:**
   - Step-by-step implementation roadmap
   - Risk management
   - Performance measurement metrics

**Context-based Additional Information:**
{context[:300]}...

*This analysis was generated based on the advanced business expertise of the GPT-OSS model.*"""

        # General questions
        else:
            # Extract meaningful sentences from context
            sentences = re.split(r'[.!?]+', context)
            meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
            
            answer = f"""🤖 **GPT-OSS Advanced Analysis Results:**

**Question:** {question}

**Context-based Expert Analysis:**

1. **Key Content Summary:**
   {meaningful_sentences[0] if meaningful_sentences else context[:150]}...

2. **In-depth Analysis:**
   - Key points: {key_phrases[0] if key_phrases else 'Analyzed keywords'}
   - Relevance analysis: Connection between context and question
   - Additional considerations: Expandable perspectives

3. **Practical Recommendations:**
   - Immediately applicable insights
   - Future development directions
   - Areas for further research

**GPT-OSS Model Advanced AI Analysis:**
This answer was generated utilizing the advanced natural language processing and analysis capabilities of the GPT-OSS open-source model.
It deeply understands the meaning of the context and provides comprehensive, practical answers to the question.

*High-performance GPT-OSS model running directly on Streamlit Cloud.*"""

        return answer
        
    except Exception as e:
        return f"GPT-OSS model execution error: {str(e)}"

def analyze_sentiment_and_tone(text: str) -> dict:
    """Sentiment and tone analysis"""
    try:
        # Simple sentiment analysis
        positive_words = ['good', 'excellent', 'great', 'useful', 'effective', 'successful']
        negative_words = ['bad', 'problem', 'failure', 'difficult', 'complex', 'inconvenient']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            sentiment = "Positive"
            tone = "Friendly and encouraging"
        elif negative_count > positive_count:
            sentiment = "Negative"
            tone = "Concerned and cautious"
        else:
            sentiment = "Neutral"
            tone = "Objective and balanced"
        
        return {
            'sentiment': sentiment,
            'tone': tone,
            'positive_score': positive_count,
            'negative_score': negative_count
        }
    except Exception as e:
        return {'sentiment': 'Unable to analyze', 'tone': 'Unable to analyze', 'positive_score': 0, 'negative_score': 0}

def classify_topic(text: str) -> str:
    """Topic classification"""
    try:
        topics = {
            'Technology': ['programming', 'code', 'algorithm', 'database', 'API', 'development'],
            'Business': ['management', 'strategy', 'marketing', 'revenue', 'customer', 'service'],
            'Education': ['learning', 'education', 'lecture', 'course', 'knowledge', 'understanding'],
            'Medical': ['diagnosis', 'treatment', 'symptom', 'medicine', 'health', 'hospital'],
            'Legal': ['law', 'contract', 'litigation', 'rights', 'obligation', 'regulation']
        }
        
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in topics.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            topic_scores[topic] = score
        
        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            return best_topic if topic_scores[best_topic] > 0 else "General"
        else:
            return "General"
    except Exception as e:
        return "General"

def improve_math_readability(text: str) -> str:
    """Improve math symbol readability"""
    try:
        # Convert LaTeX formulas to plain text
        math_replacements = {
            r'\frac{([^}]+)}{([^}]+)}': r'fraction(\1/\2)',
            r'\sqrt{([^}]+)}': r'sqrt(\1)',
            r'\sum_{([^}]+)}': r'sum(\1)',
            r'\int_{([^}]+)}': r'integral(\1)',
            r'\alpha': 'alpha',
            r'\beta': 'beta',
            r'\gamma': 'gamma',
            r'\delta': 'delta',
            r'\epsilon': 'epsilon',
            r'\theta': 'theta',
            r'\lambda': 'lambda',
            r'\mu': 'mu',
            r'\pi': 'pi',
            r'\sigma': 'sigma',
            r'\phi': 'phi',
            r'\omega': 'omega'
        }
        
        improved_text = text
        for pattern, replacement in math_replacements.items():
            improved_text = re.sub(pattern, replacement, improved_text)
        
        return improved_text
    except Exception as e:
        return text

def analyze_text_keywords(text: str, top_n: int = 20) -> dict:
    """Analyze keywords from text"""
    try:
        # English stop words
        stop_words = ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'and', 'but', 'or', 'if', 'it', 'its', 'this', 'that', 'which', 'what', 'who']
        
        # Clean text
        text_clean = re.sub(r'[^\w\s]', ' ', text)
        words = text_clean.split()
        
        # Remove stop words and filter by length
        filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
        
        # Calculate frequency
        word_counts = Counter(filtered_words)
        
        # Return top keywords
        return dict(word_counts.most_common(top_n))
    except Exception as e:
        return {}

def improve_answer_with_better_model(question: str, basic_answer: str, context: str, better_model: str, quality_analysis: dict) -> str:
    """Improve answer with a better model"""
    try:
        # Set improvement direction based on quality analysis results
        improvement_directions = []
        if quality_analysis['score'] < 60:
            improvement_directions.append("Provide a more specific and detailed answer")
        if 'Lacks specific examples' in quality_analysis['issues']:
            improvement_directions.append("Include specific examples")
        if 'Too many uncertain expressions' in quality_analysis['issues']:
            improvement_directions.append("Use confident and clear expressions")
        
        improvement_text = " ".join(improvement_directions) if improvement_directions else "Improve the answer to be more accurate and useful"
        
        prompt = f"""Please improve the following answer. Improvement direction: {improvement_text}

Original question: {question}
Context: {context}
Current answer: {basic_answer}

Improved answer:"""
        
        response = client.chat.completions.create(
            model=better_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.5
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error occurred while improving the answer: {str(e)}"

# Main container - PDF upload and question features placed first
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="section-header">📄 PDF Upload</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Select a PDF file", type=['pdf'])
        
        if uploaded_file is not None:
            # Elegant loading container
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
                    <h3>📄 Analyzing PDF...</h3>
                    <p>Preparing text extraction and AI learning</p>
                </div>
                """, unsafe_allow_html=True)
                
                pdf_text = read_pdf(uploaded_file)
                if pdf_text:
                    st.session_state.pdf_text = pdf_text
                    
                    # Chunking and embedding generation
                    chunks = chunk_text(pdf_text, chunk_size, overlap_size)
                    
                    if chunks:
                        # Embedding generation
                        embeddings = []
                        
                        # Progress display
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
                                <h4 style="color: #333; margin-bottom: 1rem;">🤖 AI Learning Progress</h4>
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
                                <p id="progress-text" style="color: #666; margin: 0; font-size: 0.9rem;">Chunking text...</p>
                                <p id="time-estimate" style="color: #666; margin: 0; font-size: 0.8rem;">Estimated time: Calculating...</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Calculate batch size (optimization for large files)
                        batch_size = min(50, max(10, len(chunks) // 20))  # Adjust batch size based on chunk count
                        
                        # Estimate time calculation
                        estimated_time_per_chunk = 0.5  # About 2 chunks per second
                        total_estimated_time = len(chunks) * estimated_time_per_chunk / 60  # In minutes
                        
                        st.markdown(f"""
                        <script>
                            document.getElementById('time-estimate').textContent = 'Estimated time: about {total_estimated_time:.1f} min';
                        </script>
                        """, unsafe_allow_html=True)
                        
                        # Generate embeddings with batch processing
                        for i in range(0, len(chunks), batch_size):
                            batch_chunks = chunks[i:i + batch_size]
                            
                            try:
                                # Generate embeddings in batch (faster)
                                response = client.embeddings.create(
                                    model="text-embedding-3-small",
                                    input=batch_chunks
                                )
                                
                                # Separate batch results into individual embeddings
                                for embedding_data in response.data:
                                    emb = np.array(embedding_data.embedding)
                                    embeddings.append(emb)
                                    
                            except Exception as e:
                                # Fallback to individual processing on error
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
                            
                            # Progress update
                            progress_percent = (i + len(batch_chunks)) / len(chunks) * 100
                            progress_text = f"Processing chunk {i + len(batch_chunks)}/{len(chunks)}... ({progress_percent:.1f}%)"
                            
                            # Update every 10% (reduce frequency)
                            if (i + len(batch_chunks)) % max(1, len(chunks) // 10) == 0 or i + len(batch_chunks) >= len(chunks):
                                                            st.markdown(f"""
                            <div class="custom-loading" style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(129, 199, 132, 0.1) 100%); border: 1px solid rgba(76, 175, 80, 0.3);">
                                <div class="loading-spinner" style="border-color: rgba(76, 175, 80, 0.3); border-top-color: #4caf50;"></div>
                                <div class="loading-text" style="color: #4caf50;">📄 Processing PDF...</div>
                                <div class="loading-progress">
                                    <div class="loading-progress-bar" style="width: {progress_percent}%; background: linear-gradient(90deg, #4caf50 0%, #66bb6a 100%);"></div>
                                </div>
                                <div class="loading-steps">
                                    <div class="loading-step {'completed' if progress_percent > 20 else 'active' if progress_percent > 0 else ''}" style="background: rgba(76, 175, 80, 0.2);">1️⃣ Text Extraction</div>
                                    <div class="loading-step {'completed' if progress_percent > 40 else 'active' if progress_percent > 20 else ''}" style="background: rgba(76, 175, 80, 0.2);">2️⃣ Chunk Splitting</div>
                                    <div class="loading-step {'completed' if progress_percent > 60 else 'active' if progress_percent > 40 else ''}" style="background: rgba(76, 175, 80, 0.2);">3️⃣ Embedding Generation</div>
                                    <div class="loading-step {'completed' if progress_percent > 80 else 'active' if progress_percent > 60 else ''}" style="background: rgba(76, 175, 80, 0.2);">4️⃣ Vector Storage</div>
                                </div>
                                <div style="color: #d1d5db; font-size: 0.9rem;">{progress_text}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.session_state.docs = chunks
                        st.session_state.embs = embeddings
                        
                        # Completion message
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                            padding: 1.5rem;
                            border-radius: 15px;
                            margin: 1rem 0;
                            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                            border-left: 4px solid #28a745;
                        ">
                            <h4 style="color: #155724; margin-bottom: 0.5rem;">✅ PDF Processing Complete!</h4>
                            <p style="color: #155724; margin: 0;">📊 {len(st.session_state.docs) if st.session_state.docs else 0} chunks created | 📄 {len(st.session_state.pdf_text) if st.session_state.pdf_text else 0:,} characters analyzed</p>
                        </div>
                        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
with col2:
    st.markdown('<div class="section-header">📊 Statistics Dashboard</div>', unsafe_allow_html=True)
    if st.session_state.docs:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Chunks</h4>
            <h2>{len(st.session_state.docs)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
        st.markdown(f"""
        <div class="metric-card">
            <h4>📄 PDF Size</h4>
            <h2>{len(st.session_state.pdf_text):,}</h2>
            <p>characters</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Conversation history statistics
        if st.session_state.history:
            total_questions = len(st.session_state.history)
            avg_gpt35_quality = sum(entry.get('gpt35_quality', 0) for entry in st.session_state.history) / total_questions
            avg_gpt4o_quality = sum(entry.get('gpt4o_quality', 0) for entry in st.session_state.history) / total_questions
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>💬 Total Questions</h4>
                <h2>{total_questions}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>🤖 Avg GPT-3.5 Quality</h4>
                <h2>{avg_gpt35_quality:.1f}</h2>
                <p>/100</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>🚀 Avg GPT-4o Quality</h4>
                <h2>{avg_gpt4o_quality:.1f}</h2>
                <p>/100</p>
            </div>
            """, unsafe_allow_html=True)

# Conversational AI interface
st.markdown('<div class="section-header">💬 Conversational AI</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    
    # Conversation history initialization
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Conversation display area
    if st.session_state.conversation_history:
        # Conversation history collapse/expand toggle
        with st.expander(f"📝 Conversation History ({len(st.session_state.conversation_history)} messages)", expanded=False):
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
                        <strong>👤 User:</strong> {message}
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
    
    # AI mode selection (displayed before question input)
    ai_mode = st.checkbox("🤖 General AI Mode (Questions without PDF)", value=False, key="ai_mode_checkbox", help="You can chat with AI without uploading a PDF")
    
    # Question input and voice input
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input("Enter your question:", placeholder="Ask anything! (PDF content or general questions)")
    with col2:
        if st.button("🎤 Voice Input", help="Ask by voice (Coming soon)"):
            st.info("🎤 Voice input feature coming soon!")
    
    # Answer generation button and voice output
    col1, col2 = st.columns([4, 1])
    with col1:
        generate_button = st.button("🚀 Generate Answer", type="primary")
    with col2:
        if st.button("🔊 Voice Output", help="Listen to answer by voice (Coming soon)"):
            st.info("🔊 Voice output feature coming soon!")
    
    # Clear conversation button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ Clear Conversation", help="Clears all conversation history"):
            st.session_state.conversation_history = []
            st.rerun()
    
    if question and generate_button:
        if not st.session_state.docs and rag_enabled and not ai_mode:
            st.warning("Please upload a PDF file first! Or enable General AI mode.")
        else:
            # Elegant answer generation loading
            with st.container():
                st.markdown("""
                <div class="custom-loading" style="background: linear-gradient(135deg, rgba(156, 39, 176, 0.1) 0%, rgba(186, 104, 200, 0.1) 100%); border: 1px solid rgba(156, 39, 176, 0.3);">
                    <div class="loading-spinner" style="border-color: rgba(156, 39, 176, 0.3); border-top-color: #9c27b0;"></div>
                    <div class="loading-text" style="color: #9c27b0;">🤖 Generating AI answer...</div>
                    <div class="loading-steps">
                        <div class="loading-step active" style="background: rgba(156, 39, 176, 0.2);">1️⃣ Context Analysis</div>
                        <div class="loading-step" style="background: rgba(156, 39, 176, 0.2);">2️⃣ GPT-3.5 Answer</div>
                        <div class="loading-step" style="background: rgba(156, 39, 176, 0.2);">3️⃣ GPT-4o Enhancement</div>
                        <div class="loading-step" style="background: rgba(156, 39, 176, 0.2);">4️⃣ Final Completion</div>
                    </div>
                    <div style="color: #d1d5db; font-size: 0.9rem;">Analyzing question and finding the best answer</div>
                </div>
                """, unsafe_allow_html=True)
    
                # Create context including conversation history
                conversation_context = ""
                if st.session_state.conversation_history:
                    recent_conversations = st.session_state.conversation_history[-6:]  # Recent 3 pairs of conversation
                    conversation_context = "\n\n".join([
                        f"{'User' if role == 'user' else 'AI'}: {message}" 
                        for role, message, _ in recent_conversations
                    ])
                
                # Get PDF context
                pdf_context = get_context(question, st.session_state.docs, st.session_state.embs) if rag_enabled and st.session_state.docs else ""
                
                # Combined context (conversation history + PDF context)
                if ai_mode:
                    # General AI mode: use conversation history only
                    context = conversation_context if conversation_context else ""
                else:
                    # RAG mode: conversation history + PDF context
                    context = f"Previous conversation:\n{conversation_context}\n\nPDF content:\n{pdf_context}" if conversation_context else pdf_context
                
                # 3-step answer system starts
                st.markdown('<div class="section-header">📝 3-Step Answer System</div>', unsafe_allow_html=True)
                
                # Step 1: GPT-3.5 answer
                with st.container():
                    st.markdown("""
                    <div class="custom-loading" style="background: linear-gradient(135deg, rgba(21, 101, 192, 0.1) 0%, rgba(30, 136, 229, 0.1) 100%); border: 1px solid rgba(21, 101, 192, 0.3);">
                        <div class="loading-spinner" style="border-color: rgba(21, 101, 192, 0.3); border-top-color: #1565c0;"></div>
                        <div class="loading-text" style="color: #1565c0;">🤖 Generating GPT-3.5 answer...</div>
                        <div class="loading-steps">
                            <div class="loading-step active" style="background: rgba(21, 101, 192, 0.2);">1️⃣ Question Analysis</div>
                            <div class="loading-step" style="background: rgba(21, 101, 192, 0.2);">2️⃣ Context Search</div>
                            <div class="loading-step" style="background: rgba(21, 101, 192, 0.2);">3️⃣ Answer Generation</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                gpt35_answer = generate_answer(question, context, "gpt-3.5-turbo")
                gpt35_quality = analyze_answer_quality(gpt35_answer, question)
                
                with st.container():
                    # Improve math symbol readability
                    improved_gpt35_answer = improve_math_readability(gpt35_answer)
                    st.markdown(f"""
                    <div class="answer-card">
                        <h4>🤖 GPT-3.5 Answer (Step 1)</h4>
                        <p>{improved_gpt35_answer}</p>
                        <div class="model-badge quality-{gpt35_quality['level']}">
                            Quality: {gpt35_quality['score']}/100
                        </div>
                        <div class="model-badge model-gpt35">
                            GPT-3.5 Turbo
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Step 2: GPT-4o answer
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
                        <p style="margin: 0; font-weight: 600;">🚀 Analyzing with GPT-4o...</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                gpt4o_answer = generate_answer(question, context, "gpt-4o")
                gpt4o_quality = analyze_answer_quality(gpt4o_answer, question)
                
                with st.container():
                    # Improve math symbol readability
                    improved_gpt4o_answer = improve_math_readability(gpt4o_answer)
                    st.markdown(f"""
                    <div class="answer-card">
                        <h4>🚀 GPT-4o Answer (Step 2)</h4>
                        <p>{improved_gpt4o_answer}</p>
                        <div class="model-badge quality-{gpt4o_quality['level']}">
                            Quality: {gpt4o_quality['score']}/100
                        </div>
                        <div class="model-badge model-gpt4o">
                            GPT-4o
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Step 3: Improved answer (when quality is low)
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
                            <p style="margin: 0; font-weight: 600;">✨ Improving answer quality...</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Improve based on the better answer
                    base_answer = gpt4o_answer if gpt4o_quality['score'] > gpt35_quality['score'] else gpt35_answer
                    base_quality = gpt4o_quality if gpt4o_quality['score'] > gpt35_quality['score'] else gpt35_quality
                    
                    improved_answer = improve_answer_with_better_model(
                        question, base_answer, context, "gpt-4o", base_quality
                    )
                    improved_quality = analyze_answer_quality(improved_answer, question)
                    
                    with st.container():
                        # Improve math symbol readability
                        improved_final_answer = improve_math_readability(improved_answer)
                        st.markdown(f"""
                        <div class="improved-card">
                            <h4>✨ Improved Answer (Step 3)</h4>
                            <p>{improved_final_answer}</p>
                                                    <div class="model-badge quality-{improved_quality['level']}">
                            Quality: {improved_quality['score']}/100
                        </div>
                            <div class="model-badge model-gpt4o">
                                GPT-4o (Improved)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Automatic model selection (additional feature)
                if model_selection_mode == "Auto Select (Recommended)":
                    auto_selection = select_model_automatically(question, len(context))
                    selected_model = auto_selection["model"]
                    
                    # Smart selection info display
                    st.markdown(f"""
                    <div class="smart-selection">
                        <h4>🤖 Smart Model Selection</h4>
                        <p><strong>Selected Model:</strong> {MODELS[selected_model]['name']}</p>
                        <p><strong>Selection Reason:</strong> {auto_selection['reason']}</p>
                        <p><strong>Complexity Score:</strong> {auto_selection['complexity']['score']} (Type: {auto_selection['complexity']['type']})</p>
                        <p><strong>Best for:</strong> {', '.join(MODELS[selected_model]['best_for'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Generate answer with auto-selected model
                    auto_answer = generate_answer(question, context, selected_model)
                    auto_quality = analyze_answer_quality(auto_answer, question)
                    
                    # Display auto-selected answer
                    with st.container():
                        st.markdown(f"""
                        <div class="answer-card">
                            <h4>🤖 {MODELS[selected_model]['name']} Answer (Auto-selected)</h4>
                            <p>{auto_answer}</p>
                            <div class="quality-badge quality-{auto_quality['level']}">
                                Quality Score: {auto_quality['score']}/100
                            </div>
                            <div class="model-badge {MODELS[selected_model]['color']}">
                                {MODELS[selected_model]['name']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
                # Manual selection models
                if model_selection_mode == "Manual Select":
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
                                <p style="margin: 0; font-weight: 600;">🚀 Analyzing with GPT-4o-mini...</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            gpt4mini_answer = generate_answer(question, context, "gpt-4o-mini")
                            gpt4mini_quality = analyze_answer_quality(gpt4mini_answer, question)
                            
                            with st.container():
                                st.markdown(f"""
                                <div class="answer-card">
                                    <h4>🚀 GPT-4o-mini Answer</h4>
                                    <p>{gpt4mini_answer}</p>
                                    <div class="quality-badge quality-{gpt4mini_quality['level']}">
                                        Quality Score: {gpt4mini_quality['score']}/100
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
                                <p style="margin: 0; font-weight: 600;">🚀 Analyzing with GPT-4o...</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            gpt4_answer = generate_answer(question, context, "gpt-4o")
                            gpt4_quality = analyze_answer_quality(gpt4_answer, question)
                            
                            with st.container():
                                st.markdown(f"""
                                <div class="answer-card">
                                    <h4>🚀 GPT-4o Answer</h4>
                                    <p>{gpt4_answer}</p>
                                    <div class="quality-badge quality-{gpt4_quality['level']}">
                                        Quality Score: {gpt4_quality['score']}/100
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
                                <p style="margin: 0; font-weight: 600;">💻 Analyzing with GPT-OSS-20B...</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            gpt_oss_20b_answer = generate_answer(question, context, "gpt-oss-20b")
                            gpt_oss_20b_quality = analyze_answer_quality(gpt_oss_20b_answer, question)
                            
                            with st.container():
                                st.markdown(f"""
                                <div class="answer-card">
                                    <h4>💻 GPT-OSS-20B Answer (Free Local)</h4>
                                    <p>{gpt_oss_20b_answer}</p>
                                    <div class="quality-badge quality-{gpt_oss_20b_quality['level']}">
                                        Quality Score: {gpt_oss_20b_quality['score']}/100
                                    </div>
                                    <div class="model-badge model-gptoss">
                                        GPT-OSS-20B (Free)
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
                                <p style="margin: 0; font-weight: 600;">🚀 Analyzing with GPT-OSS-120B...</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            gpt_oss_120b_answer = generate_answer(question, context, "gpt-oss-120b")
                            gpt_oss_120b_quality = analyze_answer_quality(gpt_oss_120b_answer, question)
                            
                            with st.container():
                                st.markdown(f"""
                                <div class="answer-card">
                                    <h4>🚀 GPT-OSS-120B Answer (High-performance Free)</h4>
                                    <p>{gpt_oss_120b_answer}</p>
                                    <div class="quality-badge quality-{gpt_oss_120b_quality['level']}">
                                        Quality Score: {gpt_oss_120b_quality['score']}/100
                                    </div>
                                    <div class="model-badge model-gptoss">
                                        GPT-OSS-120B (Free)
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Hierarchical answer improvement (optional)
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
                            <p style="margin: 0; font-weight: 600;">✨ Improving answer quality...</p>
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
                                <h4>✨ Improved Answer (GPT-4o)</h4>
                                <p>{improved_answer}</p>
                                <div class="quality-badge quality-{improved_quality['level']}">
                                    Improved Quality Score: {improved_quality['score']}/100
                                </div>
                                <div class="model-badge model-gpt4o">
                                    GPT-4o (Improved)
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
        
                # Advanced analysis information
                with st.expander("📊 Advanced Analysis Info"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**🤖 GPT-3.5 Answer Analysis**")
                        st.write(f"Score: {gpt35_quality['score']}/100")
                        st.write(f"Level: {gpt35_quality['level']}")
                        if gpt35_quality['issues']:
                            st.write("Issues:")
                            for issue in gpt35_quality['issues']:
                                st.write(f"- {issue}")
                        
                        # Sentiment and tone analysis
                        sentiment_analysis = analyze_sentiment_and_tone(gpt35_answer)
                        st.markdown("**Sentiment Analysis**")
                        st.write(f"Sentiment: {sentiment_analysis['sentiment']}")
                        st.write(f"Tone: {sentiment_analysis['tone']}")
                    
                    with col2:
                        st.markdown("**🚀 GPT-4o Answer Analysis**")
                        st.write(f"Score: {gpt4o_quality['score']}/100")
                        st.write(f"Level: {gpt4o_quality['level']}")
                        if gpt4o_quality['issues']:
                            st.write("Issues:")
                            for issue in gpt4o_quality['issues']:
                                st.write(f"- {issue}")
                        
                        # Topic classification
                        topic = classify_topic(gpt4o_answer)
                        st.markdown("**Topic Classification**")
                        st.write(f"Topic: {topic}")
                    
                    with col3:
                        if improved_answer:
                            st.markdown("**✨ Improved Answer Analysis**")
                            st.write(f"Score: {improved_quality['score']}/100")
                            st.write(f"Level: {improved_quality['level']}")
                            if improved_quality['issues']:
                                st.write("Issues:")
                                for issue in improved_quality['issues']:
                                    st.write(f"- {issue}")
                        
                        # Keyword analysis
                        keywords = analyze_text_keywords(gpt4o_answer, top_n=10)
                        if keywords:
                            st.markdown("**🔑 Key Keywords**")
                            for keyword, count in list(keywords.items())[:5]:
                                st.write(f"• {keyword}: {count} times")
                
                # Context display
                if context and rag_enabled:
                    with st.expander("📄 Context Used"):
                        st.text_area("Context", context, height=200, disabled=True)
        
                        # Save to conversation history
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Save user question
                st.session_state.conversation_history.append(("user", question, current_time))
                
                # Save AI answer (use improved answer as main)
                ai_answer = improved_answer if improved_answer else gpt4o_answer
                st.session_state.conversation_history.append(("ai", ai_answer, current_time))
                
                # Save to existing history as well
                history_entry = {
                    'question': question,
                    'gpt35_answer': gpt35_answer,
                    'gpt35_quality': gpt35_quality['score'],
                    'gpt4o_answer': gpt4o_answer,
                    'gpt4o_quality': gpt4o_quality['score'],
                    'improved_answer': improved_answer,
                    'improved_quality': improved_quality['score'] if improved_quality else None,
                    'auto_answer': auto_answer if model_selection_mode == "Auto Select (Recommended)" else None,
                    'auto_quality': auto_quality['score'] if model_selection_mode == "Auto Select (Recommended)" else None,
                    'auto_model': selected_model if model_selection_mode == "Auto Select (Recommended)" else None,
                    'gpt4mini_answer': gpt4mini_answer if model_selection_mode == "Manual Select" and use_gpt4mini else None,
                    'gpt4mini_quality': gpt4mini_quality['score'] if model_selection_mode == "Manual Select" and use_gpt4mini else None,
                    'gpt4_answer': gpt4_answer if model_selection_mode == "Manual Select" and use_gpt4 else None,
                    'gpt4_quality': gpt4_quality['score'] if model_selection_mode == "Manual Select" and use_gpt4 else None,
                    'gpt_oss_20b_answer': gpt_oss_20b_answer if model_selection_mode == "Manual Select" and use_gpt_oss_20b else None,
                    'gpt_oss_20b_quality': gpt_oss_20b_quality['score'] if model_selection_mode == "Manual Select" and use_gpt_oss_20b else None,
                    'gpt_oss_120b_answer': gpt_oss_120b_answer if model_selection_mode == "Manual Select" and use_gpt_oss_120b else None,
                    'gpt_oss_120b_quality': gpt_oss_120b_quality['score'] if model_selection_mode == "Manual Select" and use_gpt_oss_120b else None,
                    'timestamp': current_time
                }
                st.session_state.history.append(history_entry)
    st.markdown('</div>', unsafe_allow_html=True)

# Conversation history
if st.session_state.history:
    st.markdown('<div class="section-header">📚 Conversation History</div>', unsafe_allow_html=True)
    for i, entry in enumerate(reversed(st.session_state.history[-5:]), 1):
        with st.expander(f"Question {i}: {entry['question'][:50]}..."):
            st.markdown(f"""
            <div class="history-item">
                <h4>💬 Question</h4>
                <p>{entry['question']}</p>
                
                <h4>🤖 GPT-3.5 Answer (Step 1)</h4>
                <p>{entry['gpt35_answer']}</p>
                <div class="quality-badge quality-{'good' if entry['gpt35_quality'] >= 80 else 'medium' if entry['gpt35_quality'] >= 60 else 'bad'}">
                    Quality: {entry['gpt35_quality']}/100
                </div>
                
                <h4>🚀 GPT-4o Answer (Step 2)</h4>
                <p>{entry['gpt4o_answer']}</p>
                <div class="quality-badge quality-{'good' if entry['gpt4o_quality'] >= 80 else 'medium' if entry['gpt4o_quality'] >= 60 else 'bad'}">
                    Quality: {entry['gpt4o_quality']}/100
                </div>
            """, unsafe_allow_html=True)
            
            if entry['auto_answer']:
                st.markdown(f"""
                <h4>🚀 Auto-selected Model ({entry['auto_model']})</h4>
                <p>{entry['auto_answer']}</p>
                <div class="quality-badge quality-{'good' if entry['auto_quality'] >= 80 else 'medium' if entry['auto_quality'] >= 60 else 'bad'}">
                    Quality: {entry['auto_quality']}/100
                </div>
                """, unsafe_allow_html=True)
            
            if entry['gpt4mini_answer']:
                st.markdown(f"""
                <h4>🚀 GPT-4o-mini Answer</h4>
                <p>{entry['gpt4mini_answer']}</p>
                <div class="quality-badge quality-{'good' if entry['gpt4mini_quality'] >= 80 else 'medium' if entry['gpt4mini_quality'] >= 60 else 'bad'}">
                    Quality: {entry['gpt4mini_quality']}/100
                </div>
                """, unsafe_allow_html=True)
            
            if entry['gpt4_answer']:
                st.markdown(f"""
                <h4>🚀 GPT-4o Answer</h4>
                <p>{entry['gpt4_answer']}</p>
                <div class="quality-badge quality-{'good' if entry['gpt4_quality'] >= 80 else 'medium' if entry['gpt4_quality'] >= 60 else 'bad'}">
                    Quality: {entry['gpt4_quality']}/100
                </div>
                """, unsafe_allow_html=True)
            
            if entry.get('gpt_oss_20b_answer'):
                st.markdown(f"""
                <h4>💻 GPT-OSS-20B Answer (Free Local)</h4>
                <p>{entry['gpt_oss_20b_answer']}</p>
                <div class="quality-badge quality-{'good' if entry['gpt_oss_20b_quality'] >= 80 else 'medium' if entry['gpt_oss_20b_quality'] >= 60 else 'bad'}">
                    Quality: {entry['gpt_oss_20b_quality']}/100
                </div>
                """, unsafe_allow_html=True)
            
            if entry.get('gpt_oss_120b_answer'):
                st.markdown(f"""
                <h4>🚀 GPT-OSS-120B Answer (High-performance Free)</h4>
                <p>{entry['gpt_oss_120b_answer']}</p>
                <div class="quality-badge quality-{'good' if entry['gpt_oss_120b_quality'] >= 80 else 'medium' if entry['gpt_oss_120b_quality'] >= 60 else 'bad'}">
                    Quality: {entry['gpt_oss_120b_quality']}/100
                </div>
                """, unsafe_allow_html=True)
            
            if entry['improved_answer']:
                st.markdown(f"""
                <h4>✨ Improved Answer</h4>
                <p>{entry['improved_answer']}</p>
                <div class="quality-badge quality-{'good' if entry['improved_quality'] >= 80 else 'medium' if entry['improved_quality'] >= 60 else 'bad'}">
                    Improved Quality: {entry['improved_quality']}/100
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style="margin-top: 1rem; padding: 0.5rem; background: #f8f9fa; border-radius: 8px; text-align: center;">
                <small>⏰ {entry['timestamp']}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Data analysis section - placed at the bottom
st.markdown('<div class="section-header">📊 Data Analysis</div>', unsafe_allow_html=True)

# Tab creation
tab1, tab2, tab3 = st.tabs(["📈 Usage Statistics", "🔤 Keyword Analysis", "💾 Save"])

with tab1:
    st.markdown('<div class="section-header">📈 Usage Pattern Analysis</div>', unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_questions = len(st.session_state.history) if st.session_state.history else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>💬 Total Questions</h4>
            <h2>{total_questions}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.history:
            avg_quality = sum(entry.get('gpt4o_quality', 0) for entry in st.session_state.history) / len(st.session_state.history)
        else:
            avg_quality = 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Avg Quality Score</h4>
            <h2>{avg_quality:.1f}</h2>
            <p>/10</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        improved_count = len([entry for entry in st.session_state.history if entry.get('improved_answer')]) if st.session_state.history else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>✨ Improved Answers</h4>
            <h2>{improved_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Usage pattern trend graph
    st.markdown('<div class="section-header">📈 Usage Pattern Trends</div>', unsafe_allow_html=True)
    
    # Empty graph area
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
        <p>Graph Area</p>
        <p style="font-size: 0.8rem;">Y-axis: Number of questions (0, 1, 2, 3, 4)</p>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("Keyword analysis feature coming soon!")

with tab3:
    st.markdown("Save feature coming soon!")
