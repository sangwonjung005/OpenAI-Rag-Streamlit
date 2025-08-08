@echo off
echo 🚀 AI PDF Assistant 빠른 시작
cd /d C:\openai_pdf_rag
start http://localhost:8506
C:\Users\sangw\miniconda3\envs\rag_env\Scripts\streamlit.exe run app.py --server.port 8506 