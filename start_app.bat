@echo off
echo AI PDF Assistant 시작 중...
cd /d C:\openai_pdf_rag
echo 현재 디렉토리: %CD%
echo 환경 활성화 중...
call C:\Users\sangw\miniconda3\envs\rag_env\Scripts\activate.bat
echo Streamlit 앱 시작 중...
start http://localhost:8506
C:\Users\sangw\miniconda3\envs\rag_env\Scripts\streamlit.exe run app.py --server.port 8506
pause 