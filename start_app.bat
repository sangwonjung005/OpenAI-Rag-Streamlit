@echo off
echo AI PDF Assistant starting...
cd /d C:\openai_pdf_rag
echo Current directory: %CD%
echo Activating environment...
call C:\Users\sangw\miniconda3\envs\rag_env\Scripts\activate.bat
echo Starting Streamlit app...
start http://localhost:8506
C:\Users\sangw\miniconda3\envs\rag_env\Scripts\streamlit.exe run app.py --server.port 8506
pause 