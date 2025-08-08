# AI PDF Assistant

## 🚀 빠른 시작

### 방법 1: 더블클릭으로 시작
1. `quick_start.bat` 파일을 더블클릭
2. 브라우저가 자동으로 열립니다: http://localhost:8506

### 방법 2: 수동 시작
```bash
cd C:\openai_pdf_rag
C:\Users\sangw\miniconda3\envs\rag_env\Scripts\streamlit.exe run app.py --server.port 8506
```

## 📋 기능
- ✅ PDF 업로드 및 분석
- ✅ GPT-3.5와 GPT-4o 답변 비교
- ✅ 자동 모델 선택
- ✅ 답변 품질 개선
- ✅ 아름다운 UI/UX

## 🔧 문제 해결
- 앱이 안 열리면: `taskkill /F /IM streamlit.exe` 실행 후 다시 시작
- 노트북 재시작 후: `quick_start.bat` 더블클릭

## 📝 참고사항
- API 키는 `nocommit_key.txt` 파일에 저장
- 앱은 `http://localhost:8506`에서 실행
