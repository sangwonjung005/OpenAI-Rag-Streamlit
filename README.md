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
- ✅ GPT-OSS 로컬 모델 지원 (무료)
- ✅ 자동 모델 선택
- ✅ 답변 품질 개선
- ✅ 아름다운 UI/UX

## 🤖 지원 모델
- **GPT-3.5 Turbo**: 빠르고 경제적인 기본 모델
- **GPT-4o Mini**: 균형잡힌 성능과 비용
- **GPT-4o**: 최고 품질의 고급 모델
- **GPT-OSS-20B**: o3-mini 수준 성능, 무료 로컬 실행 (16GB RAM 필요)
- **GPT-OSS-120B**: o4-mini 수준 성능, 무료 로컬 실행 (80GB GPU 필요)
- **Claude 3.5 Sonnet**: Anthropic의 최신 모델
- **Gemini Pro**: Google의 고성능 모델

## 🔧 GPT-OSS 로컬 서버 설정
GPT-OSS 모델을 사용하려면 로컬 서버를 실행해야 합니다:

```bash
# vLLM 설치
pip install vllm

# GPT-OSS-20B 서버 시작 (16GB RAM 필요)
vllm serve gpt-oss-20b --host 0.0.0.0 --port 8000

# 또는 GPT-OSS-120B 서버 시작 (80GB GPU 필요)
vllm serve gpt-oss-120b --host 0.0.0.0 --port 8000
```

## 🔧 문제 해결
- 앱이 안 열리면: `taskkill /F /IM streamlit.exe` 실행 후 다시 시작
- 노트북 재시작 후: `quick_start.bat` 더블클릭
- GPT-OSS 연결 오류: 로컬 서버가 실행 중인지 확인

## 📝 참고사항
- API 키는 `nocommit_key.txt` 파일에 저장
- 앱은 `http://localhost:8506`에서 실행
- GPT-OSS 모델은 무료로 사용 가능 (하드웨어 요구사항 확인)
