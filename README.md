# AI PDF Assistant

## 🚀 Quick Start

### Method 1: Double-click to Start
1. Double-click the `quick_start.bat` file
2. Browser will open automatically: http://localhost:8506

### Method 2: Manual Start
```bash
cd C:\openai_pdf_rag
C:\Users\sangw\miniconda3\envs\rag_env\Scripts\streamlit.exe run app.py --server.port 8506
```

## 📋 Features
- ✅ PDF upload and analysis
- ✅ GPT-3.5 and GPT-4o answer comparison
- ✅ GPT-OSS local model support (free)
- ✅ Automatic model selection
- ✅ Answer quality improvement
- ✅ Beautiful UI/UX

## 🤖 Supported Models
- **GPT-3.5 Turbo**: Fast and economical basic model
- **GPT-4o Mini**: Balanced performance and cost
- **GPT-4o**: Highest quality premium model
- **GPT-OSS-20B**: o3-mini level performance, free local execution (16GB RAM required)
- **GPT-OSS-120B**: o4-mini level performance, free local execution (80GB GPU required)
- **Claude 3.5 Sonnet**: Anthropic's latest model
- **Gemini Pro**: Google's high-performance model

## 🔧 GPT-OSS Local Server Setup
To use GPT-OSS models, you need to run a local server:

```bash
# Install vLLM
pip install vllm

# Start GPT-OSS-20B server (16GB RAM required)
vllm serve gpt-oss-20b --host 0.0.0.0 --port 8000

# Or start GPT-OSS-120B server (80GB GPU required)
vllm serve gpt-oss-120b --host 0.0.0.0 --port 8000
```

## 🔧 Troubleshooting
- If the app doesn't open: Run `taskkill /F /IM streamlit.exe` then restart
- After laptop restart: Double-click `quick_start.bat`
- GPT-OSS connection error: Check if local server is running

## 📝 Notes
- API keys are stored in `nocommit_key.txt` file
- App runs at `http://localhost:8506`
- GPT-OSS models are free to use (check hardware requirements)
