@echo off
echo Starting GPT-OSS Server (Alternative Method)...
echo.

REM Activate conda environment
call C:\Users\sangw\miniconda3\Scripts\activate.bat base

REM Install simpler alternatives
echo Installing alternative packages...
pip install flask transformers torch

REM Create a simple server script
echo Creating simple server...
echo import requests > simple_server.py
echo import json >> simple_server.py
echo from flask import Flask, request, jsonify >> simple_server.py
echo app = Flask(__name__) >> simple_server.py
echo. >> simple_server.py
echo @app.route('/generate', methods=['POST']) >> simple_server.py
echo def generate(): >> simple_server.py
echo     data = request.json >> simple_server.py
echo     prompt = data.get('prompt', '') >> simple_server.py
echo     return jsonify({'response': f'GPT-OSS Mock Response: {prompt[:100]}... (Server is running!)'}) >> simple_server.py
echo. >> simple_server.py
echo @app.route('/health', methods=['GET']) >> simple_server.py
echo def health(): >> simple_server.py
echo     return jsonify({'status': 'ok'}) >> simple_server.py
echo. >> simple_server.py
echo if __name__ == '__main__': >> simple_server.py
echo     print('Starting mock GPT-OSS server on http://localhost:8000') >> simple_server.py
echo     app.run(host='0.0.0.0', port=8000) >> simple_server.py

REM Start the server
echo Starting mock GPT-OSS server...
echo Server will be available at: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

python simple_server.py

pause
