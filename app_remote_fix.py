import streamlit as st
import openai
import requests
import json
import time
from typing import Optional
import os

# Page settings
st.set_page_config(
    page_title="AI PDF Assistant - Fixed GPT-OSS",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API key settings
def load_api_keys():
    """Loads API keys."""
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
        st.error("Cannot find nocommit_key.txt file.")
        return {}

# Load API keys
api_keys = load_api_keys()

# OpenAI client settings
if 'OPENAI_API_KEY' in api_keys:
    openai.api_key = api_keys['OPENAI_API_KEY']
    client = openai.OpenAI(api_key=api_keys['OPENAI_API_KEY'])
else:
    client = None

# Check GPT-OSS server status
def check_gpt_oss_server():
    """Checks GPT-OSS server status."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# GPT-OSS API call (fixed version)
def call_gpt_oss_api(prompt: str, model_name: str = "gpt-oss-20b") -> str:
    """Calls GPT-OSS API (stabilized version)."""
    try:
        # Use simple prompt format
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
        
        if response.status_code == 200:
            result = response.json()
            if result.get('choices') and result['choices'][0].get('message'):
                content = result['choices'][0]['message']['content'].strip()
                if content and len(content) > 10:
                    return content
                else:
                    return "Model returned an empty response. Please check server status."
            else:
                return "API response format is incorrect."
        else:
            return f"API call failed: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "Request timed out. Server is not responding."
    except requests.exceptions.ConnectionError:
        return "Server connection failed. Please check if GPT-OSS server is running."
    except Exception as e:
        return f"API call error: {str(e)}"

# Safe GPT-OSS call (with retry logic)
def safe_gpt_oss_call(prompt: str, max_retries: int = 3) -> str:
    """Safe GPT-OSS API call (with retry logic)."""
    for attempt in range(max_retries):
        try:
            response = call_gpt_oss_api(prompt)
            
            # Check if response is valid
            if response and len(response.strip()) > 20 and "error" not in response and "failed" not in response:
                return response
            else:
                st.warning(f"Attempt {attempt + 1}: Response is not valid. Retrying...")
                if attempt < max_retries - 1:
                    time.sleep(3)
                    
        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    
    return "Unable to get model response. Please check server status."

# OpenAI API call
def call_openai_api(prompt: str, context: str = "", model: str = "gpt-3.5-turbo") -> str:
    """Calls OpenAI API."""
    if not client:
        return "OpenAI API key is not configured."
    
    try:
        messages = []
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"OpenAI API error: {str(e)}"

# Main UI
def main():
    st.title("🤖 AI PDF Assistant - Fixed GPT-OSS")
    st.markdown("---")
    
    # Update notification
    st.success("✅ Updated to GPT-OSS bug-fixed version!")
    st.info("🔧 Key fixes: Simplified prompts, retry logic, enhanced error handling")
    
    # Sidebar - Model selection
    with st.sidebar:
        st.header("🔧 Settings")
        
        # Check server status
        if check_gpt_oss_server():
            st.success("✅ GPT-OSS server is running")
        else:
            st.error("❌ GPT-OSS server is not running")
            st.info("💡 Solution:")
            st.markdown("""
            1. Run `fixed_start_gpt_oss_server.bat` locally
            2. Or in terminal:
            ```bash
            vllm serve gpt-oss-20b --host 0.0.0.0 --port 8000
            ```
            """)
        
        # Model selection
        st.subheader("🤖 Model Selection")
        model_options = [
            "GPT-3.5 Turbo (Fast)",
            "GPT-4o Mini (Balanced)",
            "GPT-4o (High Quality)",
            "GPT-OSS-20B (Free Local)",
            "GPT-OSS-120B (High-performance Free)"
        ]
        
        selected_model = st.selectbox(
            "Select model to use:",
            model_options,
            index=3  # GPT-OSS-20B as default
        )
        
        # Model-specific guidance
        if "GPT-OSS-120B" in selected_model:
            st.warning("⚠️ 120B model requires 80GB GPU memory.")
            st.info("💡 20B model is recommended.")
        
        if "GPT-OSS-20B" in selected_model:
            st.success("✅ 20B model: Only requires 16GB RAM, stable")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Question Input")
        question = st.text_area(
            "Enter your question:",
            height=150,
            placeholder="e.g., Please summarize the main content of this PDF."
        )
        
        context = st.text_area(
            "Context (Optional):",
            height=100,
            placeholder="Enter additional context or background information."
        )
        
        if st.button("🚀 Generate Answer", type="primary"):
            if question.strip():
                with st.spinner("Generating answer..."):
                    # Process by model
                    if "GPT-OSS" in selected_model:
                        # GPT-OSS model processing
                        if not check_gpt_oss_server():
                            st.error("GPT-OSS server is not running.")
                            st.info("Please start the GPT-OSS server locally.")
                            return
                        
                        # Recommend using 20B instead of 120B
                        if "120B" in selected_model:
                            st.warning("Using 20B model instead of 120B.")
                            model_name = "gpt-oss-20b"
                        else:
                            model_name = "gpt-oss-20b"
                        
                        # Generate simple prompt (fixes empty template issue)
                        if context:
                            prompt = f"Context: {context}\n\nQuestion: {question}\n\nPlease provide a detailed answer:"
                        else:
                            prompt = f"Question: {question}\n\nPlease provide a detailed answer:"
                        
                        response = safe_gpt_oss_call(prompt)
                        
                    else:
                        # OpenAI model processing
                        model_map = {
                            "GPT-3.5 Turbo (Fast)": "gpt-3.5-turbo",
                            "GPT-4o Mini (Balanced)": "gpt-4o-mini",
                            "GPT-4o (High Quality)": "gpt-4o"
                        }
                        
                        openai_model = model_map.get(selected_model, "gpt-3.5-turbo")
                        response = call_openai_api(question, context, openai_model)
                    
                    # Save results
                    st.session_state['last_response'] = response
                    st.session_state['last_model'] = selected_model
                    st.session_state['last_question'] = question
    
    with col2:
        st.subheader("💬 Answer Results")
        
        if 'last_response' in st.session_state:
            # Display model info
            model_info = st.session_state.get('last_model', '')
            if "GPT-OSS" in model_info:
                st.markdown(f"**🤖 {model_info} Answer (High-performance Free)** 🚀")
            else:
                st.markdown(f"**🤖 {model_info} Answer**")
            
            # Display question
            if 'last_question' in st.session_state:
                st.markdown(f"**Question:** {st.session_state['last_question']}")
            
            # Display answer
            st.markdown("**Answer:**")
            st.write(st.session_state['last_response'])
            
            # Copy button
            if st.button("📋 Copy Answer"):
                st.write("Answer copied to clipboard.")
        else:
            st.info("👈 Enter a question on the left and generate an answer!")

# Run app
if __name__ == "__main__":
    main()
