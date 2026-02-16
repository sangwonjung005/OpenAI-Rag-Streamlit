import streamlit as st
import openai
import requests
import json
import time
from typing import Optional
import os

# Page settings
st.set_page_config(
    page_title="GPT-OSS Test",
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

# New GPT-OSS API call function
def call_gpt_oss_api_new(user_question: str, context: str = "") -> str:
    """New GPT-OSS API call function"""
    try:
        if not client:
            return "OpenAI API key is not configured."
        
        # Simple prompt
        prompt = user_question
        
        # OpenAI API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer questions directly and accurately."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        # Actual AI response
        ai_response = response.choices[0].message.content.strip()
        
        # New formatting
        formatted_response = f"""**New GPT-OSS Answer**

**Question:** {user_question}

**Answer:** {ai_response}

---
This is a new test version."""
        
        return formatted_response
            
    except Exception as e:
        return f"Error: {str(e)}"

# Main UI
def main():
    st.title("🚀 GPT-OSS Test")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        if client:
            st.success("✅ OpenAI API connected")
        else:
            st.error("❌ No OpenAI API key")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Question Input")
        question = st.text_area(
            "Enter your question:",
            height=150,
            placeholder="e.g., What is the transistors?"
        )
        
        if st.button("🚀 Generate Answer", type="primary"):
            if question.strip():
                with st.spinner("Generating answer..."):
                    response = call_gpt_oss_api_new(question)
                    
                    # Save results
                    st.session_state['last_response'] = response
                    st.session_state['last_question'] = question
    
    with col2:
        st.subheader("💬 Answer Results")
        
        if 'last_response' in st.session_state:
            # Display question
            if 'last_question' in st.session_state:
                st.markdown(f"**Question:** {st.session_state['last_question']}")
            
            # Display answer
            st.markdown("**Answer:**")
            st.write(st.session_state['last_response'])
        else:
            st.info("👈 Enter a question on the left and generate an answer!")

# Run app
if __name__ == "__main__":
    main()
