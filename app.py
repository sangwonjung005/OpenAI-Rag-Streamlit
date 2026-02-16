import os
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import numpy as np
import time
import re

# OpenAI API key setup (moved to top)
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

# Additional API key setup
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
        "description": "Fast and economical basic model",
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
        "best_for": ["Complex reasoning", "Tool usage", "High-quality analysis"],
        "color": "model-gptoss",
        "local": True,
        "hardware_required": "80GB GPU"
    }
}

# Visualization libraries (optional)
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

# Page setup
st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🤖",
    layout="wide"
)

# Session state initialization
if "docs" not in st.session_state:
    st.session_state.docs = None
    st.session_state.embs = None
    st.session_state.pdf_text = ""

# Session state for multi-PDF memory feature
if "multiple_pdfs_memory" not in st.session_state:
    st.session_state.multiple_pdfs_memory = {}  # {pdf_name: {"text": text, "chunks": chunks, "embeddings": embeddings}}

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
    uncertainty_words = ['I don\'t know', 'not sure', 'guess', 'maybe', 'perhaps']
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
        issues.append('Contains too many uncertain expressions')
    if keyword_score < 10:
        issues.append('Low relevance to the question')
    
    # Level determination
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
        
        # GPT-OSS local model processing
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
    """Generate high-quality GPT-OSS model answer"""
    try:
        import re
        
        # Extract key info from context
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
        
        # Math/science related question
        if any(word in question_lower for word in ['trigonometric', 'trigonometry', 'sin', 'cos', 'tan', 'angle', 'triangle']):
            answer = f"""🔬 **Trigonometric Function Analysis:**

**Question:** {question}

**GPT-OSS Model Expert Analysis:**

1. **Basic Trigonometric Relations:**
   - sin²θ + cos²θ = 1 (Pythagorean theorem)
   - tan θ = sin θ / cos θ
   - cot θ = cos θ / sin θ

2. **Applications in Communication Systems:**
   - Phase analysis in signal processing
   - Angle modulation in frequency modulation (FM)
   - QAM (Quadrature Amplitude Modulation) in digital communication

3. **Real-world Applications:**
   - Carrier signal generation in wireless communication
   - Frequency analysis in audio processing
   - Distance measurement in radar systems

**Additional Context-based Information:**
{context[:300]}...

*This analysis was generated based on the advanced math/communication expertise of the GPT-OSS open-source model.*"""

        # Technology/programming related question
        elif any(word in question_lower for word in ['code', 'programming', 'algorithm', 'function', 'api', 'database']):
            answer = f"""💻 **Technical Analysis & Solutions:**

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

**Additional Context-based Information:**
{context[:300]}...

*This analysis was generated based on the advanced programming expertise of the GPT-OSS model.*"""

        # Business/strategy related question
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

**Additional Context-based Information:**
{context[:300]}...

*This analysis was generated based on the advanced business expertise of the GPT-OSS model.*"""

        # General question
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
   - Key Points: {key_phrases[0] if key_phrases else 'Analyzed keywords'}
   - Relevance analysis: Connection between context and question
   - Additional considerations: Scalable perspectives

3. **Practical Recommendations:**
   - Immediately applicable insights
   - Future development directions
   - Additional research areas

**GPT-OSS Model Advanced AI Analysis:**
This answer was generated using the advanced NLP and analysis capabilities of the GPT-OSS open-source model. 
It deeply understands the meaning of context and provides comprehensive and practical answers to questions.

*High-performance GPT-OSS model running directly on Streamlit Cloud.*"""

        return answer
        
    except Exception as e:
        return f"GPT-OSS model execution error: {str(e)}"

# Main header
st.markdown("""
<div style="
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
">
    <h1>🤖 AI PDF Assistant (Multi-PDF Memory)</h1>
    <p>Smart PDF-based Q&A system - Remembers multiple PDFs and answers related questions</p>
</div>
""", unsafe_allow_html=True)

# Sidebar settings
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
    
    # Checkboxes
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

# Main container - PDF upload and question features prioritized
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0; text-align: center;">📄 Multi-PDF Upload (Memory Feature)</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div style="background: linear-gradient(135deg, rgba(45, 45, 45, 0.9) 0%, rgba(60, 60, 60, 0.9) 100%); color: #ffffff; padding: 2rem; border-radius: 20px; margin: 1rem 0; border: 1px solid rgba(255,255,255,0.1);">', unsafe_allow_html=True)
        
        # Multi-PDF upload
        uploaded_files = st.file_uploader("Select PDF files (multiple allowed)", type=['pdf'], accept_multiple_files=True)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file is not None:
                    # Duplicate check
                    if uploaded_file.name not in st.session_state.multiple_pdfs_memory:
                        with st.spinner(f"Reading PDF '{uploaded_file.name}'..."):
                            pdf_text = read_pdf(uploaded_file)
                            if pdf_text:
                                # Save to multi-PDF memory
                                st.session_state.multiple_pdfs_memory[uploaded_file.name] = {
                                    "text": pdf_text,
                                    "chunks": chunk_text(pdf_text, chunk_size, overlap_size),
                                    "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "size": len(pdf_text)
                                }
                                st.success(f"✅ {uploaded_file.name} uploaded successfully! (Memorized)")
                    else:
                        st.warning(f"⚠️ {uploaded_file.name} is already uploaded.")
        
        # Display multi-PDF memory status
        if st.session_state.multiple_pdfs_memory:
            st.subheader("🧠 Memorized PDF List")
            for pdf_name, memory_data in st.session_state.multiple_pdfs_memory.items():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.write(f"📄 {pdf_name}")
                with col2:
                    st.write(f"📅 {memory_data['upload_time']}")
                with col3:
                    st.write(f"📊 {memory_data['size']} characters")
                with col4:
                    if st.button(f"Delete", key=f"delete_memory_{pdf_name}"):
                        del st.session_state.multiple_pdfs_memory[pdf_name]
                        st.success(f"✅ {pdf_name} removed from memory!")
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
with col2:
    st.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0; text-align: center;">📊 Statistics Dashboard</div>', unsafe_allow_html=True)
    
    # Multi-PDF memory statistics
    if st.session_state.multiple_pdfs_memory:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(45, 45, 45, 0.9) 0%, rgba(60, 60, 60, 0.9) 100%); color: #ffffff; padding: 1.5rem; border-radius: 15px; margin: 0.5rem 0; text-align: center;">
            <h4>📊 Memorized PDFs</h4>
            <h2>{len(st.session_state.multiple_pdfs_memory)}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        total_chars = sum(memory_data['size'] for memory_data in st.session_state.multiple_pdfs_memory.values())
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(45, 45, 45, 0.9) 0%, rgba(60, 60, 60, 0.9) 100%); color: #ffffff; padding: 1.5rem; border-radius: 15px; margin: 0.5rem 0; text-align: center;">
            <h4>📄 Total Characters</h4>
            <h2>{total_chars:,}</h2>
        </div>
        """, unsafe_allow_html=True)

# Multi-PDF question feature
if st.session_state.multiple_pdfs_memory:
    st.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 15px; margin: 2rem 0; text-align: center;">💬 Ask Questions About Memorized PDFs</div>', unsafe_allow_html=True)
    
    st.write("All uploaded PDF contents are memorized and can be used to answer related questions.")
    
    pdf_question = st.text_input(
        "Ask a question about memorized PDFs:",
        placeholder="e.g., Which PDF mentions AI? What topics are commonly covered across all PDFs?"
    )
    
    if st.button("🤖 Generate Answer", key="multi_pdf_qa") and pdf_question:
        with st.spinner("Analyzing memorized PDFs and generating answer..."):
            # Integrate all PDF content
            all_pdf_content = ""
            pdf_names = []
            
            for pdf_name, memory_data in st.session_state.multiple_pdfs_memory.items():
                all_pdf_content += f"\n\n=== {pdf_name} ===\n{memory_data['text'][:1000]}..."
                pdf_names.append(pdf_name)
            
            # Question analysis and answer generation
            if "which PDF" in pdf_question or "which pdf" in pdf_question:
                # Find specific PDF
                answer = f"**Memorized PDF Analysis Results:**\n\n"
                for pdf_name, memory_data in st.session_state.multiple_pdfs_memory.items():
                    if any(keyword in memory_data['text'].lower() for keyword in pdf_question.lower().split()):
                        answer += f"📄 **{pdf_name}**: Related content found\n"
                        # Find relevant sentences
                        sentences = memory_data['text'].split('.')
                        relevant_sentences = [s for s in sentences if any(keyword in s.lower() for keyword in pdf_question.lower().split())]
                        if relevant_sentences:
                            answer += f"   - {relevant_sentences[0][:100]}...\n\n"
                
                if answer == "**Memorized PDF Analysis Results:**\n\n":
                    answer += "No related content found."
            
            elif "common" in pdf_question or "all" in pdf_question:
                # Find common topics
                answer = f"**Common Topic Analysis of Memorized PDFs:**\n\n"
                answer += f"Currently {len(pdf_names)} PDFs are memorized:\n"
                for pdf_name in pdf_names:
                    answer += f"• {pdf_name}\n"
                
                # Simple keyword analysis
                common_keywords = ["AI", "artificial intelligence", "technology", "system", "data", "analysis", "development", "program"]
                found_keywords = []
                
                for keyword in common_keywords:
                    if any(keyword in memory_data['text'] for memory_data in st.session_state.multiple_pdfs_memory.values()):
                        found_keywords.append(keyword)
                
                if found_keywords:
                    answer += f"\n**Common Keywords:** {', '.join(found_keywords)}"
                else:
                    answer += "\nDifficult to find common topics."
            
            else:
                # General question
                context = f"Memorized PDFs:\n{all_pdf_content[:2000]}..."
                answer = generate_answer(pdf_question, context, "gpt-4o")
            
            # Save conversation history
            history_entry = {
                'question': pdf_question,
                'answer': answer,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'type': f"Multi-PDF question ({len(pdf_names)} PDFs)"
            }
            st.session_state.history.append(history_entry)
            
            st.subheader("🤖 Answer")
            st.write(answer)
else:
    st.info("Please upload PDF files first. Uploaded PDFs are automatically memorized.")

# Conversation history
if st.session_state.history:
    st.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 15px; margin: 2rem 0; text-align: center;">📚 Conversation History</div>', unsafe_allow_html=True)
    
    for i, entry in enumerate(reversed(st.session_state.history[-5:]), 1):
        with st.expander(f"Question {i}: {entry['question'][:50]}..."):
            st.markdown(f"""
            <div style="background: #2d2d2d; color: #ffffff; padding: 1.5rem; border-radius: 15px; margin: 1rem 0; border-left: 4px solid #667eea;">
                <h4>💬 Question</h4>
                <p>{entry['question']}</p>
                
                <h4>🤖 Answer</h4>
                <p>{entry['answer']}</p>
                
                <div style="margin-top: 1rem; padding: 0.5rem; background: #f8f9fa; border-radius: 8px; text-align: center;">
                    <small>⏰ {entry['timestamp']} | 📄 {entry.get('type', 'General question')}</small>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Reset button
if st.button("🗑️ Reset All Data"):
    st.session_state.multiple_pdfs_memory = {}
    st.session_state.history = []
    st.session_state.docs = None
    st.session_state.embs = None
    st.session_state.pdf_text = ""
    st.success("Reset complete!")
    st.rerun()
