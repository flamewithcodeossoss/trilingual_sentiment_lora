import streamlit as st
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="💬 Trilingual Sentiment Analyzer",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Enhanced Custom CSS Styling with Better Color Contrast
# -----------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: #0f172a;
    }
    
    .main-container {
        background: #ffffff;
        border-radius: 28px;
        box-shadow: 0 25px 70px rgba(0, 0, 0, 0.4);
        padding: 3rem 3.5rem;
        max-width: 850px;
        margin: 2rem auto;
        animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .header-section {
        text-align: center;
        margin-bottom: 2.5rem;
        position: relative;
    }
    
    .title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .subtitle {
        color: #334155 !important;
        font-size: 1.1rem;
        font-weight: 600;
        line-height: 1.6;
    }
    
    .language-badges {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    
    .badge {
        background: #1e3a8a;
        color: #ffffff;
        padding: 0.6rem 1.4rem;
        border-radius: 25px;
        font-size: 0.95rem;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.35);
        border: none;
        transition: all 0.3s ease;
    }
    
    .badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(30, 58, 138, 0.45);
        background: #1e40af;
    }
    
    .stTextArea textarea {
        border-radius: 16px !important;
        border: 2px solid #cbd5e1 !important;
        background: #ffffff !important;
        color: #0f172a !important;
        font-size: 1.05rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 4px rgba(124, 58, 237, 0.15) !important;
        background: #ffffff !important;
        color: #0f172a !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #94a3b8 !important;
    }
    
    .stTextArea label {
        font-weight: 700 !important;
        color: #0f172a !important;
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .analyze-btn {
        margin: 1.5rem 0;
    }
    
    .analyze-btn button {
        width: 100%;
        background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%) !important;
        color: white !important;
        border-radius: 16px !important;
        font-weight: 800 !important;
        border: none !important;
        padding: 1rem 2rem !important;
        font-size: 1.15rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4) !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase;
    }
    
    .analyze-btn button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(124, 58, 237, 0.5) !important;
    }
    
    .analyze-btn button:active {
        transform: translateY(-1px) !important;
    }
    
    .result-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #faf5ff 100%);
        border-radius: 24px;
        padding: 2.5rem;
        margin-top: 2rem;
        border: 3px solid #e0e7ff;
        animation: slideUp 0.5s ease-out;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .sentiment-tag {
        text-align: center;
        font-weight: 900;
        font-size: 1.8rem;
        color: white;
        padding: 1.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        letter-spacing: 1px;
        text-transform: uppercase;
        border: 3px solid rgba(255, 255, 255, 0.3);
    }
    
    .positive { 
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        box-shadow: 0 10px 30px rgba(5, 150, 105, 0.4);
    }
    
    .neutral { 
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        box-shadow: 0 10px 30px rgba(217, 119, 6, 0.4);
    }
    
    .negative { 
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        box-shadow: 0 10px 30px rgba(220, 38, 38, 0.4);
    }
    
    .confidence-box {
        background: white;
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 2rem;
        text-align: center;
        border: 2px solid #e0e7ff;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }
    
    .confidence-label {
        font-size: 1rem;
        color: #64748b !important;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .confidence-value {
        font-size: 2.2rem;
        font-weight: 900;
        color: #1e3a8a !important;
    }
    
    .scores-header {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0f172a !important;
        margin: 1.5rem 0 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #e0e7ff;
    }
    
    .stJson {
        background: white !important;
        border-radius: 14px !important;
        border: 2px solid #e0e7ff !important;
        margin-top: 1.5rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
    }
    
    .footer {
        text-align: center;
        color: #64748b !important;
        font-size: 0.95rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 2px solid #e2e8f0;
        font-weight: 600;
    }
    
    .example-section {
        background: #f1f5f9;
        border-radius: 16px;
        padding: 1.2rem 1.8rem;
        margin: 1.5rem 0;
        border-left: 5px solid #1e3a8a;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .example-title {
        font-weight: 800;
        color: #0f172a !important;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .example-text {
        color: #334155 !important;
        font-size: 0.95rem;
        font-style: italic;
        font-weight: 500;
    }
    
    /* Warning styling */
    .stAlert {
        border-radius: 14px !important;
        border-left: 5px solid #f59e0b !important;
        background: #fffbeb !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #7c3aed !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model and Tokenizer
# -----------------------------
@st.cache_resource
def load_model():
    base_model = "FacebookAI/xlm-roberta-base"
    adapter_model = "osamanaguib/trilingual-sentiment-lora"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=3)
    model = PeftModel.from_pretrained(model, adapter_model)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# Layout
# -----------------------------
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class='header-section'>
        <div class='title'>💬 Trilingual Sentiment Analyzer</div>
        <div class='subtitle'>
            Advanced AI-powered sentiment detection across three languages<br>
            Built with XLM-RoBERTa and LoRA fine-tuning
        </div>
        <div class='language-badges'>
            <span class='badge'>🇬🇧 English</span>
            <span class='badge'>🇸🇦 Arabic</span>
            <span class='badge'>🇫🇷 French</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Example section
st.markdown("""
    <div class='example-section'>
        <div class='example-title'>💡 Try these examples:</div>
        <div class='example-text'>
            "I absolutely love this product!" • "هذا رائع جداً" • "C'est un mauvais film"
        </div>
    </div>
""", unsafe_allow_html=True)

# Input
text_input = st.text_area(
    "Enter your text:",
    height=140,
    placeholder="Type or paste any text in English, Arabic, or French...",
    help="The model will automatically detect the language and analyze sentiment"
)

# Analyze button
st.markdown("<div class='analyze-btn'>", unsafe_allow_html=True)
analyze = st.button("🔍 Analyze Sentiment", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Analysis
if analyze:
    if not text_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🤖 Analyzing sentiment..."):
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

            labels = ["negative", "neutral", "positive"]
            results = {label: float(score) for label, score in zip(labels, scores)}

            predicted_label = labels[torch.argmax(scores)]
            confidence = results[predicted_label]
            
            color_class = {
                "positive": "positive",
                "negative": "negative",
                "neutral": "neutral"
            }[predicted_label]
            
            emoji_map = {
                "positive": "😊",
                "negative": "😞",
                "neutral": "😐"
            }

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            
            # Sentiment tag
            st.markdown(
                f"<div class='sentiment-tag {color_class}'>"
                f"{emoji_map[predicted_label]} {predicted_label}"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # Confidence box
            st.markdown(f"""
                <div class='confidence-box'>
                    <div class='confidence-label'>Confidence Score</div>
                    <div class='confidence-value'>{confidence:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Scores header
            st.markdown("<div class='scores-header'>📊 Detailed Score Breakdown</div>", unsafe_allow_html=True)

            # Enhanced Plotly visualization with better colors
            colors = {
                'negative': '#dc2626',
                'neutral': '#d97706', 
                'positive': '#059669'
            }
            
            bar_colors = [colors[label] for label in results.keys()]
            
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=[label.capitalize() for label in results.keys()],
                        y=list(results.values()),
                        text=[f"<b>{v:.1%}</b>" for v in results.values()],
                        textposition="outside",
                        textfont=dict(size=16, color='#0f172a', family="Inter"),
                        marker=dict(
                            color=bar_colors,
                            line=dict(color='white', width=3),
                            opacity=0.95
                        ),
                        hovertemplate="<b>%{x}</b><br>Confidence: <b>%{y:.2%}</b><extra></extra>",
                        width=[0.6, 0.6, 0.6]
                    )
                ]
            )
            
            fig.update_layout(
                xaxis_title="<b>Sentiment Class</b>",
                yaxis_title="<b>Confidence Score</b>",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(color="#0f172a", size=14, family="Inter", weight=600),
                margin=dict(t=50, b=50, l=50, r=50),
                height=380,
                xaxis=dict(
                    showgrid=False,
                    tickfont=dict(size=14, family="Inter", color="#0f172a"),
                    linecolor='#cbd5e1',
                    linewidth=2
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="#e2e8f0",
                    gridwidth=1,
                    tickformat=".0%",
                    range=[0, max(results.values()) * 1.2],
                    linecolor='#cbd5e1',
                    linewidth=2
                ),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=14,
                    font_family="Inter",
                    bordercolor="#cbd5e1"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # JSON output
            st.markdown("<div style='margin-top: 2rem;'>", unsafe_allow_html=True)
            st.json(results)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
        Built with ❤️ using Streamlit, Transformers & LoRA<br>
        © 2025 Trilingual Sentiment Analyzer
    </div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)