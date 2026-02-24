import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
from ml.predict import predict_disease
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# Page Configuration
st.set_page_config(
    page_title="MediPredict AI - Disease Prediction",
    page_icon="🏥",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    /* Main Background and Text */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Multiselect Tag Styling - BLACK BACKGROUND */
    span[data-baseweb="tag"] {
        background-color: #000000 !important;
        border: 1px solid #30363d !important;
        border-radius: 4px !important;
    }
    
    span[data-baseweb="tag"] span {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    [data-testid="stMultiSelect"] span {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #58a6ff;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #58a6ff;
    }
    
    /* Result Header - HIGH CONTRAST */
    .result-header {
        background: linear-gradient(90deg, #1f6feb 0%, #111d2c 100%);
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        border: 2px solid #58a6ff;
        text-align: center;
    }
    
    .prediction-title {
        font-size: 48px !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 0px !important;
    }
    
    .confidence-badge {
        background-color: #238636;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 18px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #238636 !important;
        color: white !important;
        width: 100% !important;
        border: none !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# Load Symptoms List
@st.cache_resource
def get_symptoms_list():
    try:
        symptom_cols = joblib.load('models/symptom_columns.pkl')
        return sorted([s.replace('_', ' ').title() for s in symptom_cols])
    except:
        return []

SYMPTOMS = get_symptoms_list()

# PDF Generation Function
def generate_pdf(result):
    report_path = 'medical_report.pdf'
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Header
    elements.append(Paragraph("MediPredict AI - Patient Report", styles['Title']))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Symptoms
    elements.append(Paragraph("Symptoms Entered:", styles['Heading2']))
    elements.append(Paragraph(", ".join(result['symptoms_entered']), styles['Normal']))
    elements.append(Spacer(1, 12))

    # Prediction
    elements.append(Paragraph(f"Primary Predicted Disease: {result['primary_prediction']}", styles['Heading2']))
    elements.append(Paragraph(f"Confidence Score: {result['confidence']}%", styles['Normal']))
    elements.append(Paragraph(f"Risk Level: {result['risk_level']}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Description
    elements.append(Paragraph("Disease Description:", styles['Heading2']))
    elements.append(Paragraph(result['description'], styles['Normal']))
    elements.append(Spacer(1, 12))

    # Precautions
    elements.append(Paragraph("Recommended Precautions:", styles['Heading2']))
    for p in result['precautions']:
        elements.append(Paragraph(f"- {p}", styles['Normal']))
    elements.append(Spacer(1, 24))

    # Disclaimer
    elements.append(Paragraph("--- MEDICAL DISCLAIMER ---", styles['Heading3']))
    disclaimer_style = styles['Normal']
    disclaimer_style.textColor = colors.red
    elements.append(Paragraph("This tool is for educational purposes only. Always consult a certified medical professional. The predictions provided by this AI should not be taken as a final medical diagnosis.", disclaimer_style))

    doc.build(elements)
    return report_path

# Sidebar
with st.sidebar:
    st.title("🏥 MediPredict AI")
    st.markdown("---")
    st.markdown("""
    ### About
    Advanced Disease Prediction System powered by Random Forest, Naive Bayes, and SVM models.
    
    ### Stats
    - **41** Diseases
    - **131** Symptoms
    - **98%+** Accuracy
    """)
    st.markdown("---")
    st.error("⚠️ **MEDICAL DISCLAIMER:** This tool is for educational purposes only. Always consult a certified medical professional.")

# Main Header
st.title("🛡️ Disease Prediction Dashboard")
st.markdown("Select your symptoms below to get an AI-powered diagnosis.")

# Form Area
with st.container():
    st.markdown("### 🤒 Symptoms Selection")
    selected_symptoms = st.multiselect(
        "Search and select symptoms you are experiencing (min 3 recommended):",
        options=SYMPTOMS,
        placeholder="e.g. Itching, Fever, Skin Rash..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_btn = st.button("🔍 Predict Disease")

# Result Display
if predict_btn:
    if not selected_symptoms or len(selected_symptoms) < 3:
        st.warning("⚠️ Please select at least 3 symptoms for an accurate prediction.")
    else:
        result = None
        with st.status("🧠 AI Models analyzing symptoms...", expanded=True) as status:
            try:
                # Prepare symptoms for prediction
                formatted_symptoms = [s.lower().replace(' ', '_') for s in selected_symptoms]
                result = predict_disease(formatted_symptoms)
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label=f"❌ Analysis Failed: {e}", state="error")
                st.error(f"❌ Prediction Error: {e}")
                st.write("Please ensure the models are trained correctly by running `run_once.py`.")

        if result:
            st.markdown("---")
            
            # Top Results Section
            res_col1, res_col2 = st.columns([2, 1])
            
            with res_col1:
                st.markdown(f"""
                <div class="result-header">
                    <p class="prediction-title">{result['primary_prediction']}</p>
                    <div style="margin-top: 15px;">
                        <span class="confidence-badge">Accuracy: {result['confidence']}%</span>
                        <span style="margin-left: 10px; font-size: 18px;">Risk Level: <b>{result['risk_level']}</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 📝 Disease Overview")
                st.info(result['description'])
                
                st.markdown("### 🛡️ Recommended Precautions")
                cols = st.columns(2)
                for i, p in enumerate(result['precautions']):
                    cols[i % 2].success(f"✅ {p.title()}")
            
            with res_col2:
                st.markdown("### 📊 Model Agreement")
                agreement = result['model_agreement']
                st.progress(agreement / 3, text=f"{agreement}/3 Models Unified")
                
                st.markdown("### 🔍 Model Breakdown")
                st.markdown(f"""
                <div style="background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d;">
                    <p style="color: #8b949e; margin-bottom: 5px;">Random Forest:</p>
                    <p style="font-weight: bold; color: #58a6ff;">{result['rf_prediction']}</p>
                    <hr style="margin: 10px 0; border-color: #30363d;">
                    <p style="color: #8b949e; margin-bottom: 5px;">Naive Bayes:</p>
                    <p style="font-weight: bold; color: #58a6ff;">{result['nb_prediction']}</p>
                    <hr style="margin: 10px 0; border-color: #30363d;">
                    <p style="color: #8b949e; margin-bottom: 5px;">SVM:</p>
                    <p style="font-weight: bold; color: #58a6ff;">{result['svm_prediction']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Report Section
            st.markdown("---")
            pdf_path = generate_pdf(result)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📥 Download Detailed Medical Report (PDF)",
                    data=f,
                    file_name=f"MediPredict_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
                    

# Performance Stats (Footer)
st.markdown("---")
p_col1, p_col2, p_col3, p_col4 = st.columns(4)
with p_col1:
    st.markdown('<div class="metric-card"><div class="metric-value">41</div><p>Diseases</p></div>', unsafe_allow_html=True)
with p_col2:
    st.markdown('<div class="metric-card"><div class="metric-value">131</div><p>Symptoms</p></div>', unsafe_allow_html=True)
with p_col3:
    st.markdown('<div class="metric-card"><div class="metric-value">3</div><p>Models</p></div>', unsafe_allow_html=True)
with p_col4:
    st.markdown('<div class="metric-card"><div class="metric-value">98.7%</div><p>Max Accuracy</p></div>', unsafe_allow_html=True)
