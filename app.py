from flask import Flask, render_template, request, jsonify, send_file
import joblib
import json
import os
import pandas as pd
from datetime import datetime
from ml.predict import predict_disease
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

app = Flask(__name__)

# Load symptom list for autocomplete
try:
    symptom_cols = joblib.load('models/symptom_columns.pkl')
    SYMPTOMS = sorted([s.replace('_', ' ').title() for s in symptom_cols])
except:
    SYMPTOMS = []

@app.route('/')
def home():
    return render_template('index.html', symptoms=SYMPTOMS)

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.getlist('symptoms')
    if not symptoms or len(symptoms) < 3:
        error = "Please select at least 3 symptoms"
        return render_template('index.html', symptoms=SYMPTOMS, error=error)
    
    try:
        result = predict_disease(symptoms)
        # Store last result in session or use a global for PDF generation (demo purpose)
        # Note: In a real app, use a proper session or database
        app.config['LAST_RESULT'] = result
        return render_template('result.html', result=result)
    except Exception as e:
        app.logger.error(f"Prediction Error: {e}")
        return render_template('index.html', symptoms=SYMPTOMS, error="An error occurred during prediction.")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for external use"""
    data = request.get_json()
    symptoms = data.get('symptoms', [])
    if not symptoms or len(symptoms) < 3:
        return jsonify({'error': 'Please select at least 3 symptoms'}), 400
    
    try:
        result = predict_disease(symptoms)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({'symptoms': SYMPTOMS})

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/diseases')
def diseases():
    try:
        desc_df = pd.read_csv('data/symptom_Description.csv')
        diseases_list = desc_df.to_dict('records')
    except:
        diseases_list = []
    return render_template('diseases.html', diseases=diseases_list)

@app.route('/download-report')
def download_report():
    result = app.config.get('LAST_RESULT')
    if not result:
        return "No report data found. Please perform a prediction first.", 404
    
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
    elements.append(Spacer(1, 12))

    # Disclaimer
    elements.append(Spacer(1, 24))
    elements.append(Paragraph("--- MEDICAL DISCLAIMER ---", styles['Heading3']))
    disclaimer_style = styles['Normal']
    disclaimer_style.textColor = colors.red
    elements.append(Paragraph("This tool is for educational purposes only. Always consult a certified medical professional. The predictions provided by this AI should not be taken as a final medical diagnosis.", disclaimer_style))

    doc.build(elements)
    
    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
