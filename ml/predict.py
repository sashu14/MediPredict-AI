import joblib
import pandas as pd
import numpy as np
import os

def load_models():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_path = os.path.join(base_path, 'models')
    
    rf = joblib.load(os.path.join(models_path, 'random_forest_model.pkl'))
    nb = joblib.load(os.path.join(models_path, 'naive_bayes_model.pkl'))
    svm = joblib.load(os.path.join(models_path, 'svm_model.pkl'))
    le = joblib.load(os.path.join(models_path, 'label_encoder.pkl'))
    cols = joblib.load(os.path.join(models_path, 'symptom_columns.pkl'))
    return rf, nb, svm, le, cols

def predict_disease(symptoms_list):
    """
    symptoms_list: list of symptom strings
    e.g. ['itching', 'skin_rash', 'fever']
    Returns: dict with predictions from all 3 models
    """
    rf, nb, svm, le, cols = load_models()
    
    # Create input vector
    input_vector = pd.DataFrame(
        np.zeros((1, len(cols))), columns=cols)
    
    for symptom in symptoms_list:
        symptom = symptom.strip().lower().replace(' ', '_')
        if symptom in cols:
            input_vector.at[0, symptom] = 1
    
    # Predictions from all 3 models
    rf_pred = le.inverse_transform(rf.predict(input_vector))[0]
    nb_pred = le.inverse_transform(nb.predict(input_vector))[0]
    svm_pred = le.inverse_transform(svm.predict(input_vector))[0]
    
    # Confidence from RF
    rf_proba = rf.predict_proba(input_vector)[0]
    rf_confidence = round(max(rf_proba) * 100, 2)
    
    # Top 3 predictions with confidence
    top3_idx = np.argsort(rf_proba)[-3:][::-1]
    top3 = [(le.inverse_transform([i])[0], 
             round(rf_proba[i]*100, 2)) for i in top3_idx]
    
    # Severity score
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    severity_df = pd.read_csv(os.path.join(base_path, 'data/symptom_severity.csv'))
    severity_map = dict(zip(
        severity_df['Symptom'].str.lower().str.replace(' ','_'),
        severity_df['weight']
    ))
    severity_score = sum(
        severity_map.get(s.lower().replace(' ','_'), 0)
        for s in symptoms_list
    )
    
    # Get description
    desc_df = pd.read_csv(os.path.join(base_path, 'data/symptom_Description.csv'))
    desc_row = desc_df[desc_df['Disease'] == rf_pred]
    description = desc_row['Description'].values[0] \
                  if len(desc_row) > 0 else "No description available."
    
    # Get precautions
    prec_df = pd.read_csv(os.path.join(base_path, 'data/symptom_precaution.csv'))
    prec_row = prec_df[prec_df['Disease'] == rf_pred]
    precautions = []
    if len(prec_row) > 0:
        for i in range(1, 5):
            col_name = f'Precaution_{i}'
            if col_name in prec_row.columns:
                p = prec_row[col_name].values[0]
                if pd.notna(p):
                    precautions.append(p)
    
    # Risk level based on severity score
    if severity_score >= 13:
        risk = "HIGH ⚠️"
        risk_color = "danger"
    elif severity_score >= 7:
        risk = "MODERATE 🔶"
        risk_color = "warning"
    else:
        risk = "LOW 🟢"
        risk_color = "success"
    
    return {
        'primary_prediction': rf_pred,
        'confidence': rf_confidence,
        'top3_predictions': top3,
        'rf_prediction': rf_pred,
        'nb_prediction': nb_pred,
        'svm_prediction': svm_pred,
        'description': description,
        'precautions': precautions,
        'severity_score': severity_score,
        'risk_level': risk,
        'risk_color': risk_color,
        'symptoms_entered': symptoms_list,
        'model_agreement': sum([
            rf_pred == nb_pred,
            rf_pred == svm_pred,
            nb_pred == svm_pred
        ])
    }
