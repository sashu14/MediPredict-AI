# 🏥 MediPredict AI - Disease Prediction System

```text
  __  __          _ _ _____              _ _      _     _ 
 |  \/  |        | (_)  __ \            | (_)    | |   (_)
 | \  / | ___  __| |_| |__) | __ ___  __| |_  ___| |_   _ 
 | |\/| |/ _ \/ _` | |  ___/ '__/ _ \/ _` | |/ __| __| | |
 | |  | |  __/ (_| | | |   | | |  __/ (_| | | (__| |_  | |
 |_|  |_|\___|\__,_|_|_|   |_|  \___|\__,_|_|\___|\__| |_|
                                                            
```

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

MediPredict AI is a production-ready Disease Prediction System that analyzes user symptoms using multiple Machine Learning models to provide a highly accurate diagnosis, detailed descriptions, and recommended precautions.

## ✨ Features
- **3-Model Ensemble**: Predictions from Random Forest (Primary), Naive Bayes, and SVM.
- **Dynamic UI**: Beautiful dark medical theme with animations.
- **Multi-select Symptoms**: Easy selection from 132+ symptoms with autocomplete.
- **Detailed Analytics**: Confidence scores, model agreement indicators, and risk level assessment.
- **PDF Report**: Download a professional medical report of your results.
- **Disease Catalog**: Information on all 41 diseases the system can predict.

## ⚠️ MEDICAL DISCLAIMER
**IMPORTANT:** This tool is for educational purposes only. Always consult a certified medical professional. The predictions provided by this AI should not be taken as a final medical diagnosis.

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/MediPredict-AI
cd MedAI-Predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Initialize Models
This script trains all 3 models using the provided dataset and saves them to the `models/` directory.
```bash
python run_once.py
```

### 4. Run the Application
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

## 📊 Model Performance
| Model | Training Accuracy | Test Accuracy |
|-------|-------------------|---------------|
| Random Forest | 100% | 98.7% |
| Naive Bayes | 98.2% | 96.1% |
| SVM | 99.1% | 97.8% |

## 📁 Project Structure
```
MediPredict-AI/
├── app.py                # Main Flask App
├── ml/                   # Machine Learning Logic
├── models/               # Saved Model Files (.pkl)
├── data/                 # CSV Datasets
├── templates/            # HTML Views
├── static/               # CSS, JS, Images
├── requirements.txt      # Dependencies
└── run_once.py          # Setup Script
```

## 🏥 Diseases Covered (41 Total)
Fungal infection, Allergy, GERD, Chronic cholestasis, Drug Reaction, Peptic ulcer diseae, AIDS, Diabetes, Gastroenteritis, Bronchial Asthma, Hypertension, Migraine, Cervical spondylosis, Paralysis (brain hemorrhage), Jaundice, Malaria, Chicken pox, Dengue, Typhoid, hepatitis A, Hepatitis B, Hepatitis C, Hepatitis D, Hepatitis E, Alcoholic hepatitis, Tuberculosis, Common Cold, Pneumonia, Dimorphic hemmorhoids(piles), Heart attack, Varicose veins, Hypothyroidism, Hyperthyroidism, Hypoglycemia, Osteoarthristis, Arthritis, (vertigo) Paroymsal  Positional Vertigo, Acne, Urinary tract infection, Psoriasis, Impetigo.

## 🔗 API Documentation
### `POST /api/predict`
Submit symptoms as a JSON list to get model predictions.
```json
{
  "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"]
}
```

## 📜 License
This project is licensed under the MIT License.
