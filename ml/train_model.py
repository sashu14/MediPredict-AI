import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

def train_all_models():
    # Ensure models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')

    # Load data
    df = pd.read_csv('data/Training.csv')
    df = df.dropna(axis=1)
    
    # Encode target
    le = LabelEncoder()
    df['prognosis'] = le.fit_transform(df['prognosis'])
    
    X = df.drop('prognosis', axis=1)
    y = df['prognosis']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # 1. Random Forest (PRIMARY MODEL)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    
    # 2. Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_acc = accuracy_score(y_test, nb.predict(X_test))
    print(f"Naive Bayes Accuracy: {nb_acc:.4f}")
    
    # 3. SVM
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))
    print(f"SVM Accuracy: {svm_acc:.4f}")
    
    # Cross validation for RF
    cv_scores = cross_val_score(rf, X, y, cv=2) # Using cv=2 for small placeholder dataset
    print(f"RF Cross-Val Score: {cv_scores.mean():.4f}")
    
    # Save all models
    joblib.dump(rf, 'models/random_forest_model.pkl')
    joblib.dump(nb, 'models/naive_bayes_model.pkl')
    joblib.dump(svm, 'models/svm_model.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    # Save symptom columns list
    symptom_cols = list(X.columns)
    joblib.dump(symptom_cols, 'models/symptom_columns.pkl')
    
    print("✅ All models saved successfully!")
    return rf_acc, nb_acc, svm_acc

if __name__ == "__main__":
    train_all_models()
