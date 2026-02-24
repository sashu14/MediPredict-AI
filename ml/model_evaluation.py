import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

def evaluate_models():
    # Ensure image directory exists
    if not os.path.exists('static/img/evaluation'):
        os.makedirs('static/img/evaluation')

    # Load data
    df_test = pd.read_csv('data/Training.csv') # Using Training for placeholder testing
    df_test = df_test.dropna(axis=1)
    
    # Load models
    rf = joblib.load('models/random_forest_model.pkl')
    nb = joblib.load('models/naive_bayes_model.pkl')
    svm = joblib.load('models/svm_model.pkl')
    le = joblib.load('models/label_encoder.pkl')
    cols = joblib.load('models/symptom_columns.pkl')
    
    X_test = df_test[cols]
    y_test = le.transform(df_test['prognosis'])
    
    models = {
        'Random Forest': rf,
        'Naive Bayes': nb,
        'SVM': svm
    }
    
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'static/img/evaluation/{name.lower().replace(" ", "_")}_cm.png')
        plt.close()

        # Print report
        print(f"\n--- {name} Classification Report ---")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Summary Table
    print("\nModel Summary:")
    print("-" * 38)
    print(f"{'Model':<20} | {'Accuracy':<10}")
    print("-" * 38)
    for name, acc in results.items():
        print(f"{name:<20} | {acc:.4%}")
    print("-" * 38)

if __name__ == "__main__":
    evaluate_models()
