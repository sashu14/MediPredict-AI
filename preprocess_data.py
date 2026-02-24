import pandas as pd
import numpy as np
import os

def preprocess():
    print("🔄 Preprocessing dataset...")
    
    # Load original dataset
    df = pd.read_csv('data/dataset.csv')
    
    # Get all unique symptoms
    symptoms = []
    for col in df.columns[1:]:  # Skip 'Disease'
        symptoms.extend(df[col].dropna().unique().tolist())
    
    # Clean symptoms: strip whitespace and replace spaces with underscores
    symptoms = sorted(list(set([s.strip().replace(' ', '_') for s in symptoms if pd.notna(s)])))
    
    # Create empty binary matrix
    processed_data = pd.DataFrame(0, index=np.arange(len(df)), columns=symptoms)
    
    # Fill matrix
    for i in range(len(df)):
        row_symptoms = df.iloc[i, 1:].dropna().tolist()
        for s in row_symptoms:
            s_clean = s.strip().replace(' ', '_')
            if s_clean in processed_data.columns:
                processed_data.at[i, s_clean] = 1
                
    # Add prognosis column
    processed_data['prognosis'] = df['Disease']
    
    # Save to data/Training.csv
    processed_data.to_csv('data/Training.csv', index=False)
    print(f"✅ Preprocessing complete! Created 'data/Training.csv' with {len(symptoms)} symptoms and {len(df)} rows.")

if __name__ == "__main__":
    preprocess()
