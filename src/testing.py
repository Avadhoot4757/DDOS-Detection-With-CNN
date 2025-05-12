import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from config import FEATURE_COLUMNS, LABEL_COLUMN
import sys

def preprocess_entry(csv_file, features, scaler, specific_label=None):
    df = pd.read_csv(csv_file)

    if specific_label:
        filtered_df = df[df[LABEL_COLUMN].str.upper() == specific_label.upper()]
        if filtered_df.empty:
            print(f"No entries found with label '{specific_label}'.")
            sys.exit(1)
        entry = filtered_df.sample(n=1)
    else:
        entry = df.sample(n=1)
        
    print("Selected Entry:\n", entry)

    X = entry[features].values
    X_scaled = scaler.transform(X)
    X_reshaped = X_scaled.reshape(1, 2, 2, 1)
    return X_reshaped, entry

if __name__ == "__main__":
    csv_file = "data/processed/filtered_output.csv"
    model = load_model('models/trained_model.h5')

    scaler = StandardScaler()
    df_full = pd.read_csv(csv_file)
    X_full = df_full[FEATURE_COLUMNS].values
    scaler.fit(X_full)

    check_label = input("Do you want to check a specific label? Type 'BENIGN' or press Enter for a random entry: ").strip()

    if check_label:
        X_test, selected_entry = preprocess_entry(csv_file, FEATURE_COLUMNS, scaler, specific_label=check_label)
    else:
        X_test, selected_entry = preprocess_entry(csv_file, FEATURE_COLUMNS, scaler)

    prediction = model.predict(X_test)
    print(f"Prediction (Probability of Attack): {prediction[0][0]:.4f}")

