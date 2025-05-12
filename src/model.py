import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from config import FEATURE_COLUMNS, LABEL_COLUMN

# Step 1: Data Preprocessing
def preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    X = df[FEATURE_COLUMNS].values
    y = (df[LABEL_COLUMN].str.upper() != 'BENIGN').astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 2, 2, 1)

    return X_reshaped, y

# Step 2: Model Architecture
def create_model():
    model = Sequential([
        Conv2D(128, (2, 2), activation='relu', input_shape=(2, 2, 1), padding='same'),
        Conv2D(64, (2, 2), activation='relu', padding='same'),
        Conv2D(64, (2, 2), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(1, 1)),
        Conv2D(32, (2, 2), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(1, 1)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Step 3: Train
def train_model(X, y, model, epochs=10, batch_size=32):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        verbose=1)
    return model, history, X_test, y_test

# Step 4: Evaluate
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test loss: {loss:.4f}")

# Main
if __name__ == "__main__":
    csv_file = "data/processed/filtered_output.csv"
    X, y = preprocess_data(csv_file)
    model = create_model()
    trained_model, history, X_test, y_test = train_model(X, y, model)
    trained_model.save('models/trained_model.h5')
    evaluate_model(trained_model, X_test, y_test)

