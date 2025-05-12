import os
import subprocess

if not os.path.exists('../data/processed/filtered_output.csv'):
    print("Running preprocessing.py to prepare dataset...")
    subprocess.run(['python', 'src/preprocessing.py'])

if not os.path.exists('../models/trained_model.h5'):
    print("Running model.py to train model...")
    subprocess.run(['python', 'src/model.py'])

print("Launching testing interface...")
subprocess.run(['python', 'src/testing.py'])

