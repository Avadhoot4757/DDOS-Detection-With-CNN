import pandas as pd
import os
from config import FEATURE_COLUMNS, LABEL_COLUMN

# Create default column mapping structure
default_columns = {col: None for col in FEATURE_COLUMNS + [LABEL_COLUMN]}

print("Expected features:")
for col in default_columns:
    print(f" - {col}")

# Ask for dataset file
dataset_path = input("\nEnter the CSV filename (e.g., final.csv): ").strip()

# Check if file exists
if not os.path.exists(dataset_path):
    print(f"Error: File '{dataset_path}' not found.")
    exit(1)

# Load the first chunk to inspect columns
first_chunk = pd.read_csv(dataset_path, nrows=1)
first_chunk.columns = first_chunk.columns.str.strip()
available_columns = list(first_chunk.columns)

# Ask user for mapping if necessary
print("\nChecking for column names in dataset...\n")
for default_name in default_columns:
    if default_name in available_columns:
        default_columns[default_name] = default_name
    else:
        print(f"Column '{default_name}' not found.")
        print("Available columns:")
        for i, col in enumerate(available_columns):
            print(f"  [{i}] {col}")
        while True:
            user_input = input(f"Enter corresponding column for '{default_name}': ").strip()
            if user_input in available_columns:
                default_columns[default_name] = user_input
                break
            else:
                print("Invalid column name. Try again.")

# Process dataset in chunks
chunk_size = 10000
chunks = pd.read_csv(dataset_path, chunksize=chunk_size)
filtered_chunks = []

for chunk in chunks:
    chunk.columns = chunk.columns.str.strip()
    selected_chunk = chunk[[default_columns[col] for col in default_columns]]
    selected_chunk.columns = list(default_columns.keys())  # Standardize names
    filtered_chunks.append(selected_chunk)

# Concatenate and save
final_df = pd.concat(filtered_chunks)
final_df.to_csv('data/processed/filtered_output.csv', index=False)

print("\n✅ Filtered data saved to 'filtered_output.csv' with standardized column names.")

