# DDoS Detection using CNN

A deep learning-based DDoS detection system built using Convolutional Neural Networks (CNN). This project performs data preprocessing, trains a CNN model, and evaluates it on network traffic data to identify potential DDoS attacks.

## Features

- Preprocessing support for large datasets (tested on 7GB+ CSV files using pandas chunking)
- Automatically handles varying column names and standardizes them using `config.py`
- CNN architecture with 2D convolution layers, pooling, and fully connected layers
- Modular scripts: preprocessing, training (`model.py`), and evaluation (`testing.py`)
- `run.py` script automates full pipeline

## Getting Started

### 1. Clone the repository

```
git clone https://github.com/Avadhoot4757/DDOS-Detection-With-CNN.git
cd DDOS-Detection-With-CNN
```

### 2. Create a virtual environment (optional but recommended)

```
python3 -m venv myenv
source myenv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Add your dataset

Place your large dataset (CSV file). We have included a balanced sample dataset `sample_dataset.csv` (under 10MB) containing equal entries from benign and DDoS classes, useful for quick testing under data/raw directory in root.

### 5. Run the project pipeline

```
python3 run.py
```

This will:
- Preprocess the dataset
- Train the CNN model
- Save and evaluate the model
- Start the testing interface

## Resources

- Balanced Dataset: https://www.kaggle.com/datasets/devendra416/ddos-datasets
- Research Paper (Referenced): [A_Deep_CNN_Ensemble_Framework_for_Efficient_DDoS_Attack_Detection_in_Software_Defined_Networks.pdf](./A_Deep_CNN_Ensemble_Framework_for_Efficient_DDoS_Attack_Detection_in_Software_Defined_Networks.pdf)
