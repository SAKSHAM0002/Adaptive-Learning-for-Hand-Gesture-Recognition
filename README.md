# Adaptive-Learning-for-Hand-Gesture-Recognition
This project explores an Active Learning approach using Empatica E4 accelerometer data and two additional datasets. It reduces annotation efforts while maintaining or surpassing the accuracy of a Personalized Model. Includes data preprocessing, model training, and evaluation scripts for benchmarking model performance across multiple datasets.


# Personalised, Generalised, and Active Learning Models

## Overview
This repository contains the implementation of:
- **Personalised Models**
- **Generalised Models**
- **Active Learning Models**

It includes code for training and evaluating these models on three publicly available datasets and one custom dataset collected by the author.

## Repository Structure
```
📂 Repository Root
├── 📄 Personalised_Model.py       # Code for the personalised learning model
├── 📄 Generalised_Model.py        # Code for the generalised learning model
├── 📄 LSTM_Active_Learning.py     # LSTM-based active learning implementation
├── 📄 XGBoost_Active_Learning.py  # XGBoost-based active learning implementation
├── 📊 Final_results.xlsx          # Evaluation results of the models
└── 📄 README.md                   # This file
```

## Datasets
The models have been trained and evaluated on:
1. **Public Dataset 1**
2. **Public Dataset 2**
3. **Public Dataset 3**
4. **Custom Dataset** (collected by the author)

## Model Descriptions
### 1. **Personalised Model**
- Tailors predictions based on individual user data.
- Implemented in `Personalised_Model.py`.

### 2. **Generalised Model**
- Trains on diverse data to make predictions applicable across different users.
- Implemented in `Generalised_Model.py`.

### 3. **Active Learning Models**
- Uses selective sampling to improve model efficiency.
- Implemented using:
  - `LSTM_Active_Learning.py` (LSTM-based approach)
  - `XGBoost_Active_Learning.py` (XGBoost-based approach)

## Results
The performance of these models is documented in `Final_results.xlsx`, which contains evaluation metrics across all datasets.

## Usage
1. Install required dependencies:
   ```bash
   pip install -r requirements.txt  # (If you have a requirements file)
   ```
2. Run the models as needed:
   ```bash
   python Personalised_Model.py
   python Generalised_Model.py
   python LSTM_Active_Learning.py
   python XGBoost_Active_Learning.py
   ```


