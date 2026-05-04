# 📍 Indian Address Validation System (Version 2)

An advanced machine learning system for validating Indian addresses using XGBoost classifier with comprehensive geographical verification and rule-based validation.

## 🎯 Overview

This project provides a complete ML pipeline to validate Indian addresses by:
- **Generating** synthetic Indian address datasets (valid and invalid patterns)
- **Preprocessing** and normalizing address text
- **Engineering** features using TF-IDF vectors and numerical address characteristics
- **Training** an XGBoost classifier with 80/20 train-test split
- **Validating** predictions using a multi-rule engine:
  - Validates pincodes against a comprehensive India Post database
  - Checks if the mentioned state is valid for the provided pincode
  - Checks if the mentioned city or district matches the provided pincode
  - Flags non-India countries and rejects immediately
- **Suggesting** smart fixes when mismatches are detected
- **Deploying** via a Streamlit web interface for interactive predictions

## 📂 Project Structure

```
version-2/
│
├── data/                           # Data storage
│   ├── raw_addresses.csv           # Generated synthetic dataset
│   ├── indian_address_dataset.csv  # Processed dataset for training
│   ├── pincode_clean.csv           # Clean pincode-to-state mappings
│   ├── pincode_raw.csv             # Raw pincode data
│   └── state_list.py               # Cached state list from pincode data
│
├── src/                            # Source code modules
│   ├── generate_data.py            # Generate synthetic address data
│   ├── clean_normalized.py         # Data cleaning and normalization
│   ├── features.py                 # Feature extraction (TF-IDF + numerical)
│   ├── train.py                    # Model training pipeline
│   ├── predict.py                  # Prediction and validation functions
│   ├── pincode_lookup.py           # Pincode, city, and district validation
│   ├── state_list.py               # Indian state name list
│   └── __init__.py                 # Package initialization
│
├── models/                         # Trained model artifacts
│   ├── model.pkl                   # XGBoost classifier
│   └── vectorizer.pkl              # TF-IDF vectorizer
│
├── app.py                          # Streamlit web interface
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 🚀 Quick Start

### 1. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Training Data

```bash
python -m src.generate_data
```

### 4. Train Model

```bash
python -m src.train
```

### 5. Make Predictions

#### Programmatic Usage

```python
from src.predict import predict

# Valid address example
result = predict("435, Sector 47, Panipat, Haryana 132103")
print(result)

# Output contains prediction, score, corrections, rules fired, and suggestions if any.
```

### 6. Launch Web App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with an interactive address validation interface.

## 🔧 Features

### Data Generation (`generate_data.py`)
- Realistic Indian addresses with proper format
- Multiple invalid patterns:
  - Random gibberish text
  - Invalid pincode formats (4 digits, alphabetic characters)
  - Missing fields (city, state)
  - Special characters and symbols

### Feature Engineering (`features.py`)
Uses TF-IDF vectorization + numerical features:
- **TF-IDF**: 200-dimensional vectors with 1-2 word ngrams
- **Numerical Features**: Address length, word count, pincode presence, state/city keyword matching, special character count

### Model (`train.py`)
- **Algorithm**: XGBoost Classifier
- **Train/Test Split**: 80/20

### Advanced Rules Engine (`predict.py`)
- **`rule_state_pincode_mismatch`**: Verifies if the mentioned state in the address matches the provided pincode's official states.
- **`rule_city_pincode_mismatch`**: Cross-checks the city or district in the address text against the valid districts/cities of the provided pincode. Supports synonyms (e.g., *Gurugram* / *Gurgaon*, *Bangalore* / *Bengaluru*).
- **`rule_non_india_country`**: Flags any address explicitly mentioning another country. Rejects it immediately with a score of `0.0`.

### Smart Fix & Suggestions
When there is a mismatch between the pincode and the provided state/city/district, the system provides automatic **suggestions** via the returned output:
- **`corrected_city_and_state`**: Generates a suggested address string with the correct city and state matching the given pincode.
- **`corrected_pincode`**: Generates a suggested address string with the correct pincode matching the given city and state.

## 📋 Requirements

- Python 3.7+
- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: TF-IDF vectorization, model evaluation
- xgboost: Gradient boosting classifier
- streamlit: Web interface
- joblib: Model serialization

See `requirements.txt` for versions.

## 📱 Web Interface (Streamlit)

The `app.py` provides:
- **Single Address Validation**: Enter address, get instant prediction
- **Confidence Score & Reason**: Visual indicators of model confidence and rule evaluation
- **Smart Fix / Recommendations**: Displays automatic address suggestions in case of mismatches

## 🔐 Data Privacy

- No personal data is stored
- Synthetic data for training/testing only
- All processing local and offline

## 📝 License

MIT License - Feel free to use and modify for your projects.
