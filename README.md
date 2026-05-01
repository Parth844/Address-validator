# 📍 Indian Address Validation System (Version 2)

An advanced machine learning system for validating Indian addresses using XGBoost classifier with pincode-to-state validation.

## 🎯 Overview

This project provides a complete ML pipeline to validate Indian addresses by:
- **Generating** synthetic Indian address datasets (valid and invalid patterns)
- **Preprocessing** and normalizing address text
- **Engineering** features using TF-IDF vectors and numerical address characteristics
- **Training** an XGBoost classifier with 80/20 train-test split
- **Validating** predictions against a pincode database
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
│   ├── pincode_lookup.py           # Pincode-to-state validation
│   ├── state_list.py               # Indian state name list
│   └── __init__.py                 # Package initialization
│
├── models/                         # Trained model artifacts
│   ├── model.pkl                   # XGBoost classifier
│   └── vectorizer.pkl              # TF-IDF vectorizer
│
├── app.py                          # Streamlit web interface
├── requirements.txt                # Python dependencies
└── README.md                       # This file
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

This creates `data/raw_addresses.csv` with 10,000 synthetic addresses:
- 50% valid addresses (realistic Indian format)
- 50% invalid addresses (random text, wrong pincodes, malformed)

### 4. Train Model

```bash
python -m src.train
```

This will:
- Load and clean the dataset
- Extract TF-IDF features (200 features, 1-2 ngrams)
- Extract numerical address features
- Train XGBoost classifier
- Save model and vectorizer to `models/`
- Print accuracy, precision, recall, F1 score

### 5. Make Predictions

#### Command Line

```bash
python -m src.predict
```

#### Programmatic Usage

```python
from src.predict import predict

# Single address
result = predict("123, MG Road, Bangalore, Karnataka 560001")
print(result)
# Output: {'prediction': 1, 'confidence': 0.95, 'valid': True, ...}
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
- **Address Length**: Character count
- **Word Count**: Number of words
- **Pincode Presence**: Has valid 6-digit pincode
- **Pincode Format**: Digits-only check
- **State Detection**: Contains known Indian state name
- **City Detection**: Contains known city names
- **Special Characters**: Count of unusual characters

### Model (`train.py`)
- **Algorithm**: XGBoost Classifier
- **Features**: 200 TF-IDF + 8 numerical = 208 total features
- **Train/Test Split**: 80/20
- **Hyperparameters**: Default XGBoost settings

### Prediction Validation (`predict.py`)
- **ML Prediction**: XGBoost model output
- **Pincode Validation**: Cross-reference with pincode-to-state database
- **State Verification**: Matches extracted state against pincode prefix
- **Confidence Score**: Probability from model
- **Detailed Analysis**: Feature breakdown and validation flags

### Pincode Lookup (`pincode_lookup.py`)
- Maps first 2 digits of pincode to Indian states
- Validates if address state matches pincode state
- Includes support for:
  - All 28 states and 8 union territories
  - Army Post Offices
  - Special postal codes

## 📊 Expected Performance

| Metric | Range |
|--------|-------|
| Accuracy | 90-95% |
| Precision | 90-95% |
| Recall | 90-95% |
| F1 Score | 90-95% |

## 📋 Requirements

- Python 3.7+
- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: TF-IDF vectorization, model evaluation
- xgboost: Gradient boosting classifier
- streamlit: Web interface
- joblib: Model serialization

See `requirements.txt` for versions.

## 🛠️ API Reference

### Prediction Functions

```python
from src.predict import predict

# Main prediction function
result = predict(address_string)

# Returns dictionary:
# {
#     'prediction': 0 or 1,
#     'confidence': float (0-1),
#     'valid': bool,
#     'pincode': str or None,
#     'state_from_text': str or None,
#     'state_from_pincode': str or None,
#     'pincode_valid': bool,
#     'features': dict
# }
```

### Feature Extraction

```python
from src.features import extract_features

features = extract_features("123, MG Road, Bangalore 560001")
# Returns: numpy array of numerical features
```

### Data Generation

```python
from src.generate_data import generate_valid, generate_invalid

valid_addr = generate_valid()      # "123, MG Road, Mumbai..."
invalid_addr = generate_invalid()  # "asdfgh jkl"
```

## 📱 Web Interface (Streamlit)

The `app.py` provides:
- **Single Address Validation**: Enter address, get instant prediction
- **Confidence Score**: Visual indicator of model confidence
- **Validation Details**: Breakdown of features and pincode matching
- **Error Handling**: User-friendly error messages
- **Dark Theme**: Professional dark mode interface

## ⚙️ Configuration

### TF-IDF Parameters (in `train.py`)
```python
vectorizer = TfidfVectorizer(
    max_features=200,      # Number of features
    ngram_range=(1,2)      # Unigrams and bigrams
)
```

### Pincode State Mappings (in `predict.py`)
```python
PINCODE_STATE_PREFIXES = {
    "11": "Delhi",
    "40": "Maharashtra",
    "56": "Karnataka",
    # ... more mappings
}
```

## 🐛 Troubleshooting

### Model Not Found
```
FileNotFoundError: models/model.pkl
```
**Solution**: Run `python -m src.train` to train and save the model.

### Import Errors
```
ModuleNotFoundError: No module named 'xgboost'
```
**Solution**: Activate venv and run `pip install -r requirements.txt`.

### Data File Not Found
```
FileNotFoundError: data/indian_address_dataset.csv
```
**Solution**: Run `python -m src.generate_data` to create training data.

### Streamlit Port Already in Use
```
Port 8501 already in use
```
**Solution**: `streamlit run app.py --server.port 8502`

## 📈 Pipeline Workflow

```
1. generate_data.py
   └─→ data/raw_addresses.csv

2. clean_normalized.py
   └─→ data/indian_address_dataset.csv

3. train.py
   ├─→ models/model.pkl
   └─→ models/vectorizer.pkl

4. predict.py + app.py
   └─→ Address Validation Predictions
```

## 🔐 Data Privacy

- No personal data is stored
- Synthetic data for training/testing only
- No API calls to external services
- All processing local and offline

## 📝 License

MIT License - Feel free to use and modify for your projects.

## 👨‍💻 Author

Built as an advanced demonstration of ML-based address validation for Indian addresses with geographical verification.

---

**Happy validating! 🎉**
