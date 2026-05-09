# Stress-ML: Wearable-Based Stress Detection Platform

## 📌 Project Overview
Stress-ML is a production-ready Machine Learning application designed to detect psychological stress using multimodal data from wearable sensors. This project is the practical implementation of the research paper: **"A Unified Multi-Layered Framework for Continuous Mental Stress Quantification using Multimodal Wearable Sensors"**.

**Author:** Antra Das  
**Field:** Artificial Intelligence and Machine Learning  

## 🚀 Features
- **Real-Time Prediction:** Input physiological data (EDA, HR, Temp, ACC) to get instant stress classification.
- **Multimodal Fusion:** Implements logic to fuse data from multiple sensors for higher accuracy.
- **Interactive Dashboards:** Visualize model performance and physiological metrics using Plotly.
- **Production Ready:** Optimized for deployment on Streamlit Cloud with a layered architecture.

## 🛠️ Technical Stack
- **Language:** Python 3.9+
- **Framework:** Streamlit
- **ML Libraries:** Scikit-learn, Pandas, NumPy, Joblib
- **Visualization:** Plotly, Go

## 📂 Project Structure
```
stress-ml/
├── app.py                # Main Streamlit application
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
├── models/               # Serialized model and scaler artifacts
├── scripts/              # Training and utility scripts
│   └── train_model.py    # Mock training script for end-to-end functionality
└── assets/               # Static assets and images
```

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone <your-repo-link>
cd stress-ml
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Model Artifacts
Before running the app, you must generate the trained model and scaler:
```bash
python scripts/train_model.py
```

### 4. Run the Application
```bash
streamlit run app.py
```

## 🧪 Model Performance
Based on the WESAD dataset benchmarks:
- **Binary Accuracy:** 94.1%
- **3-State Accuracy:** 86.8%
- **F1-Score:** 0.94

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

---
*Created by Antra Das - 2026*
