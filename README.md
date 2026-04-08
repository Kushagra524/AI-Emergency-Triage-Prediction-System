# 🏥 AI Emergency Triage Prediction System

> **Kaggle Competition: Triagegist** — Predicting medical emergency urgency levels (1–5) from patient vitals and clinical inputs using LightGBM.

---

## 📌 Overview

The **AI Emergency Triage Prediction System** is a machine learning pipeline that predicts the **urgency level** of a medical emergency on a scale of:

| Level | Meaning |
|-------|---------|
| 1 | 🔴 Critical — Immediate life-threatening |
| 2 | 🟠 Emergent — High risk, urgent care needed |
| 3 | 🟡 Urgent — Stable but needs prompt attention |
| 4 | 🟢 Semi-Urgent — Non-critical, can wait |
| 5 | ⚪ Non-Urgent — Routine, no immediate danger |

The system takes patient vitals and medical inputs, runs them through a trained **LightGBM classifier**, and outputs the predicted triage class — mimicking real-world emergency department triage protocols.

---

## 📂 Project Structure

```
AI-Emergency-Triage-Prediction/
│
├── data/
│   ├── train.csv                  # Training dataset (800,000 × 40)
│   └── test.csv                   # Test dataset (200,000 × 40)
│
├── notebooks/
│   └── eda_and_cleaning.ipynb     # EDA & data preprocessing (VS Code)
│   └── model_training.ipynb       # LightGBM training (Google Colab)
│
├── models/
│   └── lightgbm_model_triage.txt      # Saved LightGBM model
│
├── app/
│   └── app.py                     # Streamlit UI
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Source | Kaggle Competition — *Triagegist* |
| Data Type | Patient records & clinical vitals |
| Training Set | 800,000 rows × 40 columns |
| Test Set | 200,000 rows × 40 columns |
| Target Variable | `triage` (classes: 1, 2, 3, 4, 5) |
| Task Type | Multi-class Classification |

> **Note:** The dataset is provided by the Kaggle competition. Download it from the [Triagegist competition page](https://www.kaggle.com/competitions/triagegeist/data) and place the files inside the `data/` folder.

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10.11 |
| ML Model | LightGBM |
| Data Handling | Pandas, NumPy |
| Visualization | Seaborn, Matplotlib |
| Model Saving | .txt format |
| UI | Streamlit |
| Training Environment | Google Colab |
| Development Environment | VS Code |

---

## 🤖 Model Details

- **Algorithm:** LightGBM (Light Gradient Boosting Machine)
- **Type:** Single multi-class classifier
- **Target:** `triage` — 5 urgency levels (1 to 5)
- **Key Challenge Handled:** Class imbalance across triage levels

LightGBM was chosen for its speed and efficiency on large tabular datasets, native support for categorical features, and strong performance on imbalanced classification problems.

---

## ⚙️ Workflow

```
Raw Data (Kaggle)
      │
      ▼
EDA & Data Cleaning ──── VS Code
      │  (feature inspection, null handling
      │   encoding)
      ▼
Model Training ──────── Google Colab
      │  (LightGBM, class imbalance handling)
      │  
      ▼
Model Saving ────────── txt format (.txt)
      │
      ▼
Streamlit UI ─────────── VS Code
      │  (input vitals → predict triage level)
      ▼
Deployment ──────────── Streamlit Cloud
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have **Python 3.8+** installed on your system.

### 1. Clone the Repository

```bash
git clone https://github.com/Kushagra524/AI-Emergency-Triage-Prediction.git
cd AI-Emergency-Triage-Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the competition dataset from Kaggle and place `train.csv` and `test.csv` inside the `data/` folder.

```bash
# Using Kaggle CLI (optional)
kaggle competitions download -c triagegist
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📋 Requirements

```
lightgbm
scikit-learn
numpy
pandas
streamlit
seaborn
matplotlib
joblib
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🌐 Live Demo

The application is deployed on **Streamlit Cloud**.

👉 [Launch the App](https://ai-emergency-triage-prediction-system-by-kushagra.streamlit.app/)

---

## ⚠️ Challenges

### Class Imbalance
The triage dataset is heavily skewed — critical cases (level 1) are far fewer than non-urgent ones (level 5), mirroring real-world emergency data. Techniques applied to mitigate this include:

- Adjusting `class_weight` in LightGBM
- Using stratified splits during validation
- Evaluating with classification report instead of raw accuracy

---

## 🔮 Future Improvements

- **Ensemble Models** — Combine LightGBM with XGBoost or CatBoost via stacking for better generalization
- **Feature Engineering** — Derive interaction features (e.g., shock index = heart rate / systolic BP) from raw vitals
- **SHAP Explainability** — Integrate SHAP values into the Streamlit UI so clinicians can understand *why* a triage level was predicted
- **SMOTE / Oversampling** — Apply synthetic minority oversampling during training to better handle class imbalance
- **Real-time Input Validation** — Add clinical range checks in the UI (e.g., flag a heart rate of 300 as invalid)
- **Multi-language Support** — Extend the Streamlit UI to support regional languages for broader accessibility
- **REST API** — Wrap the model in a FastAPI backend for integration with hospital management systems

---

## 📄 License

This project is for educational and competition purposes. Dataset usage is subject to Kaggle's [competition rules](https://www.kaggle.com/competitions/triagegeist/rules).

---

## 🙌 Acknowledgements

- [Kaggle](https://www.kaggle.com/) for hosting the Triagegist competition and providing the dataset
- [LightGBM](https://lightgbm.readthedocs.io/) by Microsoft for the blazing-fast gradient boosting framework
- [Streamlit](https://streamlit.io/) for making ML app deployment effortless

---

<p align="center">Made by Kushagra Srivastava</p
