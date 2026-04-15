# 📱 Telecom Customer Churn Prediction — End-to-End ML Project

---

## 🤔 What Problem Are We Solving?

Imagine you run a telecom company (like Airtel or Jio).  
Every month, some customers **cancel their subscription and leave** — this is called **CHURN**.

Losing customers = Losing money.

If you could **predict in advance** which customers are likely to leave, you could:
- Offer them a discount 🎁
- Call them proactively 📞
- Give them a free upgrade 🚀

This ML project builds a **brain** (a machine learning model) that looks at a customer's
usage patterns and tells you:  
> _"This customer has an 80% chance of leaving next month."_

---

## 🗂️ Project Folder Structure (What Each Folder Does)

```
Telecom-Churn-ML/
│
├── 📄 README.md              ← You are here! Project guide.
├── 📄 requirements.txt       ← List of Python libraries to install.
├── 📄 train.py               ← ONE script to train the model. Run this first!
│
├── 📁 config/
│   └── config.yaml           ← Central settings file (like a control panel).
│
├── 📁 data/
│   └── raw/
│       └── telecom_churn.csv ← The original dataset (7043 customers).
│   └── processed/            ← Cleaned data gets saved here (auto-created).
│
├── 📁 src/                   ← The "engine room" — all ML code lives here.
│   ├── __init__.py           ← Tells Python "this folder is a package".
│   ├── utils.py              ← Helper tools (logging, saving files, charts).
│   ├── preprocessing.py      ← Cleans and prepares raw data for the model.
│   ├── feature_engineering.py← Creates smarter features from existing ones.
│   └── model_training.py     ← Trains, evaluates, and saves the ML model.
│
├── 📁 app/                   ← The web application to use the trained model.
│   ├── app.py                ← Flask web server.
│   └── templates/
│       └── index.html        ← The webpage users see in their browser.
│
├── 📁 models/                ← Saved trained model gets stored here.
├── 📁 logs/                  ← Log files for debugging.
└── 📁 notebooks/             ← Jupyter notebooks for exploration & learning.
```

---

## 🔄 How the Whole System Works (Simple Explanation)

```
RAW DATA (CSV)
     ↓
  STEP 1: PREPROCESSING
  "Clean the data — fix missing values, convert Yes/No to 1/0"
     ↓
  STEP 2: FEATURE ENGINEERING
  "Create smarter clues — e.g. monthly bill ÷ months = avg spend per month"
     ↓
  STEP 3: MODEL TRAINING
  "Teach 3 algorithms to recognize patterns of churning customers"
     ↓
  STEP 4: EVALUATION
  "Test how accurate the model is on data it has never seen before"
     ↓
  STEP 5: SAVE MODEL
  "Save the trained brain to a .pkl file for later use"
     ↓
  STEP 6: FLASK WEB APP
  "Load the saved model and serve predictions through a website"
```

---

## ⚡ Quick Start (Run in 3 Steps)

### Step 1 — Install all required libraries
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model
```bash
python train.py
```
This will print accuracy, F1-score, etc. and save `models/pipeline.pkl`.

### Step 3 — Start the web app
```bash
python app/app.py
```
Open your browser at: **http://localhost:5000**

---

## 📊 The Dataset

**File:** `data/raw/telecom_churn.csv`  
**Rows:** 7,043 customers  
**Columns:** 21 (20 features + 1 target)

| Column | Type | What it means |
|---|---|---|
| `customerID` | Text | Unique ID — not useful for ML, we drop it |
| `gender` | Text | Male / Female |
| `SeniorCitizen` | 0 or 1 | Is the customer 65+? |
| `Partner` | Yes/No | Do they have a spouse/partner? |
| `Dependents` | Yes/No | Do they support children/family? |
| `tenure` | Number | How many months they've been a customer |
| `PhoneService` | Yes/No | Do they have phone service? |
| `MultipleLines` | Text | Do they have multiple phone lines? |
| `InternetService` | Text | DSL / Fiber optic / No |
| `OnlineSecurity` | Yes/No | Subscribed to online security addon? |
| `OnlineBackup` | Yes/No | Subscribed to backup addon? |
| `DeviceProtection` | Yes/No | Subscribed to device protection? |
| `TechSupport` | Yes/No | Subscribed to tech support? |
| `StreamingTV` | Yes/No | Streams TV via the service? |
| `StreamingMovies` | Yes/No | Streams movies via the service? |
| `Contract` | Text | Month-to-month / 1-year / 2-year |
| `PaperlessBilling` | Yes/No | Gets paperless bills? |
| `PaymentMethod` | Text | How they pay |
| `MonthlyCharges` | Number | Monthly bill amount in $ |
| `TotalCharges` | Number | Total amount paid so far in $ |
| `Churn` | Yes/No | **TARGET: Did they leave?** |

**Key Insight:** ~26.5% customers churned. This means data is **imbalanced** 
(more "No" than "Yes") — our model handles this carefully.

---

## 🧠 Machine Learning Models Used

We train **3 models** and pick the best one:

| Model | What it's like in real life | Strength |
|---|---|---|
| **Logistic Regression** | A simple scoring formula (like a credit score) | Fast, easy to explain |
| **Random Forest** | A committee of 100 decision trees voting | Handles messy data well |
| **XGBoost** | A turbo-charged version of Random Forest | Usually the most accurate |

---

## 📈 Performance Targets

| Metric | What it means | Target |
|---|---|---|
| **Accuracy** | Out of all predictions, how many were right? | > 80% |
| **Precision** | Of all "will churn" predictions, how many actually churned? | > 75% |
| **Recall** | Of all actual churners, how many did we catch? | > 70% |
| **F1-Score** | Balance between Precision and Recall | > 0.75 |
| **ROC-AUC** | Overall model quality score | > 0.85 |

---

## 🌐 Web App Endpoints

Once `app/app.py` is running:

| URL | Method | What it does |
|---|---|---|
| `http://localhost:5000/` | GET | Opens the prediction webpage |
| `http://localhost:5000/health` | GET | Checks if server is running |
| `http://localhost:5000/api/predict` | POST | Predict for 1 customer (JSON) |
| `http://localhost:5000/api/batch-predict` | POST | Predict for many customers |
| `http://localhost:5000/api/feature-info` | GET | See what features the model uses |

---

## 🛠️ Tech Stack

| Tool | Why we use it |
|---|---|
| **Python** | The language of data science |
| **Pandas** | Like Excel in Python — for tables/dataframes |
| **NumPy** | Super-fast math operations |
| **Scikit-learn** | Ready-made ML algorithms + preprocessing tools |
| **XGBoost** | Best algorithm for tabular (table) data |
| **Matplotlib / Seaborn** | Drawing charts and graphs |
| **Flask** | Lightweight web framework to serve predictions |
| **PyYAML** | Read the config.yaml settings file |
| **Pickle** | Save/load the trained model to/from disk |

**Total cost: $0 — all open source!**
