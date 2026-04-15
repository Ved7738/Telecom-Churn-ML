"""
src/utils.py  —  Shared Helper Utilities
=========================================

WHAT IS THIS FILE?
    This is the "toolbox" of the project. It contains small helper functions
    that are used by multiple other files. Instead of copy-pasting the same
    code everywhere, we write it once here and import it wherever needed.

ANALOGY:
    Think of it like a kitchen drawer with scissors, tape, and a pen.
    Every room in the house uses them, so you keep them in one central place.

CONTENTS:
    1. load_config()       → Reads config.yaml into a Python dictionary
    2. get_logger()        → Sets up a logging system (like a diary for the app)
    3. ensure_dirs()       → Creates folders if they don't exist
    4. save_object()       → Saves any Python object to disk using pickle
    5. load_object()       → Loads a pickled object back from disk
    6. load_data()         → Loads a CSV file into a pandas DataFrame
    7. basic_eda()         → Prints a quick summary of the dataset
    8. plot_*()            → Helper functions for common chart types
"""

import os
import yaml
import logging
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# 1. CONFIGURATION LOADER
# =============================================================================

def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Read the YAML config file and return it as a Python dictionary.

    WHY:
        Instead of hard-coding settings like test_size=0.2 in every file,
        we read them from config.yaml. Changing one file updates everything.

    EXAMPLE:
        cfg = load_config()
        print(cfg["preprocessing"]["test_size"])  # → 0.2
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)   # safe_load = read YAML safely (no code execution)
    return config


# =============================================================================
# 2. LOGGER
# =============================================================================

def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Create and return a logger that writes messages to BOTH:
      - The terminal (console) — so you see what's happening live
      - A log file in the logs/ folder — so you can review it later

    WHY LOGGING INSTEAD OF PRINT?
        print() statements disappear once the program finishes.
        A logger saves every message to a file WITH timestamps,
        making it easy to debug issues that happened yesterday.

    ANALOGY:
        It's like a black box recorder on an airplane.
        You might not need it most of the time, but when something goes
        wrong you're very glad it was recording everything.

    EXAMPLE:
        logger = get_logger("preprocessing")
        logger.info("Data loaded successfully")    # INFO  = normal message
        logger.warning("Missing values found")     # WARNING = something to note
        logger.error("File not found!")            # ERROR = something broke
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    # Prevent adding duplicate handlers if the function is called twice
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # Capture ALL levels of messages

    # Format: "2025-01-15 10:30:00  INFO      preprocessing  Data loaded"
    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler 1: Print to terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)      # Only INFO and above in terminal
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler 2: Write to log file (new file each day)
    log_filename = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)        # Everything goes to the file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# =============================================================================
# 3. DIRECTORY HELPER
# =============================================================================

def ensure_dirs(*dirs: str):
    """
    Create one or more folders if they don't already exist.

    WHY:
        When saving the trained model to models/pipeline.pkl, the models/
        folder must exist first. This function handles that automatically.

    EXAMPLE:
        ensure_dirs("models", "logs", "data/processed")
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)  # exist_ok=True = no error if already exists


# =============================================================================
# 4 & 5. MODEL PERSISTENCE (SAVE / LOAD)
# =============================================================================

def save_object(obj, filepath: str):
    """
    Save any Python object (model, preprocessor, list, dict…) to a .pkl file.

    WHAT IS PICKLE?
        Pickle is Python's way of converting any object into a stream of bytes
        that can be written to a file. Later, you can load it back and get the
        exact same object — like a photocopy of the object's memory.

    ANALOGY:
        Like freezing food. You prepare a meal (train a model), freeze it
        (pickle it), and later thaw it (load it) — same meal, no re-cooking.

    EXAMPLE:
        save_object(trained_model, "models/best_model.pkl")
    """
    ensure_dirs(os.path.dirname(filepath))
    with open(filepath, "wb") as f:   # "wb" = write binary
        pickle.dump(obj, f)
    print(f"  ✅ Saved  →  {filepath}")


def load_object(filepath: str):
    """
    Load a previously pickled object back from disk.

    EXAMPLE:
        model = load_object("models/best_model.pkl")
        predictions = model.predict(X_new)
    """
    with open(filepath, "rb") as f:   # "rb" = read binary
        obj = pickle.load(f)
    print(f"  ✅ Loaded ←  {filepath}")
    return obj


# =============================================================================
# 6. DATA LOADER
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Read a CSV file and return it as a pandas DataFrame.

    WHAT IS A DATAFRAME?
        A DataFrame is a table — just like an Excel spreadsheet — but inside Python.
        Each row = one customer. Each column = one feature (age, charges, etc.).

    EXAMPLE:
        df = load_data("data/raw/telecom_churn.csv")
        print(df.head())   # Show first 5 rows
    """
    df = pd.read_csv(filepath)
    print(f"\n  📂 Loaded: {os.path.basename(filepath)}")
    print(f"     Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
    return df


# =============================================================================
# 7. EXPLORATORY DATA ANALYSIS (EDA) SUMMARY
# =============================================================================

def basic_eda(df: pd.DataFrame):
    """
    Print a quick overview of the dataset.

    WHY DO EDA?
        Before feeding data to a model, you need to UNDERSTAND it:
        - Are there missing values? (empty cells the model can't handle)
        - Are there duplicate rows? (could bias the model)
        - What's the class balance? (if 99% "No Churn", model could just guess "No" always)

    This function answers all those questions in one call.

    EXAMPLE:
        basic_eda(df)
        # Prints: shape, dtypes, missing values, duplicates, target distribution
    """
    divider = "=" * 60
    print(f"\n{divider}")
    print("  DATASET OVERVIEW")
    print(divider)

    print(f"\n  Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Duplicates   : {df.duplicated().sum()}")

    # Missing values (only show columns that have them)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\n  ⚠️  Missing Values:\n{missing.to_string()}")
    else:
        print("\n  ✅ No missing values found")

    # Data types
    print(f"\n  Data Types:\n{df.dtypes.to_string()}")

    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n  Numeric Statistics:\n{df[numeric_cols].describe().round(2).to_string()}")

    # Target variable distribution (look for 'Churn', 'churn', 'target', etc.)
    for candidate in ["Churn", "churn", "target", "label"]:
        if candidate in df.columns:
            vc = df[candidate].value_counts()
            pct = (vc / len(df) * 100).round(1)
            print(f"\n  🎯 Target Column '{candidate}':")
            for label, count in vc.items():
                print(f"     {label:5} → {count:,} ({pct[label]}%)")
            break

    print(f"\n{divider}\n")


# =============================================================================
# 8. PLOTTING HELPERS
# =============================================================================

def plot_feature_importance(importance_df: pd.DataFrame,
                             top_n: int = 15,
                             title: str = "Top Feature Importances"):
    """
    Draw a horizontal bar chart showing which features matter most to the model.

    WHY:
        Feature importance tells you WHICH columns the model relies on most.
        Example: "Contract type" might be the #1 predictor of churn.
        This helps businesses focus on the right customer signals.

    PARAM importance_df: DataFrame with columns ['feature', 'importance']
    PARAM top_n: How many features to show
    """
    data = importance_df.head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=data,
                palette="viridis", orient="h")
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def plot_class_distribution(y: pd.Series, title: str = "Churn Distribution"):
    """
    Show how many customers churned vs stayed — as bar and pie charts.

    WHY:
        Class imbalance (e.g., only 26% churned) affects model training.
        Visualising it helps you decide if you need to handle the imbalance.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    vc = y.value_counts()
    colors = ["#4CAF50", "#F44336"]  # Green = stay, Red = churn

    # Bar chart
    ax1.bar(vc.index.astype(str), vc.values, color=colors, edgecolor="white", width=0.5)
    ax1.set_title(title, fontsize=13)
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Customer Count")
    for i, v in enumerate(vc.values):
        ax1.text(i, v + 30, f"{v:,}", ha="center", fontweight="bold")

    # Pie chart
    ax2.pie(vc.values,
            labels=[f"{l}\n({v:,})" for l, v in zip(vc.index.astype(str), vc.values)],
            autopct="%1.1f%%",
            colors=colors,
            startangle=90)
    ax2.set_title("Proportion", fontsize=13)

    plt.suptitle("Class Balance Check", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Feature Correlation Heatmap"):
    """
    Draw a colour-coded grid showing how strongly each numeric feature
    is related to every other numeric feature.

    WHY:
        If two features are 99% correlated, keeping both adds no value.
        High correlation with the target column = good predictor.
        Red = positive correlation, Blue = negative.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    # Only show the lower triangle (to avoid duplicate info)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5,
                annot_kws={"size": 9})
    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()
