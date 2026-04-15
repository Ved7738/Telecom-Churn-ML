"""
src/preprocessing.py  —  Data Cleaning & Preparation
======================================================

WHAT DOES "PREPROCESSING" MEAN?
    Raw data from the real world is messy. Before we can feed it to a machine
    learning model, we need to:

    1. DROP useless columns (customerID is just a name, not a pattern)
    2. FIX broken data (TotalCharges is stored as text in the CSV, not numbers)
    3. FILL missing values (a model can't handle empty cells)
    4. ENCODE categories (models only understand numbers, not "Yes"/"Male"/"DSL")
    5. SCALE numbers (tenure goes 0–72, MonthlyCharges goes 18–118 — very different
       ranges. Scaling brings them to the same scale so no column unfairly dominates)

ANALOGY:
    Raw data is like a job application form filled in by 7,043 people in different
    handwriting, some left blanks, some wrote "N/A". Preprocessing is the HR
    team standardising all forms into a clean digital database.

THE CLASS: DataPreprocessor
    We use a CLASS so the preprocessing logic learned from training data
    (e.g., "the average tenure is 32") can be SAVED and reused on new data.
    This is critical — you must apply IDENTICAL transformations at inference time.

KEY CONCEPT — FIT vs TRANSFORM:
    fit()       → LEARN parameters from training data (e.g., mean of MonthlyCharges)
    transform() → APPLY those learned parameters to any data (train or new)

    WHY SEPARATE?
        If you compute the mean on the TEST data too, you're "cheating" —
        the model indirectly sees the test data. This is called DATA LEAKAGE.
        We only learn statistics from the TRAINING set, then apply to everything.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from src.utils import load_config, get_logger

# Create a logger specifically for this module
logger = get_logger("preprocessing")


class DataPreprocessor:
    """
    Handles all data cleaning and transformation steps.

    ATTRIBUTES (things this class remembers after fitting):
        scaler          → StandardScaler fitted on numeric training columns
        label_encoders  → Dict of LabelEncoder, one per categorical column
        numeric_cols    → List of numeric column names
        categorical_cols→ List of categorical column names
        feature_names   → Final ordered list of all feature columns
    """

    # ── Column definitions (hardcoded for telecom churn dataset) ──────────────

    # Columns that contain numbers — will be SCALED
    NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

    # Already 0 or 1, no encoding needed — just kept as-is
    BINARY_COLS = ["SeniorCitizen"]

    # Not useful for prediction — just a customer identifier
    DROP_COLS = ["customerID"]

    # The column we're trying to PREDICT
    TARGET_COL = "Churn"

    # Columns containing text categories — will be LABEL ENCODED (text → integer)
    CATEGORICAL_COLS = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
    ]

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialise the preprocessor. Reads settings from config.yaml.
        Nothing is trained yet — this just sets up empty containers.
        """
        cfg = load_config(config_path)
        pp = cfg["preprocessing"]

        self.test_size    = pp["test_size"]       # e.g., 0.2 = 20% for testing
        self.random_state = pp["random_state"]    # e.g., 42
        self.stratify     = pp["stratify"]        # True = keep class balance

        # These will be filled when fit() is called
        self.scaler = StandardScaler()
        self.label_encoders: dict = {}

        # Copies of the column lists (so they can be overridden if needed)
        self.numeric_cols     = self.NUMERIC_COLS[:]
        self.categorical_cols = self.CATEGORICAL_COLS[:]
        self.feature_names: list = []

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame):
        """
        Complete pipeline: clean → encode target → split → fit on train → transform.

        This is the MAIN method called by train.py. It does everything in order.

        RETURNS:
            X_train, X_test  → Feature tables (numbers only, ready for model)
            y_train, y_test  → Target labels (0=No Churn, 1=Churn)

        EXAMPLE:
            prep = DataPreprocessor()
            X_train, X_test, y_train, y_test = prep.run(df)
        """
        logger.info("Starting preprocessing pipeline...")

        # Step 1: Remove junk columns and fix data types
        df = self._initial_clean(df)

        # Step 2: Separate features (X) from the answer we want to predict (y)
        y = self._encode_target(df[self.TARGET_COL])   # "Yes"/"No" → 1/0
        X = df.drop(columns=[self.TARGET_COL])

        # Step 3: Split into training set and test set
        # stratify=y means: keep same churn ratio (26%/74%) in both sets
        strat = y if self.stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=strat
        )
        logger.info(f"Split complete → Train: {len(X_train):,} | Test: {len(X_test):,}")

        # Step 4: LEARN scaling and encoding parameters FROM TRAINING DATA ONLY
        self.fit(X_train)

        # Step 5: Apply those learned parameters to both sets
        X_train_processed = self.transform(X_train)
        X_test_processed  = self.transform(X_test)

        logger.info(f"Preprocessing complete. Final shape: {X_train_processed.shape}")
        return X_train_processed, X_test_processed, y_train, y_test

    def fit(self, X: pd.DataFrame) -> "DataPreprocessor":
        """
        LEARN all transformation parameters from the training data.

        What we learn:
          - Mean and std of each numeric column (for StandardScaler)
          - All possible category values (for LabelEncoder — e.g., ["Male","Female"])

        IMPORTANT: Only call this on TRAINING data, never on test data.
        """
        X = X.copy()
        X = self._fix_total_charges(X)
        X = self._fill_missing(X)

        # Learn the scaling parameters (mean and standard deviation)
        # StandardScaler formula: scaled_value = (value - mean) / std
        # This centres data around 0. E.g., tenure 29 months → 0 (average)
        self.scaler.fit(X[self.numeric_cols])

        # Learn all possible category values for each text column
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
            logger.debug(f"  Encoded '{col}': {list(le.classes_)}")

        # Record the final feature order (important for consistent predictions)
        self.feature_names = self.numeric_cols + self.BINARY_COLS + self.categorical_cols

        logger.info(f"Preprocessor fitted. Total features: {len(self.feature_names)}")
        return self  # Return self so you can chain: preprocessor.fit(X).transform(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        APPLY learned transformations to any dataset (train, test, or new data).

        Steps performed:
          1. Fix TotalCharges if it's still text
          2. Fill any missing values
          3. Scale numeric columns using the fitted scaler
          4. Encode categorical columns using the fitted label encoders
          5. Return all columns in consistent order

        EXAMPLE:
            # At prediction time:
            X_new = pd.DataFrame([{"tenure": 24, "MonthlyCharges": 65, ...}])
            X_processed = preprocessor.transform(X_new)
            model.predict(X_processed)
        """
        X = X.copy()
        X = self._fix_total_charges(X)
        X = self._fill_missing(X)

        # Scale: brings all numeric columns to mean=0, std=1 range
        scaled_array = self.scaler.transform(X[self.numeric_cols])
        scaled_df = pd.DataFrame(scaled_array,
                                 columns=self.numeric_cols,
                                 index=X.index)

        # Encode: "Male" → 0, "Female" → 1, "Yes" → 1, "No" → 0, etc.
        cat_df = X[self.categorical_cols].copy()
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            cat_df[col] = X[col].astype(str).map(
                # If we see an unknown category at inference, use -1 as a safe default
                lambda val, _le=le: int(_le.transform([val])[0])
                    if val in _le.classes_ else -1
            )

        # Binary column stays as-is (already 0 or 1)
        binary_df = X[self.BINARY_COLS].copy()

        # Combine all three parts in a fixed column order
        result = pd.concat([scaled_df, binary_df, cat_df], axis=1)
        return result[self.feature_names]   # Enforce column order

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPER METHODS  (not called directly by users)
    # ──────────────────────────────────────────────────────────────────────────

    def _initial_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        First pass of cleaning:
          - Remove customerID (not useful)
          - Fix TotalCharges type issue
        """
        df = df.copy()

        # Drop identifier columns that carry no predictive value
        df.drop(columns=[c for c in self.DROP_COLS if c in df.columns], inplace=True)

        # Fix TotalCharges (some rows have " " instead of a number — new customers)
        df = self._fix_total_charges(df)

        logger.info(f"After initial clean: {df.shape[1]} columns remaining")
        return df

    @staticmethod
    def _fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
        """
        TotalCharges ISSUE EXPLANATION:
            In the raw CSV, customers with tenure=0 (brand new) have TotalCharges=" "
            (a space character). This causes the entire column to be read as text (object).

            pd.to_numeric(..., errors='coerce') converts:
              - "1234.56"  →  1234.56  (works fine)
              - " "        →  NaN      (blank becomes "Not a Number")
              - "hello"    →  NaN      (any non-number → NaN)

            We then fill NaN with the column median (a safe default).
        """
        if "TotalCharges" in df.columns and df["TotalCharges"].dtype == object:
            df = df.copy()
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        return df

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill any remaining empty (NaN) cells.

        Strategy:
          - Numeric columns  → fill with MEDIAN (robust to outliers)
          - Text columns     → fill with MODE (most common value)

        WHY NOT MEAN FOR NUMERIC?
            If monthly charges are mostly $50–80 but one outlier is $5000,
            the mean is pulled toward $5000. Median is unaffected by outliers.
        """
        df = df.copy()
        for col in self.numeric_cols:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.debug(f"  Filled '{col}' NaN with median={median_val:.2f}")

        for col in self.categorical_cols:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                logger.debug(f"  Filled '{col}' NaN with mode='{mode_val}'")
        return df

    @staticmethod
    def _encode_target(series: pd.Series) -> pd.Series:
        """
        Convert the target column from text to numbers:
            "Yes" → 1  (churned)
            "No"  → 0  (stayed)

        WHY:
            ML models need numbers. "Yes"/"No" are meaningless to a mathematical formula.
        """
        if series.dtype == object:
            return (series.str.strip().str.lower() == "yes").astype(int)
        return series.astype(int)
