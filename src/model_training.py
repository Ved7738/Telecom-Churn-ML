"""
src/model_training.py  —  Training, Evaluating & Saving the ML Model
======================================================================

WHAT HAPPENS IN THIS FILE?
    This is where the actual "learning" happens.

    We train THREE different algorithms on the prepared data and compare them:
      1. Logistic Regression  — Simple, fast, interpretable
      2. Random Forest        — Ensemble of 100 decision trees, robust
      3. XGBoost              — Gradient boosting, typically most accurate

    Then we pick the BEST one, evaluate it thoroughly, and SAVE it to disk.

ANALOGY FOR EACH MODEL:
    → Logistic Regression: Like a scoring formula. "If monthly charges > $80
      AND contract is month-to-month → score = 75% churn probability."
      Simple, explainable to business stakeholders.

    → Random Forest: A committee of 100 experts, each looking at slightly
      different data and casting a vote. Majority wins.
      Good at handling messy, mixed data.

    → XGBoost: Like Random Forest but smarter. Each new tree specifically
      fixes the mistakes of the previous tree. "Boosting" = sequential improvement.
      Usually wins Kaggle competitions on structured data.

KEY CONCEPTS:
    Cross-Validation:
        Instead of one train/test split, we do 5 rounds:
          Round 1: Train on folds 2,3,4,5 → Test on fold 1
          Round 2: Train on folds 1,3,4,5 → Test on fold 2
          ...
        Final score = average of 5 rounds.
        WHY: More reliable than a single split. Reduces luck of the draw.

    F1-Score (our primary metric):
        F1 = 2 × (Precision × Recall) / (Precision + Recall)
        Balances two concerns:
          - Precision: "Of customers we flagged as churning, how many actually did?"
          - Recall:    "Of all churning customers, how many did we catch?"
        WHY NOT ACCURACY? If 74% didn't churn, always predicting "No" gives 74% accuracy!
        F1 exposes this problem.

    Hyperparameter Tuning:
        Models have "dials" (hyperparameters) like:
          - n_estimators: How many trees? (50? 100? 200?)
          - max_depth: How deep each tree? (5? 10? 15?)
        Grid Search tries ALL combinations and picks the best.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics        import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import xgboost as xgb

from src.utils import get_logger, load_config, save_object

logger = get_logger("model_training")


# =============================================================================
# CLASS 1: ModelTrainer — Trains multiple algorithms and compares them
# =============================================================================

class ModelTrainer:
    """
    Trains all three models with cross-validation and identifies the best one.

    HOW TO USE:
        trainer = ModelTrainer()
        trainer.train_all(X_train, y_train)
        best_name, best_model = trainer.best_model()
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        cfg = load_config(config_path)
        mt  = cfg["model_training"]

        self.cv_folds     = mt["cv_folds"]       # Number of cross-validation folds
        self.scoring      = mt["scoring"]         # Metric to compare models (f1)
        self.random_state = mt["random_state"]

        # ── Define the three models ──────────────────────────────────────────

        lr  = mt["models"]["logistic_regression"]
        rf  = mt["models"]["random_forest"]
        xgb_cfg = mt["models"]["xgboost"]

        self.models = {

            # LOGISTIC REGRESSION
            # Despite its name, it's a CLASSIFICATION model (not regression).
            # It computes a probability using a sigmoid function.
            # Best for: linearly separable data, when you need to explain predictions.
            "Logistic Regression": LogisticRegression(
                max_iter=lr["max_iter"],       # Max iterations to converge
                random_state=lr["random_state"],
                class_weight="balanced"        # Adjusts for our 74%/26% imbalance
            ),

            # RANDOM FOREST
            # Builds n_estimators decision trees on random subsets of data and features.
            # Prediction = majority vote of all trees.
            # Best for: general-purpose, handles missing patterns, less overfitting.
            "Random Forest": RandomForestClassifier(
                n_estimators=rf["n_estimators"],  # Number of trees
                max_depth=rf["max_depth"],         # Max depth (prevent memorizing)
                random_state=rf["random_state"],
                n_jobs=rf["n_jobs"],               # -1 = use all CPU cores
                class_weight="balanced"
            ),

            # XGBOOST
            # Gradient Boosting: builds trees sequentially, each fixing previous errors.
            # learning_rate: how big a step each new tree takes (small = careful)
            # Best for: highest accuracy on tabular data, Kaggle competitions.
            "XGBoost": xgb.XGBClassifier(
                n_estimators=xgb_cfg["n_estimators"],
                max_depth=xgb_cfg["max_depth"],
                learning_rate=xgb_cfg["learning_rate"],
                random_state=xgb_cfg["random_state"],
                n_jobs=xgb_cfg["n_jobs"],
                eval_metric="logloss",
                verbosity=0
            ),
        }

        self.trained_models: dict = {}
        self.cv_scores: dict      = {}

    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """
        Train every model and measure performance using stratified cross-validation.

        STRATIFIED K-FOLD:
            Ensures each fold has the same proportion of churners (26%) as the full dataset.
            Regular K-Fold might accidentally put all churners in one fold.

        RETURNS: Dictionary of {model_name: trained_model}
        """
        # StratifiedKFold = cross-validation that preserves class ratios
        cv = StratifiedKFold(n_splits=self.cv_folds,
                             shuffle=True,
                             random_state=self.random_state)

        print("\n" + "═" * 60)
        print("  MODEL TRAINING & CROSS-VALIDATION")
        print("═" * 60)

        for name, model in self.models.items():
            print(f"\n  🔄 Training: {name} ...")

            # Train on full training set
            model.fit(X_train, y_train)
            self.trained_models[name] = model

            # Measure how well it generalises using cross-validation
            # cross_val_score: trains + tests 5 times, returns 5 scores
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv,
                scoring=self.scoring,
                n_jobs=-1
            )
            self.cv_scores[name] = scores

            # Report the average and variability
            print(f"     CV {self.scoring.upper()}: {scores.mean():.4f}  "
                  f"(±{scores.std():.4f})  "
                  f"[min={scores.min():.4f}, max={scores.max():.4f}]")
            logger.info(f"{name}: CV mean={scores.mean():.4f}, std={scores.std():.4f}")

        return self.trained_models

    def best_model(self):
        """
        Return the model with the highest average cross-validation F1-score.

        RETURNS: (name: str, model: fitted estimator)
        """
        name = max(self.cv_scores, key=lambda n: self.cv_scores[n].mean())
        print(f"\n  🏆 Best model: {name}  "
              f"(CV F1 = {self.cv_scores[name].mean():.4f})")
        return name, self.trained_models[name]


# =============================================================================
# CLASS 2: ModelEvaluator — Calculates metrics and draws charts
# =============================================================================

class ModelEvaluator:
    """
    Evaluates how well a model performs on TEST data (data it has never seen).

    WHY SEPARATE EVALUATION ON TEST DATA?
        Cross-validation gives a training estimate.
        Test data gives the REAL-WORLD estimate.
        If test performance drops a lot → the model is overfitting (memorised training data).

    METRICS EXPLAINED:
        Accuracy  = correct predictions / total predictions
                    Problem: misleading for imbalanced data

        Precision = TP / (TP + FP)
                    "Of all customers flagged for churn, how many actually churned?"
                    High precision → few false alarms → retention team not wasting calls

        Recall    = TP / (TP + FN)
                    "Of all actual churners, how many did we flag?"
                    High recall → catches more churners → fewer customers lost undetected

        F1-Score  = harmonic mean of Precision and Recall
                    Best single metric for imbalanced classification

        ROC-AUC   = Area under the ROC curve (0.5 = random, 1.0 = perfect)
                    Measures model's ability to rank churners above non-churners.
                    Does NOT depend on a threshold.

    CONFUSION MATRIX:
                        Predicted No  Predicted Yes
        Actual No    [  True Neg   |  False Pos  ]   ← False alarm (annoying)
        Actual Yes   [  False Neg  |  True Pos   ]   ← Missed churn (costly!)
    """

    @staticmethod
    def evaluate(y_true, y_pred, y_pred_proba=None,
                 model_name: str = "Model") -> dict:
        """
        Compute and print all key metrics.

        PARAMS:
            y_true       → Actual labels (0 or 1)
            y_pred       → Model's predicted labels
            y_pred_proba → Model's predicted probabilities (needed for ROC-AUC)
            model_name   → Just for display

        RETURNS: dict of metric_name → value
        """
        metrics = {
            "Accuracy" : accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall"   : recall_score(y_true, y_pred, zero_division=0),
            "F1-Score" : f1_score(y_true, y_pred, zero_division=0),
        }
        if y_pred_proba is not None:
            metrics["ROC-AUC"] = roc_auc_score(y_true, y_pred_proba)

        print("\n" + "═" * 60)
        print(f"  EVALUATION METRICS — {model_name}")
        print("═" * 60)
        for metric, value in metrics.items():
            bar = "▓" * int(value * 20)
            print(f"  {metric:<12}: {value:.4f}  {bar}")

        print(f"\n  Detailed Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=["No Churn", "Churn"],
                                    zero_division=0))
        return metrics

    @staticmethod
    def compare_all_models(trained_models: dict,
                           X_test: pd.DataFrame,
                           y_test: pd.Series) -> pd.DataFrame:
        """
        Run all trained models on test data and produce a comparison table.

        RETURNS: DataFrame sorted by F1-Score (best first)
        """
        rows = []
        for name, model in trained_models.items():
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            rows.append({
                "Model"    : name,
                "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "Recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
                "F1-Score" : round(f1_score(y_test, y_pred, zero_division=0), 4),
                "ROC-AUC"  : round(roc_auc_score(y_test, y_proba), 4),
            })

        df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)
        print("\n  📊 MODEL COMPARISON (Test Set):")
        print(df.to_string(index=False))
        return df

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model"):
        """
        Draw a 2×2 grid showing correct and incorrect predictions.

        HOW TO READ IT:
            Top-left (TN): Predicted No Churn, Actually No Churn ← Good
            Top-right (FP): Predicted Churn, Actually No Churn   ← False alarm
            Bottom-left (FN): Predicted No Churn, Actually Churn ← MISSED! Costly
            Bottom-right (TP): Predicted Churn, Actually Churn   ← Good
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Predicted: Stay", "Predicted: Churn"],
                    yticklabels=["Actual: Stay", "Actual: Churn"],
                    linewidths=1, linecolor="white")
        plt.title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba, model_name: str = "Model"):
        """
        ROC CURVE EXPLANATION:
            X-axis = False Positive Rate (how often we wrongly flag loyal customers)
            Y-axis = True Positive Rate (how often we correctly flag churners)

            A perfect model goes straight up then right → AUC = 1.0
            A random model is the diagonal line    → AUC = 0.5
            Our model should be above the diagonal.

            The AUC (Area Under Curve) summarises the whole curve in one number.
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, lw=2.5, color="#667eea",
                 label=f"{model_name} (AUC = {auc:.3f})")
        plt.fill_between(fpr, tpr, alpha=0.1, color="#667eea")
        plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier (AUC = 0.5)")
        plt.xlabel("False Positive Rate  (1 - Specificity)")
        plt.ylabel("True Positive Rate  (Sensitivity / Recall)")
        plt.title("ROC Curve", fontsize=13, pad=12)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


# =============================================================================
# CLASS 3: HyperparameterOptimizer — Fine-tunes the best model
# =============================================================================

class HyperparameterOptimizer:
    """
    Grid Search Hyperparameter Tuning.

    WHAT ARE HYPERPARAMETERS?
        They are the "settings" of a model that YOU choose (not learned from data).
        Like: how many trees? How deep? How fast to learn?

    WHAT IS GRID SEARCH?
        It tries every possible combination from a list of values.
        E.g., {n_estimators: [50,100,200], max_depth: [5,10,15]}
        → 3 × 3 = 9 combinations, each cross-validated 5 times = 45 total fits.
        Picks the best.

    WARNING: This is SLOW. Comment it out if you just want a quick run.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        cfg = load_config(config_path)
        ht  = cfg["hyperparameter_tuning"]
        mt  = cfg["model_training"]

        self.cv       = mt["cv_folds"]
        self.rf_grid  = ht["rf_param_grid"]
        self.xgb_grid = ht["xgb_param_grid"]

    def tune_rf(self, X_train, y_train):
        """Grid search over Random Forest hyperparameters."""
        logger.info("Starting Random Forest grid search...")
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
        gs = GridSearchCV(rf, self.rf_grid, cv=self.cv,
                          scoring="f1", n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)

        print(f"\n  Best RF params : {gs.best_params_}")
        print(f"  Best CV F1     : {gs.best_score_:.4f}")
        return gs.best_estimator_

    def tune_xgb(self, X_train, y_train):
        """Grid search over XGBoost hyperparameters."""
        logger.info("Starting XGBoost grid search...")
        model = xgb.XGBClassifier(random_state=42, n_jobs=-1,
                                   eval_metric="logloss", verbosity=0)
        gs = GridSearchCV(model, self.xgb_grid, cv=self.cv,
                          scoring="f1", n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)

        print(f"\n  Best XGB params: {gs.best_params_}")
        print(f"  Best CV F1     : {gs.best_score_:.4f}")
        return gs.best_estimator_


# =============================================================================
# PIPELINE SAVE / LOAD FUNCTIONS
# =============================================================================

def save_pipeline(model, preprocessor, feature_engineer,
                  filepath: str = "models/pipeline.pkl"):
    """
    Save the COMPLETE inference pipeline to a single file.

    WHY SAVE EVERYTHING TOGETHER?
        At prediction time, you need all 3 components in the exact right order:
          1. preprocessor    → clean and encode raw input
          2. feature_engineer → create interaction features + select top N
          3. model           → predict churn probability

        Saving them together ensures they always stay in sync.
        The metadata records WHEN the model was trained and WHICH type it is.

    PICKLE FORMAT:
        We use Python's pickle module. It serialises Python objects into bytes.
        The .pkl file is like a "frozen snapshot" of the entire trained pipeline.
    """
    pipeline = {
        "model"           : model,
        "preprocessor"    : preprocessor,
        "feature_engineer": feature_engineer,
        "metadata": {
            "model_type" : type(model).__name__,
            "created_at" : pd.Timestamp.now().isoformat(),
            "description": "Telecom Churn Prediction Pipeline"
        }
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    save_object(pipeline, filepath)
    logger.info(f"Pipeline saved → {filepath}")


def load_pipeline(filepath: str = "models/pipeline.pkl") -> dict:
    """
    Load a previously saved pipeline from disk.

    RETURNS:
        dict with keys: 'model', 'preprocessor', 'feature_engineer', 'metadata'
    """
    with open(filepath, "rb") as f:
        pipeline = pickle.load(f)
    logger.info(f"Pipeline loaded ← {filepath}")
    print(f"  📦 Pipeline metadata: {pipeline.get('metadata', {})}")
    return pipeline
