"""
train.py  —  The One Script to Train Everything
=================================================

HOW TO RUN:
    python train.py

WHAT IT DOES (in order):
    1. Loads the raw CSV data
    2. Cleans and preprocesses it
    3. Creates new smart features
    4. Trains 3 ML models with cross-validation
    5. Picks the best model
    6. Evaluates it on test data (draws charts)
    7. Saves the full pipeline to models/pipeline.pkl

AFTER THIS RUNS:
    You'll have a trained model ready to serve predictions.
    Run:  python app/app.py
    Then: http://localhost:5000

TIME TO RUN:
    Approximately 2–5 minutes on a normal laptop (no GPU needed).
"""

import os
import sys

# ── Make sure Python can find the src/ package ─────────────────────────────
# When running "python train.py" from the project root, Python needs to know
# where to find "src.preprocessing", "src.utils", etc.
# Adding the current directory to the path solves this.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils             import load_data, basic_eda, get_logger, ensure_dirs
from src.preprocessing     import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training    import (
    ModelTrainer, ModelEvaluator, save_pipeline
)

logger = get_logger("train")


def main():
    """
    Full end-to-end training pipeline.
    Each step is clearly labelled so you can follow the flow.
    """

    print("\n" + "█" * 65)
    print("  TELECOM CHURN PREDICTION — MODEL TRAINING PIPELINE")
    print("█" * 65)

    # ── Ensure output directories exist ─────────────────────────────────────
    ensure_dirs("models", "logs", "data/processed")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1: LOAD DATA
    # ════════════════════════════════════════════════════════════════════════
    print("\n\n[STEP 1/6]  Loading raw data...")
    print("─" * 40)

    data_path = "data/raw/telecom_churn.csv"
    if not os.path.exists(data_path):
        print(f"\n  ❌ ERROR: Dataset not found at '{data_path}'")
        print("     Please copy telecom_churn.csv to data/raw/")
        print("     Download from: https://www.kaggle.com/blastchar/telco-customer-churn")
        sys.exit(1)

    df = load_data(data_path)

    # Print a full overview of the raw data
    basic_eda(df)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2: PREPROCESS
    # ════════════════════════════════════════════════════════════════════════
    print("\n\n[STEP 2/6]  Preprocessing data...")
    print("─" * 40)
    print("  Tasks: fix TotalCharges type, drop customerID,")
    print("         fill missing values, scale numbers, encode categories,")
    print("         split into 80% train / 20% test.")

    preprocessor = DataPreprocessor(config_path="config/config.yaml")

    # run() does everything: clean → encode target → split → fit → transform
    X_train, X_test, y_train, y_test = preprocessor.run(df)

    print(f"\n  ✅ Train set : {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
    print(f"  ✅ Test set  : {X_test.shape[0]:,} rows × {X_test.shape[1]} features")
    print(f"  ✅ Churn rate in train: {y_train.mean()*100:.1f}%")
    print(f"  ✅ Churn rate in test : {y_test.mean()*100:.1f}%")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3: FEATURE ENGINEERING
    # ════════════════════════════════════════════════════════════════════════
    print("\n\n[STEP 3/6]  Feature engineering...")
    print("─" * 40)
    print("  Creating: charges_per_tenure, total_vs_expected, tenure_x_monthly")
    print("  Then selecting top features by Random Forest importance.")

    fe = FeatureEngineer(config_path="config/config.yaml")

    # fit_transform: create features + rank + select top N
    X_train_fe = fe.fit_transform(X_train, y_train)

    # transform: apply SAME operations to test set (no re-learning)
    X_test_fe  = fe.transform(X_test)

    print(f"\n  ✅ Final feature count: {X_train_fe.shape[1]}")
    print(f"  ✅ Selected features  : {fe.selected_features}")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 4: TRAIN MODELS
    # ════════════════════════════════════════════════════════════════════════
    print("\n\n[STEP 4/6]  Training models...")
    print("─" * 40)
    print("  Training Logistic Regression, Random Forest, XGBoost")
    print("  with 5-fold stratified cross-validation.")

    trainer = ModelTrainer(config_path="config/config.yaml")
    trainer.train_all(X_train_fe, y_train)

    # Identify the best model by cross-validation F1
    best_name, best_model = trainer.best_model()

    # ════════════════════════════════════════════════════════════════════════
    # STEP 5: EVALUATE
    # ════════════════════════════════════════════════════════════════════════
    print("\n\n[STEP 5/6]  Evaluating all models on test data...")
    print("─" * 40)

    evaluator = ModelEvaluator()

    # Compare all models side-by-side on test data
    evaluator.compare_all_models(trainer.trained_models, X_test_fe, y_test)

    # Deep evaluation of the best model
    y_pred  = best_model.predict(X_test_fe)
    y_proba = best_model.predict_proba(X_test_fe)[:, 1]

    metrics = evaluator.evaluate(y_pred=y_pred,
                                  y_true=y_test,
                                  y_pred_proba=y_proba,
                                  model_name=best_name)

    # Draw confusion matrix and ROC curve
    try:
        evaluator.plot_confusion_matrix(y_test, y_pred, best_name)
        evaluator.plot_roc_curve(y_test, y_proba, best_name)
    except Exception as e:
        logger.warning(f"Could not display charts (headless environment?): {e}")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 6: SAVE PIPELINE
    # ════════════════════════════════════════════════════════════════════════
    print("\n\n[STEP 6/6]  Saving the pipeline...")
    print("─" * 40)
    print("  Saving: preprocessor + feature engineer + model → models/pipeline.pkl")

    save_pipeline(
        model             = best_model,
        preprocessor      = preprocessor,
        feature_engineer  = fe,
        filepath          = "models/pipeline.pkl"
    )

    # ════════════════════════════════════════════════════════════════════════
    # DONE!
    # ════════════════════════════════════════════════════════════════════════
    print("\n\n" + "█" * 65)
    print("  TRAINING COMPLETE!")
    print("█" * 65)
    print(f"\n  Best Model  : {best_name}")
    print(f"  Accuracy    : {metrics.get('Accuracy', 0):.4f}")
    print(f"  F1-Score    : {metrics.get('F1-Score', 0):.4f}")
    print(f"  ROC-AUC     : {metrics.get('ROC-AUC', 0):.4f}")
    print(f"\n  Model saved : models/pipeline.pkl")
    print("\n  ─────────────────────────────────────────────────")
    print("  NEXT STEP: Run the web app")
    print("  Command   : python app/app.py")
    print("  Browser   : http://localhost:5000")
    print("  ─────────────────────────────────────────────────\n")


# ── Entry point ─────────────────────────────────────────────────────────────
# This block runs ONLY when you execute "python train.py" directly.
# If another file imports train.py, this block is skipped.
if __name__ == "__main__":
    main()
