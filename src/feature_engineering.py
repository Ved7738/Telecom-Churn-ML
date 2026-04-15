"""
src/feature_engineering.py  —  Creating Smarter Features
==========================================================

WHAT IS FEATURE ENGINEERING?
    The original dataset has 20 columns. But sometimes the raw columns
    don't tell the full story. Feature engineering creates NEW columns
    that capture patterns the model might miss.

REAL-WORLD ANALOGY:
    Imagine predicting exam performance. You have:
      - Hours studied per week
      - Total weeks before exam

    You could ALSO create:
      - Total study hours = hours_per_week × weeks  (more informative!)

    That new column "total study hours" is a ENGINEERED FEATURE.

WHAT WE CREATE FOR CHURN:
    1. charges_per_tenure  → Average monthly spend (TotalCharges ÷ tenure)
       WHY: A customer paying $100/month for 2 years vs 10 years is very different.
       The ratio reveals spending trends better than raw total charges.

    2. total_vs_expected   → Did total charges match expected?
       = TotalCharges ÷ (tenure × MonthlyCharges)
       WHY: If actual total < expected, the customer may have had discounts/pauses.
       This reveals billing irregularities that might predict churn.

    3. tenure_x_monthly    → tenure × MonthlyCharges (interaction feature)
       WHY: Long tenure + high bills = strong retention. The combination is
       more meaningful than either column alone.

FEATURE SELECTION:
    After creating new features, we might have 20+ columns. But more features
    ≠ better model. Irrelevant features add noise.

    We use a Random Forest to quickly RANK features by importance
    (how much each feature reduces prediction error), then keep only the TOP N.

    WHY RANDOM FOREST FOR SELECTION?
        It's fast, handles any data type, and gives reliable importance scores.

THE CLASS: FeatureEngineer
    fit_transform(X_train, y_train) → Create features + select top N → return
    transform(X_test)               → Create same features + select same N → return
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.utils import get_logger, load_config, plot_feature_importance

logger = get_logger("feature_engineering")


class FeatureEngineer:
    """
    Creates domain-specific features and selects the most important ones.

    ATTRIBUTES:
        n_top           → How many top features to keep (from config.yaml)
        selected_features → List of chosen feature names (filled after fit_transform)
        importance_df   → Full ranking of all features by importance
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        cfg = load_config(config_path)
        fe  = cfg["feature_engineering"]

        self.n_top            = fe["n_top_features"]       # e.g., 15
        self.do_interactions  = fe["create_interactions"]  # True/False
        self.do_ratios        = fe["create_ratio_features"]# True/False

        self.selected_features: list = []
        self.importance_df: pd.DataFrame = pd.DataFrame()

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        TRAINING TIME:
          1. Create new features from existing ones
          2. Use Random Forest to rank all features by importance
          3. Keep only the top N features
          4. Remember which features were selected (for use at inference time)

        RETURNS: DataFrame with only the selected top features
        """
        logger.info("Starting feature engineering (fit_transform)...")

        # Step 1: Create new features
        X_enriched = self._create_features(X)
        logger.info(f"Features after creation: {X_enriched.shape[1]}")

        # Step 2: Select top N by importance
        self._select_features(X_enriched, y)

        # Step 3: Return only selected columns
        return X_enriched[self.selected_features]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        INFERENCE/TEST TIME:
          1. Create the same new features as during training
          2. Select the same features that were chosen during training

        WHY NOT RE-SELECT AT INFERENCE TIME?
            The selection must be IDENTICAL to training.
            Re-running selection on test data would cause data leakage and
            potentially choose different features.

        RETURNS: DataFrame with the same selected features as training
        """
        if not self.selected_features:
            raise RuntimeError("Call fit_transform() first before transform().")

        # Create the same engineered features
        X_enriched = self._create_features(X)

        # Keep only the features selected during training
        # (safe subset in case any column is missing)
        available = [col for col in self.selected_features if col in X_enriched.columns]

        if len(available) < len(self.selected_features):
            missing = set(self.selected_features) - set(available)
            logger.warning(f"Missing features at transform time: {missing}")

        return X_enriched[available]

    # ──────────────────────────────────────────────────────────────────────────
    # FEATURE CREATION  (private)
    # ──────────────────────────────────────────────────────────────────────────

    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create new columns from existing ones.
        All operations are safe (handle division-by-zero, missing columns gracefully).
        """
        X = X.copy()

        # Column name references (after preprocessing these are scaled numeric)
        TENURE  = "tenure"
        MONTHLY = "MonthlyCharges"
        TOTAL   = "TotalCharges"

        # ── RATIO FEATURES ──────────────────────────────────────────────────
        if self.do_ratios:

            # Feature: Average spend per month of tenure
            # Interpretation: customer A paid $10,000 over 10 years = $83/mo (loyal, low bill)
            #                 customer B paid $2,000 over 2 years   = $83/mo (new, same bill)
            # The raw TotalCharges is the same ratio — but tenure context matters!
            if TENURE in X.columns and MONTHLY in X.columns:
                denom = X[TENURE].replace(0, np.nan)  # avoid division by zero
                X["charges_per_tenure"] = X[MONTHLY] / denom
                X["charges_per_tenure"].fillna(X[MONTHLY], inplace=True)  # new customers

            # Feature: Did total charges match what was expected?
            # expected = tenure × monthly. If ratio < 1, customer had discounts/pauses.
            if all(c in X.columns for c in [TENURE, MONTHLY, TOTAL]):
                expected = X[TENURE] * X[MONTHLY]
                denom = expected.replace(0, np.nan)
                X["total_vs_expected"] = X[TOTAL] / denom
                X["total_vs_expected"].fillna(1.0, inplace=True)  # default: matches expected

        # ── INTERACTION FEATURES ─────────────────────────────────────────────
        if self.do_interactions:

            # Feature: tenure × monthly charges
            # High value = long-term, high-value customer (less likely to churn)
            # Low value  = new or cheap customer (more at risk)
            if TENURE in X.columns and MONTHLY in X.columns:
                X["tenure_x_monthly"] = X[TENURE] * X[MONTHLY]

        return X

    # ──────────────────────────────────────────────────────────────────────────
    # FEATURE SELECTION  (private)
    # ──────────────────────────────────────────────────────────────────────────

    def _select_features(self, X: pd.DataFrame, y: pd.Series):
        """
        Train a fast Random Forest to rank features by importance.

        HOW DOES RANDOM FOREST RANK FEATURES?
            Each tree in the forest makes splits on features.
            A feature that creates "purer" splits (better separates churn vs no-churn)
            gets a higher importance score.

            Think of it like a quiz: "Which question best separates students who pass
            from those who fail?" The best questions = most important features.
        """
        logger.info(f"Ranking {X.shape[1]} features by Random Forest importance...")

        # Quick Random Forest just for ranking (not the final model)
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1    # Use all CPU cores for speed
        )
        rf.fit(X, y)

        # Build a sorted DataFrame of feature importances
        self.importance_df = (
            pd.DataFrame({
                "feature"   : X.columns,
                "importance": rf.feature_importances_
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        # Select top N features
        top_n = min(self.n_top, len(X.columns))
        self.selected_features = self.importance_df.head(top_n)["feature"].tolist()

        logger.info(f"Selected top {top_n} features:")
        print("\n  📊 Feature Importance Ranking (Top {}):\n".format(top_n))
        print(self.importance_df.head(top_n).to_string(index=False))

        # Optionally visualise (only if running interactively)
        try:
            plot_feature_importance(self.importance_df, top_n=top_n)
        except Exception:
            pass  # Skip if running headless (no display available)
