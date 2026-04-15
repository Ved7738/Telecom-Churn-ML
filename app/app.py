"""
app/app.py  —  Flask Web Application
======================================

WHAT IS THIS FILE?
    This is the WEB SERVER for our ML model. It:
      1. Loads the trained pipeline (models/pipeline.pkl) when it starts
      2. Serves a webpage where anyone can enter customer details
      3. Accepts JSON API calls for programmatic predictions
      4. Returns predictions as JSON responses

WHAT IS FLASK?
    Flask is a lightweight Python web framework. You define ROUTES
    (URL paths) and the Python function that runs when someone visits that URL.

    Example:
        @app.route("/health")        ← When someone visits /health
        def health():                ← Run this function
            return "OK"              ← Send this back

HOW TO RUN:
    python app/app.py
    → Visit http://localhost:5000

API ENDPOINTS:
    GET  /                    → Returns the HTML prediction form
    GET  /health              → Health check (is the server alive?)
    POST /api/predict         → Predict churn for ONE customer
    POST /api/batch-predict   → Predict churn for MANY customers at once
    GET  /api/feature-info    → Returns list of expected input features

IMPORTANT — THE PREDICTION PIPELINE ORDER:
    Every prediction MUST go through the same 3 steps used during training:
        raw input → preprocessor.transform() → feature_engineer.transform() → model.predict()
    Skipping a step = garbage predictions.
"""

import os
import sys
import logging
import pickle
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request, jsonify

# ── Allow imports from the project root ─────────────────────────────────────
# When running "python app/app.py", Python's working context is the app/ folder.
# We add the parent folder (project root) to the path so we can import from src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── CRITICAL: Import the pipeline classes BEFORE loading the pickle ──────────
# WHY: When Python pickled the pipeline, it stored references like:
#      "src.preprocessing.DataPreprocessor"
#      When unpickling, Python needs to FIND that class.
#      Importing it first registers it so pickle.load() can resolve it.
from src.preprocessing      import DataPreprocessor      # noqa: F401
from src.feature_engineering import FeatureEngineer       # noqa: F401

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════

os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(ROOT, "logs", "app.log")),
        logging.StreamHandler(),   # Also print to terminal
    ]
)
logger = logging.getLogger("flask_app")

# ═══════════════════════════════════════════════════════════════════════════
# FLASK APP INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════

# template_folder: where Flask looks for HTML files
app = Flask(__name__, template_folder="templates")
app.config["JSON_SORT_KEYS"] = False   # Keep JSON response keys in insertion order

# ═══════════════════════════════════════════════════════════════════════════
# LOAD THE TRAINED PIPELINE AT STARTUP
# ═══════════════════════════════════════════════════════════════════════════

PIPELINE_PATH = os.path.join(ROOT, "models", "pipeline.pkl")


def _load_pipeline():
    """
    Load models/pipeline.pkl when the server starts.

    WHY AT STARTUP (not per request)?
        Loading a model takes time. We load it ONCE into memory.
        Every prediction request reuses the same in-memory objects —
        making predictions fast (milliseconds).

    If the file doesn't exist, the server still starts but predictions
    return a friendly error message instead of crashing.
    """
    if not os.path.exists(PIPELINE_PATH):
        logger.warning(
            f"Pipeline not found at {PIPELINE_PATH}. "
            "Run 'python train.py' first to train the model."
        )
        return None

    try:
        with open(PIPELINE_PATH, "rb") as f:
            pipeline = pickle.load(f)
        logger.info(f"Pipeline loaded. Model: {pipeline['metadata']['model_type']}, "
                    f"Trained: {pipeline['metadata']['created_at'][:10]}")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        return None


# Load once at module import time
PIPELINE         = _load_pipeline()
MODEL            = PIPELINE["model"]             if PIPELINE else None
PREPROCESSOR     = PIPELINE["preprocessor"]     if PIPELINE else None
FEATURE_ENGINEER = PIPELINE["feature_engineer"] if PIPELINE else None


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES (URL ENDPOINTS)
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    """
    Serve the main web page.

    render_template("index.html") reads app/templates/index.html
    and sends it to the browser.
    """
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.

    USE CASE: Monitoring systems ping /health every minute to verify
              the server is running and the model is loaded.
              Returns HTTP 200 if healthy.

    EXAMPLE RESPONSE:
        {
          "status": "healthy",
          "model_loaded": true,
          "model_type": "XGBClassifier",
          "timestamp": "2025-01-15T10:30:00"
        }
    """
    return jsonify({
        "status"      : "healthy",
        "model_loaded": MODEL is not None,
        "model_type"  : PIPELINE["metadata"]["model_type"] if PIPELINE else None,
        "trained_at"  : PIPELINE["metadata"]["created_at"][:10] if PIPELINE else None,
        "timestamp"   : datetime.now().isoformat()
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Single-customer churn prediction.

    HOW TO CALL (from JavaScript or curl):
        POST /api/predict
        Content-Type: application/json
        Body: {
            "tenure": 24,
            "MonthlyCharges": 65.50,
            "TotalCharges": 1500,
            "gender": "Male",
            "Contract": "Month-to-month",
            ... (all 19 features)
        }

    RESPONSE:
        {
            "status": "success",
            "prediction": 1,
            "prediction_label": "Will Churn",
            "churn_probability": 0.82,
            "no_churn_probability": 0.18,
            "confidence": 0.82
        }

    WHY BOTH prediction AND churn_probability?
        prediction = 0 or 1 (binary decision at 50% threshold)
        churn_probability = exact probability (business can set their own threshold)
        Example: A business might act on customers with >30% churn risk (not just >50%)
    """
    try:
        # Get the JSON body sent by the browser/caller
        data = request.get_json(force=True)
        logger.info(f"Predict request received: {data}")

        # Guard: model must be loaded
        if MODEL is None:
            return jsonify({
                "status" : "error",
                "message": "Model not loaded. Run python train.py first."
            }), 503   # 503 = Service Unavailable

        # Wrap single record in a DataFrame (models expect 2D table input)
        input_df = pd.DataFrame([data])

        # Validate: are all required columns present?
        error_msg = _validate_input(input_df)
        if error_msg:
            return jsonify({"status": "error", "message": error_msg}), 400

        # Run the full inference pipeline
        result = _run_inference(input_df)

        logger.info(f"Prediction: {result['prediction_label']} "
                    f"(churn prob: {result['churn_probability']:.2%})")
        return jsonify(result)

    except Exception as exc:
        logger.exception("Unexpected error in /api/predict")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    """
    Predict churn for MULTIPLE customers at once.

    USE CASE: A business analyst uploads a CSV of 500 at-risk customers.
              The app scores all 500 at once and returns a ranked list.

    REQUEST BODY: JSON array of customer objects
        [
            {"tenure": 24, "MonthlyCharges": 65, ...},
            {"tenure": 5,  "MonthlyCharges": 95, ...},
            ...
        ]

    RESPONSE:
        {
            "status": "success",
            "count": 2,
            "results": [
                {"prediction": 0, "prediction_label": "Will Not Churn", "churn_probability": 0.18},
                {"prediction": 1, "prediction_label": "Will Churn",     "churn_probability": 0.91}
            ]
        }
    """
    try:
        data = request.get_json(force=True)

        if not isinstance(data, list) or len(data) == 0:
            return jsonify({
                "status" : "error",
                "message": "Expected a non-empty JSON array of customer records."
            }), 400

        if MODEL is None:
            return jsonify({
                "status" : "error",
                "message": "Model not loaded. Run python train.py first."
            }), 503

        results = []
        for i, record in enumerate(data):
            try:
                input_df = pd.DataFrame([record])
                r = _run_inference(input_df)
                results.append({
                    "prediction"        : r["prediction"],
                    "prediction_label"  : r["prediction_label"],
                    "churn_probability" : r["churn_probability"],
                })
            except Exception as row_err:
                # Don't let one bad record kill the whole batch
                results.append({"error": str(row_err), "record_index": i})

        logger.info(f"Batch prediction: {len(results)} records processed.")
        return jsonify({
            "status" : "success",
            "count"  : len(results),
            "results": results
        })

    except Exception as exc:
        logger.exception("Unexpected error in /api/batch-predict")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/feature-info", methods=["GET"])
def feature_info():
    """
    Return information about what input features the model expects.

    USE CASE: A developer integrating the API can call this to discover
              which fields to send in the predict request — no documentation needed.

    RESPONSE:
        {
            "numeric_features": ["tenure", "MonthlyCharges", "TotalCharges"],
            "categorical_features": ["gender", "Contract", ...],
            "selected_model_features": [...top N features...]
        }
    """
    if PREPROCESSOR is None:
        return jsonify({"error": "Model not loaded."}), 503

    return jsonify({
        "numeric_features"      : PREPROCESSOR.numeric_cols,
        "categorical_features"  : PREPROCESSOR.categorical_cols,
        "binary_features"       : PREPROCESSOR.BINARY_COLS,
        "total_input_features"  : len(PREPROCESSOR.feature_names),
        "selected_model_features": FEATURE_ENGINEER.selected_features
                                   if FEATURE_ENGINEER else []
    })


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _validate_input(df: pd.DataFrame):
    """
    Check that the incoming request has all required columns.

    WHY VALIDATE?
        If a required column is missing (e.g., someone forgets "Contract"),
        the preprocessor will fail with a confusing KeyError.
        Validation gives a clear, friendly error message instead.

    RETURNS: error string if invalid, None if valid
    """
    if PREPROCESSOR is None:
        return "Preprocessor not available."

    required = PREPROCESSOR.numeric_cols + PREPROCESSOR.categorical_cols
    for col in required:
        if col not in df.columns:
            return (f"Missing required field: '{col}'. "
                    f"Call /api/feature-info to see all required fields.")
    return None   # None = no error = valid


def _run_inference(input_df: pd.DataFrame) -> dict:
    """
    Run a single record through the full 3-step inference pipeline.

    STEP 1: preprocessor.transform()
        - Fixes TotalCharges type (if needed)
        - Scales numeric columns using TRAINING statistics
        - Encodes categorical columns using TRAINING encoders

    STEP 2: feature_engineer.transform()
        - Creates ratio and interaction features
        - Selects SAME top-N features chosen during training

    STEP 3: model.predict() + model.predict_proba()
        - Returns binary prediction (0 or 1)
        - Returns probability [no_churn_prob, churn_prob]

    RETURNS: dict ready to be serialised as a JSON response
    """
    # Step 1
    processed = PREPROCESSOR.transform(input_df)

    # Step 2
    engineered = FEATURE_ENGINEER.transform(processed)

    # Step 3
    prediction = int(MODEL.predict(engineered)[0])
    proba      = MODEL.predict_proba(engineered)[0]   # [prob_class_0, prob_class_1]

    return {
        "status"              : "success",
        "prediction"          : prediction,
        "prediction_label"    : "Will Churn" if prediction == 1 else "Will Not Churn",
        "churn_probability"   : round(float(proba[1]), 4),
        "no_churn_probability": round(float(proba[0]), 4),
        "confidence"          : round(float(max(proba)), 4),
        "timestamp"           : datetime.now().isoformat()
    }


# ═══════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(_):
    """Friendly JSON response for unknown URLs."""
    return jsonify({
        "error"    : "Endpoint not found.",
        "available": ["/", "/health", "/api/predict",
                      "/api/batch-predict", "/api/feature-info"]
    }), 404


@app.errorhandler(405)
def method_not_allowed(_):
    """Friendly JSON response for wrong HTTP method (e.g., GET instead of POST)."""
    return jsonify({"error": "Method not allowed for this endpoint."}), 405


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  TELECOM CHURN PREDICTION — WEB APP")
    print("═" * 55)
    print(f"  Model loaded : {'✅ Yes' if MODEL else '❌ No (run train.py first)'}")
    print(f"  URL          : http://localhost:5000")
    print(f"  Health check : http://localhost:5000/health")
    print("  Press Ctrl+C to stop\n")

    # debug=True: auto-reloads when you edit code (development only)
    # threaded=True: handles multiple requests simultaneously
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
