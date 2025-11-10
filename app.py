# basic front end
from flask import Flask, render_template, request
import os
import subprocess
import joblib
import re
import pandas as pd
from typing import Optional
import json
import logging
from datetime import datetime

app = Flask(__name__)
def setup_logging():
    # Create logs directory
    os.makedirs('logs', exist_ok=True)

    # Remove all handlers associated with the root logger object
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure the root logger with console and file handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/app_activity.log')
        ]
    )
# initialize logging before routes
setup_logging()


def _get_user_ip():
    """Return a best-effort client IP for the current request context."""
    try:
        # prefer X-Forwarded-For if behind a proxy
        forwarded = request.headers.get('X-Forwarded-For', None)
        if forwarded:
            # X-Forwarded-For can be a comma-separated list
            return forwarded.split(',')[0].strip()
    except Exception:
        pass
    return request.remote_addr or 'unknown'


def get_user_logger(user_ip: str):
    """Return a per-user logger that writes to logs/user_<ip>.log.

    Reuses the logger if already configured to avoid duplicate handlers.
    """
    # sanitize ip for filename
    safe_ip = user_ip.replace(':', '_').replace('.', '_')
    logger_name = f"user_{safe_ip}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    # create file handler for this user
    user_log_dir = os.path.join('logs')
    os.makedirs(user_log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(user_log_dir, f"{logger_name}.log"))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    return logger
# ----- Feature engineering helpers (same as in data_transformation.py) -----
def _has_phone_number(text: str) -> int:
    pattern = r"\b[\+]?[0-9][0-9\-]{9,}\b"
    return int(bool(re.search(pattern, str(text))))

def _has_link(text: str) -> int:
    pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return int(bool(re.search(pattern, str(text))))

def _keywords(text: str) -> int:
    keywords = [
        "मुफ़्त", "दावा", "जीत", "नकद", "प्रस्ताव", "सीमित", "पुरस्कार", "पैसा",
        "मौका", "लिखकर", "लाख", "stop", "हज़ार", "claim", "free",
        "urgent", "act now", "winner",
        'फ्री', 'जल्दी', 'लिमिटेड', 'विजेता', 'इनाम', 'ऑफर', 'कॉल', 'क्लिक', 'लकी',
        'खरीदें', 'बधाई', 'शीघ्र'
    ]
    txt = str(text).lower()
    for kw in keywords:
        if kw.lower() in txt:
            return 1
    return 0

def _special_characters(text: str) -> int:
    pattern = r"(?:₹|RS|INR|\$)\s*\d+(?:,\d+)*(?:\.\d{2})?|[!@#$%^&*(),.?\":{}|<>]"
    return int(bool(re.search(pattern, str(text))))

def _cash_amount(text: str) -> int:
    cash_keywords = ["1 लाख", "दस लाख", "1 हज़ार", "दस हज़ार", "करोड़", "दस करोड़", "मिलियन", "बिलियन", "सौ", "लाख", "हज़ार"]
    txt = str(text)
    for c in cash_keywords:
        if c in txt:
            return 1
    return 0

def _length_sms(text: str) -> int:
    return len(str(text))

def _sms_number(text: str) -> int:
    pattern = r"\b[5-9]\d{4,5}\b"
    return int(bool(re.search(pattern, str(text))))

def _word_count(text: str) -> int:
    return len(str(text).split())

def _compute_features_from_message(message: str) -> pd.DataFrame:
    """Given a raw message string, compute the same engineered features as training.
    
    Returns a DataFrame with one row and all feature columns, ready to pass to model.predict_proba.
    The order and names match what the trainer expects.
    """
    features = {
        'contains_phone_number': [_has_phone_number(message)],
        'contains_URL_link': [_has_link(message)],
        'Keywords': [_keywords(message)],
        'Special_Characters': [_special_characters(message)],
        'Amount': [_cash_amount(message)],
        'Length': [_length_sms(message)],
        'SMS_Number': [_sms_number(message)],
        'word_count': [_word_count(message)],
    }
    return pd.DataFrame(features)


def _calculate_proba_from_features(features_dict: dict, label: str) -> float:
    """Calculate confidence as average of feature values.
    
    For spam label: higher average = higher spam confidence.
    For ham label: lower average = higher ham confidence (so we return 100 - avg).
    """
    feature_values = list(features_dict.values())
    avg_features = sum(feature_values) / len(feature_values) if feature_values else 0
    
    if label == 'spam':
        # Higher features -> higher spam confidence
        proba = min(100, avg_features * 12.5)  # Scale to 0-100
    else:
        # Lower features -> higher ham confidence
        proba = max(0, 100 - avg_features * 12.5)
    
    return round(proba, 2)


MODEL_PATH = os.path.join("artifacts", "model_trainer", "model.joblib")
MODEL_TRAINED_AT = None  # Track when model was last trained
MODEL_ACCURACY = None  # Track model accuracy from training
# Use the evaluation metrics file produced by the pipeline
METRICS_FILE = os.path.join("artifacts", "model_evaluation", "metrics.json")

def load_model_accuracy() -> Optional[float]:
    """Load model accuracy from the pipeline's metrics.json if present.

    Expects JSON with an 'accuracy' key (value between 0 and 1).
    Returns percentage (0-100) or None on failure.
    """
    try:
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'accuracy' in data:
                    acc = float(data['accuracy'])
                    return round(acc * 100, 2)
    except Exception as e:
        app.logger.warning(f"Could not load accuracy from {METRICS_FILE}: {e}")
    return None


try:
    MODEL = joblib.load(MODEL_PATH)
    MODEL_TRAINED_AT = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).strftime('%Y-%m-%d %H:%M:%S')
    MODEL_ACCURACY = load_model_accuracy()
    app.logger.info(f"Loaded model at startup from {MODEL_PATH} (trained at {MODEL_TRAINED_AT}, accuracy {MODEL_ACCURACY}%)")
except Exception as e:
    app.logger.info(f"Model not found at startup ({MODEL_PATH}): {e}")
    MODEL = None
    MODEL_TRAINED_AT = None
    MODEL_ACCURACY = None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    # Run training synchronously and reload the model if created
    user_ip = _get_user_ip()
    ulog = get_user_logger(user_ip)
    ulog.info("Training requested by user")
    try:
        proc = subprocess.run([os.sys.executable, "main.py"], cwd=os.getcwd())
    except Exception as e:
        app.logger.exception(e)
        ulog.exception(f"Failed to start training: {e}")
        return "Failed to start training", 500

    if proc.returncode != 0:
        ulog.error(f"Training exited with code {proc.returncode}")
        return f"Training failed (exit code {proc.returncode})", 500

    # Attempt to reload the trained model
    global MODEL, MODEL_TRAINED_AT, MODEL_ACCURACY
    try:
        MODEL = joblib.load(MODEL_PATH)
        MODEL_TRAINED_AT = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        MODEL_ACCURACY = load_model_accuracy()
        ulog.info("Training completed and model loaded")
        return render_template('index.html', message="Training Successful, model loaded and ready to predict")
    except Exception as e:
        app.logger.exception(e)
        ulog.exception(f"Training finished but failed to load model artifact: {e}")
        MODEL = None
        MODEL_TRAINED_AT = None
        MODEL_ACCURACY = None
        return "Training finished but failed to load model artifact", 500

def _model_predict_proba_for_message(model, message: str) -> Optional[float]:
    try:
        if hasattr(model, "predict_proba"):
            # The model expects engineered features, not raw strings.
            # Compute the same features the trainer used.
            X_features = _compute_features_from_message(message)
            probs = model.predict_proba(X_features)
            # find spam class probability; training used label 0 for spam
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 0 in classes:
                    idx = classes.index(0)
                    return float(probs[0, idx])
            return float(probs[0, 0])

        if hasattr(model, "predict"):
            X_features = _compute_features_from_message(message)
            pred = model.predict(X_features)[0]
            return 1.0 if int(pred) == 0 else 0.0

    except Exception as e:
        app.logger.debug(f"predict_proba failed for raw message: {e}")
        return None
    return None


@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        message = request.form.get('message', '').strip()
        if not message:
            return render_template('index.html')

        # create per-user logger
        user_ip = _get_user_ip()
        ulog = get_user_logger(user_ip)
        ulog.info(f"Prediction requested; message length={len(message)}")

        global MODEL, MODEL_TRAINED_AT
        
        # If no model available, show error and tell user to train
        if MODEL is None:
            ulog.warning('No model loaded; user needs to train')
            return render_template('result.html', error='No model available. Please go to <a href="/train">/train</a> to train the model first.')

        # At this point MODEL should be available
        try:
            proba = _model_predict_proba_for_message(MODEL, message)
        except Exception as e:
            app.logger.exception(e)
            ulog.exception(f"Prediction exception: {e}")
            proba = None

        if proba is None:
            ulog.warning('Prediction failed for message')
            return render_template('result.html', error='Prediction failed; model may expect engineered features')

        label = "spam" if proba >= 0.5 else "ham"
        
        # Calculate confidence as average of features
        features_df = _compute_features_from_message(message)
        features_dict = features_df.iloc[0].to_dict()
        confidence = _calculate_proba_from_features(features_dict, label)
        
        ulog.info(f"Prediction result: label={label}, confidence={confidence}%")
        return render_template('result.html', label=label, proba=confidence, message=message, 
                             trained_at=MODEL_TRAINED_AT, model_accuracy=MODEL_ACCURACY)
    # GET
    return render_template('index.html')
    

# ... all your routes and functions ...
if __name__ == '__main__':
    try:
        print("Attempting to start Flask server...")
        app.run(host='0.0.0.0', port=8080, debug=False)  # Use debug=True for more info
    except Exception as e:
        print(f"Failed to start server: {e}")