# basic front end
from flask import Flask, render_template, request, redirect, url_for
import os
import subprocess
import joblib
import re
import pandas as pd
from typing import Optional
from datetime import datetime
import logging
app = Flask(__name__)

# Configure a simple file logger for predictions and training

logging.basicConfig(
    filename='logs/app_activity.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ----- Helper to get client IP -----
def get_client_ip():
    """Return the client's IP address, accounting for proxies."""
    if request.environ.get('HTTP_X_FORWARDED_FOR'):
        return request.environ.get('HTTP_X_FORWARDED_FOR').split(',')[0].strip()
    return request.remote_addr or 'UNKNOWN'

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


MODEL_PATH = os.path.join("artifacts", "model_trainer", "model.joblib")
try:
    MODEL = joblib.load(MODEL_PATH)
    app.logger.info(f"Loaded model at startup from {MODEL_PATH}")
except Exception as e:
    app.logger.info(f"Model not found at startup ({MODEL_PATH}): {e}")
    MODEL = None

@app.route('/', methods=['GET'])
def home():
    """Homepage: display the input form. Show success message if training just completed."""
    training_status = request.args.get('training_status', None)
    return render_template('index.html', training_status=training_status)

@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    """
    Train the spam detection model.
    
    - Runs the full ML pipeline (data ingestion, validation, transformation, training, evaluation).
    - Saves the trained RandomForest model to artifacts/model_trainer/model.joblib.
    - Reloads the model into memory for use by /predict.
    - Logs training start/end with client IP and timestamp.
    - On success, redirects back to the homepage with a success message.
    - On failure, returns error with HTTP 500 status.
    """
    client_ip = get_client_ip()
    logging.info(f"TRAIN_START | IP: {client_ip}")
    
    # Run training synchronously and reload the model if created
    try:
        proc = subprocess.run([os.sys.executable, "main.py"], cwd=os.getcwd())
    except Exception as e:
        app.logger.exception(e)
        logging.error(f"TRAIN_FAILED | IP: {client_ip} | Error: {e}")
        return "Failed to start training", 500

    if proc.returncode != 0:
        logging.error(f"TRAIN_FAILED | IP: {client_ip} | Exit code: {proc.returncode}")
        return f"Training failed (exit code {proc.returncode})", 500

    # Attempt to reload the trained model
    global MODEL
    try:
        MODEL = joblib.load(MODEL_PATH) 
        logging.info(f"TRAIN_SUCCESS | IP: {client_ip} | Model loaded from {MODEL_PATH}")
        # Redirect back to index with success message as query param
        return redirect(url_for('home', training_status='success'))
    except Exception as e:
        app.logger.exception(e)
        logging.error(f"TRAIN_FAILED | IP: {client_ip} | Model load failed: {e}")
        MODEL = None
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
    """
    Predict spam/ham classification for a given message.
    
    GET: Return the input form (index.html).
    POST: 
      1. Extract message from form.
      2. If no model is loaded, trigger training automatically.
      3. Compute engineered features from the raw message.
      4. Call model.predict_proba() to get spam probability.
      5. Return result (spam/ham + confidence) or error message.
      6. Log all predictions with client IP, message snippet, and result.
    """
    if request.method == 'POST':
        message = request.form.get('message', '').strip()
        client_ip = get_client_ip()
        message_snippet = (message[:50] + '...') if len(message) > 50 else message
        
        if not message:
            logging.warning(f"PREDICT_EMPTY | IP: {client_ip} | Empty message submitted")
            return render_template('index.html')

        # If model not loaded yet, run training to produce the artifact and reload
        global MODEL
        if MODEL is None:
            logging.info(f"PREDICT_NO_MODEL | IP: {client_ip} | Auto-triggering training")
            try:
                proc = subprocess.run([os.sys.executable, "main.py"], cwd=os.getcwd())
            except Exception as e:
                app.logger.exception(e)
                logging.error(f"PREDICT_TRAIN_FAILED | IP: {client_ip} | Error: {e}")
                return render_template('result.html', error='Failed to start training')

            if proc.returncode != 0:
                logging.error(f"PREDICT_TRAIN_FAILED | IP: {client_ip} | Exit code: {proc.returncode}")
                return render_template('result.html', error=f'Training failed (exit {proc.returncode})')

            try:
                MODEL = joblib.load(MODEL_PATH)
                logging.info(f"PREDICT_MODEL_LOADED | IP: {client_ip} | After training")
            except Exception as e:
                app.logger.exception(e)
                logging.error(f"PREDICT_MODEL_LOAD_FAILED | IP: {client_ip} | Error: {e}")
                return render_template('result.html', error='Training finished but model could not be loaded')

        # At this point MODEL should be available
        try:
            proba = _model_predict_proba_for_message(MODEL, message)
        except Exception as e:
            app.logger.exception(e)
            logging.error(f"PREDICT_ERROR | IP: {client_ip} | Message: '{message_snippet}' | Error: {e}")
            proba = None

        if proba is None:
            logging.error(f"PREDICT_FAILED | IP: {client_ip} | Message: '{message_snippet}' | Proba is None")
            return render_template('result.html', error='Prediction failed; model may expect engineered features')

        label = "spam" if proba >= 0.5 else "ham"
        logging.info(f"PREDICT_SUCCESS | IP: {client_ip} | Message: '{message_snippet}' | Label: {label} | Confidence: {proba:.4f}")
        return render_template('result.html', label=label, proba=f"{proba:.4f}", message=message)
    # GET
    return render_template('index.html')
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)