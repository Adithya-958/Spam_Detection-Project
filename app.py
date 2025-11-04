#basic front end
from flask import Flask, render_template, request
import os
import json
import joblib
import numpy as np
from typing import Optional

from src.Spam_Detection_Project.utils.feature_engineering import compute_features, feature_columns


app = Flask(__name__)


def _find_and_load_model() -> Optional[object]:
    """Try to find a model artifact in common locations and load it with joblib.

    Returns the loaded model or None if not found.
    """
    candidates = [
        os.path.join("artifacts", "model", "model.joblib"),
        os.path.join("artifacts", "model", "full_pipeline.joblib"),
        os.path.join("src", "Spam_Detection_Project", "artifacts", "model_trainer", "model.joblib"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return joblib.load(p)
            except Exception:
                # continue searching
                pass
    return None


# Load once at startup
MODEL = _find_and_load_model()


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/train', methods=['GET'])
def training():
    # Run training synchronously (keeps original behavior).
    os.system("python main.py")
    return "Training is successful"


def _model_predict_proba(model, X: np.ndarray) -> Optional[float]:
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            # predict_proba returns shape (n_samples, n_classes)
            # we want a spam probability — training used label 0 for spam, 1 for ham.
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 0 in classes:
                    idx = classes.index(0)
                    return float(probs[0, idx])
            # fallback: choose max prob of first column
            return float(probs[0, 0])
        else:
            return None
    except Exception:
        return None


def _heuristic_score(features: dict) -> float:
    # Simple score: each keyword/link/phone/amount adds weight
    score = 0
    score += features.get("Keywords", 0) * 2
    score += features.get("contains_URL_link", 0) * 2
    score += features.get("contains_phone_number", 0) * 1
    score += features.get("Special_Characters", 0) * 1
    score += features.get("Amount", 0) * 1
    # penalize long messages (less likely spam)
    if features.get("word_count", 0) > 50:
        score -= 1
    # normalize into 0..1 range roughly
    proba = min(max(score / 5.0, 0.0), 1.0)
    return proba


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')

    # POST
    # Support form POST (index.html) and JSON POST for API clients
    message = None
    if request.form and 'message' in request.form:
        message = request.form.get('message', '').strip()
    else:
        try:
            body = request.get_json(silent=True)
            if body and 'message' in body:
                message = str(body['message']).strip()
        except Exception:
            message = None

    if not message:
        return render_template('result.html', error='Please provide a message to classify')

    # Compute handcrafted features
    feats = compute_features(message)
    cols = feature_columns()

    # If a model is available, try to use it
    if MODEL is not None:
        try:
            X = np.array([[feats[c] for c in cols]])
            proba = _model_predict_proba(MODEL, X)
            if proba is None:
                # fall back to predict()
                pred = MODEL.predict(X)[0]
                # map pred to probability-like value
                proba = 1.0 if int(pred) == 0 else 0.0
            label = 'spam' if proba > 0.5 else 'ham'
            return render_template('result.html', label=label, proba=round(float(proba), 4), message=message)
        except Exception:
            # fall through to heuristic
            pass

    # No model — use a heuristic
    proba = _heuristic_score(feats)
    label = 'spam' if proba > 0.5 else 'ham'
    return render_template('result.html', label=label, proba=round(float(proba), 4), message=message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)