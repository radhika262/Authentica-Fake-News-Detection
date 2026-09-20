import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
from pprint import pprint  

# --- MONGO CONFIG (minimal) ---
# Replace with your Atlas URI or local URI as needed
MONGO_URI = "mongodb+srv://devanshkapoor5_fake_news:fake-news-26180209@cluster0.npvkut6.mongodb.net/?retryWrites=true&w=majority"

mongo_client = MongoClient(MONGO_URI)           # lazy connect
mongo_db = mongo_client["fake_news_db"]        # choose a database name
predictions_col = mongo_db["predictions"]      # collection to store predictions

# create simple index (idempotent) for faster recent queries
try:
    predictions_col.create_index([("created_at", -1)])
except Exception as e:
    print("Warning: could not create index on predictions_col:", e)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Map short name -> filename
MODEL_FILES = {
    'lr': 'fake_news_pipeline_lr.joblib',
    'dt': 'fake_news_pipeline_dt.joblib',
    'gb': 'fake_news_pipeline_gb.joblib',
    'rf': 'fake_news_pipeline_rf.joblib'
}

# Optional label mapping (adjust if your model uses different labels)
LABEL_MAP = {
    0: 'real',
    1: 'fake'
}

app = Flask(__name__)
CORS(app)  # dev: allow requests from frontend; restrict in production

# Load all available models into memory
models = {}
for key, fname in MODEL_FILES.items():
    path = os.path.join(MODEL_DIR, fname)
    if os.path.exists(path):
        try:
            models[key] = joblib.load(path)
            print(f"Loaded model [{key}] from {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    else:
        print(f"Model file not found: {path}")

print("Models Loaded:")
pprint({k: type(v).__name__ for k, v in models.items()})

@app.route('/', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'available_models': list(models.keys())
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not models:
        return jsonify({'error': 'no models loaded on server'}), 500

    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({'error': 'invalid or missing JSON body'}), 400

    print(body)
    text = body.get('text') or body.get('input') or ""
    model_name = body.get('model', 'lr')  # default to lr
    if not text:
        return jsonify({'error': 'no text provided'}), 400
    if model_name not in models:
        return jsonify({'error': 'unknown model', 'available': list(models.keys())}), 400

    pipeline = models[model_name]
    try:
        # ---- prediction ----
        pred = pipeline.predict([text])[0]
        raw_prediction = str(pred)

        # friendly label
        if isinstance(pred, (int, float)):
            label = LABEL_MAP.get(int(pred), raw_prediction)
        else:
            label = raw_prediction

        # probabilities (optional)
        try:
            probs = pipeline.predict_proba([text])[0].tolist()
            confidence = float(max(probs))
        except Exception:
            probs = None
            confidence = None

        # build response
        result = {
            'model': model_name,
            'raw_prediction': raw_prediction,
            'label': label,
            'probabilities': probs,
            'confidence': confidence
        }

        # ---- persist to MongoDB (best-effort) ----
        try:
            doc = {
                "text": text,
                "model": model_name,
                "label": label,
                "raw_prediction": raw_prediction,
                "probabilities": probs,
                "confidence": confidence,
                "created_at": datetime.utcnow()
            }
            ins = predictions_col.insert_one(doc)
            result['saved_id'] = str(ins.inserted_id)
        except Exception as db_e:
            # do not fail the prediction if DB insert fails; include the error for debugging
            result['db_error'] = str(db_e)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': 'prediction failed', 'details': str(e)}), 500

if __name__ == "__main__":
    # dev server; use a proper WSGI server for production
    app.run(debug=True, host='127.0.0.1', port=5000)