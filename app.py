from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Dynamically locate model file in deployment
MODEL_FILENAME = "pcos_model.pkl"
model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Define expected features
expected_features = ["age", "bmi", "fsh", "lh", "irregular_periods", "acne", "hair_fall"]

# Treatment recommendations
treatment_guidelines = {
    "irregular_periods": "Lifestyle changes, Hormone therapy, Regular exercise",
    "acne": "Topical retinoids, Oral contraceptives, Dermatologist consultation",
    "hair_fall": "Anti-androgen medications, Nutritional supplements, PRP therapy"
}

@app.route("/")
def home():
    return "Flask PCOS Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        missing_features = [feature for feature in expected_features if feature not in data]
        if missing_features:
            return jsonify({"error": f"Missing required features: {missing_features}"}), 400

        try:
            input_features = [float(data.get(feature, 0)) for feature in expected_features[:4]]
        except ValueError:
            return jsonify({"error": "Invalid input types. All values must be numeric"}), 400

        input_array = np.array(input_features).reshape(1, -1)

        if not hasattr(model, "predict_proba"):
            return jsonify({"error": "Model does not support probability prediction"}), 500

        prob = model.predict_proba(input_array)[0][1]
        prediction = 1 if prob > 0.45 else 0

        recommendations = [
            treatment_guidelines[symptom]
            for symptom in expected_features[4:]
            if str(data.get(symptom, "0")).lower() in ["1", "true"]
        ]

        response = {
            "PCOS_Prediction": prediction,
            "Confidence": round(prob, 2),
            "Treatment_Recommendations": recommendations if prediction == 1 else "No treatment needed"
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
