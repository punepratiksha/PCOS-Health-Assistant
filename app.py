from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Dynamically locate model file in deployment
MODEL_FILENAME = "pcos_model.pkl"
model_path = os.path.join(os.getcwd(), MODEL_FILENAME)  # FIX: Using os.getcwd() instead

if not os.path.exists(model_path):
    print(f"❌ ERROR: Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")

try:
    print("✅ Loading model...")
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
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
    return jsonify({"message": "Flask PCOS Prediction API is Running!"})  # FIX: JSON response

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Check for missing features
        missing_features = [feature for feature in expected_features if feature not in data]
        if missing_features:
            return jsonify({"error": f"Missing required features: {missing_features}"}), 400

        # Convert input data to float values
        try:
            input_features = [float(data.get(feature, 0)) for feature in expected_features]

        except ValueError:
            return jsonify({"error": "Invalid input types. All values must be numeric"}), 400

        input_array = np.array(input_features).reshape(1, -1)

        # Ensure model has probability prediction capability
        if not hasattr(model, "predict_proba"):
            return jsonify({"error": "Model does not support probability prediction"}), 500

        # Get prediction probability
        prob = model.predict_proba(input_array)[0][1]
        prediction = 1 if prob > 0.45 else 0

        # Get treatment recommendations
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
        print(f"❌ ERROR in /predict: {e}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # FIX: Render uses port 10000
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
