from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODEL_FILENAME = "pcos_model.pkl"
model_path = os.path.join(os.getcwd(), MODEL_FILENAME)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found at {model_path}")

model = joblib.load(model_path)

# Mapping from API input to model-trained feature names
feature_map = {
    "age": "Age (yrs)",
    "bmi": "BMI",
    "fsh": "FSH(mIU/mL)",
    "lh": "LH(mIU/mL)",
    "irregular_periods": "irregular_periods",
    "acne": "Pimples(Y/N)",
    "hair_fall": "Hair loss(Y/N)"
}

# Features expected by the model in this order
model_features_order = list(feature_map.values())

# Treatment recommendations
treatment_guidelines = {
    "irregular_periods": "Lifestyle changes, Hormone therapy, Regular exercise",
    "acne": "Topical retinoids, Oral contraceptives, Dermatologist consultation",
    "hair_fall": "Anti-androgen medications, Nutritional supplements, PRP therapy"
}

@app.route("/")
def home():
    return jsonify({"message": "Flask PCOS Prediction API is Running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Check for missing inputs
        missing = [key for key in feature_map if key not in data]
        if missing:
            return jsonify({"error": f"Missing input(s): {missing}"}), 400

        # Build input for the model in correct order
        try:
            input_values = [
                float(data.get(api_key)) for api_key in feature_map
            ]
        except ValueError:
            return jsonify({"error": "Invalid input types. Must be numeric."}), 400

        input_array = np.array(input_values).reshape(1, -1)

        # Predict and get confidence
        prob = model.predict_proba(input_array)[0][1]
        prediction = int(prob > 0.45)
        confidence = round(prob, 2)

        # Build recommendations
        recommendations = []
        for symptom in ["irregular_periods", "acne", "hair_fall"]:
            if str(data.get(symptom)).lower() in ["1", "true"]:
                recommendations.append(treatment_guidelines[symptom])

        return jsonify({
            "PCOS_Prediction": prediction,
            "Confidence": confidence,
            "Treatment_Recommendations": recommendations if prediction else [
                "Maintain a balanced diet to reduce hormonal imbalances.",
                "Exercise regularly to stay healthy and manage weight.",
                "Manage stress through yoga, meditation, or counseling.",
                "Go for routine health checkups to monitor hormonal levels."
            ]
        })

    except Exception as e:
        print(f"❌ ERROR in /predict: {e}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
