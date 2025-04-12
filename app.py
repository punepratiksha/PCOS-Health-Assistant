from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model
MODEL_FILENAME = "pcos_model.pkl"
model_path = os.path.join(os.getcwd(), MODEL_FILENAME)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found at {model_path}")

model = joblib.load(model_path)

# Map API input keys to actual model feature order
feature_map = {
    "age": "Age (yrs)",
    "bmi": "BMI",
    "fsh": "FSH(mIU/mL)",
    "lh": "LH(mIU/mL)",
    "irregular_periods": "irregular_periods",
    "acne": "Pimples(Y/N)",
    "hair_fall": "Hair loss(Y/N)"
}

expected_features = list(feature_map.keys())

# Treatment recommendation map
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

        # Validate inputs
        missing = [f for f in expected_features if f not in data]
        if missing:
            return jsonify({"error": f"Missing input(s): {missing}"}), 400

        # Build input array in correct order for the model
        try:
            input_values = [
                float(data.get(key)) for key in expected_features
            ]
        except ValueError:
            return jsonify({"error": "Invalid input types. Must be numeric."}), 400

        input_array = np.array(input_values).reshape(1, -1)

        # Prediction with probability
        prob = model.predict_proba(input_array)[0][1]
        prediction = 1 if prob > 0.45 else 0

        # Treatment logic
        recommendations = []
        for symptom in ["irregular_periods", "acne", "hair_fall"]:
            if str(data.get(symptom)) in ["1", "true", "True"]:
                recommendations.append(treatment_guidelines[symptom])

        return jsonify({
            "PCOS_Prediction": prediction,
            "Confidence": round(prob, 2),
            "Treatment_Recommendations": recommendations if prediction else "No treatment needed"
        })

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
