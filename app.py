from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('pcos_model.pkl', 'rb'))

# Expected input features
expected_features = ["age", "bmi", "fsh", "lh", "irregular_periods", "acne", "hair_fall"]

# Treatment recommendations
treatment_guidelines = {
    "irregular_periods": "Track your cycles, Take hormonal medication if prescribed, Avoid stress",
    "acne": "Use dermatologically approved medication, Maintain hygiene, Avoid oily foods",
    "hair_fall": "Take biotin supplements, Consult a trichologist, Use gentle hair care products"
}

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "PCOS Prediction API is running."})

@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Check for missing features
        missing = [f for f in expected_features if f not in data]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # Parse numerical inputs
        numerical_features = ["age", "bmi", "fsh", "lh"]
        try:
            input_data = [float(data[feature]) for feature in numerical_features]
        except ValueError:
            return jsonify({"error": "Numerical features must be valid numbers"}), 400

        # Parse binary symptom features
        symptom_features = ["irregular_periods", "acne", "hair_fall"]
        try:
            symptom_data = [
                1.0 if str(data[feature]).lower() in ["1", "true", "yes"] else 0.0
                for feature in symptom_features
            ]
        except Exception:
            return jsonify({"error": "Symptom features must be 1/0 or true/false"}), 400

        # Final input array
        input_array = np.array(input_data + symptom_data).reshape(1, -1)

        if not hasattr(model, "predict_proba"):
            return jsonify({"error": "Model missing predict_proba method"}), 500

        prediction = model.predict(input_array)[0]
        proba = model.predict_proba(input_array)[0][prediction]
        confidence = round(proba, 2)

        # Prepare treatment
        if prediction == 1:
            recommendations = [
                f"• {line}" for symptom in symptom_features
                if str(data[symptom]).lower() in ["1", "true", "yes"]
                for line in treatment_guidelines[symptom].split(", ")
            ]
            if not recommendations:
                recommendations = [
                    "• Consult a gynecologist for a personalized treatment plan.",
                    "• Maintain a healthy lifestyle and regular menstrual tracking.",
                    "• Consider hormone evaluation and medical supervision."
                ]
        else:
            recommendations = [
                "• Maintain a balanced diet to reduce hormonal imbalances.",
                "• Exercise regularly to stay healthy and manage weight.",
                "• Manage stress through yoga, meditation, or counseling.",
                "• Go for routine health checkups to monitor hormonal levels."
            ]

        return jsonify({
            "PCOS_Prediction": int(prediction),
            "Confidence": confidence,
            "Treatment_Recommendations": "\n".join(recommendations)
        })

    except Exception as e:
        print(f"❌ ERROR in /predict: {e}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
