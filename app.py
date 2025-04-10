from flask import Flask, request, jsonify
import numpy as np
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load your trained model
model = pickle.load(open('pcos_model.pkl', 'rb'))

# Define expected input features
expected_features = ["age", "bmi", "fsh", "lh", "irregular_periods", "acne", "hair_fall"]

# Define symptom-based treatment guidelines
treatment_guidelines = {
    "irregular_periods": "Track your cycles, Take hormonal medication if prescribed, Avoid stress",
    "acne": "Use dermatologically approved medication, Maintain hygiene, Avoid oily foods",
    "hair_fall": "Take biotin supplements, Consult a trichologist, Use gentle hair care products"
}

# Your route and prediction function (no changes needed here)
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
            input_features = [float(data.get(feature, 0)) for feature in expected_features]
        except ValueError:
            return jsonify({"error": "Invalid input types. All values must be numeric"}), 400

        input_array = np.array(input_features).reshape(1, -1)

        if not hasattr(model, "predict_proba"):
            return jsonify({"error": "Model does not support probability prediction"}), 500

        prediction = model.predict(input_array)[0]
        proba = model.predict_proba(input_array)[0][prediction]
        confidence = round(proba, 2)

        if prediction == 1:
            recommendations = [
                f"• {line}" for symptom in expected_features[4:]
                if str(data.get(symptom, "0")).lower() in ["1", "true"]
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

        response = {
            "PCOS_Prediction": int(prediction),
            "Confidence": confidence,
            "Treatment_Recommendations": "\n".join(recommendations)
        }

        return jsonify(response)

    except Exception as e:
        print(f"❌ ERROR in /predict: {e}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# Required to run app locally or via WSGI like gunicorn
if __name__ == '__main__':
    app.run(debug=True)
