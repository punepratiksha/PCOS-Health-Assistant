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

        # Get prediction and confidence
        prediction = model.predict(input_array)[0]
        proba = model.predict_proba(input_array)[0][prediction]
        confidence = round(proba, 2)

        # Generate treatment or preventive recommendations
        if prediction == 1:
            # PCOS detected
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
            # No PCOS — share general preventive tips
            recommendations = [
                "• Maintain a balanced diet to reduce hormonal imbalances.",
                "• Exercise regularly to stay healthy and manage weight.",
                "• Manage stress through yoga, meditation, or counseling.",
                "• Go for routine health checkups to monitor hormonal levels."
            ]

        recommendations_str = "\n".join(recommendations)

        response = {
            "PCOS_Prediction": int(prediction),
            "Confidence": confidence,
            "Treatment_Recommendations": recommendations_str
        }

        return jsonify(response)

    except Exception as e:
        print(f"❌ ERROR in /predict: {e}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
