
import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=BASE_DIR)

# Load the trained model
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

# The model expects 14 numerical inputs.
# Verify this order against the notebook training columns.
FEATURE_COLUMNS = [
    "student_id",
    "age",
    "gender",
    "gaming_hours",
    "study_hours",
    "sleep_hours",
    "attendance",
    "gaming_genre",
    "social_activity",
    "device_usage",
    "reaction_time_ms",
    "addiction_score",
    "grades",
    "genre_label"
]

STRESS_LABELS = {
    0: "Low",
    1: "Medium",
    2: "High"
}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Please send JSON data"
            }), 400

        features = []

        for column in FEATURE_COLUMNS:
            if column not in data or data[column] == "":
                return jsonify({
                    "error": f"Missing field: {column}"
                }), 400

            features.append(float(data[column]))

        X = np.array([features], dtype=float)

        prediction = model.predict(X)[0]
        prediction = int(prediction)

        return jsonify({
            "prediction": prediction,
            "stress_level": STRESS_LABELS.get(
                prediction, str(prediction)
            )
        })

    except (ValueError, TypeError) as e:
        return jsonify({
            "error": str(e)
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check Flask terminal."
        }), 500


if __name__ == "__main__":
    print("Model loaded successfully")
    print("Model features:", model.n_features_in_)
    app.run(debug=True)
